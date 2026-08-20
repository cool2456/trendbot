"""Experiment 003's two new diagnostics, and its decision rule.

Everything experiment 002 built is reused rather than reimplemented - the noise
generators, the engine, the bucket study, ``worst_months`` - through the generalised
entry points in :mod:`trendbot.engine.xs_validation`. What is genuinely new here is:

1. :func:`market_regression`. PREREG_003.md section 7.5 promotes the market-beta
   attribution from a diagnostic to **standard equipment**, and section 8 turns it into
   both a support clause (positive alpha at t > 2) and an abandon clause (negative
   alpha, whatever the Sharpe). Experiment 002's :func:`factor_attribution` regressed on
   an equal-weight basket of synthetic instruments across seeds; this regresses one real
   return series on one real market series and reports the standard errors, which is a
   different calculation and needs its own implementation.

2. :func:`evaluate_decision_rule_003`. Section 8 gained two clauses over 002's rule -
   a t-statistic on the spread, and the alpha condition - and relaxed the monotonicity
   requirement to tolerate one inversion. Encoding that as a fresh function rather than
   parameterising 002's keeps each experiment's rule readable next to its own document.

**Which series is "the market".** Section 7.5 says "market excess returns" and does not
name an index. Two readings are defensible: SPY, the cap-weighted market, which is the
literal reading of "market beta" and the standard CAPM proxy; or the equal-weight
buy-and-hold of the same universe, which is what 002's diagnostic used and which nets
out the equal-weighting tilt. **Both are computed and reported. SPY is the headline and
section 8's clauses are adjudicated on it** - fixed before either number existed, so
neither can be selected after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..config_003 import Config003
from .metrics import TRADING_DAYS_PER_YEAR

__all__ = [
    "MarketRegression",
    "market_regression",
    "Decision003",
    "evaluate_decision_rule_003",
]


# --------------------------------------------------------------------------------------
# section 7.5 - market-beta attribution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketRegression:
    """OLS of strategy excess return on market excess return, with standard errors.

    ``r_strategy(t) = alpha + beta * r_market(t) + e(t)``, both series in excess of the
    same risk-free rate and both daily. ``alpha`` is reported per period and annualised
    by multiplying by 252, which is the convention used everywhere else in this package.

    The t-statistic on alpha is the plain OLS one. Daily strategy residuals are close
    enough to serially uncorrelated for this to be the honest simple answer; a
    Newey-West correction is not applied and its absence is stated rather than hidden,
    because it would widen the standard error and section 8 turns on ``t > 2``.
    """

    market_label: str
    n_observations: int
    alpha_per_period: float
    alpha_annualised: float
    alpha_t_stat: float
    alpha_p_value: float
    beta: float
    beta_t_stat: float
    r_squared: float
    residual_vol_annualised: float

    @property
    def alpha_positive(self) -> bool:
        return self.alpha_per_period > 0.0

    def clears(self, min_t: float) -> bool:
        """Section 8's clause: alpha positive AND its t-statistic above ``min_t``.

        A NaN t-statistic - a degenerate fit - never clears; the comparison would be
        False anyway, and stating it makes that deliberate rather than incidental.
        """
        return (
            self.alpha_positive
            and math.isfinite(self.alpha_t_stat)
            and self.alpha_t_stat > min_t
        )

    def __str__(self) -> str:
        return (
            f"vs {self.market_label}: beta {self.beta:+.3f} (t {self.beta_t_stat:+.1f}), "
            f"alpha {self.alpha_annualised:+.2%}/yr (t {self.alpha_t_stat:+.2f}, "
            f"p {self.alpha_p_value:.4f}), R² {self.r_squared:.3f}, "
            f"residual vol {self.residual_vol_annualised:.2%}"
        )


def market_regression(
    strategy_excess: pd.Series,
    market_excess: pd.Series,
    *,
    market_label: str,
) -> MarketRegression:
    """Regress strategy excess return on market excess return.

    Both series are aligned on their shared dates before fitting; a length mismatch is
    an alignment bug rather than something to paper over with a fill, so the
    intersection is used and its size reported.
    """
    frame = pd.concat(
        {"strategy": strategy_excess, "market": market_excess}, axis=1, join="inner"
    ).dropna()
    n = len(frame)
    if n < 3:
        raise ValueError(f"need at least 3 overlapping observations, got {n}")

    y = frame["strategy"].to_numpy(dtype=float)
    x = frame["market"].to_numpy(dtype=float)
    # A market series with no dispersion makes beta unidentified and the normal
    # equations singular; numpy would return an arbitrary solution with a NaN standard
    # error rather than complain. Section 8 abandons on the sign of alpha, so an alpha
    # whose standard error is NaN must be an error, not a number.
    market_sd = float(np.std(x, ddof=1))
    if market_sd <= max(float(np.max(np.abs(x))), 1.0) * 1e-12:
        raise ValueError(
            "the market series has no dispersion, so beta is unidentified and alpha "
            "cannot be separated from it"
        )
    design = np.column_stack([np.ones(n), x])

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta = float(coefficients[0]), float(coefficients[1])

    residuals = y - design @ coefficients
    degrees_of_freedom = n - 2
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    alpha_se, beta_se = float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1]))

    # A residual dispersion indistinguishable from floating-point noise makes every
    # t-statistic here a ratio of one rounding error to another. On an exact linear
    # relationship that produced alpha = 5e-19 at t = +3.8, which would have cleared
    # section 8's t > 2 clause on nothing at all. Same convention as
    # :func:`trendbot.engine.metrics.sharpe`: undefined, therefore NaN, never a large
    # number that looks like a result.
    scale = float(np.max(np.abs(y))) if n else 0.0
    if math.sqrt(residual_variance) <= max(scale, 1.0) * 1e-12:
        alpha_se = beta_se = float("nan")

    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else float("nan")

    alpha_t = alpha / alpha_se if np.isfinite(alpha_se) and alpha_se > 0 else float("nan")
    return MarketRegression(
        market_label=market_label,
        n_observations=n,
        alpha_per_period=alpha,
        alpha_annualised=alpha * TRADING_DAYS_PER_YEAR,
        alpha_t_stat=alpha_t,
        alpha_p_value=float(2.0 * stats.t.sf(abs(alpha_t), degrees_of_freedom))
        if np.isfinite(alpha_t)
        else float("nan"),
        beta=beta,
        beta_t_stat=beta / beta_se if np.isfinite(beta_se) and beta_se > 0 else float("nan"),
        r_squared=r_squared,
        residual_vol_annualised=math.sqrt(residual_variance * TRADING_DAYS_PER_YEAR),
    )


# --------------------------------------------------------------------------------------
# section 8 - the pre-committed decision rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision003:
    """The verdict, evaluated mechanically against PREREG_003.md section 8."""

    strategy_sharpe: float
    benchmark_sharpe: float
    n_inversions: int
    spread_mean: float
    spread_t_stat: float
    alpha_annualised: float
    alpha_t_stat: float
    verdict: str
    support_clauses: tuple[tuple[str, bool, str], ...]
    abandon_clauses: tuple[tuple[str, bool, str], ...]

    @property
    def supported(self) -> bool:
        return self.verdict == "SUPPORTED"

    def __str__(self) -> str:
        lines = [self.verdict, "  support clauses (all four required):"]
        lines += [
            f"    [{'YES' if ok else 'NO '}] {name} — {detail}"
            for name, ok, detail in self.support_clauses
        ]
        lines += ["  abandon clauses (any one is sufficient):"]
        lines += [
            f"    [{'YES' if ok else 'NO '}] {name} — {detail}"
            for name, ok, detail in self.abandon_clauses
        ]
        return "\n".join(lines)


def evaluate_decision_rule_003(
    cfg: Config003,
    *,
    strategy_sharpe: float,
    benchmark_sharpe: float,
    n_inversions: int,
    spread_mean: float,
    spread_t_stat: float,
    alpha_annualised: float,
    alpha_t_stat: float,
) -> Decision003:
    """Apply section 8 exactly as written, with no interpretation at the margin.

    Section 8, verbatim::

        Supported only if all four hold:
        - Net Sharpe (excess of T-bill, 10 bps) exceeds 0.40, AND
        - Net Sharpe exceeds equal-weight buy-and-hold of the same universe by at
          least 0.15, AND
        - Decile monotonicity: forward returns decrease monotonically D1 -> D10,
          allowing at most one adjacent inversion, with D1-D10 spread positive at
          t > 2.0, AND
        - Alpha to market beta is positive with t > 2.0.

        Abandon if any of:
        - Fails to beat equal-weight buy-and-hold at all, OR
        - More than one decile inversion, or D1-D10 t-statistic below 1.0, OR
        - Alpha to market is negative, OR
        - Net Sharpe below 0.15.

    Comparators follow the document's own words: "exceeds" and "above" are strict,
    "at least" and "at most" are inclusive, "below" is strict.
    """
    exceeds_min = strategy_sharpe > cfg.support_min_sharpe
    margin = strategy_sharpe - benchmark_sharpe
    beats_by_margin = margin >= cfg.support_min_sharpe_excess_over_buy_and_hold
    beats_at_all = strategy_sharpe > benchmark_sharpe

    inversions_ok = n_inversions <= cfg.max_inversions
    spread_ok = spread_mean > 0.0 and spread_t_stat > cfg.min_spread_t_stat
    monotonicity_ok = inversions_ok and spread_ok

    alpha_positive = alpha_annualised > 0.0
    alpha_ok = alpha_positive and alpha_t_stat > cfg.min_alpha_t_stat

    below_abandon_sharpe = strategy_sharpe < cfg.abandon_below_sharpe
    too_many_inversions = n_inversions > cfg.max_inversions
    spread_t_too_low = spread_t_stat < cfg.abandon_below_spread_t_stat
    alpha_negative = alpha_annualised < 0.0

    supported = exceeds_min and beats_by_margin and monotonicity_ok and alpha_ok
    abandon = (
        (not beats_at_all)
        or too_many_inversions
        or spread_t_too_low
        or alpha_negative
        or below_abandon_sharpe
    )

    support_clauses = (
        (
            f"net Sharpe exceeds {cfg.support_min_sharpe:.2f}",
            exceeds_min,
            f"{strategy_sharpe:+.3f}",
        ),
        (
            f"exceeds buy-and-hold by at least {cfg.support_min_sharpe_excess_over_buy_and_hold:+.2f}",
            beats_by_margin,
            f"margin {margin:+.3f} ({strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f})",
        ),
        (
            f"decile monotonicity: at most {cfg.max_inversions} inversion and spread "
            f"positive at t > {cfg.min_spread_t_stat:.1f}",
            monotonicity_ok,
            f"{n_inversions} inversion(s), spread {spread_mean:+.4%}/month at "
            f"t {spread_t_stat:+.3f}",
        ),
        (
            f"alpha to market positive at t > {cfg.min_alpha_t_stat:.1f}",
            alpha_ok,
            f"alpha {alpha_annualised:+.2%}/yr at t {alpha_t_stat:+.3f}",
        ),
    )
    abandon_clauses = (
        (
            "fails to beat equal-weight buy-and-hold at all",
            not beats_at_all,
            f"{strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f}",
        ),
        (
            f"more than {cfg.max_inversions} decile inversion",
            too_many_inversions,
            f"{n_inversions} inversion(s)",
        ),
        (
            f"D1-D10 t-statistic below {cfg.abandon_below_spread_t_stat:.1f}",
            spread_t_too_low,
            f"t {spread_t_stat:+.3f}",
        ),
        ("alpha to market is negative", alpha_negative, f"{alpha_annualised:+.2%}/yr"),
        (
            f"net Sharpe below {cfg.abandon_below_sharpe:.2f}",
            below_abandon_sharpe,
            f"{strategy_sharpe:+.3f}",
        ),
    )

    if supported and abandon:  # pragma: no cover - section 8 makes this unreachable
        verdict = "CONTRADICTORY - the decision rule is internally inconsistent here"
    elif supported:
        verdict = "SUPPORTED"
    elif abandon:
        verdict = "ABANDON"
    else:
        verdict = "INCONCLUSIVE - do not trade"

    return Decision003(
        strategy_sharpe=strategy_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        n_inversions=n_inversions,
        spread_mean=spread_mean,
        spread_t_stat=spread_t_stat,
        alpha_annualised=alpha_annualised,
        alpha_t_stat=alpha_t_stat,
        verdict=verdict,
        support_clauses=support_clauses,
        abandon_clauses=abandon_clauses,
    )
