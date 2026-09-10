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


@dataclass(frozen=True, slots=True)
class MarketRegression:
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
    frame = pd.concat(
        {"strategy": strategy_excess, "market": market_excess}, axis=1, join="inner"
    ).dropna()
    n = len(frame)
    if n < 3:
        raise ValueError(f"need at least 3 overlapping observations, got {n}")

    y = frame["strategy"].to_numpy(dtype=float)
    x = frame["market"].to_numpy(dtype=float)
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


@dataclass(frozen=True, slots=True)
class Decision003:
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

    if supported and abandon:  # pragma: no cover
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
