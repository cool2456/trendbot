"""Experiment 005's two new pieces, and its decision rule.

Everything experiments 002 and 003 built is reused rather than reimplemented - the
noise generators, the engine, :func:`~trendbot.engine.xs_validation.bucket_study`,
:func:`~trendbot.engine.xs_validation.worst_months`, and
:func:`~trendbot.engine.eq_validation.market_regression`. What is genuinely new here is:

1. :func:`worst_month_clustering`. PREREG_005.md section 9 removes the mandatory crash
   month that 002 and 003 both carried - "currency momentum has no single canonical
   crash date" - and replaces it with a weaker but still falsifiable statement: the five
   worst months should cluster around known FX dislocations, and **the absence of any
   clustering is a warning about the implementation**. That is a diagnostic with a
   stated failure direction, so it is computed rather than eyeballed.

2. :func:`carry_adjusted_prices` and :func:`short_rate_panel`. Section 4 requires a
   diagnostic that re-runs the headline with the interest-rate component the spot-only
   return definition leaves out. It is explicitly **not a configuration**: the signal
   is not recomputed, the same target weights are replayed against a different price
   panel, and the difference is reported. See :func:`carry_adjusted_prices` for why
   that is the arithmetic that isolates the approximation.

3. :func:`evaluate_decision_rule_005`. Section 8's four support clauses and five
   abandon clauses, applied mechanically. Structurally this is section 8 of PREREG_003
   with the market proxy replaced by the dollar factor, but it is written out rather
   than parameterised, for the reason :mod:`trendbot.engine.eq_validation` gives: each
   experiment's rule should be readable next to its own document.

Which series is "the dollar factor"
------------------------------------
Section 5 defines it as "equal-weight long all currencies in the universe against USD"
and names it the dollar factor. Three constructions are defensible - buy-and-hold,
rebalanced on the strategy's monthly schedule, or rebalanced every period - and they
are not the same series. **The daily equal-weighted mean of the normalised currency
returns is the headline**, fixed before any number existed. It is the construction
Lustig/Roussanov/Verdelhan and Menkhoff et al. mean by DOL, which is what section 5's
parenthetical invokes; and being path-independent it lets section 5's Sharpe comparison
and section 8's alpha regression run against literally the same object rather than two
subtly different ones. It lives in :func:`trendbot.fx.dollar_factor_returns`. The other
two constructions are reported alongside so the choice can be seen rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config_005 import Config005
from .metrics import TRADING_DAYS_PER_YEAR

__all__ = [
    "MonthClustering",
    "worst_month_clustering",
    "ShortRateCoverage",
    "short_rate_panel",
    "carry_adjusted_prices",
    "Decision005",
    "evaluate_decision_rule_005",
]


# --------------------------------------------------------------------------------------
# 1. section 9 - do the worst months land on known FX dislocations?
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonthClustering:
    """Whether the worst months coincide with the dislocations section 9 names.

    ``matched`` pairs each worst month with the dislocation label whose month window it
    falls in. ``n_matched == 0`` is section 9's warning condition: it does not fail the
    experiment, but it says the implementation is suspect and must be reported as such
    rather than passed over.
    """

    worst: pd.DataFrame
    matched: tuple[tuple[str, float, str | None], ...]
    windows: dict[str, tuple[str, ...]]

    @property
    def n_matched(self) -> int:
        return sum(1 for _, _, label in self.matched if label is not None)

    @property
    def any_clustering(self) -> bool:
        return self.n_matched > 0

    @property
    def labels_hit(self) -> tuple[str, ...]:
        seen: list[str] = []
        for _, _, label in self.matched:
            if label is not None and label not in seen:
                seen.append(label)
        return tuple(seen)

    def __str__(self) -> str:
        if self.any_clustering:
            return (
                f"{self.n_matched} of {len(self.matched)} worst months fall on a named FX "
                f"dislocation ({', '.join(self.labels_hit)})"
            )
        return (
            f"NONE of the {len(self.matched)} worst months falls on a named FX dislocation "
            "— section 9 calls this a warning sign about the implementation"
        )


def worst_month_clustering(worst: pd.DataFrame, windows: dict[str, tuple[str, ...]]) -> MonthClustering:
    """Match each of the worst months against section 9's dislocation windows."""
    lookup = {month: label for label, months in windows.items() for month in months}
    matched = tuple(
        (str(period), float(row["return"]), lookup.get(str(period)))
        for period, row in worst.iterrows()
    )
    return MonthClustering(worst=worst, matched=matched, windows=dict(windows))


# --------------------------------------------------------------------------------------
# 2. section 4's required diagnostic - the interest-rate approximation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShortRateCoverage:
    """Which currencies got a FRED short rate, which did not, and over what span."""

    rates: pd.DataFrame  # calendar x series_id, annualised decimal foreign short rate
    sources: dict[str, str]  # series_id -> the FRED series used
    covered: tuple[str, ...]
    uncovered: tuple[str, ...]
    spans: dict[str, tuple[str, str]]
    coverage_fraction: dict[str, float]

    def __str__(self) -> str:
        return (
            f"short rates found for {len(self.covered)} of "
            f"{len(self.covered) + len(self.uncovered)} currencies; "
            f"missing: {', '.join(self.uncovered) if self.uncovered else 'none'}"
        )


def short_rate_panel(
    fetched: dict[str, tuple[str, pd.Series]],
    universe: tuple[str, ...],
    calendar: pd.DatetimeIndex,
) -> ShortRateCoverage:
    """Align whatever short rates were obtained onto the panel calendar.

    Rates arrive at mixed frequencies - most of what FRED carries for these economies is
    monthly - so each is forward-filled onto the daily calendar. Forward filling a
    *rate* is not the same kind of assumption as forward filling a price: a policy or
    money-market rate genuinely is the prevailing rate until the next observation, and
    the fill is backwards-looking either way.

    A currency with no rate contributes **zero** carry, which is the spot-only
    approximation for that currency. That biases the diagnostic toward understating the
    size of the approximation, and :attr:`ShortRateCoverage.uncovered` names exactly
    which currencies it understates it for.
    """
    columns: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    spans: dict[str, tuple[str, str]] = {}
    fraction: dict[str, float] = {}
    covered: list[str] = []
    uncovered: list[str] = []

    for series_id in universe:
        entry = fetched.get(series_id)
        if entry is None:
            uncovered.append(series_id)
            columns[series_id] = pd.Series(np.nan, index=calendar)
            fraction[series_id] = 0.0
            continue
        source, raw = entry
        observed = raw.dropna()
        aligned = observed.reindex(observed.index.union(calendar)).ffill().reindex(calendar)
        columns[series_id] = aligned
        sources[series_id] = source
        spans[series_id] = (str(observed.index[0].date()), str(observed.index[-1].date()))
        fraction[series_id] = float(aligned.notna().mean())
        covered.append(series_id)

    return ShortRateCoverage(
        rates=pd.DataFrame(columns, index=calendar, columns=list(universe)),
        sources=sources,
        covered=tuple(covered),
        uncovered=tuple(uncovered),
        spans=spans,
        coverage_fraction=fraction,
    )


def carry_adjusted_prices(
    spot: pd.DataFrame, rates: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Turn spot rates into the USD value of a foreign money-market deposit.

    Why this is the right arithmetic. The headline book is fully invested in foreign
    currency, so the engine's cash balance is zero and
    :attr:`~trendbot.engine.panel_backtest.PanelBacktestResult.excess_returns` already
    computes ``spot change - i_US``: a spot position financed at the US rate, earning
    nothing on the foreign leg. The true currency excess return is
    ``spot change + i_foreign - i_US``. The two differ by exactly ``i_foreign``, so the
    whole of the correction is the foreign interest accrual - and the natural way to
    express it is to hold, instead of the currency, a deposit denominated in it::

        P_total_i(t) = P_spot_i(t) * exp( sum_{u <= t} i_foreign_i(u) / 252 )

    The US leg needs no change: the engine subtracts the T-bill rate for the strategy
    and the benchmark alike, exactly as it did for the headline.

    The accrual at row ``t`` uses the rate known at row ``t-1``, so no rate observation
    can move a return dated before it. A currency with no rate accrues nothing and its
    total-return series is its spot series, unchanged.
    """
    if not spot.index.equals(rates.index) or list(spot.columns) != list(rates.columns):
        raise ValueError("spot and rate panels must share an index and a column order")
    daily = (rates.shift(1) / float(periods_per_year)).fillna(0.0)
    return spot * np.exp(daily.cumsum())


# --------------------------------------------------------------------------------------
# 3. section 8 - the pre-committed decision rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision005:
    """The verdict, evaluated mechanically against PREREG_005.md section 8."""

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


def evaluate_decision_rule_005(
    cfg: Config005,
    *,
    strategy_sharpe: float,
    benchmark_sharpe: float,
    n_inversions: int,
    spread_mean: float,
    spread_t_stat: float,
    alpha_annualised: float,
    alpha_t_stat: float,
) -> Decision005:
    """Apply section 8 exactly as written, with no interpretation at the margin.

    Section 8, verbatim::

        Supported only if all four hold:
        - Net Sharpe (excess of T-bill, 5 bps) exceeds 0.40, AND
        - Net Sharpe exceeds the equal-weight dollar-factor benchmark by at least
          0.15, AND
        - Quintile monotonicity: forward returns decrease monotonically Q1 -> Q5 with
          at most one adjacent inversion, and Q1-Q5 spread positive at t > 2.0, AND
        - Alpha to the dollar factor is positive with t > 2.0.

        Abandon if any of: fails to beat the benchmark at all; more than one inversion,
        or Q1-Q5 t below 1.0; alpha to the dollar factor negative; net Sharpe below
        0.15.

    Comparators follow the document's own words: "exceeds" and "above" are strict, "at
    least" and "at most" are inclusive, "below" is strict.
    """
    exceeds_min = strategy_sharpe > cfg.support_min_sharpe
    margin = strategy_sharpe - benchmark_sharpe
    beats_by_margin = margin >= cfg.support_min_sharpe_excess_over_benchmark
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
            f"exceeds the dollar factor by at least {cfg.support_min_sharpe_excess_over_benchmark:+.2f}",
            beats_by_margin,
            f"margin {margin:+.3f} ({strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f})",
        ),
        (
            f"quintile monotonicity: at most {cfg.max_inversions} inversion and spread "
            f"positive at t > {cfg.min_spread_t_stat:.1f}",
            monotonicity_ok,
            f"{n_inversions} inversion(s), spread {spread_mean:+.4%}/month at t {spread_t_stat:+.3f}",
        ),
        (
            f"alpha to the dollar factor positive at t > {cfg.min_alpha_t_stat:.1f}",
            alpha_ok,
            f"alpha {alpha_annualised:+.2%}/yr at t {alpha_t_stat:+.3f}",
        ),
    )
    abandon_clauses = (
        (
            "fails to beat the dollar factor at all",
            not beats_at_all,
            f"{strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f}",
        ),
        (
            f"more than {cfg.max_inversions} quintile inversion",
            too_many_inversions,
            f"{n_inversions} inversion(s)",
        ),
        (
            f"Q1-Q5 t-statistic below {cfg.abandon_below_spread_t_stat:.1f}",
            spread_t_too_low,
            f"t {spread_t_stat:+.3f}",
        ),
        ("alpha to the dollar factor is negative", alpha_negative, f"{alpha_annualised:+.2%}/yr"),
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

    return Decision005(
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
