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


@dataclass(frozen=True, slots=True)
class MonthClustering:
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
    lookup = {month: label for label, months in windows.items() for month in months}
    matched = tuple(
        (str(period), float(row["return"]), lookup.get(str(period)))
        for period, row in worst.iterrows()
    )
    return MonthClustering(worst=worst, matched=matched, windows=dict(windows))


@dataclass(frozen=True, slots=True)
class ShortRateCoverage:
    rates: pd.DataFrame
    sources: dict[str, str]
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
    if not spot.index.equals(rates.index) or list(spot.columns) != list(rates.columns):
        raise ValueError("spot and rate panels must share an index and a column order")
    daily = (rates.shift(1) / float(periods_per_year)).fillna(0.0)
    return spot * np.exp(daily.cumsum())


@dataclass(frozen=True, slots=True)
class Decision005:
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

    if supported and abandon:  # pragma: no cover
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
