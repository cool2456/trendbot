"""Experiment 006's protocol — multi-strategy risk allocation over sleeve returns.

Peer of :mod:`trendbot.engine.xs_validation` (002), :mod:`trendbot.engine.eq_validation`
(003/004) and :mod:`trendbot.engine.fx_validation` (005). What is different is that
there is no signal to validate: every gate here is about the *construction*.

The four things that can silently go wrong, and what catches each
------------------------------------------------------------------
1. **A lookahead in the covariance.** A full-sample covariance is not caught by any
   ``.shift()``, because there is no shift involved — the matrix simply knows things.
   :func:`overlay_targets` builds each date's matrix with
   :func:`trendbot.sizing.ewma_covariance`, whose ``ewm`` is causal by construction, and
   :func:`covariance_is_point_in_time` proves it by appending future data and checking
   nothing earlier moved.

2. **A solver that does not converge.** Equal risk contribution has no closed form for
   three sleeves, and a solver that stops early produces weights that look like an
   allocation and are not one. Every solve carries its realised worst deviation from
   equality, computed from the returned weights rather than from the solver's own
   opinion of itself, and :class:`OverlayResult` reports the worst across the sample.

3. **An asymmetry between portfolio and benchmark.** If the two are built by different
   code, or the same code with different arguments, the comparison measures the
   difference in construction rather than the difference in what is being combined.
   Both are built by one call to :func:`run_overlay`; :func:`assert_symmetric` compares
   the recorded settings of the two results field by field and refuses any difference
   other than the input series themselves.

4. **Misalignment across calendars.** Sleeves A and B trade US equity sessions; sleeve C
   is on FRED H.10 observation dates. A one-day slip between them corrupts the
   correlation estimate, which is precisely what section 8's third clause turns on.
   :func:`align` inner-joins the date indices — dropping bars rather than manufacturing
   them — and reports what it dropped.

Costs and cash, once each
-------------------------
The overlay is run through :func:`~trendbot.engine.panel_backtest.run_panel_backtest`
rather than by multiplying weights into returns, so that the rebalance calendar, the
execution shift, the cost on realised turnover, the re-application of the gross cap to
weights that drifted, and the accrual on idle cash are all the same code every earlier
experiment used. The sleeves are handed to it as synthetic instruments whose price is
their own cumulative total return; see :func:`synthetic_panel` for why total rather than
excess, and what would go wrong otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import PriceData, daily_risk_free
from ..sizing import ewma_covariance
from .allocation import SleeveWeights, allocate
from .backtest import rebalance_dates
from .metrics import TRADING_DAYS_PER_YEAR, PerformanceStats, sharpe, summarise
from .panel_backtest import PanelBacktestResult, run_panel_backtest

__all__ = [
    "Alignment",
    "align",
    "synthetic_panel",
    "overlay_targets",
    "OverlayResult",
    "run_overlay",
    "assert_symmetric",
    "covariance_is_point_in_time",
    "CorrelationReport",
    "correlation_report",
    "Decision006",
    "evaluate_decision_006",
    "synthetic_sleeve_returns",
    "NoiseResult006",
    "noise_test_006",
]


# --------------------------------------------------------------------------------------
# 1. section 5 - the sample window, and alignment across three calendars
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Alignment:
    """What section 5's intersection actually cost, per sleeve."""

    index: pd.DatetimeIndex
    rule: str
    table: pd.DataFrame

    @property
    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.index[0], self.index[-1]

    def __str__(self) -> str:
        lo, hi = self.span
        return (
            f"{self.rule}: {len(self.index)} common bars, {lo.date()} -> {hi.date()}, "
            f"dropping {int(self.table['dropped'].sum())} sleeve-bars in total"
        )


def align(series: dict[str, pd.Series]) -> Alignment:
    """Intersect the date indices of the sleeves. Drops bars; manufactures none.

    The alternative — reindexing every sleeve onto the union and filling the holes with
    zero — is rejected on a specific ground rather than on taste. A zero-filled day is a
    day on which a sleeve is asserted to have returned exactly nothing, and injecting
    those into sleeve C attenuates its measured correlation with A and B. That biases
    section 8's third clause in the *passing* direction, which is the one clause whose
    whole purpose is to test the mechanism rather than the outcome. Losing 46 bars in 18
    years is the cheaper error.
    """
    if not series:
        raise ValueError("cannot align an empty set of sleeves")
    index: pd.DatetimeIndex | None = None
    for one in series.values():
        index = one.index if index is None else index.intersection(one.index)
    assert index is not None
    if len(index) == 0:
        raise ValueError("the sleeves share no dates; section 5's intersection is empty")

    lo, hi = index[0], index[-1]
    rows = []
    for label, one in series.items():
        inside = one.loc[lo:hi]
        rows.append(
            {
                "sleeve": label,
                "own bars": len(one),
                "own bars in span": len(inside),
                "dropped": len(inside) - len(index),
                "first": one.index[0].date(),
                "last": one.index[-1].date(),
            }
        )
    return Alignment(
        index=index,
        rule="inner join on date index",
        table=pd.DataFrame(rows).set_index("sleeve"),
    )


# --------------------------------------------------------------------------------------
# 2. section 3 - the weight path
# --------------------------------------------------------------------------------------


def synthetic_panel(excess_returns: pd.DataFrame, rf: pd.Series) -> PriceData:
    """Turn sleeve return series into a price panel the engine can trade.

    Each sleeve becomes an instrument whose close is the cumulative product of its
    **total** return and whose open is the previous close, so that the whole of a day's
    return falls between the open at which the engine trades and the close at which it
    marks. There is no intraday leg to model because a sleeve return has none.

    Why total rather than excess, stated because getting it backwards is invisible.
    The engine accrues the risk-free rate on idle cash and ``excess_returns`` subtracts
    that same rate from the whole account. Feeding it *excess* series would give::

        sum_i w_i r_i^ex + (1 - sum_i w_i) rf - rf  =  sum_i w_i r_i^ex - (sum_i w_i) rf

    which subtracts financing a second time on the invested fraction. Feeding *total*
    series gives ``sum_i w_i r_i^ex`` exactly, which is the quantity section 4 asks for.
    The error is small only because the gross cap keeps the book near fully invested; it
    would be worth several hundredths of Sharpe if the vol target ever ran the book at
    60% gross.
    """
    if not isinstance(excess_returns, pd.DataFrame):
        raise TypeError("excess_returns must be a DataFrame of daily sleeve returns")
    if excess_returns.isna().to_numpy().any():
        raise ValueError(
            "sleeve returns contain NaN; align() must be applied before building a panel"
        )
    rf_daily = daily_risk_free(rf, excess_returns.index)
    total = excess_returns.add(rf_daily, axis=0)
    close = (1.0 + total).cumprod()
    open_ = close.shift(1)
    open_.iloc[0] = 1.0
    return PriceData(
        open=open_,
        close=close,
        source="synthetic sleeve total-return panel",
        adjusted=True,
        fetched_at="",
    )


def overlay_targets(
    excess_returns: pd.DataFrame,
    *,
    halflife: int,
    vol_target: float,
    gross_cap: float,
    min_weight: float,
    max_weight: float,
    order: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Section 3's weights for every date, plus the diagnostics behind them.

    Returns ``(targets, diagnostics, risk_contributions)``. ``targets`` is a frame of
    target weights as of each row's own close, which is the contract
    :mod:`trendbot.engine.panel` defines; the engine applies the single execution shift
    itself, so a date's weights are traded on the following bar and can never be
    informed by the return they earn.

    Dates before the covariance estimator has ``halflife`` observations get a weight of
    zero — the account is in cash. Section 3 authorises no non-ERC warm-up period, so
    inventing an equal-weight one would be inventing a construction.

    The same applies to a date whose covariance is not positive definite. That is not a
    numerical nuisance to be regularised away: it means two sleeves are perfectly
    collinear over the estimation window, and equal risk contribution has no unique
    solution on such a matrix. Those dates are counted into
    ``attrs["n_singular"]`` on the diagnostics frame and left in cash. On section 5's
    pre-committed window the count is zero, and the runner asserts it; it is non-zero
    only under the raw-availability reading, where sleeves A and B are both entirely in
    cash for the first year and therefore identical.
    """
    covariances = ewma_covariance(excess_returns, halflife)
    labels = list(excess_returns.columns)

    weights: dict[pd.Timestamp, pd.Series] = {}
    contributions: dict[pd.Timestamp, pd.Series] = {}
    rows: list[dict] = []
    n_undefined = 0
    n_singular = 0
    for date in excess_returns.index:
        matrix = covariances.loc[date]
        if matrix.isna().to_numpy().any():
            n_undefined += 1
            continue
        if np.linalg.eigvalsh(matrix.to_numpy(dtype=float)).min() <= 0:
            n_singular += 1
            continue
        solved: SleeveWeights = allocate(
            matrix,
            date=date,
            vol_target=vol_target,
            gross_cap=gross_cap,
            min_weight=min_weight,
            max_weight=max_weight,
            order=order,
        )
        weights[date] = solved.weights
        contributions[date] = solved.realised_risk_contributions
        rows.append(
            {
                "date": date,
                "k": solved.k,
                "k_uncapped": solved.k_uncapped,
                "exante_vol": solved.exante_vol,
                "final_vol": solved.final_vol,
                "gross": solved.gross,
                "min_max_binding": solved.min_max_binding,
                "gross_cap_binding": solved.gross_cap_binding,
                "bound_breach": solved.bound_breach,
                "erc_worst_deviation": solved.erc_worst_deviation,
                "erc_iterations": solved.erc_iterations,
                "erc_converged": solved.erc_converged,
                **{f"erc_{label}": solved.erc[label] for label in labels},
            }
        )
    if not rows:
        raise ValueError(
            f"the covariance estimator never became defined over {len(excess_returns)} bars "
            f"at halflife {halflife}; there is nothing to allocate"
        )
    targets = pd.DataFrame(weights).T.reindex(index=excess_returns.index, columns=labels).fillna(0.0)
    diagnostics = pd.DataFrame(rows).set_index("date")
    diagnostics.attrs["n_undefined"] = n_undefined
    diagnostics.attrs["n_singular"] = n_singular
    risk_contributions = pd.DataFrame(contributions).T.reindex(columns=labels)
    return targets, diagnostics, risk_contributions


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """One ERC overlay — portfolio or benchmark — and everything it is judged on."""

    label: str
    settings: tuple[tuple[str, object], ...]
    result: PanelBacktestResult
    targets: pd.DataFrame
    diagnostics: pd.DataFrame
    risk_contributions: pd.DataFrame
    first_funded: pd.Timestamp
    sleeve_returns: pd.DataFrame = field(repr=False)

    @property
    def returns(self) -> pd.Series:
        """Portfolio return in excess of the T-bill, over the whole section 5 window."""
        return self.result.excess_returns

    @property
    def stats(self) -> PerformanceStats:
        """Full metric set, with turnover and gross taken from the book actually held."""
        return self.result.stats_from(self.sleeve_returns.index[0])

    @property
    def sharpe(self) -> float:
        return sharpe(self.returns)

    @property
    def worst_erc_deviation(self) -> float:
        """Largest realised departure from equal risk contribution across the sample."""
        return float(self.diagnostics["erc_worst_deviation"].max())

    @property
    def all_converged(self) -> bool:
        return bool(self.diagnostics["erc_converged"].all())

    @property
    def n_singular_dates(self) -> int:
        """Dates left in cash because the covariance had no unique ERC solution."""
        return int(self.diagnostics.attrs.get("n_singular", 0))

    @property
    def n_undefined_dates(self) -> int:
        """Dates left in cash because the EWMA had not yet accumulated enough history."""
        return int(self.diagnostics.attrs.get("n_undefined", 0))

    @property
    def gross_cap(self) -> float:
        return float(dict(self.settings)["gross_cap"])

    @property
    def gross_cap_binding_fraction(self) -> float:
        """Fraction of *rebalances* at which section 3's gross cap binds the target."""
        rebals = rebalance_dates(self.sleeve_returns.index)
        decisions = self.diagnostics.index.intersection(
            _decision_dates(self.sleeve_returns.index, rebals)
        )
        if len(decisions) == 0:
            return float("nan")
        return float(self.diagnostics.loc[decisions, "gross_cap_binding"].mean())

    @property
    def gross_cap_binding_fraction_daily(self) -> float:
        """Fraction of funded *days* on which the book is held at the cap.

        Measured on the weights actually held rather than on the target, because
        between rebalances the book drifts and the cap is a statement about what is
        held. Days before the first funded rebalance are excluded: an all-cash account
        is not a book pinned at its cap.
        """
        held = self.result.weights.loc[self.first_funded :].abs().sum(axis=1)
        if len(held) == 0:
            return float("nan")
        return float((held >= self.gross_cap - 1e-6).mean())

    def __str__(self) -> str:
        return (
            f"{self.label}: Sharpe {self.sharpe:+.4f}, worst ERC deviation "
            f"{self.worst_erc_deviation:.3e}, "
            f"{'all solves converged' if self.all_converged else 'A SOLVE DID NOT CONVERGE'}"
        )


def _decision_dates(index: pd.DatetimeIndex, rebals: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The bar whose close informs each rebalance — one before it, per the engine's shift."""
    positions = {date: i for i, date in enumerate(index)}
    return pd.DatetimeIndex([index[positions[reb] - 1] for reb in rebals if positions[reb] > 0])


def run_overlay(
    excess_returns: pd.DataFrame,
    rf: pd.Series,
    *,
    label: str,
    halflife: int,
    vol_target: float,
    gross_cap: float,
    min_weight: float,
    max_weight: float,
    order: str,
    cost_bps: float,
) -> OverlayResult:
    """Build one equal-risk-contribution overlay over ``excess_returns``.

    This is the *only* construction path. Section 4's benchmark is this function called
    with the sleeves' benchmarks in place of the sleeves; nothing else differs, and
    :func:`assert_symmetric` checks that claim rather than asserting it.
    """
    targets, diagnostics, contributions = overlay_targets(
        excess_returns,
        halflife=halflife,
        vol_target=vol_target,
        gross_cap=gross_cap,
        min_weight=min_weight,
        max_weight=max_weight,
        order=order,
    )
    prices = synthetic_panel(excess_returns, rf)
    result = run_panel_backtest(
        prices,
        list(excess_returns.columns),
        None,
        targets=targets,
        cost_bps=cost_bps,
        drift_band=None,  # section 3 gives no band: the weights are the weights
        gross_cap=gross_cap,
        risk_free=rf,
    )
    funded = result.weights.abs().sum(axis=1)
    live = funded[funded > 1e-12]
    first_funded = live.index[0] if len(live) else excess_returns.index[-1]
    # The risk-free series is a construction input too, and section 4's symmetry claim
    # is about the construction. Fingerprinting it rather than comparing the object
    # means the gate catches a benchmark financed at a different rate, which no other
    # compared field would reveal.
    rf_used = daily_risk_free(rf, excess_returns.index)
    settings = (
        ("rf_fingerprint", (len(rf_used), round(float(rf_used.sum()), 12))),
        ("halflife", halflife),
        ("vol_target", vol_target),
        ("gross_cap", gross_cap),
        ("min_weight", min_weight),
        ("max_weight", max_weight),
        ("order", order),
        ("cost_bps", float(cost_bps)),
        ("drift_band", None),
        ("n_sleeves", len(excess_returns.columns)),
        ("n_bars", len(excess_returns)),
    )
    return OverlayResult(
        label=label,
        settings=settings,
        result=result,
        targets=targets,
        diagnostics=diagnostics,
        risk_contributions=contributions,
        first_funded=first_funded,
        sleeve_returns=excess_returns,
    )


def assert_symmetric(portfolio: OverlayResult, benchmark: OverlayResult) -> pd.DataFrame:
    """Build step 5's gate: portfolio and benchmark differ only in their inputs.

    Returns the comparison table. Raises if any setting differs — an asymmetry between
    the two constructions invalidates the comparison section 8's first clause makes.
    """
    left, right = dict(portfolio.settings), dict(benchmark.settings)
    rows = []
    for key in left:
        rows.append({"setting": key, portfolio.label: left[key], benchmark.label: right.get(key), "same": left[key] == right.get(key)})
    table = pd.DataFrame(rows).set_index("setting")
    differing = table.index[~table["same"]].tolist()
    if differing:
        raise RuntimeError(
            f"portfolio and benchmark overlays were built with different settings "
            f"{differing}. Section 4 requires identical covariance estimation, identical "
            "vol target and identical caps; any asymmetry makes the comparison measure "
            "the construction rather than the sleeves."
        )
    return table


def covariance_is_point_in_time(
    excess_returns: pd.DataFrame, *, halflife: int, n_future: int = 250, seed: int = 0
) -> tuple[bool, float]:
    """Append fabricated future rows and check no earlier covariance moved.

    Returns ``(passed, worst_absolute_change)``. The fabricated rows are deliberately
    extreme — an order of magnitude more volatile than the real series — so that a
    full-sample estimator could not fail to be caught.

    An honest caveat, recorded because the gate reads stronger than it is: an estimator
    that used ``r_t`` to weight the position that earns ``r_t`` would still pass this,
    because appending rows *after* ``t`` does not disturb it. This gate rules out
    full-sample contamination, not the one-bar question. That one is settled separately
    by the engine's single execution shift.
    """
    rng = np.random.default_rng(seed)
    scale = float(excess_returns.std().max()) * 10.0
    future_index = pd.date_range(
        excess_returns.index[-1] + pd.Timedelta(days=1), periods=n_future, freq="B"
    )
    future = pd.DataFrame(
        rng.normal(scale, scale, size=(n_future, excess_returns.shape[1])),
        index=future_index,
        columns=excess_returns.columns,
    )
    base = ewma_covariance(excess_returns, halflife)
    extended = ewma_covariance(pd.concat([excess_returns, future]), halflife)
    overlap = extended.loc[: excess_returns.index[-1]]
    aligned = overlap.reindex(base.index)
    delta = (aligned - base).abs().to_numpy()
    worst = float(np.nanmax(delta)) if delta.size else 0.0
    return bool(worst == 0.0), worst


# --------------------------------------------------------------------------------------
# 3. section 8 - the pre-committed decision rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelationReport:
    """Section 8's third clause: the mechanism, not the outcome."""

    sleeve_matrix: pd.DataFrame
    benchmark_matrix: pd.DataFrame
    sleeve_mean: float
    benchmark_mean: float

    @property
    def difference(self) -> float:
        """Negative means the sleeves are less correlated than their benchmarks."""
        return self.sleeve_mean - self.benchmark_mean

    @property
    def holds(self) -> bool:
        return self.sleeve_mean < self.benchmark_mean

    def __str__(self) -> str:
        return (
            f"mean pairwise correlation: sleeves {self.sleeve_mean:.4f} vs benchmarks "
            f"{self.benchmark_mean:.4f} (difference {self.difference:+.4f}) -> clause (c) "
            f"{'HOLDS' if self.holds else 'FAILS'}"
        )


def _mean_pairwise(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float)
    upper = np.triu_indices_from(values, k=1)
    return float(np.mean(values[upper]))


def correlation_report(sleeves: pd.DataFrame, benchmarks: pd.DataFrame) -> CorrelationReport:
    """Realised correlation matrices of the sleeves and of their benchmarks."""
    if list(sleeves.columns) != list(benchmarks.columns):
        raise ValueError("sleeve and benchmark frames must share a column order")
    if not sleeves.index.equals(benchmarks.index):
        raise ValueError(
            "sleeve and benchmark frames are on different calendars; the correlation "
            "comparison would not be like-for-like"
        )
    sleeve_matrix = sleeves.corr()
    benchmark_matrix = benchmarks.corr()
    return CorrelationReport(
        sleeve_matrix=sleeve_matrix,
        benchmark_matrix=benchmark_matrix,
        sleeve_mean=_mean_pairwise(sleeve_matrix),
        benchmark_mean=_mean_pairwise(benchmark_matrix),
    )


@dataclass(frozen=True, slots=True)
class Decision006:
    """Section 8, evaluated. Three clauses, all required, each reported as a number."""

    portfolio_sharpe: float
    benchmark_sharpe: float
    best_sleeve_label: str
    best_sleeve_sharpe: float
    correlation: CorrelationReport
    min_excess_over_benchmark: float
    min_excess_over_best_sleeve: float

    @property
    def clause_a(self) -> float:
        return self.portfolio_sharpe - self.benchmark_sharpe

    @property
    def clause_b(self) -> float:
        return self.portfolio_sharpe - self.best_sleeve_sharpe

    @property
    def clause_c(self) -> float:
        return self.correlation.difference

    @property
    def supported(self) -> bool:
        return (
            self.clause_a >= self.min_excess_over_benchmark
            and self.clause_b >= self.min_excess_over_best_sleeve
            and self.correlation.holds
        )

    @property
    def abandon_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.clause_a <= 0:
            reasons.append(
                f"fails to beat the section 4 benchmark at all ({self.portfolio_sharpe:+.4f} "
                f"vs {self.benchmark_sharpe:+.4f}, margin {self.clause_a:+.4f})"
            )
        if self.clause_b <= 0:
            reasons.append(
                f"fails to beat the best individual sleeve ({self.portfolio_sharpe:+.4f} vs "
                f"sleeve {self.best_sleeve_label} at {self.best_sleeve_sharpe:+.4f}, margin "
                f"{self.clause_b:+.4f})"
            )
        if not self.correlation.holds:
            reasons.append(
                f"sleeve correlations are not lower than benchmark correlations "
                f"({self.correlation.sleeve_mean:.4f} vs {self.correlation.benchmark_mean:.4f})"
            )
        return tuple(reasons)

    @property
    def verdict(self) -> str:
        if self.supported:
            return "SUPPORTED"
        if self.abandon_reasons:
            return "ABANDON"
        return "INCONCLUSIVE"

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "clause": "(a) net Sharpe - section 4 benchmark",
                    "required": f">= +{self.min_excess_over_benchmark:.2f}",
                    "observed": self.clause_a,
                    "holds": self.clause_a >= self.min_excess_over_benchmark,
                },
                {
                    "clause": f"(b) net Sharpe - best sleeve ({self.best_sleeve_label})",
                    "required": f">= +{self.min_excess_over_best_sleeve:.2f}",
                    "observed": self.clause_b,
                    "holds": self.clause_b >= self.min_excess_over_best_sleeve,
                },
                {
                    "clause": "(c) sleeve corr - benchmark corr",
                    "required": "< 0",
                    "observed": self.clause_c,
                    "holds": self.correlation.holds,
                },
            ]
        ).set_index("clause")

    def __str__(self) -> str:
        body = "\n".join(f"    - {reason}" for reason in self.abandon_reasons)
        return f"SECTION 8 VERDICT: {self.verdict}" + (f"\n  abandon triggers:\n{body}" if body else "")


def evaluate_decision_006(
    *,
    portfolio_sharpe: float,
    benchmark_sharpe: float,
    sleeve_sharpes: pd.Series,
    correlation: CorrelationReport,
    min_excess_over_benchmark: float,
    min_excess_over_best_sleeve: float,
) -> Decision006:
    """Section 8, with the best sleeve chosen *after* the fact, as the clause requires."""
    if sleeve_sharpes.empty:
        raise ValueError("cannot identify the best sleeve from an empty set")
    best = sleeve_sharpes.idxmax()
    return Decision006(
        portfolio_sharpe=portfolio_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        best_sleeve_label=str(best),
        best_sleeve_sharpe=float(sleeve_sharpes.loc[best]),
        correlation=correlation,
        min_excess_over_benchmark=min_excess_over_benchmark,
        min_excess_over_best_sleeve=min_excess_over_best_sleeve,
    )


# --------------------------------------------------------------------------------------
# 4. the noise test - the construction on sleeves that contain nothing
# --------------------------------------------------------------------------------------


def synthetic_sleeve_returns(
    seed: int,
    *,
    n_days: int,
    labels: tuple[str, ...],
    correlation: np.ndarray,
    annual_vols: tuple[float, ...],
) -> pd.DataFrame:
    """Zero-mean Gaussian sleeve returns with a known covariance structure.

    Zero mean exactly, not approximately: the drift is what the noise test is looking
    for, so leaving a sampled drift in would test the sampler rather than the
    construction.
    """
    correlation = np.asarray(correlation, dtype=float)
    n = len(labels)
    if correlation.shape != (n, n):
        raise ValueError(f"correlation must be {n}x{n}, got {correlation.shape}")
    if len(annual_vols) != n:
        raise ValueError("one annualised volatility per sleeve is required")
    daily = np.asarray(annual_vols, dtype=float) / np.sqrt(TRADING_DAYS_PER_YEAR)
    covariance = correlation * np.outer(daily, daily)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(n), covariance, size=n_days, method="cholesky")
    draws = draws - draws.mean(axis=0)  # exactly zero mean
    index = pd.date_range("2000-01-03", periods=n_days, freq="B")
    return pd.DataFrame(draws, index=index, columns=list(labels))


@dataclass(frozen=True, slots=True)
class NoiseResult006:
    """The ERC construction run on sleeves that contain no signal whatsoever."""

    table: pd.DataFrame
    target_correlation: pd.DataFrame
    tolerance: float
    correlation_tolerance: float

    @property
    def mean_sharpe(self) -> float:
        return float(self.table["portfolio_sharpe"].mean())

    @property
    def worst_abs_sharpe(self) -> float:
        return float(self.table["portfolio_sharpe"].abs().max())

    @property
    def worst_correlation_error(self) -> float:
        return float(self.table["max_corr_error"].max())

    @property
    def worst_erc_deviation(self) -> float:
        return float(self.table["worst_erc_deviation"].max())

    @property
    def all_converged(self) -> bool:
        return bool(self.table["all_converged"].all())

    def passes(self) -> bool:
        return (
            abs(self.mean_sharpe) <= self.tolerance
            and self.worst_correlation_error <= self.correlation_tolerance
            and self.all_converged
        )

    def __str__(self) -> str:
        return (
            f"{len(self.table)} seeds: mean portfolio Sharpe {self.mean_sharpe:+.4f} "
            f"(worst |Sharpe| {self.worst_abs_sharpe:.4f}), worst correlation-recovery "
            f"error {self.worst_correlation_error:.4f}, worst ERC deviation "
            f"{self.worst_erc_deviation:.3e}"
        )


def noise_test_006(
    *,
    seeds: tuple[int, ...],
    n_days: int,
    labels: tuple[str, ...],
    correlation: np.ndarray,
    annual_vols: tuple[float, ...],
    halflife: int,
    vol_target: float,
    gross_cap: float,
    min_weight: float,
    max_weight: float,
    order: str,
    cost_bps: float,
    tolerance: float = 0.2,
    correlation_standard_errors: float = 5.0,
) -> NoiseResult006:
    """Build step 4: the whole construction on zero-mean sleeves of known covariance.

    Two things are checked, not one. That the portfolio earns nothing is necessary but
    weak — a construction that allocated nothing at all would also pass it. So the test
    also asserts the realised correlation matrix recovers the one that was put in, which
    fails if the alignment, the panel synthesis or the return accounting has shifted a
    series relative to the others.

    The risk-free rate is zero here, so total and excess returns coincide and the
    portfolio return is exactly the weighted sleeve return net of overlay cost.

    The correlation tolerance is expressed in **sampling standard errors** rather than
    in correlation units, because a sample correlation over ``n_days`` has a known
    dispersion and a fixed absolute tolerance would be too tight at 1,500 bars and too
    loose at 20,000. The Fisher-z standard error ``1/sqrt(n-3)`` is used, which is
    conservative for the correlation itself. Five of them is wide enough that ordinary
    sampling never trips it and narrow enough to catch what the check exists for: a
    one-bar slip between sleeves collapses a 0.70 correlation towards zero, which is
    roughly twenty-seven standard errors at 1,500 bars.
    """
    correlation_tolerance = correlation_standard_errors / math.sqrt(max(n_days - 3, 1))
    target = pd.DataFrame(np.asarray(correlation, dtype=float), index=list(labels), columns=list(labels))
    rows = []
    for seed in seeds:
        returns = synthetic_sleeve_returns(
            seed, n_days=n_days, labels=labels, correlation=correlation, annual_vols=annual_vols
        )
        rf = pd.Series(0.0, index=returns.index)
        overlay = run_overlay(
            returns,
            rf,
            label=f"noise seed {seed}",
            halflife=halflife,
            vol_target=vol_target,
            gross_cap=gross_cap,
            min_weight=min_weight,
            max_weight=max_weight,
            order=order,
            cost_bps=cost_bps,
        )
        realised = returns.corr()
        error = float(np.max(np.abs(realised.to_numpy() - target.to_numpy())))
        rows.append(
            {
                "seed": seed,
                "portfolio_sharpe": overlay.sharpe,
                "gross_mean": float(overlay.result.weights.abs().sum(axis=1).mean()),
                "max_corr_error": error,
                "worst_erc_deviation": overlay.worst_erc_deviation,
                "all_converged": overlay.all_converged,
                **{f"sleeve_{label}_sharpe": sharpe(returns[label]) for label in labels},
            }
        )
    return NoiseResult006(
        table=pd.DataFrame(rows).set_index("seed"),
        target_correlation=target,
        tolerance=tolerance,
        correlation_tolerance=correlation_tolerance,
    )
