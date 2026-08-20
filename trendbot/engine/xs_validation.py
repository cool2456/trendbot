"""Diagnostics whose purpose is to argue that experiment 002's result is not real.

PREREG_002.md section 7, in order, with the two pieces that are new relative to
experiment 001 called out:

1. :func:`panel_noise_test` on **independent** random walks - the same null
   experiment 001 ran.
2. :func:`panel_noise_test` on **correlated** random walks driven by a shared common
   factor, built by :func:`common_factor_prices`. This is the test that matters here
   and it has no counterpart in experiment 001. A cross-sectional rule sorts
   instruments against each other; if the instruments share a factor and load on it
   with different betas, then "which names went up most over the past year" is partly
   "which names have the highest beta", and the rule ends up holding a beta tilt.
   On independent noise that failure mode cannot appear at all, so test 1 alone would
   pass while the strategy was doing something the hypothesis never claimed.
3. :func:`quintile_study` - section 8's monotonicity gate. This is a **pass/fail
   criterion on the hypothesis**, not a diagnostic: section 8 abandons the strategy
   outright if the ordering is not monotonic.
4. :func:`worst_months` - section 9 predicts momentum crashes in March 2009 and April
   2020 by name. Their *absence* would be evidence the implementation is not doing
   what it claims.
5. :func:`evaluate_decision_rule_002` - section 8, applied mechanically.

Reporting choices that PREREG_002.md does not fix, disclosed here rather than
chosen silently. None of them can change a position:

* The common-factor generator's beta spread and the share of variance the factor
  carries. Both are arguments with stated defaults, and the noise test is run across
  a range of factor loadings so the result is not a single point.
* The forward-return convention in the quintile study: **open of the rebalance bar to
  open of the next rebalance bar**, which is the same fill convention section 5 gives
  the strategy, so Q1's series is directly comparable to what the strategy earns.
* The t-statistic on the Q1-Q5 spread is a plain one-sample t on the monthly spread.
  Monthly overlapping-window autocorrelation is not corrected for; the windows do not
  overlap, so there is nothing obvious to correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..config_002 import Config002
from ..data import PriceData
from ..strategies import CrossSectionalMomentum
from ..xsmom import quantile_labels
from .backtest import rebalance_dates
from .metrics import TRADING_DAYS_PER_YEAR, sharpe

__all__ = [
    "FactorAttribution",
    "bucket_study",
    "factor_attribution",
    "UniverseVerification",
    "universe_verification",
    "common_factor_prices",
    "PanelNoiseResult",
    "panel_noise_test",
    "QuintileStudy",
    "quintile_study",
    "worst_months",
    "Decision002",
    "evaluate_decision_rule_002",
]


# --------------------------------------------------------------------------------------
# 0. universe verification (section 2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniverseVerification:
    """Section 2's data-availability check, per ticker."""

    table: pd.DataFrame
    required_from: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_window_bars: int

    @property
    def dropped(self) -> tuple[str, ...]:
        """Tickers whose history does not begin on or before the required date."""
        return tuple(self.table.index[~self.table["starts_early_enough"]])

    @property
    def kept(self) -> tuple[str, ...]:
        return tuple(self.table.index[self.table["starts_early_enough"]])

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def with_holes(self) -> pd.Series:
        """Tickers listed throughout the window that the vendor still skipped a bar for."""
        holes = self.table.loc[self.table["starts_early_enough"], "missing_bars_in_window"]
        return holes[holes > 0]

    def __str__(self) -> str:
        return (
            f"universe verification against {self.required_from.date()}: "
            f"{self.n_kept} of {len(self.table)} tickers kept"
            + (f", dropped {list(self.dropped)}" if self.dropped else ", none dropped")
        )


def universe_verification(prices: PriceData, cfg: Config002) -> UniverseVerification:
    """Section 2: confirm every ticker has continuous data from the required date.

    "Continuous" is read as *continuously listed*, not "the vendor published a bar on
    every single session". Those are different claims and only the first is a property
    of the instrument. A one-day hole in a vendor feed for a fund that traded that day
    is a data artefact; treating it as a discontinuity and dropping the ticker would
    let the vendor's housekeeping choose the universe. Both are reported, separately,
    so the reader applies whichever reading they prefer to a number rather than to a
    description.
    """
    required = pd.Timestamp(cfg.universe_history_required_from)
    close = prices.close
    window = close.loc[required:]
    n_bars = len(window)

    rows = []
    for ticker in cfg.universe:
        series = close[ticker]
        first, last = series.first_valid_index(), series.last_valid_index()
        present = int(window[ticker].notna().sum())
        rows.append(
            {
                "ticker": ticker,
                "sleeve": cfg.sleeve_of(ticker),
                "first_bar": first,
                "last_bar": last,
                "starts_early_enough": bool(first is not None and first <= required),
                "bars_in_window": present,
                "missing_bars_in_window": n_bars - present,
            }
        )
    table = pd.DataFrame(rows).set_index("ticker")
    table["starts_early_enough"] = table["starts_early_enough"].astype(bool)
    table["bars_in_window"] = table["bars_in_window"].astype(int)
    table["missing_bars_in_window"] = table["missing_bars_in_window"].astype(int)
    return UniverseVerification(
        table=table,
        required_from=required,
        window_start=window.index[0],
        window_end=window.index[-1],
        n_window_bars=n_bars,
    )


# --------------------------------------------------------------------------------------
# synthetic data with a shared common factor
# --------------------------------------------------------------------------------------


def common_factor_prices(
    tickers: Sequence[str],
    *,
    seed: int,
    n_days: int = 4000,
    annual_vol: float = 0.16,
    common_variance_share: float = 0.5,
    beta_low: float = 0.6,
    beta_high: float = 1.4,
    annual_factor_drift: float = 0.0,
    overnight_gap_fraction: float = 0.4,
    start: str = "1998-01-02",
) -> PriceData:
    """Correlated driftless random walks sharing one common factor.

    ``r_i(t) = beta_i * f(t) + eps_i(t)``, with ``f`` a driftless factor and the
    idiosyncratic vol set per instrument so that **every instrument has the same total
    volatility**. That last part is what makes this a clean null: if the betas changed
    the total vol as well, a cross-sectional momentum rule would be sorting partly on
    volatility, and any result would be ambiguous between "loads on the factor" and
    "loads on vol".

    ``common_variance_share`` is the share of the *average* instrument's variance the
    factor carries. Because ``sigma_eps_i^2 = sigma^2 - beta_i^2 sigma_f^2`` must stay
    non-negative, the achievable share is bounded by the beta spread; an infeasible
    combination raises rather than silently clipping.
    """
    if n_days < 2:
        raise ValueError("need at least 2 days")
    if not 0.0 <= common_variance_share < 1.0:
        raise ValueError("common_variance_share must lie in [0, 1)")
    if beta_low <= 0 or beta_high < beta_low:
        raise ValueError("betas must be positive with beta_high >= beta_low")

    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    n_assets = len(tickers)
    index = pd.bdate_range(start, periods=n_days)

    sd = annual_vol / math.sqrt(TRADING_DAYS_PER_YEAR)
    betas = np.linspace(beta_low, beta_high, n_assets) if n_assets > 1 else np.array([1.0])
    factor_sd = sd * math.sqrt(common_variance_share / float(np.mean(betas**2)))
    idio_var = sd**2 - (betas * factor_sd) ** 2
    if (idio_var < 0).any():
        raise ValueError(
            f"common_variance_share={common_variance_share} is infeasible with betas in "
            f"[{beta_low}, {beta_high}]: the highest-beta instrument would need negative "
            "idiosyncratic variance"
        )
    idio_sd = np.sqrt(idio_var)

    factor_mu = annual_factor_drift / TRADING_DAYS_PER_YEAR
    factor = rng.normal(factor_mu, factor_sd, size=(n_days, 1))
    idio = rng.normal(0.0, 1.0, size=(n_days, n_assets)) * idio_sd[None, :]
    # -0.5 sigma^2 keeps the arithmetic drift at zero once exponentiated, so the null
    # is "earns nothing", not "earns minus half a variance".
    log_returns = factor * betas[None, :] + idio - 0.5 * sd**2

    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=index, columns=tickers
    )
    gaps = rng.normal(0.0, sd * overnight_gap_fraction, size=(n_days, n_assets))
    open_ = close.shift(1) * np.exp(gaps)
    open_.iloc[0] = 100.0
    return PriceData(
        open=open_,
        close=close,
        source=f"common-factor(seed={seed}, share={common_variance_share})",
        adjusted=True,
        fetched_at="",
    )


# --------------------------------------------------------------------------------------
# 1 & 2. the two noise tests
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PanelNoiseResult:
    """Sharpe of the cross-sectional rule on synthetic data that contains nothing."""

    label: str
    seeds: tuple[int, ...]
    strategy_sharpes: tuple[float, ...]
    gross_sharpes: tuple[float, ...]
    buy_and_hold_sharpes: tuple[float, ...]
    n_days: int
    cost_bps: float
    mean_turnover: float
    detail: str = ""

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.strategy_sharpes))

    @property
    def std_sharpe(self) -> float:
        if len(self.strategy_sharpes) < 2:
            return float("nan")
        return float(np.std(self.strategy_sharpes, ddof=1))

    @property
    def t_stat(self) -> float:
        """t of the mean Sharpe across seeds against zero. |t| > 2 is a red flag."""
        if len(self.strategy_sharpes) < 2 or not np.isfinite(self.std_sharpe):
            return float("nan")
        return self.mean_sharpe / (self.std_sharpe / math.sqrt(len(self.strategy_sharpes)))

    @property
    def mean_gross_sharpe(self) -> float:
        """The rule's Sharpe with costs switched off - the pure "earns nothing" claim.

        The net figure on driftless data is not expected to be zero: this strategy
        replaces most of its book every month, so at 5 bps it loses the costs. Testing
        the net number alone would conflate "the rule finds nothing" with "the costs
        are large", and only the first is what section 7 step 1 is asking about.
        """
        return float(np.mean(self.gross_sharpes))

    @property
    def gross_t_stat(self) -> float:
        if len(self.gross_sharpes) < 2:
            return float("nan")
        sd = float(np.std(self.gross_sharpes, ddof=1))
        if sd == 0:
            return float("nan")
        return self.mean_gross_sharpe / (sd / math.sqrt(len(self.gross_sharpes)))

    @property
    def mean_buy_and_hold_sharpe(self) -> float:
        return float(np.mean(self.buy_and_hold_sharpes))

    @property
    def min_sharpe(self) -> float:
        return float(np.min(self.strategy_sharpes))

    @property
    def max_sharpe(self) -> float:
        return float(np.max(self.strategy_sharpes))

    def passes(self, tolerance: float = 0.2) -> bool:
        """Gross of costs, the rule and buy-and-hold must both earn approximately nothing.

        Measured gross deliberately: see :attr:`mean_gross_sharpe`. The net figure is
        reported alongside and is expected to sit below zero by the cost drag.
        """
        return (
            abs(self.mean_gross_sharpe) <= tolerance
            and abs(self.mean_buy_and_hold_sharpe) <= tolerance
        )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "net": self.strategy_sharpes,
                "gross": self.gross_sharpes,
                "buy_and_hold": self.buy_and_hold_sharpes,
            },
            index=pd.Index(self.seeds, name="seed"),
        )

    def __str__(self) -> str:
        return (
            f"{self.label}: {len(self.seeds)} seeds x {self.n_days} days -> Sharpe "
            f"gross {self.mean_gross_sharpe:+.3f} (t {self.gross_t_stat:+.2f}), "
            f"net of {self.cost_bps:g}bps {self.mean_sharpe:+.3f} "
            f"(sd {self.std_sharpe:.3f}, range {self.min_sharpe:+.3f}..{self.max_sharpe:+.3f}), "
            f"buy-and-hold {self.mean_buy_and_hold_sharpe:+.3f}, "
            f"turnover {self.mean_turnover:.1f}x/yr"
        )


def panel_noise_test(
    cfg: Config002 | None = None,
    *,
    label: str,
    seeds: Sequence[int] = tuple(range(8)),
    n_days: int = 4000,
    correlated: bool = False,
    annual_vol: float = 0.16,
    detail: str = "",
    universe: Sequence[str] | None = None,
    strategy=None,
    cost_bps: float | None = None,
    gross_cap: float | None = None,
    **generator_kwargs,
) -> PanelNoiseResult:
    """Run the real engine and a real ranking strategy on synthetic data.

    ``correlated=False`` draws independent walks with
    :func:`trendbot.engine.validation.synthetic_prices`; ``correlated=True`` draws
    common-factor walks with :func:`common_factor_prices`. Both are driftless, so the
    strategy and buy-and-hold must both earn approximately nothing.

    Passing ``cfg`` alone reproduces experiment 002 exactly. Experiment 003 has no
    ticker list in its document - its universe resolves from an index snapshot - so
    ``universe``, ``strategy``, ``cost_bps`` and ``gross_cap`` may be supplied instead.
    The loop, the generators and the engine are the same either way, which is what
    stops the two experiments' noise tests from drifting apart.
    """
    from .panel_backtest import panel_buy_and_hold, run_panel_backtest  # avoids a cycle
    from .validation import synthetic_prices  # avoids a cycle

    if cfg is None and (universe is None or strategy is None or cost_bps is None):
        raise ValueError(
            "supply either cfg (experiment 002) or all of universe/strategy/cost_bps"
        )
    universe = tuple(universe) if universe is not None else cfg.universe
    strategy = strategy if strategy is not None else CrossSectionalMomentum(cfg)
    cost_bps = cost_bps if cost_bps is not None else cfg.cost_bps_per_side
    gross_cap = gross_cap if gross_cap is not None else cfg.gross_exposure_cap

    strat, gross, bh, turnover = [], [], [], []
    for seed in seeds:
        generator = common_factor_prices if correlated else synthetic_prices
        prices = generator(
            universe, seed=seed, n_days=n_days, annual_vol=annual_vol, **generator_kwargs
        )
        result = run_panel_backtest(
            prices,
            universe,
            strategy,
            cost_bps=cost_bps,
            drift_band=None,
            gross_cap=gross_cap,
        )
        strat.append(sharpe(result.returns))
        gross.append(sharpe(result.gross_returns))
        turnover.append(result.stats.ann_turnover)
        bh.append(sharpe(panel_buy_and_hold(prices, universe)))
    return PanelNoiseResult(
        label=label,
        seeds=tuple(seeds),
        strategy_sharpes=tuple(strat),
        gross_sharpes=tuple(gross),
        buy_and_hold_sharpes=tuple(bh),
        n_days=n_days,
        cost_bps=float(cost_bps),
        mean_turnover=float(np.mean(turnover)),
        detail=detail,
    )


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    """How much of the rule's return on correlated noise is just the common factor.

    The worry the correlated test exists for is not only "does it earn something" but
    "does whatever it earns come from a beta tilt". A Sharpe near zero could still hide
    a rule that is a leveraged factor bet in a period when the factor happened to go
    nowhere. Regressing the strategy on the equal-weight basket separates the two: beta
    says how much of the book is the factor, alpha says what is left over.
    """

    table: pd.DataFrame

    @property
    def mean_beta(self) -> float:
        return float(self.table["beta"].mean())

    @property
    def mean_alpha_annualised(self) -> float:
        return float(self.table["alpha_annualised"].mean())

    @property
    def alpha_t_stat(self) -> float:
        alpha = self.table["alpha_annualised"]
        if len(alpha) < 2:
            return float("nan")
        sd = float(alpha.std(ddof=1))
        if sd == 0:
            return float("nan")
        return float(alpha.mean()) / (sd / math.sqrt(len(alpha)))

    @property
    def sharpe_correlation(self) -> float:
        """Cross-seed correlation between the rule's Sharpe and the basket's."""
        return float(self.table["strategy_sharpe"].corr(self.table["basket_sharpe"]))

    def __str__(self) -> str:
        return (
            f"factor attribution over {len(self.table)} seeds: beta to the equal-weight "
            f"basket {self.mean_beta:.3f}, alpha {self.mean_alpha_annualised:+.3%}/yr "
            f"(t {self.alpha_t_stat:+.2f}), cross-seed corr(strategy Sharpe, basket "
            f"Sharpe) {self.sharpe_correlation:+.3f}"
        )


def factor_attribution(
    cfg: Config002 | None = None,
    *,
    seeds: Sequence[int] = tuple(range(8)),
    n_days: int = 4000,
    universe: Sequence[str] | None = None,
    strategy=None,
    gross_cap: float | None = None,
    **generator_kwargs,
) -> FactorAttribution:
    """Regress the rule's daily return on the equal-weight basket, one seed at a time.

    Run gross of costs: the question is what the *rule* produces, and a cost drag is
    not a factor loading.

    As with :func:`panel_noise_test`, ``cfg`` alone reproduces experiment 002 and the
    explicit ``universe``/``strategy`` arguments let experiment 003 - whose universe is
    not in its document - run the identical diagnostic.
    """
    from .panel_backtest import panel_buy_and_hold, run_panel_backtest  # avoids a cycle

    if cfg is None and (universe is None or strategy is None):
        raise ValueError("supply either cfg (experiment 002) or both universe and strategy")
    universe = tuple(universe) if universe is not None else cfg.universe
    strategy = strategy if strategy is not None else CrossSectionalMomentum(cfg)
    gross_cap = gross_cap if gross_cap is not None else cfg.gross_exposure_cap

    rows = []
    for seed in seeds:
        prices = common_factor_prices(universe, seed=seed, n_days=n_days, **generator_kwargs)
        strategy_returns = run_panel_backtest(
            prices, universe, strategy, cost_bps=0.0, gross_cap=gross_cap
        ).returns
        basket = panel_buy_and_hold(prices, universe)
        beta, intercept = np.polyfit(basket.to_numpy(), strategy_returns.to_numpy(), 1)
        rows.append(
            {
                "seed": seed,
                "beta": float(beta),
                "alpha_annualised": float(intercept) * TRADING_DAYS_PER_YEAR,
                "strategy_sharpe": sharpe(strategy_returns),
                "basket_sharpe": sharpe(basket),
            }
        )
    return FactorAttribution(table=pd.DataFrame(rows).set_index("seed"))


# --------------------------------------------------------------------------------------
# 3. section 8's quintile monotonicity gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuintileStudy:
    """Forward returns by momentum quintile — section 8's pass/fail criterion."""

    returns: pd.DataFrame  # one column per bucket, one row per rebalance
    membership: pd.DataFrame  # instrument count per bucket per rebalance
    n_quantiles: int
    convention: str
    label_prefix: str = "Q"

    @property
    def labels(self) -> list[str]:
        return [f"{self.label_prefix}{q + 1}" for q in range(self.n_quantiles)]

    @property
    def mean_monthly(self) -> pd.Series:
        return self.returns.mean()

    @property
    def annualised(self) -> pd.Series:
        """Geometric annualisation of each quintile's monthly series."""
        n = len(self.returns)
        if n == 0:
            return pd.Series(np.nan, index=self.returns.columns)
        years = n / 12.0
        return (1.0 + self.returns).prod() ** (1.0 / years) - 1.0

    @property
    def t_stats(self) -> pd.Series:
        n = len(self.returns)
        if n < 2:
            return pd.Series(np.nan, index=self.returns.columns)
        return self.returns.mean() / (self.returns.std(ddof=1) / math.sqrt(n))

    @property
    def spread(self) -> pd.Series:
        """The Q1 - Q5 monthly series."""
        return (self.returns.iloc[:, 0] - self.returns.iloc[:, -1]).rename("Q1-Q5")

    @property
    def spread_mean(self) -> float:
        return float(self.spread.mean())

    @property
    def spread_t_stat(self) -> float:
        s = self.spread
        if len(s) < 2:
            return float("nan")
        sd = float(s.std(ddof=1))
        if sd == 0:
            return float("nan")
        return float(s.mean()) / (sd / math.sqrt(len(s)))

    @property
    def spread_positive(self) -> bool:
        return self.spread_mean > 0.0

    @property
    def is_monotonic(self) -> bool:
        """True when mean forward return falls, step by step, from Q1 to Q5.

        Strictly decreasing. Section 8 says "the ordering must be monotonically
        decreasing from Q1 to Q5"; a tie is not a decrease, and with continuous
        returns an exact tie does not arise, so nothing turns on the strictness.
        """
        means = self.mean_monthly.to_numpy()
        return bool(np.all(np.diff(means) < 0))

    @property
    def n_inversions(self) -> int:
        """How many of the four adjacent steps go the wrong way."""
        return int((np.diff(self.mean_monthly.to_numpy()) >= 0).sum())

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mean monthly": self.mean_monthly,
                "annualised": self.annualised,
                "t-stat": self.t_stats,
                "monthly vol": self.returns.std(ddof=1),
                "hit rate": (self.returns > 0).mean(),
                "avg names": self.membership.mean(),
            }
        )

    @property
    def spread_name(self) -> str:
        return f"{self.label_prefix}1-{self.label_prefix}{self.n_quantiles}"

    def __str__(self) -> str:
        if self.is_monotonic:
            ordering = "MONOTONIC"
        else:
            ordering = (
                f"NON-MONOTONIC ({self.n_inversions} of {self.n_quantiles - 1} steps invert)"
            )
        return (
            f"bucket monotonicity over {len(self.returns)} rebalances: {ordering}, "
            f"{self.spread_name} spread {self.spread_mean:+.4%}/month "
            f"(t {self.spread_t_stat:+.2f})"
        )


def bucket_study(
    prices: PriceData,
    universe: Sequence[str],
    *,
    formation_days: int,
    skip_days: int,
    n_quantiles: int,
    method: str = "floor",
    label_prefix: str = "Q",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> QuintileStudy:
    """Section 8's monotonicity gate, for any universe and any number of buckets.

    :func:`quintile_study` is experiment 002's call into this with five buckets;
    experiment 003 calls it with ten and the ``"even"`` bucket convention. One
    implementation, so the two experiments' gates cannot disagree about what a bucket
    is.
    """
    from ..xsmom import cross_sectional_momentum  # noqa: PLC0415

    universe = list(universe)
    close = prices.close[universe]
    open_ = prices.open[universe]
    mark_close = close.ffill()
    mark_open = open_.where(open_.notna(), mark_close.shift(1)).ffill()

    momentum = cross_sectional_momentum(close, formation_days, skip_days)
    labels = quantile_labels(momentum, n_quantiles, method)

    index = close.index
    pos = {d: i for i, d in enumerate(index)}
    rebals = [d for d in rebalance_dates(index) if (start is None or d >= start)]
    if end is not None:
        rebals = [d for d in rebals if d <= end]

    open_v = mark_open.to_numpy(dtype=float)
    label_v = labels.to_numpy(dtype=float)

    rows: list[dict] = []
    counts: list[dict] = []
    dates: list[pd.Timestamp] = []
    skipped = 0
    for reb, nxt in zip(rebals, rebals[1:]):
        i, j = pos[reb], pos[nxt]
        buckets = label_v[i - 1]
        forward = open_v[j] / open_v[i] - 1.0
        usable = np.isfinite(buckets) & np.isfinite(forward)
        if not usable.any():
            skipped += 1
            continue
        row, count = {}, {}
        for q in range(n_quantiles):
            member = usable & (buckets == float(q))
            name = f"{label_prefix}{q + 1}"
            count[name] = int(member.sum())
            row[name] = float(forward[member].mean()) if member.any() else np.nan
        if any(not np.isfinite(v) for v in row.values()):
            skipped += 1
            continue
        rows.append(row)
        counts.append(count)
        dates.append(reb)

    columns = [f"{label_prefix}{q + 1}" for q in range(n_quantiles)]
    idx = pd.DatetimeIndex(dates, name="rebalance")
    return QuintileStudy(
        returns=pd.DataFrame(rows, index=idx, columns=columns),
        membership=pd.DataFrame(counts, index=idx, columns=columns),
        n_quantiles=n_quantiles,
        label_prefix=label_prefix,
        convention=(
            "sorted on momentum known at the previous close; forward return from the "
            "open of the rebalance bar to the open of the next rebalance bar; equal "
            f"weight within each bucket; gross of costs; {method!r} bucket sizing; "
            f"{len(rebals) - 1 - skipped} of {max(len(rebals) - 1, 0)} rebalance "
            "intervals measured"
        ),
    )


def quintile_study(
    prices: PriceData,
    cfg: Config002,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> QuintileStudy:
    """Section 8: sort into quintiles at each rebalance, measure forward returns.

    The sort uses the momentum known at the **previous** close - the same information
    the strategy is allowed to trade on - and the forward return runs from the open of
    the rebalance bar to the open of the next one, which is the fill convention
    section 5 gives the strategy. So Q1's series is what a portfolio holding the top
    quintile actually earns, gross of costs, and the comparison across quintiles is
    like for like.

    No negative shift is used anywhere: the forward return is read off consecutive
    rebalance dates by position.

    Experiment 002's parameters, handed to the shared :func:`bucket_study`. Keeping one
    implementation is what stops 002's five-bucket gate and 003's ten-bucket gate from
    quietly meaning different things by "a bucket".
    """
    return bucket_study(
        prices,
        cfg.universe,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method="floor",
        label_prefix="Q",
        start=start,
        end=end,
    )


# --------------------------------------------------------------------------------------
# 4. behavioural validation
# --------------------------------------------------------------------------------------


def worst_months(returns: pd.Series, n: int = 5) -> pd.DataFrame:
    """The ``n`` worst calendar months of a daily return series, compounded."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("worst_months needs a DatetimeIndex")
    monthly = returns.groupby(returns.index.to_period("M")).apply(lambda x: float((1 + x).prod() - 1))
    monthly.index.name = "month"
    ordered = monthly.sort_values()
    return ordered.head(n).to_frame("return")


# --------------------------------------------------------------------------------------
# 5. section 8's pre-committed decision rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision002:
    """The verdict, evaluated mechanically against PREREG_002.md section 8."""

    strategy_sharpe: float
    benchmark_sharpe: float
    monotonic: bool
    spread_positive: bool
    exceeds_min_sharpe: bool
    beats_benchmark_by_margin: bool
    beats_benchmark_at_all: bool
    below_abandon_threshold: bool
    verdict: str
    support_clauses: tuple[tuple[str, bool, str], ...] = field(default=())
    abandon_clauses: tuple[tuple[str, bool, str], ...] = field(default=())

    def __str__(self) -> str:
        lines = [self.verdict, "  support clauses (all three required):"]
        lines += [f"    [{'YES' if ok else 'NO '}] {name} — {detail}" for name, ok, detail in self.support_clauses]
        lines += ["  abandon clauses (any one is sufficient):"]
        lines += [f"    [{'YES' if ok else 'NO '}] {name} — {detail}" for name, ok, detail in self.abandon_clauses]
        return "\n".join(lines)


def evaluate_decision_rule_002(
    cfg: Config002,
    *,
    strategy_sharpe: float,
    benchmark_sharpe: float,
    monotonic: bool,
    spread_positive: bool,
) -> Decision002:
    """Apply section 8 exactly as written, with no interpretation at the margin.

    Section 8, verbatim::

        The hypothesis is supported only if all three hold:
        - Net Sharpe (excess of T-bill, 5 bps) exceeds 0.40, AND
        - Net Sharpe exceeds equal-weight buy-and-hold ... by at least 0.15, AND
        - Quintile monotonicity holds. ... the ordering must be monotonically
          decreasing from Q1 to Q5, and the Q1-Q5 spread must be positive.

        Abandon if any of:
        - It fails to beat equal-weight buy-and-hold at all, OR
        - Quintile ordering is non-monotonic, OR
        - Net Sharpe is below 0.15.

    Comparators follow the document's own words: "exceeds" is strict, "at least" is
    inclusive, "below" is strict.
    """
    exceeds_min = strategy_sharpe > cfg.support_min_sharpe
    margin = strategy_sharpe - benchmark_sharpe
    beats_by_margin = margin >= cfg.support_min_sharpe_excess_over_buy_and_hold
    beats_at_all = strategy_sharpe > benchmark_sharpe
    below_abandon = strategy_sharpe < cfg.abandon_below_sharpe
    monotonicity_holds = bool(monotonic) and bool(spread_positive)

    supported = exceeds_min and beats_by_margin and monotonicity_holds
    abandon = (not beats_at_all) or (not monotonic) or below_abandon

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
            "quintile monotonicity holds",
            monotonicity_holds,
            f"ordering {'monotonic' if monotonic else 'NOT monotonic'}, "
            f"Q1-Q5 spread {'positive' if spread_positive else 'NOT positive'}",
        ),
    )
    abandon_clauses = (
        (
            "fails to beat equal-weight buy-and-hold at all",
            not beats_at_all,
            f"{strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f}",
        ),
        ("quintile ordering is non-monotonic", not monotonic, "section 8's own words"),
        (
            f"net Sharpe is below {cfg.abandon_below_sharpe:.2f}",
            below_abandon,
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

    return Decision002(
        strategy_sharpe=strategy_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        monotonic=bool(monotonic),
        spread_positive=bool(spread_positive),
        exceeds_min_sharpe=exceeds_min,
        beats_benchmark_by_margin=beats_by_margin,
        beats_benchmark_at_all=beats_at_all,
        below_abandon_threshold=below_abandon,
        verdict=verdict,
        support_clauses=support_clauses,
        abandon_clauses=abandon_clauses,
    )
