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


@dataclass(frozen=True, slots=True)
class UniverseVerification:
    table: pd.DataFrame
    required_from: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_window_bars: int

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(self.table.index[~self.table["starts_early_enough"]])

    @property
    def kept(self) -> tuple[str, ...]:
        return tuple(self.table.index[self.table["starts_early_enough"]])

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def with_holes(self) -> pd.Series:
        holes = self.table.loc[self.table["starts_early_enough"], "missing_bars_in_window"]
        return holes[holes > 0]

    def __str__(self) -> str:
        return (
            f"universe verification against {self.required_from.date()}: "
            f"{self.n_kept} of {len(self.table)} tickers kept"
            + (f", dropped {list(self.dropped)}" if self.dropped else ", none dropped")
        )


def universe_verification(prices: PriceData, cfg: Config002) -> UniverseVerification:
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


@dataclass(frozen=True, slots=True)
class PanelNoiseResult:
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
        if len(self.strategy_sharpes) < 2 or not np.isfinite(self.std_sharpe):
            return float("nan")
        return self.mean_sharpe / (self.std_sharpe / math.sqrt(len(self.strategy_sharpes)))

    @property
    def mean_gross_sharpe(self) -> float:
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
    from .panel_backtest import panel_buy_and_hold, run_panel_backtest
    from .validation import synthetic_prices

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
    from .panel_backtest import panel_buy_and_hold, run_panel_backtest

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


@dataclass(frozen=True, slots=True)
class QuintileStudy:
    returns: pd.DataFrame
    membership: pd.DataFrame
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
        means = self.mean_monthly.to_numpy()
        return bool(np.all(np.diff(means) < 0))

    @property
    def n_inversions(self) -> int:
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


def worst_months(returns: pd.Series, n: int = 5) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("worst_months needs a DatetimeIndex")
    monthly = returns.groupby(returns.index.to_period("M")).apply(lambda x: float((1 + x).prod() - 1))
    monthly.index.name = "month"
    ordered = monthly.sort_values()
    return ordered.head(n).to_frame("return")


@dataclass(frozen=True, slots=True)
class Decision002:
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

    if supported and abandon:  # pragma: no cover
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
