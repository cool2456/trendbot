from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..data import PriceData
from .metrics import TRADING_DAYS_PER_YEAR, sharpe

__all__ = [
    "synthetic_prices",
    "NoiseTestResult",
    "noise_test",
    "SplitResult",
    "in_sample_out_of_sample",
    "WalkForwardResult",
    "walk_forward",
    "DeflatedSharpe",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "noise_sweep",
    "NoiseSweepResult",
    "noise_sweep_study",
    "NoiseSweepStudy",
    "DecisionRuleResult",
    "evaluate_decision_rule",
    "standalone_instrument_sharpes",
]

EULER_MASCHERONI = 0.5772156649015329


def synthetic_prices(
    tickers: Sequence[str],
    *,
    seed: int,
    n_days: int = 6000,
    annual_vol: float = 0.16,
    annual_drift: float = 0.0,
    overnight_gap_fraction: float = 0.4,
    start: str = "1998-01-02",
) -> PriceData:
    if n_days < 2:
        raise ValueError("need at least 2 days")
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    index = pd.bdate_range(start, periods=n_days)
    sd = annual_vol / math.sqrt(TRADING_DAYS_PER_YEAR)
    mu = annual_drift / TRADING_DAYS_PER_YEAR - 0.5 * sd**2

    log_returns = rng.normal(mu, sd, size=(n_days, len(tickers)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=index, columns=tickers
    )
    gaps = rng.normal(0.0, sd * overnight_gap_fraction, size=(n_days, len(tickers)))
    open_ = close.shift(1) * np.exp(gaps)
    open_.iloc[0] = 100.0
    return PriceData(
        open=open_, close=close, source=f"synthetic(seed={seed})", adjusted=True, fetched_at=""
    )


@dataclass(frozen=True, slots=True)
class NoiseTestResult:
    seeds: tuple[int, ...]
    strategy_sharpes: tuple[float, ...]
    buy_and_hold_sharpes: tuple[float, ...]
    n_days: int
    annual_drift: float

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.strategy_sharpes))

    @property
    def std_sharpe(self) -> float:
        return float(np.std(self.strategy_sharpes, ddof=1)) if len(self.strategy_sharpes) > 1 else float("nan")

    @property
    def mean_buy_and_hold_sharpe(self) -> float:
        return float(np.mean(self.buy_and_hold_sharpes))

    def passes(self, tolerance: float = 0.2) -> bool:
        return abs(self.mean_sharpe) <= tolerance and abs(self.mean_buy_and_hold_sharpe) <= tolerance

    def __str__(self) -> str:
        return (
            f"noise test over {len(self.seeds)} seeds x {self.n_days} days: "
            f"strategy Sharpe {self.mean_sharpe:+.3f} (sd {self.std_sharpe:.3f}), "
            f"buy-and-hold {self.mean_buy_and_hold_sharpe:+.3f}"
        )


def noise_test(
    cfg: Config,
    *,
    seeds: Sequence[int] = tuple(range(8)),
    n_days: int = 6000,
    annual_drift: float = 0.0,
    annual_vol: float = 0.16,
) -> NoiseTestResult:
    from .backtest import buy_and_hold, run_backtest

    strat, bh = [], []
    for seed in seeds:
        prices = synthetic_prices(
            cfg.universe, seed=seed, n_days=n_days, annual_vol=annual_vol, annual_drift=annual_drift
        )
        strat.append(sharpe(run_backtest(prices, cfg).returns))
        bh.append(sharpe(buy_and_hold(prices, cfg)))
    return NoiseTestResult(
        seeds=tuple(seeds),
        strategy_sharpes=tuple(strat),
        buy_and_hold_sharpes=tuple(bh),
        n_days=n_days,
        annual_drift=annual_drift,
    )


@dataclass(frozen=True, slots=True)
class SplitResult:
    split_date: pd.Timestamp
    first_half_sharpe: float
    second_half_sharpe: float
    first_half_n: int
    second_half_n: int

    @property
    def asymmetry(self) -> float:
        return self.first_half_sharpe - self.second_half_sharpe

    def __str__(self) -> str:
        return (
            f"split at {self.split_date.date()}: "
            f"first half {self.first_half_sharpe:+.3f} ({self.first_half_n} days), "
            f"second half {self.second_half_sharpe:+.3f} ({self.second_half_n} days), "
            f"gap {self.asymmetry:+.3f}"
        )


def in_sample_out_of_sample(returns: pd.Series) -> SplitResult:
    returns = returns.dropna()
    if len(returns) < 4:
        raise ValueError("need at least 4 observations to split")
    mid = len(returns) // 2
    first, second = returns.iloc[:mid], returns.iloc[mid:]
    return SplitResult(
        split_date=pd.Timestamp(returns.index[mid]),
        first_half_sharpe=sharpe(first),
        second_half_sharpe=sharpe(second),
        first_half_n=len(first),
        second_half_n=len(second),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: pd.DataFrame
    train_years: float
    test_years: float
    step_years: float

    @property
    def mean_train_sharpe(self) -> float:
        return float(self.windows["train_sharpe"].mean())

    @property
    def mean_test_sharpe(self) -> float:
        return float(self.windows["test_sharpe"].mean())

    @property
    def gap(self) -> float:
        return self.mean_train_sharpe - self.mean_test_sharpe

    @property
    def fraction_positive(self) -> float:
        return float((self.windows["test_sharpe"] > 0).mean())

    def __str__(self) -> str:
        return (
            f"walk-forward {len(self.windows)} windows "
            f"({self.train_years:g}y train / {self.test_years:g}y live, step {self.step_years:g}y): "
            f"train {self.mean_train_sharpe:+.3f}, live {self.mean_test_sharpe:+.3f}, "
            f"gap {self.gap:+.3f}, {self.fraction_positive:.0%} of live windows positive"
        )


def walk_forward(
    returns: pd.Series,
    *,
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> WalkForwardResult:
    returns = returns.dropna()
    train_n = int(round(train_years * periods_per_year))
    test_n = int(round(test_years * periods_per_year))
    step_n = int(round(step_years * periods_per_year))
    if min(train_n, test_n, step_n) < 1:
        raise ValueError("window lengths must be at least one period")
    if len(returns) < train_n + test_n:
        raise ValueError(
            f"need at least {train_n + test_n} observations for a "
            f"{train_years}y/{test_years}y window, have {len(returns)}"
        )

    rows = []
    start = 0
    while start + train_n + test_n <= len(returns):
        train = returns.iloc[start : start + train_n]
        test = returns.iloc[start + train_n : start + train_n + test_n]
        rows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "train_sharpe": sharpe(train, periods_per_year),
                "test_sharpe": sharpe(test, periods_per_year),
                "test_return": float((1 + test).prod() - 1),
            }
        )
        start += step_n
    return WalkForwardResult(
        windows=pd.DataFrame(rows),
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    if n_observations < 2:
        raise ValueError("need at least 2 observations")
    denominator = 1.0 - skewness * observed_sharpe + 0.25 * (kurtosis - 1.0) * observed_sharpe**2
    if denominator <= 0:
        raise ValueError(
            f"PSR variance term is non-positive ({denominator:.4g}); the return "
            "distribution's higher moments make the estimator undefined here"
        )
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_configurations: int, sharpe_variance: float) -> float:
    if n_configurations < 1:
        raise ValueError("n_configurations must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("variance cannot be negative")
    if n_configurations == 1:
        return 0.0
    n = float(n_configurations)
    term = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n) + EULER_MASCHERONI * stats.norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return float(math.sqrt(sharpe_variance) * term)


@dataclass(frozen=True, slots=True)
class DeflatedSharpe:
    annualised_sharpe: float
    per_period_sharpe: float
    n_observations: int
    skewness: float
    kurtosis: float
    n_configurations: int
    benchmark_sharpe: float
    psr_vs_zero: float
    deflated_sharpe: float
    significance_level: float = 0.95
    note: str = ""

    @property
    def is_significant(self) -> bool:
        return self.deflated_sharpe > self.significance_level

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.is_significant else "not significant"
        return (
            f"Sharpe {self.annualised_sharpe:+.3f} over {self.n_observations} obs, "
            f"{self.n_configurations} configuration(s) tried -> "
            f"PSR(0) {self.psr_vs_zero:.4f}, deflated {self.deflated_sharpe:.4f} "
            f"({verdict} at {self.significance_level:.0%})"
        )


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_configurations: int,
    *,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    significance_level: float = 0.95,
) -> DeflatedSharpe:
    if not isinstance(n_configurations, (int, np.integer)) or isinstance(n_configurations, bool):
        raise TypeError("n_configurations must be an int")
    if n_configurations < 1:
        raise ValueError("n_configurations must be at least 1")

    r = returns.dropna()
    n = len(r)
    if n < 4:
        raise ValueError("need at least 4 observations")
    sd = float(r.std(ddof=1))
    if sd == 0:
        raise ValueError("zero-variance return series")

    sr_period = float(r.mean()) / sd
    skewness = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    psr0 = probabilistic_sharpe_ratio(sr_period, n, skewness, kurt, 0.0)

    if n_configurations == 1:
        sr_star = 0.0
        note = (
            "N=1: the E[max SR] term is degenerate (Z^-1(0)); with one configuration "
            "there is no selection bias, so the benchmark is 0 and the deflated "
            "Sharpe equals the probabilistic Sharpe against zero."
        )
    else:
        if trial_sharpes is None:
            raise ValueError(
                f"n_configurations={n_configurations} requires trial_sharpes: the "
                "deflation term needs the variance of the Sharpe ratios across the "
                "configurations that were actually tried, and it cannot be inferred "
                "from the winner alone."
            )
        trials = np.asarray(list(trial_sharpes), dtype=float)
        if len(trials) < 2:
            raise ValueError("need at least 2 trial Sharpes to estimate their variance")
        sr_star = expected_max_sharpe(n_configurations, float(np.var(trials, ddof=1)))
        note = f"benchmark SR* = E[max SR] over {n_configurations} trials"

    dsr = probabilistic_sharpe_ratio(sr_period, n, skewness, kurt, sr_star)

    return DeflatedSharpe(
        annualised_sharpe=sr_period * math.sqrt(periods_per_year),
        per_period_sharpe=sr_period,
        n_observations=n,
        skewness=skewness,
        kurtosis=kurt,
        n_configurations=n_configurations,
        benchmark_sharpe=sr_star,
        psr_vs_zero=psr0,
        deflated_sharpe=dsr,
        significance_level=significance_level,
        note=note,
    )


@dataclass(frozen=True, slots=True)
class NoiseSweepResult:
    grid: pd.DataFrame = field(repr=False)
    best_config: dict = field(repr=False)
    best_sharpe: float = 0.0
    best_returns: pd.Series = field(default=None, repr=False)
    n_days: int = 0
    seed: int = 0

    @property
    def n_configurations(self) -> int:
        return len(self.grid)

    @property
    def per_period_sharpes(self) -> tuple[float, ...]:
        return tuple(self.grid["per_period_sharpe"])

    def deflated(self) -> DeflatedSharpe:
        return deflated_sharpe_ratio(
            self.best_returns,
            self.n_configurations,
            trial_sharpes=self.per_period_sharpes,
        )

    def __str__(self) -> str:
        d = self.deflated()
        cfg = ", ".join(f"{k}={v}" for k, v in self.best_config.items())
        return (
            f"swept {self.n_configurations} configurations on pure noise: best is "
            f"({cfg}) at Sharpe {self.best_sharpe:+.3f}; deflated {d.deflated_sharpe:.4f} "
            f"({'SIGNIFICANT' if d.is_significant else 'not significant'})"
        )


def noise_sweep(
    cfg: Config,
    *,
    seed: int = 0,
    lookbacks: Sequence[int] = (20, 40, 60, 90, 120, 160, 200, 250, 300, 400),
    halflives: Sequence[int] = (10, 20, 30, 60, 120),
    variants: Sequence[bool] = (True, False),
    n_days: int = 1250,
    annual_vol: float = 0.16,
) -> NoiseSweepResult:
    from dataclasses import replace as _replace  # noqa: PLC0415

    from .backtest import run_backtest
    from ..signal import trend_signal  # noqa: PLC0415

    prices = synthetic_prices(cfg.universe, seed=seed, n_days=n_days, annual_vol=annual_vol)

    rows: list[dict] = []
    best: tuple[float, dict, pd.Series] | None = None
    for long_only in variants:
        variant_cfg = _replace(cfg, variant="long-only" if long_only else "long-short")
        for lookback in lookbacks:
            sig = trend_signal(prices.close, int(lookback), long_only=long_only)
            for halflife in halflives:
                trial_cfg = _replace(variant_cfg, ewma_halflife_days=int(halflife))
                r = run_backtest(prices, trial_cfg, signal=sig).returns
                sd = float(r.std(ddof=1))
                sr = sharpe(r)
                config = {
                    "lookback": int(lookback),
                    "halflife": int(halflife),
                    "variant": "long-only" if long_only else "long-short",
                }
                rows.append({**config, "sharpe": sr, "per_period_sharpe": float(r.mean()) / sd if sd > 0 else 0.0})
                if best is None or sr > best[0]:
                    best = (sr, config, r)

    assert best is not None
    return NoiseSweepResult(
        grid=pd.DataFrame(rows),
        best_config=best[1],
        best_sharpe=best[0],
        best_returns=best[2],
        n_days=n_days,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class NoiseSweepStudy:
    sweeps: tuple[NoiseSweepResult, ...]

    @property
    def table(self) -> pd.DataFrame:
        rows = []
        for sweep in self.sweeps:
            d = sweep.deflated()
            rows.append(
                {
                    "seed": sweep.seed,
                    "n_configurations": sweep.n_configurations,
                    "best_sharpe": sweep.best_sharpe,
                    "best_config": ", ".join(f"{k}={v}" for k, v in sweep.best_config.items()),
                    "benchmark_sharpe_ann": d.benchmark_sharpe * math.sqrt(TRADING_DAYS_PER_YEAR),
                    "psr_vs_zero": d.psr_vs_zero,
                    "deflated_sharpe": d.deflated_sharpe,
                    "significant": d.is_significant,
                }
            )
        return pd.DataFrame(rows)

    @property
    def max_best_sharpe(self) -> float:
        return float(max(s.best_sharpe for s in self.sweeps))

    @property
    def n_mined_above_one(self) -> int:
        return sum(1 for s in self.sweeps if s.best_sharpe > 1.0)

    @property
    def any_significant(self) -> bool:
        return any(s.deflated().is_significant for s in self.sweeps)

    def __str__(self) -> str:
        return (
            f"{len(self.sweeps)} independent noise datasets, "
            f"{self.sweeps[0].n_configurations} configurations each: mined in-sample Sharpe "
            f"exceeded 1.0 on {self.n_mined_above_one} of them (best {self.max_best_sharpe:+.3f}); "
            f"the deflated Sharpe called {'AT LEAST ONE' if self.any_significant else 'none'} of them significant"
        )


def noise_sweep_study(
    cfg: Config,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    n_days: int = 750,
    **kwargs,
) -> NoiseSweepStudy:
    return NoiseSweepStudy(
        sweeps=tuple(noise_sweep(cfg, seed=seed, n_days=n_days, **kwargs) for seed in seeds)
    )


@dataclass(frozen=True, slots=True)
class DecisionRuleResult:
    strategy_sharpe: float
    benchmark_sharpe: float
    n_positive_instruments: int
    n_instruments: int
    exceeds_min_sharpe: bool
    enough_positive_instruments: bool
    beats_benchmark_by_margin: bool
    beats_benchmark_at_all: bool
    below_abandon_threshold: bool
    verdict: str
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.verdict}\n" + "\n".join(f"  - {r}" for r in self.reasons)


def evaluate_decision_rule(
    cfg: Config,
    *,
    strategy_sharpe: float,
    benchmark_sharpe: float,
    n_positive_instruments: int,
) -> DecisionRuleResult:
    exceeds_min = strategy_sharpe > cfg.support_min_sharpe
    enough_positive = n_positive_instruments >= cfg.support_min_positive_instruments
    margin = strategy_sharpe - benchmark_sharpe
    beats_by_margin = margin >= cfg.support_min_sharpe_excess_over_buy_and_hold
    beats_at_all = strategy_sharpe > benchmark_sharpe
    below_abandon = strategy_sharpe < cfg.abandon_below_sharpe

    supported = exceeds_min and enough_positive and beats_by_margin
    abandon = below_abandon or not beats_at_all

    reasons = [
        f"net Sharpe {strategy_sharpe:+.3f} {'exceeds' if exceeds_min else 'does NOT exceed'} "
        f"the {cfg.support_min_sharpe:.2f} support threshold",
        f"sign positive in {n_positive_instruments} of {n_instruments_str(cfg)} instruments, "
        f"{'meeting' if enough_positive else 'short of'} the required "
        f"{cfg.support_min_positive_instruments}",
        f"margin over equal-weight buy-and-hold {margin:+.3f} "
        f"({'>=' if beats_by_margin else '<'} the required "
        f"+{cfg.support_min_sharpe_excess_over_buy_and_hold:.2f})",
        f"{'beats' if beats_at_all else 'does NOT beat'} equal-weight buy-and-hold at all "
        f"({strategy_sharpe:+.3f} vs {benchmark_sharpe:+.3f})",
    ]

    if supported and abandon:  # pragma: no cover
        verdict = "CONTRADICTORY - the decision rule is internally inconsistent here"
    elif supported:
        verdict = "SUPPORTED"
    elif abandon:
        verdict = "ABANDON"
        reasons.append(
            f"abandonment triggered by: "
            + " and ".join(
                filter(
                    None,
                    [
                        f"Sharpe below {cfg.abandon_below_sharpe:.2f}" if below_abandon else "",
                        "failure to beat equal-weight buy-and-hold at all" if not beats_at_all else "",
                    ],
                )
            )
        )
    else:
        verdict = "INCONCLUSIVE - do not trade"

    return DecisionRuleResult(
        strategy_sharpe=strategy_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        n_positive_instruments=n_positive_instruments,
        n_instruments=cfg.n_universe,
        exceeds_min_sharpe=exceeds_min,
        enough_positive_instruments=enough_positive,
        beats_benchmark_by_margin=beats_by_margin,
        beats_benchmark_at_all=beats_at_all,
        below_abandon_threshold=below_abandon,
        verdict=verdict,
        reasons=tuple(reasons),
    )


def n_instruments_str(cfg: Config) -> int:
    return cfg.n_universe


def standalone_instrument_sharpes(
    prices,
    cfg: Config,
    *,
    risk_free=None,
    cost_bps: float | None = None,
) -> pd.Series:
    from dataclasses import replace as _replace  # noqa: PLC0415

    from ..data import PriceData  # noqa: PLC0415
    from .backtest import run_backtest  # noqa: PLC0415

    out = {}
    for ticker in cfg.universe:
        single = _replace(
            cfg,
            sleeves={cfg.sleeve_of(ticker): (ticker,)},
            universe=(ticker,),
        )
        sub = PriceData(
            open=prices.open[[ticker]],
            close=prices.close[[ticker]],
            source=prices.source,
            adjusted=prices.adjusted,
            fetched_at=prices.fetched_at,
        )
        result = run_backtest(sub, single, risk_free=risk_free, cost_bps=cost_bps)
        out[ticker] = sharpe(result.excess_returns)
    return pd.Series(out, name="standalone_sharpe")
