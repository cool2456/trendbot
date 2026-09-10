from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import load_config
from .config_002 import load_config_002
from .config_005 import load_config_005
from .data import PriceData, daily_risk_free, load_prices, load_risk_free_rate
from .engine.backtest import buy_and_hold, full_universe_start, run_backtest
from .engine.fx_validation import carry_adjusted_prices, short_rate_panel
from .engine.metrics import sharpe
from .engine.panel import price_panel
from .engine.panel_backtest import panel_buy_and_hold, run_panel_backtest
from .fx import build_fx_universe, dollar_factor_returns, fetch_short_rates, fx_price_data
from .regression import EXPERIMENT_001_BENCHMARK_SHARPE, EXPERIMENT_001_NET_SHARPE
from .strategies import CrossSectionalMomentum, CurrencyCrossSectionalMomentum, TimeSeriesTrend

__all__ = [
    "Sleeve",
    "SleeveSet",
    "SLEEVE_MANIFEST",
    "BENCHMARK_VARIANTS",
    "load_sleeve_a",
    "load_sleeve_b",
    "load_sleeve_c",
    "load_sleeves",
    "reproduce",
    "ReproductionCheck",
]

_SLEEVE_C_BUCKET_METHOD = "even"

BENCHMARK_VARIANTS = ("carry-corrected", "spot-only")

SLEEVE_MANIFEST: tuple[dict, ...] = (
    {
        "label": "A",
        "experiment": "001",
        "name": "time-series trend, 12 ETFs, long-only",
        "recorded_sharpe": EXPERIMENT_001_NET_SHARPE,
        "recorded_benchmark_sharpe": EXPERIMENT_001_BENCHMARK_SHARPE,
        "decimals": 3,
        "exact": True,
        "benchmark_name": "equal-weight 12-ETF buy-and-hold",
        "source": "FINDINGS.md / trendbot/regression.py",
    },
    {
        "label": "B",
        "experiment": "002",
        "name": "cross-sectional momentum, 41 ETFs, top quintile",
        "recorded_sharpe": 0.463,
        "recorded_benchmark_sharpe": 0.487,
        "decimals": 3,
        "exact": False,
        "benchmark_name": "equal-weight 41-ETF buy-and-hold",
        "source": "FINDINGS_002.md section 1",
    },
    {
        "label": "C",
        "experiment": "005",
        "name": "cross-sectional currency momentum, 22 pairs, carry-corrected",
        "recorded_sharpe": 0.209,
        "recorded_benchmark_sharpe": 0.090,
        "decimals": 3,
        "exact": False,
        "benchmark_name": "daily-rebalanced dollar factor",
        "source": "FINDINGS_005.md section 9, 'The delta'",
    },
)

SLEEVE_C_SPOT_BENCHMARK_SHARPE = -0.431
SLEEVE_C_SPOT_STRATEGY_SHARPE = -0.271


@dataclass(frozen=True, slots=True)
class Sleeve:
    label: str
    experiment: str
    name: str
    benchmark_name: str
    returns: pd.Series
    benchmark: pd.Series
    recorded_sharpe: float
    recorded_benchmark_sharpe: float
    decimals: int
    exact: bool
    source: str

    @property
    def realised_sharpe(self) -> float:
        return sharpe(self.returns)

    @property
    def realised_benchmark_sharpe(self) -> float:
        return sharpe(self.benchmark)

    @property
    def window(self) -> tuple[str, str]:
        return str(self.returns.index[0].date()), str(self.returns.index[-1].date())

    def __str__(self) -> str:
        lo, hi = self.window
        return (
            f"sleeve {self.label} ({self.experiment}) {self.name}: "
            f"Sharpe {self.realised_sharpe:+.4f} vs recorded {self.recorded_sharpe:+.4f}, "
            f"{len(self.returns)} bars {lo}..{hi}"
        )


@dataclass(frozen=True, slots=True)
class ReproductionCheck:
    rows: pd.DataFrame
    passed: bool

    def __str__(self) -> str:
        return (
            f"sleeve reproduction: {'PASS' if self.passed else 'FAIL'} "
            f"({int(self.rows['ok'].sum())}/{len(self.rows)} sleeves reproduce their "
            "recorded headline)"
        )


@dataclass(frozen=True, slots=True)
class SleeveSet:
    sleeves: tuple[Sleeve, ...]
    benchmark_variant: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.sleeves)

    @property
    def returns(self) -> pd.DataFrame:
        return pd.DataFrame({s.label: s.returns for s in self.sleeves})

    @property
    def benchmarks(self) -> pd.DataFrame:
        return pd.DataFrame({s.label: s.benchmark for s in self.sleeves})

    def __getitem__(self, label: str) -> Sleeve:
        for sleeve in self.sleeves:
            if sleeve.label == label:
                return sleeve
        raise KeyError(f"no sleeve labelled {label!r}; have {self.labels}")


def _manifest(label: str) -> dict:
    for entry in SLEEVE_MANIFEST:
        if entry["label"] == label:
            return entry
    raise KeyError(f"no manifest entry for sleeve {label!r}")


def load_sleeve_a(*, source: str = "yahoo", full_history: bool = False) -> Sleeve:
    cfg = load_config()
    prices = load_prices(cfg.universe, source=source)
    rf = load_risk_free_rate()
    rf_daily = daily_risk_free(rf, prices.close.index)
    start = prices.close.index[0] if full_history else full_universe_start(prices, cfg)

    result = run_backtest(prices, cfg, risk_free=rf)
    entry = _manifest("A")
    return Sleeve(
        label="A",
        experiment="001",
        name=entry["name"],
        benchmark_name=entry["benchmark_name"],
        returns=result.excess_returns.loc[start:],
        benchmark=(buy_and_hold(prices, cfg, rebalance="none", start=start) - rf_daily).loc[start:],
        recorded_sharpe=entry["recorded_sharpe"],
        recorded_benchmark_sharpe=entry["recorded_benchmark_sharpe"],
        decimals=entry["decimals"],
        exact=entry["exact"],
        source=entry["source"],
    )


def load_sleeve_b(*, source: str = "yahoo", full_history: bool = False) -> Sleeve:
    cfg = load_config_002()
    prices = load_prices(cfg.universe, source=source)
    rf = load_risk_free_rate()
    rf_daily = daily_risk_free(rf, prices.close.index)
    start = prices.close.index[0] if full_history else pd.Timestamp(cfg.sample_start)

    targets = CrossSectionalMomentum(cfg)(price_panel(prices, cfg.universe))
    result = run_panel_backtest(
        prices,
        cfg.universe,
        None,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )
    entry = _manifest("B")
    return Sleeve(
        label="B",
        experiment="002",
        name=entry["name"],
        benchmark_name=entry["benchmark_name"],
        returns=result.excess_returns.loc[start:],
        benchmark=(
            panel_buy_and_hold(prices, cfg.universe, rebalance="none", start=start) - rf_daily
        ).loc[start:],
        recorded_sharpe=entry["recorded_sharpe"],
        recorded_benchmark_sharpe=entry["recorded_benchmark_sharpe"],
        decimals=entry["decimals"],
        exact=entry["exact"],
        source=entry["source"],
    )


def _last_complete_session(index: pd.DatetimeIndex) -> pd.Timestamp:
    earlier = index[index < pd.Timestamp.today().normalize()]
    if len(earlier) == 0:
        raise ValueError("no completed session in the exchange rate history")
    return earlier[-1]


def load_sleeve_c(*, benchmark_variant: str, refresh: bool = False, full_history: bool = False) -> Sleeve:
    if benchmark_variant not in BENCHMARK_VARIANTS:
        raise ValueError(
            f"unknown benchmark variant {benchmark_variant!r}; expected one of {BENCHMARK_VARIANTS}"
        )
    cfg = load_config_005()
    universe = build_fx_universe(sample_start=cfg.sample_start, refresh=refresh)
    prices = fx_price_data(universe)
    end = _last_complete_session(prices.close.index)
    prices = prices.slice(None, str(end.date()))
    rf = load_risk_free_rate()
    rf_daily = daily_risk_free(rf, prices.close.index)
    start = prices.close.index[0] if full_history else pd.Timestamp(cfg.sample_start)
    members = list(universe.universe)

    targets = CurrencyCrossSectionalMomentum(cfg, universe.universe, _SLEEVE_C_BUCKET_METHOD)(
        price_panel(prices, members)
    )
    coverage = short_rate_panel(
        fetch_short_rates(universe.universe, refresh=refresh), universe.universe, prices.close.index
    )
    total = PriceData(
        open=carry_adjusted_prices(prices.open[members], coverage.rates[members]),
        close=carry_adjusted_prices(prices.close[members], coverage.rates[members]),
        source=prices.source + " + foreign short-rate accrual",
        adjusted=True,
        fetched_at="",
    )
    result = run_panel_backtest(
        total,
        members,
        None,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )
    if result.strategy_name != "precomputed":
        raise RuntimeError(
            f"sleeve C was run with strategy {result.strategy_name!r} rather than replayed "
            "target weights. Recomputing the signal on the carry-adjusted panel is a "
            "re-parameterisation PREREG_006 section 2 forbids, and it does not fail "
            "loudly: it returns a higher Sharpe than the correct one."
        )

    entry = _manifest("C")
    if benchmark_variant == "carry-corrected":
        benchmark = (dollar_factor_returns(universe, total) - rf_daily).loc[start:end]
        recorded_benchmark = entry["recorded_benchmark_sharpe"]
        benchmark_name = entry["benchmark_name"] + ", carry-corrected"
    else:
        benchmark = (dollar_factor_returns(universe, prices) - rf_daily).loc[start:end]
        recorded_benchmark = SLEEVE_C_SPOT_BENCHMARK_SHARPE
        benchmark_name = entry["benchmark_name"] + ", spot-only"

    return Sleeve(
        label="C",
        experiment="005",
        name=entry["name"],
        benchmark_name=benchmark_name,
        returns=result.excess_returns.loc[start:end],
        benchmark=benchmark,
        recorded_sharpe=entry["recorded_sharpe"],
        recorded_benchmark_sharpe=recorded_benchmark,
        decimals=entry["decimals"],
        exact=entry["exact"],
        source=entry["source"],
    )


def load_sleeves(
    *,
    benchmark_variant: str,
    source: str = "yahoo",
    refresh: bool = False,
    full_history: bool = False,
) -> SleeveSet:
    return SleeveSet(
        sleeves=(
            load_sleeve_a(source=source, full_history=full_history),
            load_sleeve_b(source=source, full_history=full_history),
            load_sleeve_c(
                benchmark_variant=benchmark_variant, refresh=refresh, full_history=full_history
            ),
        ),
        benchmark_variant=benchmark_variant,
    )


def reproduce(sleeve_set: SleeveSet) -> ReproductionCheck:
    rows = []
    for sleeve in sleeve_set.sleeves:
        realised = sleeve.realised_sharpe
        realised_bench = sleeve.realised_benchmark_sharpe
        strategy_ok = round(realised, sleeve.decimals) == round(
            sleeve.recorded_sharpe, sleeve.decimals
        )
        benchmark_ok = round(realised_bench, sleeve.decimals) == round(
            sleeve.recorded_benchmark_sharpe, sleeve.decimals
        )
        if sleeve.exact:
            strategy_ok = strategy_ok and realised == sleeve.recorded_sharpe
            benchmark_ok = benchmark_ok and realised_bench == sleeve.recorded_benchmark_sharpe
        lo, hi = sleeve.window
        rows.append(
            {
                "sleeve": sleeve.label,
                "experiment": sleeve.experiment,
                "bars": len(sleeve.returns),
                "window": f"{lo}..{hi}",
                "recorded": sleeve.recorded_sharpe,
                "realised": realised,
                "recorded benchmark": sleeve.recorded_benchmark_sharpe,
                "realised benchmark": realised_bench,
                "ok": bool(strategy_ok and benchmark_ok),
                "source": sleeve.source,
            }
        )
    frame = pd.DataFrame(rows).set_index("sleeve")
    return ReproductionCheck(rows=frame, passed=bool(frame["ok"].all()))
