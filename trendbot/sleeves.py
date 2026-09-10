"""The three sleeve return series of PREREG_006.md section 2, and their benchmarks.

Experiment 006 introduces no signal. Every series this module returns is produced by
running an earlier experiment's **committed, unmodified** implementation, which is why
this file contains no strategy logic at all: it is a manifest and a set of call sites.

The rule the module exists to enforce
--------------------------------------
Section 6: "Sleeve returns are taken from each experiment's existing, committed
implementation. No re-implementation, no re-parameterisation." That is easy to state
and easy to breach by accident, so :func:`reproduce` re-derives each sleeve's own
headline Sharpe and refuses to hand the series onward unless it matches the number that
experiment recorded in its findings document. A sleeve whose Sharpe has drifted means a
*prior* result has drifted, and the correct response is to stop and report it rather
than to allocate to it.

One breach in particular has no natural safety net. Sleeve C is replayed on a
carry-adjusted price panel using target weights computed on the **spot** panel — that
replay is what keeps experiment 005's configuration counter at four, because no signal
is recomputed and so no selection is possible. Recomputing the signal on the
carry-adjusted panel instead does not fail, does not warn, and does not produce a NaN:
it produces ``0.315``, which is *better* than the correct ``0.209``. Nothing in the
engine will catch it. :func:`load_sleeve_c` therefore passes ``targets=`` explicitly and
asserts the engine recorded the run as precomputed.

Excess returns, once
--------------------
Every series returned here is already in excess of the 13-week T-bill: the engines
accrue the rate on idle cash and ``excess_returns`` subtracts the same rate, and the
hand-built benchmark series subtract :func:`~trendbot.data.daily_risk_free` exactly
once. Nothing downstream may subtract it again — see FINDINGS_006.md on the double
subtraction.
"""

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

# Experiment 005's quintile bucket convention, declared in scripts/run_experiment_005.py
# as a reporting choice rather than a pre-registered parameter. Sleeve C must be built
# with the same one the headline used or it is a different sleeve.
_SLEEVE_C_BUCKET_METHOD = "even"

# Section 4 says only "005's daily-rebalanced dollar factor" and does not say whether it
# carries the same carry correction section 2 applies to sleeve C itself. Both readings
# are implemented; neither is a default. See FINDINGS_006.md.
BENCHMARK_VARIANTS = ("carry-corrected", "spot-only")

# Recorded outputs of the earlier experiments — NOT parameters, and not editable to make
# a gate pass. Provenance of each number is given: 001's two come from
# trendbot/regression.py at full precision, which is itself the committed regression
# target; the rest are stated in their findings documents to three decimal places, which
# is therefore the precision the gate can assert.
SLEEVE_MANIFEST: tuple[dict, ...] = (
    {
        "label": "A",
        "experiment": "001",
        "name": "time-series trend, 12 ETFs, long-only",
        "recorded_sharpe": EXPERIMENT_001_NET_SHARPE,
        "recorded_benchmark_sharpe": EXPERIMENT_001_BENCHMARK_SHARPE,
        "decimals": 3,
        "exact": True,  # trendbot/regression.py carries full precision
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

# The spot-only reading of section 4 for sleeve C, recorded so the alternative benchmark
# is gated exactly as the headline one is. FINDINGS_005.md section 9.
SLEEVE_C_SPOT_BENCHMARK_SHARPE = -0.431
SLEEVE_C_SPOT_STRATEGY_SHARPE = -0.271


@dataclass(frozen=True, slots=True)
class Sleeve:
    """One sleeve: its own returns, its own benchmark, and its own recorded result."""

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
    """Section 6's gate: every sleeve reproduces its own committed headline."""

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
    """The three sleeves of section 2, on their own calendars, before alignment."""

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


# --------------------------------------------------------------------------------------
# sleeve A - experiment 001
# --------------------------------------------------------------------------------------


def load_sleeve_a(*, source: str = "yahoo", full_history: bool = False) -> Sleeve:
    """Experiment 001's headline: time-series trend over the twelve ETFs of section 2.

    The window is ``full_universe_start`` — the first bar on which every instrument has
    a complete 252-day lookback — through the last bar of the data, which is exactly
    what ``EXPERIMENT_001_WINDOW`` records. The backtest is run over the *full* history
    and sliced afterwards, because slicing the prices first would change the state the
    engine carries into the window.
    """
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
        # Section 8 of PREREGISTRATION.md, read literally: bought once at the start of
        # the window, then held. This is the construction FINDINGS.md headlines.
        benchmark=(buy_and_hold(prices, cfg, rebalance="none", start=start) - rf_daily).loc[start:],
        recorded_sharpe=entry["recorded_sharpe"],
        recorded_benchmark_sharpe=entry["recorded_benchmark_sharpe"],
        decimals=entry["decimals"],
        exact=entry["exact"],
        source=entry["source"],
    )


# --------------------------------------------------------------------------------------
# sleeve B - experiment 002
# --------------------------------------------------------------------------------------


def load_sleeve_b(*, source: str = "yahoo", full_history: bool = False) -> Sleeve:
    """Experiment 002's headline: top-quintile cross-sectional momentum over 41 ETFs.

    The window is section 6's pre-committed ``2008-01-01 to present``; history before it
    forms the signal and is measured for nothing.
    """
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
        drift_band=None,  # PREREG_002 section 5: full rebalance each month
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


# --------------------------------------------------------------------------------------
# sleeve C - experiment 005, carry-corrected
# --------------------------------------------------------------------------------------


def _last_complete_session(index: pd.DatetimeIndex) -> pd.Timestamp:
    """The last session strictly before today — experiment 005's own end-of-data rule."""
    earlier = index[index < pd.Timestamp.today().normalize()]
    if len(earlier) == 0:
        raise ValueError("no completed session in the exchange rate history")
    return earlier[-1]


def load_sleeve_c(*, benchmark_variant: str, refresh: bool = False, full_history: bool = False) -> Sleeve:
    """Experiment 005's **carry-corrected** result — PREREG_005 section 4's diagnostic.

    The correction is the foreign interest accrual and nothing else: the same target
    weights, computed once on the spot panel, are replayed against a price panel in
    which each currency is a money-market deposit rather than the bare currency. That
    replay is what keeps 005's configuration counter at four, and re-deriving the signal
    on the carry-adjusted panel would break it silently — see the module docstring.

    ``benchmark_variant`` selects the reading of PREREG_006 section 4 for this sleeve's
    benchmark and has no default, because the two readings move section 8's first clause
    across zero.
    """
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

    # THE signal, computed once, on the SPOT panel.
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
        targets=targets,  # REPLAYED, never recomputed - see the module docstring
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


# --------------------------------------------------------------------------------------
# the set, and section 6's reproduction gate
# --------------------------------------------------------------------------------------


def load_sleeves(
    *,
    benchmark_variant: str,
    source: str = "yahoo",
    refresh: bool = False,
    full_history: bool = False,
) -> SleeveSet:
    """Load all three sleeves of section 2. No selection step exists, by design.

    ``full_history`` reads section 5's "available histories" as the raw extent of each
    experiment's computable return series rather than as its pre-committed window. It is
    a **sensitivity, not the headline**, and it changes what sleeve B is: only 20 of
    002's 41 ETFs had listed by 1999, so an earlier start makes sleeve B a
    top-quintile-of-20 strategy. Section 2 forbids that as a construction; it is run
    here only so section 9's stated expectation can be tested rather than merely
    contradicted, and the reproduction gate is not applied to it.
    """
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
    """Assert every sleeve still produces the Sharpe its own findings document records.

    A failure here is not a failure of experiment 006. It means a *prior* result has
    drifted, and the only correct response is to stop and report that.
    """
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
        # 001 carries its recorded value at full precision, so it is held to it.
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
