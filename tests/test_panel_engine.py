"""The generalised engine: the protocol, the shift, and causality.

Experiment 001 proved its engine could not see the future three ways — a causality
sweep, a load-bearing-lag test, and a static ban on negative shifts. The panel engine
inherits the third for free (it lives under ``trendbot/`` and
``tests/test_no_lookahead.py`` sweeps the whole package) and reproduces the first two
here, against the strategy that actually needs them.

The panel engine has no equivalent of experiment 001's test-only lookahead escape
hatch, so the "is the lag load-bearing" question is answered directly instead: a
hand-built target frame is fed in and the row the engine consumes at rebalance ``d``
is asserted to be the row dated ``d - 1``. That is a stronger statement than "turning
the shift off changes the answer", because it names which row is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_002 import load_config_002
from trendbot.data import PriceData
from trendbot.engine.backtest import rebalance_dates
from trendbot.engine.panel import panel_field, price_panel, validate_panel
from trendbot.engine.panel_backtest import run_panel_backtest
from trendbot.engine.validation import synthetic_prices
from trendbot.strategies import CrossSectionalMomentum


@pytest.fixture(scope="session")
def cfg002():
    return load_config_002()


@pytest.fixture(scope="session")
def synth41(cfg002):
    return synthetic_prices(cfg002.universe, seed=7, n_days=1500)


# --------------------------------------------------------------------------------------
# the protocol's input contract
# --------------------------------------------------------------------------------------


def _flat_prices(names=("A", "B", "C"), periods=70) -> PriceData:
    """Constant prices, so nothing but trading can move the equity curve."""
    index = pd.bdate_range("2020-01-01", periods=periods)
    frame = pd.DataFrame(100.0, index=index, columns=list(names))
    return PriceData(open=frame.copy(), close=frame.copy(), source="flat", adjusted=True, fetched_at="")


def test_price_panel_round_trips_to_the_wide_frames():
    prices = _flat_prices()
    panel = price_panel(prices, ["A", "B", "C"])
    assert list(panel) == ["A", "B", "C"]
    assert list(panel["A"].columns) == ["open", "close"]
    close = panel_field(panel, "close")
    assert list(close.columns) == ["A", "B", "C"]
    assert close.equals(prices.close)
    assert panel_field(panel, "open").equals(prices.open)


def test_price_panel_refuses_a_symbol_it_does_not_have():
    with pytest.raises(ValueError, match="missing panel members"):
        price_panel(_flat_prices(), ["A", "Z"])


def test_a_panel_on_two_calendars_is_refused():
    """A cross-sectional rank is meaningless if row t is a different date per symbol."""
    index = pd.bdate_range("2020-01-01", periods=10)
    panel = {
        "A": pd.DataFrame({"close": 1.0}, index=index),
        "B": pd.DataFrame({"close": 1.0}, index=index + pd.Timedelta(days=1)),
    }
    with pytest.raises(ValueError, match="different trading calendar"):
        validate_panel(panel)


def test_a_panel_missing_the_field_or_the_index_is_refused():
    index = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError, match="no 'close' column"):
        validate_panel({"A": pd.DataFrame({"open": 1.0}, index=index)})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_panel({"A": pd.DataFrame({"close": 1.0}, index=range(10))})
    with pytest.raises(ValueError, match="sorted"):
        validate_panel({"A": pd.DataFrame({"close": 1.0}, index=index[::-1])})
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel({"A": pd.DataFrame({"close": 1.0}, index=index.append(index[:1]).sort_values())})
    with pytest.raises(ValueError, match="empty"):
        validate_panel({})


def test_the_engine_takes_exactly_one_of_strategy_or_targets(cfg002):
    prices = _flat_prices()
    with pytest.raises(ValueError, match="exactly one"):
        run_panel_backtest(prices, ["A", "B", "C"], cost_bps=0.0)
    targets = pd.DataFrame(0.0, index=prices.close.index, columns=["A", "B", "C"])
    with pytest.raises(ValueError, match="exactly one"):
        run_panel_backtest(
            prices, ["A", "B", "C"], CrossSectionalMomentum(cfg002), targets=targets, cost_bps=0.0
        )


def test_targets_naming_an_instrument_outside_the_universe_are_refused():
    prices = _flat_prices()
    targets = pd.DataFrame(0.0, index=prices.close.index, columns=["A", "B", "Z"])
    with pytest.raises(ValueError, match="outside the universe"):
        run_panel_backtest(prices, ["A", "B", "C"], targets=targets, cost_bps=0.0)


# --------------------------------------------------------------------------------------
# THE shift — which row does a rebalance consume?
# --------------------------------------------------------------------------------------


def test_a_rebalance_consumes_the_previous_bars_target_row():
    """Hand-built: a target that exists on exactly one bar, traded on exactly the next.

    Prices are constant, so the only thing that can move a weight is a trade. The
    target frame is zero everywhere except the last bar of January, where it asks for
    100% of A. If the engine consumed its own bar's row, the February rebalance would
    see zero and nothing would ever be bought.
    """
    prices = _flat_prices()
    index = prices.close.index
    rebals = rebalance_dates(index)
    assert len(rebals) >= 3, "fixture needs at least three month boundaries"

    first_reb = rebals[0]
    signal_bar = index[index.get_loc(first_reb) - 1]

    targets = pd.DataFrame(0.0, index=index, columns=["A", "B", "C"])
    targets.loc[signal_bar, "A"] = 1.0

    result = run_panel_backtest(prices, ["A", "B", "C"], targets=targets, cost_bps=0.0)

    # bought at the first rebalance, i.e. exactly one bar after the target appeared
    assert result.weights.loc[first_reb, "A"] == pytest.approx(1.0)
    assert result.weights.loc[signal_bar, "A"] == pytest.approx(0.0)
    # and sold at the next rebalance, because by then the lagged row is zero again
    assert result.weights.loc[rebals[1], "A"] == pytest.approx(0.0)
    # the diagnostics say which bar the decision came from
    assert result.diagnostics.loc[first_reb, "decision_date"] == signal_bar
    assert result.diagnostics.loc[first_reb, "turnover"] == pytest.approx(1.0)
    assert result.diagnostics.loc[rebals[1], "turnover"] == pytest.approx(1.0)


def test_a_target_on_the_rebalance_bar_itself_is_not_tradeable():
    """The negative control for the test above."""
    prices = _flat_prices()
    index = prices.close.index
    first_reb = rebalance_dates(index)[0]

    targets = pd.DataFrame(0.0, index=index, columns=["A", "B", "C"])
    targets.loc[first_reb, "A"] = 1.0  # known only at that bar's close

    result = run_panel_backtest(prices, ["A", "B", "C"], targets=targets, cost_bps=0.0)
    assert result.weights.loc[first_reb, "A"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# causality — no future bar can change a past decision
# --------------------------------------------------------------------------------------


def _replace_prices_from(prices: PriceData, j: int, *, seed: int) -> PriceData:
    """Overwrite every bar from row ``j`` onward with unrelated random prices."""
    rng = np.random.default_rng(seed)
    close = prices.close.copy()
    open_ = prices.open.copy()
    shape = close.iloc[j:].shape
    close.iloc[j:] = rng.uniform(10.0, 500.0, size=shape)
    open_.iloc[j:] = rng.uniform(10.0, 500.0, size=shape)
    return PriceData(
        open=open_, close=close, source=prices.source, adjusted=True, fetched_at=""
    )


@pytest.mark.parametrize("cut", [0.4, 0.6, 0.8])
def test_no_future_bar_can_change_a_past_decision(cfg002, synth41, cut):
    j = int(len(synth41.close) * cut)
    base = run_panel_backtest(
        synth41,
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=cfg002.cost_bps_per_side,
        gross_cap=cfg002.gross_exposure_cap,
    )
    alt = run_panel_backtest(
        _replace_prices_from(synth41, j, seed=99),
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=cfg002.cost_bps_per_side,
        gross_cap=cfg002.gross_exposure_cap,
    )
    cut_date = synth41.close.index[j]
    for field in ("returns", "equity", "costs"):
        a = getattr(base, field).loc[:cut_date].iloc[:-1]
        b = getattr(alt, field).loc[:cut_date].iloc[:-1]
        assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), (
            f"{field} before {cut_date.date()} moved when only later bars changed"
        )
    a = base.weights.loc[:cut_date].iloc[:-1]
    b = alt.weights.loc[:cut_date].iloc[:-1]
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
    a = base.targets.loc[:cut_date].iloc[:-1]
    b = alt.targets.loc[:cut_date].iloc[:-1]
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_the_causality_harness_detects_a_clairvoyant_strategy(cfg002, synth41):
    """Negative control: the same sweep, pointed at a strategy that reads ahead.

    Without this, a broken harness would look like a clean engine.
    """
    universe = list(cfg002.universe)

    def clairvoyant(panel):
        close = panel_field(panel, "close")
        # 30 bars, not 5: the last rebalance before the cut can be three weeks earlier,
        # and a peek that does not reach past the cut would leave the control inert.
        ahead = close.shift(-30) / close - 1.0  # deliberate lookahead; tests may do this
        ranked = ahead.rank(axis=1, ascending=False, method="first")
        selected = ranked <= 8
        counts = selected.sum(axis=1)
        return selected.astype(float).div(counts.where(counts > 0), axis=0).fillna(0.0)

    clairvoyant.name = "clairvoyant control"

    j = int(len(synth41.close) * 0.6)
    base = run_panel_backtest(synth41, universe, clairvoyant, cost_bps=0.0)
    alt = run_panel_backtest(_replace_prices_from(synth41, j, seed=5), universe, clairvoyant, cost_bps=0.0)
    cut_date = synth41.close.index[j]
    a = base.weights.loc[:cut_date].iloc[:-1].to_numpy()
    b = alt.weights.loc[:cut_date].iloc[:-1].to_numpy()
    assert not np.array_equal(a, b), (
        "the causality sweep failed to notice a strategy that reads thirty bars ahead; "
        "the harness, not the engine, is what the other test would then be measuring"
    )


# --------------------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------------------


def test_the_book_is_fully_invested_whenever_the_ranking_exists(cfg002, synth41):
    result = run_panel_backtest(
        synth41,
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=cfg002.cost_bps_per_side,
        gross_cap=cfg002.gross_exposure_cap,
    )
    ranked = result.diagnostics["n_selected"] > 0
    assert ranked.any()
    gross = result.diagnostics.loc[ranked, "gross_target"]
    assert np.allclose(gross.to_numpy(), 1.0)
    assert (result.diagnostics.loc[ranked, "n_selected"] == 8).all()
    # section 4 caps gross at 1.0 and equal weights hit it exactly, so nothing is forced
    assert not result.diagnostics["gross_cap_forced"].any()
    assert not result.diagnostics["instrument_cap_forced"].any()


def test_costs_reconcile_with_the_turnover_actually_traded(cfg002, synth41):
    """The gross and net curves must differ by exactly the charged cost, not roughly."""
    rate = cfg002.cost_bps_per_side / 10_000.0
    result = run_panel_backtest(
        synth41,
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=cfg002.cost_bps_per_side,
        gross_cap=cfg002.gross_exposure_cap,
    )
    diag = result.diagnostics
    assert np.allclose(
        diag["cost"].to_numpy(), rate * diag["turnover"].to_numpy() * diag["equity_open"].to_numpy()
    )
    free = run_panel_backtest(
        synth41,
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=0.0,
        gross_cap=cfg002.gross_exposure_cap,
    )
    # the zero-cost run and the costed run's own gross path are the same curve
    assert np.allclose(
        free.returns.to_numpy(), result.gross_returns.to_numpy(), rtol=1e-12, atol=1e-14
    )
    assert result.equity.iloc[-1] < free.equity.iloc[-1]


def test_a_negative_cost_is_refused(cfg002):
    with pytest.raises(ValueError, match="cannot be negative"):
        run_panel_backtest(
            _flat_prices(), ["A", "B", "C"],
            targets=pd.DataFrame(0.0, index=_flat_prices().close.index, columns=["A", "B", "C"]),
            cost_bps=-1.0,
        )
