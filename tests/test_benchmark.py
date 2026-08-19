"""The section 8 benchmark.

This is the quantity the pre-committed decision rule is measured against, and a
0.11-of-Sharpe difference between constructions is enough to change the verdict, so
it gets its own tests rather than riding along with the engine's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.data import PriceData
from trendbot.engine.backtest import buy_and_hold
from trendbot.engine.metrics import equity_curve


def _panel(close: pd.DataFrame) -> PriceData:
    return PriceData(open=close.copy(), close=close, source="test", adjusted=True, fetched_at="")


@pytest.fixture
def two_asset_cfg(cfg):
    import dataclasses

    return dataclasses.replace(cfg, sleeves={"Equity": ("A", "B")}, universe=("A", "B"))


def test_unknown_rebalance_mode_raises(synth, cfg):
    with pytest.raises(ValueError, match="unknown rebalance mode"):
        buy_and_hold(synth, cfg, rebalance="weekly")


def test_buy_and_hold_lets_weights_drift(two_asset_cfg):
    # A doubles, B is flat. Bought 50/50, the book ends 1.5x and A ends at 2/3 weight.
    index = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"A": [100.0, 150.0, 200.0], "B": [100.0, 100.0, 100.0]}, index=index)
    returns = buy_and_hold(_panel(close), two_asset_cfg, rebalance="none")
    curve = equity_curve(returns)
    assert curve.iloc[-1] == pytest.approx(1.5)
    # day 1: +25% (half of A's +50%), day 2: A is 150/250 of the book and rises 33.3%
    assert returns.iloc[1] == pytest.approx(0.25)
    assert returns.iloc[2] == pytest.approx(0.2)


def test_daily_rebalancing_differs_from_buy_and_hold_on_the_same_data(two_asset_cfg):
    index = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"A": [100.0, 150.0, 200.0], "B": [100.0, 100.0, 100.0]}, index=index)
    panel = _panel(close)
    held = equity_curve(buy_and_hold(panel, two_asset_cfg, rebalance="none")).iloc[-1]
    daily = equity_curve(buy_and_hold(panel, two_asset_cfg, rebalance="daily")).iloc[-1]
    # rebalancing back to 50/50 after day 1 sells the winner, so it lags a trending asset
    assert daily == pytest.approx(1.0 * 1.25 * (1 + 0.5 * (200 / 150 - 1)))
    assert daily != pytest.approx(held)


def test_a_late_listing_instrument_is_bought_when_it_appears(two_asset_cfg):
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [np.nan, np.nan, 50.0, 55.0]}, index=index
    )
    returns = buy_and_hold(_panel(close), two_asset_cfg, rebalance="none")
    # While only A exists the book is 100% A and earns A's return.
    assert returns.iloc[1] == pytest.approx(0.10)
    # B becomes tradeable once it has a previous close, at which point the book is
    # re-levelled to 50/50 and both rise 10%.
    assert returns.iloc[-1] == pytest.approx(0.10)
    assert not returns.isna().any()


def test_the_investable_set_changing_is_the_only_thing_that_triggers_a_re_level(cfg, synth):
    # On a panel where all twelve exist throughout, buy-and-hold must never trade, so
    # its weights must genuinely drift apart over time.
    returns = buy_and_hold(synth, cfg, rebalance="none")
    monthly = buy_and_hold(synth, cfg, rebalance="monthly")
    assert not np.allclose(returns.to_numpy(), monthly.to_numpy())


def test_all_three_constructions_are_finite_and_start_at_zero(cfg, synth):
    for mode in ("none", "monthly", "daily"):
        returns = buy_and_hold(synth, cfg, rebalance=mode)
        assert returns.index.equals(synth.close.index)
        assert np.isfinite(returns.to_numpy()).all()
        assert returns.iloc[0] == 0.0


def test_start_argument_trims_the_series(cfg, synth):
    cut = synth.close.index[500]
    trimmed = buy_and_hold(synth, cfg, rebalance="none", start=cut)
    assert trimmed.index[0] == cut
    assert len(trimmed) == len(synth.close.loc[cut:])


def test_buy_and_hold_earns_nothing_on_driftless_noise(cfg):
    from trendbot.engine.metrics import sharpe
    from trendbot.engine.validation import synthetic_prices

    sharpes = [
        sharpe(buy_and_hold(synthetic_prices(cfg.universe, seed=s, n_days=3000), cfg))
        for s in range(4)
    ]
    # observed mean on this machine: -0.06; a wide bound keeps it non-brittle
    assert abs(float(np.mean(sharpes))) < 0.4


def test_the_benchmark_must_be_bought_at_the_window_being_reported(cfg):
    """Slicing a 1993-bought basket to 2008 does not give an equal-weight 2008 basket.

    Passing ``start`` buys 1/12 of each on the first day of the reported window.
    Omitting it and slicing afterwards reports a basket carrying fifteen years of
    drift, which is a different portfolio and a different Sharpe.
    """
    from trendbot.engine.metrics import sharpe
    from trendbot.engine.validation import synthetic_prices

    prices = synthetic_prices(cfg.universe, seed=3, n_days=2000)
    cut = prices.close.index[1000]
    windowed = buy_and_hold(prices, cfg, start=cut)
    sliced = buy_and_hold(prices, cfg).loc[cut:]
    assert windowed.index[0] == cut
    assert not np.allclose(windowed.to_numpy(), sliced.to_numpy())
    assert sharpe(windowed) != pytest.approx(sharpe(sliced))


def test_a_one_day_vendor_hole_does_not_liquidate_the_instrument(two_asset_cfg):
    """A missing close is a missing print, not a delisting.

    Treating it as one would re-level the basket across the survivors and buy back a
    day later - the opposite of buy-and-hold, and worth measurable Sharpe.
    """
    index = pd.bdate_range("2020-01-01", periods=5)
    intact = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [100.0, 100.0, 100.0, 100.0, 100.0]},
        index=index,
    )
    holed = intact.copy()
    holed.loc[index[2], "B"] = np.nan

    base = equity_curve(buy_and_hold(_panel(intact), two_asset_cfg, rebalance="none"))
    with_hole = equity_curve(buy_and_hold(_panel(holed), two_asset_cfg, rebalance="none"))
    # B is flat throughout, so blanking one of its prints must not move the book at all
    assert with_hole.iloc[-1] == pytest.approx(base.iloc[-1])
