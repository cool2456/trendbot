from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_003 import load_config_003
from trendbot.config_005 import load_config_005
from trendbot.engine.panel import price_panel
from trendbot.engine.validation import synthetic_prices
from trendbot.strategies import CurrencyCrossSectionalMomentum, EquityCrossSectionalMomentum
from trendbot.xsmom import quantile_sizes

UNIVERSE = tuple(f"FX{i:02d}" for i in range(22))


@pytest.fixture(scope="module")
def cfg005():
    return load_config_005()


@pytest.fixture(scope="module")
def panel():
    prices = synthetic_prices(UNIVERSE, seed=7, n_days=1500, annual_vol=0.10)
    return price_panel(prices, UNIVERSE)


def test_the_signal_is_the_same_object_as_experiment_003s(cfg005, panel):
    cfg003 = load_config_003()
    assert (cfg005.formation_days, cfg005.skip_days) == (cfg003.formation_days, cfg003.skip_days)
    currency = CurrencyCrossSectionalMomentum(cfg005, UNIVERSE, "even")
    equity = EquityCrossSectionalMomentum(cfg003, UNIVERSE, "even")
    pd.testing.assert_frame_equal(currency.momentum(panel), equity.momentum(panel))


def test_the_traded_book_is_the_top_quintile_and_sums_to_one(cfg005, panel):
    weights = CurrencyCrossSectionalMomentum(cfg005, UNIVERSE, "even")(panel)
    live = weights.loc[weights.abs().sum(axis=1) > 0]
    assert not live.empty
    assert np.allclose(live.sum(axis=1), 1.0)
    assert (weights >= 0).all().all(), "section 4 is long-only"
    held = (live.abs() > 0).sum(axis=1).unique()
    assert set(held) == {quantile_sizes(len(UNIVERSE), cfg005.n_quantiles, "even")[0]}


def test_the_even_split_keeps_the_two_gated_ends_the_same_size(cfg005):
    sizes = quantile_sizes(22, cfg005.n_quantiles, "even")
    assert abs(sizes[0] - sizes[-1]) <= 1
    floor_sizes = quantile_sizes(22, cfg005.n_quantiles, "floor")
    assert floor_sizes[-1] - floor_sizes[0] == 2


def test_the_book_size_matches_what_section_4_predicted(cfg005):
    for n in range(cfg005.expected_universe_low, cfg005.expected_universe_high + 1):
        assert quantile_sizes(n, cfg005.n_quantiles, "even")[0] in (4, 5)


def test_the_strategy_never_lags_its_own_output(cfg005, panel):
    close = pd.DataFrame({s: f["close"] for s, f in panel.items()})
    weights = CurrencyCrossSectionalMomentum(cfg005, UNIVERSE, "even")(panel)
    momentum = CurrencyCrossSectionalMomentum(cfg005, UNIVERSE, "even").momentum(panel)
    first_signal = momentum.dropna(how="all").index[0]
    first_weight = weights.loc[weights.abs().sum(axis=1) > 0].index[0]
    assert first_weight == first_signal
    assert len(close) == len(weights)


def test_a_currency_with_insufficient_history_is_excluded_not_zeroed(cfg005):
    prices = synthetic_prices(UNIVERSE, seed=3, n_days=600, annual_vol=0.10)
    late = prices.close.copy()
    late.iloc[:400, 0] = np.nan
    opens = prices.open.copy()
    opens.iloc[:400, 0] = np.nan
    panel = price_panel(prices.__class__(open=opens, close=late, source="t", adjusted=True, fetched_at=""), UNIVERSE)
    momentum = CurrencyCrossSectionalMomentum(cfg005, UNIVERSE, "even").momentum(panel)
    early = momentum.iloc[450]
    assert np.isnan(early.iloc[0]), "the late starter was ranked before it had history"
    assert early.iloc[1:].notna().any(), "the other currencies should still rank"
