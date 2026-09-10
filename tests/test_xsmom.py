from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_002 import load_config_002
from trendbot.engine.validation import synthetic_prices
from trendbot.xsmom import (
    cross_sectional_momentum,
    quantile_labels,
    quantile_sizes,
    top_quantile_weights,
)


@pytest.fixture(scope="session")
def cfg002():
    return load_config_002()


HAND_NAMES = ["A", "B", "C", "D", "E", "F"]
HAND_ROW3 = {"A": 95.0, "B": 90.0, "C": 85.0, "D": 80.0, "E": 75.0, "F": 98.0}
HAND_EXPECTED_MOMENTUM = {"A": -0.05, "B": -0.10, "C": -0.15, "D": -0.20, "E": -0.25, "F": -0.02}
HAND_EXPECTED_ORDER = ["F", "A", "B", "C", "D", "E"]
HAND_EXPECTED_BUCKETS = {"F": 0, "A": 0, "B": 1, "C": 1, "D": 2, "E": 2}


@pytest.fixture
def hand_prices() -> pd.DataFrame:
    index = pd.date_range("2020-01-06", periods=6, freq="B")
    frame = pd.DataFrame(index=index, columns=HAND_NAMES, dtype=float)
    frame.iloc[0] = 100.0
    frame.iloc[1] = 999.0
    frame.iloc[2] = 888.0
    frame.iloc[3] = [HAND_ROW3[n] for n in HAND_NAMES]
    frame.iloc[4] = 777.0
    frame.iloc[5] = 666.0
    return frame


def test_momentum_is_the_ratio_the_document_states(hand_prices):
    momentum = cross_sectional_momentum(hand_prices, formation_days=4, skip_days=1)
    row = momentum.iloc[4]
    for name, expected in HAND_EXPECTED_MOMENTUM.items():
        assert row[name] == pytest.approx(expected), name
    assert momentum.iloc[:4].isna().all().all()


def test_the_skip_means_the_current_bar_is_not_read(hand_prices):
    momentum = cross_sectional_momentum(hand_prices, formation_days=4, skip_days=1)
    assert momentum.iloc[4].nunique() == len(HAND_NAMES)

    tampered = hand_prices.copy()
    tampered.iloc[4] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tampered.iloc[5] = 1e6
    again = cross_sectional_momentum(tampered, formation_days=4, skip_days=1)
    assert np.array_equal(
        momentum.iloc[4].to_numpy(), again.iloc[4].to_numpy(), equal_nan=True
    ), "changing the current bar changed a momentum that is supposed to skip it"


def test_the_selected_set_matches_by_hand(hand_prices):
    momentum = cross_sectional_momentum(hand_prices, formation_days=4, skip_days=1)

    order = momentum.iloc[4].sort_values(ascending=False).index.tolist()
    assert order == HAND_EXPECTED_ORDER

    labels = quantile_labels(momentum, n_quantiles=3)
    for name, bucket in HAND_EXPECTED_BUCKETS.items():
        assert labels.iloc[4][name] == float(bucket), name

    weights = top_quantile_weights(momentum, n_quantiles=3)
    held = weights.iloc[4]
    assert set(held[held > 0].index) == {"F", "A"}
    assert held["F"] == pytest.approx(0.5)
    assert held["A"] == pytest.approx(0.5)
    assert held[["B", "C", "D", "E"]].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]
    assert held.sum() == pytest.approx(1.0)


def test_an_instrument_without_history_is_excluded_not_zeroed(hand_prices):
    with_gap = hand_prices.copy()
    with_gap["G"] = np.nan

    momentum = cross_sectional_momentum(with_gap, formation_days=4, skip_days=1)
    assert np.isnan(momentum.iloc[4]["G"])

    labels = quantile_labels(momentum, n_quantiles=3)
    assert np.isnan(labels.iloc[4]["G"]), "an instrument with no momentum was given a bucket"
    for name, bucket in HAND_EXPECTED_BUCKETS.items():
        assert labels.iloc[4][name] == float(bucket), name

    weights = top_quantile_weights(momentum, n_quantiles=3)
    held = weights.iloc[4]
    assert set(held[held > 0].index) == {"F", "A"}
    assert held["G"] == 0.0

    zeroed = momentum.fillna(0.0).iloc[[4]]
    zeroed_labels = quantile_labels(zeroed, n_quantiles=3)
    assert zeroed_labels.iloc[0]["G"] == 0.0, "the counterfactual no longer bites"
    zeroed_weights = top_quantile_weights(zeroed, n_quantiles=3)
    assert zeroed_weights.iloc[0]["G"] == pytest.approx(0.5)
    assert set(zeroed_weights.iloc[0][zeroed_weights.iloc[0] > 0].index) == {"G", "F"}


def test_a_non_positive_price_is_treated_as_missing_history(hand_prices):
    broken = hand_prices.copy()
    broken.iloc[0, broken.columns.get_loc("A")] = 0.0
    momentum = cross_sectional_momentum(broken, formation_days=4, skip_days=1)
    assert np.isnan(momentum.iloc[4]["A"])


def test_ties_are_broken_deterministically_by_column_order():
    index = pd.date_range("2020-01-06", periods=5, freq="B")
    frame = pd.DataFrame(100.0, index=index, columns=["A", "B", "C", "D"])
    frame.iloc[3] = 110.0
    momentum = cross_sectional_momentum(frame, formation_days=4, skip_days=1)
    assert momentum.iloc[4].nunique() == 1

    labels = quantile_labels(momentum, n_quantiles=2)
    assert labels.iloc[4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert quantile_labels(momentum, n_quantiles=2).iloc[4].tolist() == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "n_valid,expected",
    [
        (41, (8, 8, 8, 8, 9)),
        (40, (8, 8, 8, 8, 8)),
        (44, (8, 8, 8, 8, 12)),
        (10, (2, 2, 2, 2, 2)),
        (5, (1, 1, 1, 1, 1)),
        (4, (0, 0, 0, 0, 0)),
        (0, (0, 0, 0, 0, 0)),
    ],
)


def test_quantile_sizes_put_the_remainder_in_the_bottom_bucket(n_valid, expected):
    sizes = quantile_sizes(n_valid, 5)
    assert sizes == expected
    if n_valid >= 5:
        assert sum(sizes) == n_valid


def test_the_top_bucket_holds_the_eight_names_section_3_names(cfg002):
    assert cfg002.n_universe == 41
    assert quantile_sizes(cfg002.n_universe, cfg002.n_quantiles)[0] == cfg002.declared_quantile_size

    prices = synthetic_prices(cfg002.universe, seed=0, n_days=600)
    momentum = cross_sectional_momentum(
        prices.close, cfg002.formation_days, cfg002.skip_days
    )
    weights = top_quantile_weights(momentum, cfg002.n_quantiles)

    ranked = momentum.notna().sum(axis=1) >= cfg002.n_quantiles
    held = weights[ranked]
    assert len(held) > 0
    assert ((held > 0).sum(axis=1) == 8).all()
    held_v = held.to_numpy()
    assert np.allclose(held_v[held_v > 0], 1.0 / 8.0)
    assert np.allclose(held.sum(axis=1).to_numpy(), 1.0)
    assert np.allclose(weights[~ranked].to_numpy(), 0.0)


def test_the_traded_set_is_exactly_q1(cfg002):
    prices = synthetic_prices(cfg002.universe, seed=1, n_days=600)
    momentum = cross_sectional_momentum(prices.close, cfg002.formation_days, cfg002.skip_days)
    labels = quantile_labels(momentum, cfg002.n_quantiles)
    weights = top_quantile_weights(momentum, cfg002.n_quantiles)
    assert ((labels == 0.0) == (weights > 0)).all().all()


def test_every_ranked_instrument_lands_in_exactly_one_bucket(cfg002):
    prices = synthetic_prices(cfg002.universe, seed=2, n_days=600)
    momentum = cross_sectional_momentum(prices.close, cfg002.formation_days, cfg002.skip_days)
    labels = quantile_labels(momentum, cfg002.n_quantiles)
    ranked = momentum.notna().sum(axis=1) >= cfg002.n_quantiles
    counts = labels[ranked].apply(lambda r: r.value_counts().reindex(range(5)).fillna(0), axis=1)
    assert (counts.to_numpy() == np.array([8, 8, 8, 8, 9])).all()
    assert labels[ranked].notna().sum(axis=1).eq(41).all()


def test_a_formation_window_inside_the_skip_is_refused():
    frame = pd.DataFrame(100.0, index=pd.date_range("2020-01-06", periods=5, freq="B"), columns=["A"])
    with pytest.raises(ValueError, match="must exceed"):
        cross_sectional_momentum(frame, formation_days=5, skip_days=5)
    with pytest.raises(ValueError, match="must exceed"):
        cross_sectional_momentum(frame, formation_days=3, skip_days=4)
    with pytest.raises(ValueError, match=">= 0"):
        cross_sectional_momentum(frame, formation_days=5, skip_days=-1)


def test_an_unsorted_or_duplicated_calendar_is_refused():
    index = pd.to_datetime(["2020-01-08", "2020-01-06", "2020-01-07"])
    frame = pd.DataFrame(100.0, index=index, columns=["A"])
    with pytest.raises(ValueError, match="sorted"):
        cross_sectional_momentum(frame, formation_days=2, skip_days=1)

    dupes = pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-07"])
    with pytest.raises(ValueError, match="duplicate"):
        cross_sectional_momentum(pd.DataFrame(100.0, index=dupes, columns=["A"]), 2, 1)


def test_a_single_bucket_sort_is_refused():
    frame = pd.DataFrame(
        100.0, index=pd.date_range("2020-01-06", periods=5, freq="B"), columns=["A", "B"]
    )
    momentum = cross_sectional_momentum(frame, formation_days=4, skip_days=1)
    with pytest.raises(ValueError, match="at least two buckets"):
        quantile_labels(momentum, 1)
    with pytest.raises(ValueError, match="at least two buckets"):
        quantile_sizes(10, 1)
