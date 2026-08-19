"""Signal tests — build step 1 gate, second half:
"signal returns the documented values on a hand-built fixture".

PREREGISTRATION.md section 3::

    trend_i(t) = sign( P_i(t) / P_i(t-252) - 1 )
    - Lookback: 252 trading days.
    - Computed on the close of bar t using only data through bar t.
    - Undefined (insufficient history) -> position 0.
    - Long-only variant: trend_i in {0, 1}.

Everything here is hand-built or seeded. The lookback used in the small fixtures is
deliberately short so the expected vector can be written out in full and checked by
eye; the real 252 is exercised in the boundary and universe tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trendbot.config import Config
from trendbot.signal import SIGNAL_ID, trend_signal, validate_price_frame


def _frame(**columns: object) -> pd.DataFrame:
    """Build a price frame on consecutive business days from column value lists."""
    lengths = {len(v) for v in columns.values()}  # type: ignore[arg-type]
    assert len(lengths) == 1, "all columns must be the same length"
    n = lengths.pop()
    return pd.DataFrame(
        {k: np.asarray(v, dtype=float) for k, v in columns.items()},
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def _col(signal: pd.DataFrame, name: str) -> list[float]:
    return [float(x) for x in signal[name].to_numpy()]


# --------------------------------------------------------------------------------------
# 1. hand-built fixtures with known answers
# --------------------------------------------------------------------------------------


def test_monotonically_rising_series_is_long_after_the_lookback() -> None:
    prices = _frame(UP=[100.0 * 1.01**i for i in range(12)])
    signal = trend_signal(prices, 4, long_only=True)
    assert _col(signal, "UP") == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # the same series is +1 under long-short: the clip is the only difference
    assert _col(trend_signal(prices, 4, long_only=False), "UP") == _col(signal, "UP")


def test_monotonically_falling_series_is_flat_long_only_and_short_long_short() -> None:
    prices = _frame(DOWN=[100.0 * 0.99**i for i in range(12)])

    long_only = trend_signal(prices, 4, long_only=True)
    assert _col(long_only, "DOWN") == [0.0] * 12  # clipped to {0, 1}

    long_short = trend_signal(prices, 4, long_only=False)
    assert _col(long_short, "DOWN") == [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]


def test_perfectly_flat_series_is_exactly_zero_because_sign_of_zero_is_zero() -> None:
    prices = _frame(FLAT=[100.0] * 12)
    for long_only in (True, False):
        signal = trend_signal(prices, 4, long_only=long_only)
        assert _col(signal, "FLAT") == [0.0] * 12
        # not merely "close to zero" - exactly zero, and signed zero would be a smell
        assert (signal.to_numpy() == 0.0).all()
        assert not np.signbit(signal.to_numpy()).any()


def test_series_that_crosses_zero_flips_sign_on_the_documented_bar() -> None:
    # lookback 3. ratios: 110/100+, 105/100+, 95/100-, 90/110-, 100/105-, 120/95+
    prices = _frame(X=[100, 100, 100, 110, 105, 95, 90, 100, 120])

    assert _col(trend_signal(prices, 3, long_only=False), "X") == [0, 0, 0, 1, 1, -1, -1, -1, 1]
    assert _col(trend_signal(prices, 3, long_only=True), "X") == [0, 0, 0, 1, 1, 0, 0, 0, 1]


def test_price_exactly_equal_to_its_lookback_value_is_zero_not_long() -> None:
    # a round trip back to the starting price is not an uptrend
    prices = _frame(RT=[100, 120, 140, 100, 90])
    assert _col(trend_signal(prices, 3, long_only=False), "RT") == [0, 0, 0, 0, -1]


def test_columns_are_independent() -> None:
    prices = _frame(
        UP=[100.0 * 1.01**i for i in range(10)],
        DOWN=[100.0 * 0.99**i for i in range(10)],
        FLAT=[100.0] * 10,
    )
    signal = trend_signal(prices, 3, long_only=False)
    assert _col(signal, "UP") == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert _col(signal, "DOWN") == [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    assert _col(signal, "FLAT") == [0.0] * 10


def test_column_order_does_not_change_any_value() -> None:
    prices = _frame(
        A=[100.0 * 1.02**i for i in range(9)],
        B=[100.0 * 0.98**i for i in range(9)],
    )
    straight = trend_signal(prices, 2, long_only=False)
    swapped = trend_signal(prices[["B", "A"]], 2, long_only=False)
    assert list(swapped.columns) == ["B", "A"]
    assert_frame_equal(swapped[["A", "B"]], straight, check_exact=True)


# --------------------------------------------------------------------------------------
# 2. the history boundary - off by one here is a silent disaster
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("lookback", [1, 2, 3, 5, 63, 252])
def test_exactly_lookback_rows_are_zero_for_want_of_history(lookback: int) -> None:
    n = lookback + 5
    prices = _frame(UP=[100.0 * 1.001**i for i in range(n)])
    signal = trend_signal(prices, lookback, long_only=True)
    values = signal["UP"].to_numpy()

    assert (values[:lookback] == 0.0).all(), "warm-up rows must be flat"
    assert (values[lookback:] == 1.0).all(), "every row with full history must be long"
    # state the boundary twice, from both sides
    assert values[lookback - 1] == 0.0
    assert values[lookback] == 1.0
    assert int((values == 0.0).sum()) == lookback


def test_lookback_equal_to_the_number_of_rows_gives_no_signal_at_all() -> None:
    prices = _frame(UP=[100.0 * 1.01**i for i in range(10)])
    assert _col(trend_signal(prices, 10, long_only=True), "UP") == [0.0] * 10
    # one row shorter and exactly the final bar becomes tradeable
    assert _col(trend_signal(prices, 9, long_only=True), "UP") == [0.0] * 9 + [1.0]


def test_lookback_longer_than_the_history_is_all_zero_not_an_error() -> None:
    prices = _frame(UP=[100.0 * 1.01**i for i in range(10)])
    signal = trend_signal(prices, 5000, long_only=True)
    assert (signal.to_numpy() == 0.0).all()


def test_a_late_listing_only_trades_a_full_lookback_after_its_first_price() -> None:
    """Insufficient history is per instrument, not per frame."""
    early = [100.0 * 1.01**i for i in range(12)]
    late = [np.nan] * 5 + [100.0 * 1.01**i for i in range(7)]
    signal = trend_signal(_frame(EARLY=early, LATE=late), 3, long_only=True)

    assert _col(signal, "EARLY") == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # LATE's first price is row 5, so rows 5..7 lack a lookback value and are 0
    assert _col(signal, "LATE") == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


# --------------------------------------------------------------------------------------
# 3. missing and impossible prices
# --------------------------------------------------------------------------------------


def test_nan_at_t_zeroes_only_that_cell() -> None:
    values = [100.0 * 1.01**i for i in range(10)]
    values[5] = np.nan
    signal = trend_signal(_frame(X=values), 3, long_only=True)
    # row 5 has no current price; row 8 has no price at t-3. Nothing else moves.
    assert _col(signal, "X") == [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]


def test_nan_does_not_contaminate_other_columns() -> None:
    clean = [100.0 * 1.01**i for i in range(10)]
    holed = list(clean)
    holed[4] = np.nan
    signal = trend_signal(_frame(HOLED=holed, CLEAN=clean), 3, long_only=True)

    assert _col(signal, "HOLED") == [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    assert _col(signal, "CLEAN") == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_an_all_nan_column_is_all_zero_and_leaves_its_neighbour_alone() -> None:
    signal = trend_signal(
        _frame(DEAD=[np.nan] * 8, ALIVE=[100.0 * 1.01**i for i in range(8)]),
        2,
        long_only=False,
    )
    assert _col(signal, "DEAD") == [0.0] * 8
    assert _col(signal, "ALIVE") == [0, 0, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("bad", [0.0, -1.0, -250.0])
def test_non_positive_prices_yield_zero_at_both_ends_of_the_window(bad: float) -> None:
    values = [100.0 * 1.01**i for i in range(10)]
    values[6] = bad
    signal = trend_signal(_frame(X=values), 2, long_only=False)
    # row 6 (price is bad now) and row 8 (price was bad a lookback ago) are 0
    assert _col(signal, "X") == [0, 0, 1, 1, 1, 1, 0, 1, 0, 1]


def test_a_column_of_zeros_never_produces_a_position() -> None:
    signal = trend_signal(_frame(Z=[0.0] * 8, P=[100.0] * 8), 2, long_only=False)
    assert (signal.to_numpy() == 0.0).all()


# --------------------------------------------------------------------------------------
# 4. structural guarantees of the output
# --------------------------------------------------------------------------------------


def test_output_mirrors_the_input_shape_and_is_clean_float(synth) -> None:
    prices = synth.close
    signal = trend_signal(prices, 252, long_only=True)

    assert signal.index.equals(prices.index)
    assert list(signal.columns) == list(prices.columns)
    assert signal.shape == prices.shape
    assert (signal.dtypes == np.float64).all()
    assert not signal.isna().any().any()


def test_long_only_output_is_confined_to_zero_and_one(synth, cfg: Config) -> None:
    signal = trend_signal(synth.close, cfg.lookback_days, long_only=cfg.long_only)
    assert set(np.unique(signal.to_numpy())) <= {0.0, 1.0}
    assert (signal.iloc[: cfg.lookback_days].to_numpy() == 0.0).all()
    # a driftless random walk should still be long about half the time; observed 0.493
    active = float(signal.iloc[cfg.lookback_days :].to_numpy().mean())
    assert 0.2 < active < 0.8, active


def test_long_short_output_is_confined_to_minus_one_zero_and_one(synth) -> None:
    signal = trend_signal(synth.close, 252, long_only=False)
    assert set(np.unique(signal.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert (signal.iloc[:252].to_numpy() == 0.0).all()


def test_long_only_is_the_positive_part_of_long_short(synth) -> None:
    long_short = trend_signal(synth.close, 252, long_only=False)
    long_only = trend_signal(synth.close, 252, long_only=True)
    assert_frame_equal(long_only, long_short.clip(lower=0.0), check_exact=True)


def test_signal_does_not_mutate_its_input(synth) -> None:
    prices = synth.close.copy()
    before = prices.copy()
    trend_signal(prices, 252, long_only=True)
    assert_frame_equal(prices, before, check_exact=True)


def test_signal_id_names_the_documented_rule() -> None:
    assert "sign(" in SIGNAL_ID and "lookback" in SIGNAL_ID


# --------------------------------------------------------------------------------------
# 5. causality - row t depends only on rows <= t
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("j", [300, 900, 1500])
def test_perturbing_a_future_price_cannot_change_an_earlier_signal(synth, j: int) -> None:
    prices = synth.close
    baseline = trend_signal(prices, 252, long_only=True)

    perturbed_prices = prices.copy()
    perturbed_prices.iloc[j] *= 1.5  # a 50% shock to every instrument on bar j
    perturbed = trend_signal(perturbed_prices, 252, long_only=True)

    assert_frame_equal(perturbed.iloc[:j], baseline.iloc[:j], check_exact=True)
    # and the shock really did do something at and after j, or the test proves nothing
    assert not perturbed.iloc[j:].equals(baseline.iloc[j:])


def test_signal_on_a_truncated_history_equals_the_prefix_of_the_full_run(synth) -> None:
    """The live path sees only data up to today; it must get today's backtest value."""
    prices = synth.close
    full = trend_signal(prices, 252, long_only=True)
    for cutoff in (400, 1000, 1999):
        so_far = trend_signal(prices.iloc[: cutoff + 1], 252, long_only=True)
        assert_frame_equal(so_far, full.iloc[: cutoff + 1], check_exact=True)


def test_reversing_time_changes_the_answer(synth) -> None:
    """A sanity check that the function is directional at all: a centred or
    forward-looking window would be much less sensitive to reversal."""
    prices = synth.close.iloc[:600]
    forward = trend_signal(prices, 252, long_only=False)
    backward = trend_signal(
        pd.DataFrame(prices.to_numpy()[::-1], index=prices.index, columns=prices.columns),
        252,
        long_only=False,
    )
    assert not forward.iloc[252:].equals(backward.iloc[252:])


# --------------------------------------------------------------------------------------
# 6. input validation
# --------------------------------------------------------------------------------------


def test_validate_accepts_a_well_formed_frame() -> None:
    assert validate_price_frame(_frame(A=[1.0, 2.0, 3.0])) is None


@pytest.mark.parametrize("not_a_frame", [None, [1, 2, 3], pd.Series([1.0, 2.0]), np.zeros((3, 2))])
def test_validate_rejects_anything_that_is_not_a_dataframe(not_a_frame: object) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        validate_price_frame(not_a_frame)  # type: ignore[arg-type]


def test_validate_rejects_a_non_datetime_index() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_price_frame(prices)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        trend_signal(prices, 1, long_only=True)


def test_validate_rejects_an_unsorted_index() -> None:
    prices = _frame(A=[1.0, 2.0, 3.0])
    shuffled = prices.iloc[[2, 0, 1]]
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_price_frame(shuffled)
    with pytest.raises(ValueError, match="sorted ascending"):
        trend_signal(shuffled, 1, long_only=True)


def test_validate_rejects_a_duplicated_date() -> None:
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_price_frame(prices)
    with pytest.raises(ValueError, match="duplicate dates"):
        trend_signal(prices, 1, long_only=True)


def test_validate_rejects_duplicated_columns() -> None:
    prices = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.bdate_range("2020-01-01", periods=2),
        columns=["SPY", "SPY"],
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_price_frame(prices)
    with pytest.raises(ValueError, match="duplicate columns"):
        trend_signal(prices, 1, long_only=True)


@pytest.mark.parametrize("lookback", [0, -1, -252])
def test_non_positive_lookback_raises(lookback: int) -> None:
    with pytest.raises(ValueError, match="lookback_days must be >= 1"):
        trend_signal(_frame(A=[1.0, 2.0, 3.0]), lookback, long_only=True)


@pytest.mark.parametrize("lookback", [True, False, 252.0, 251.5, "252", None, np.float64(252.0)])
def test_non_integer_lookback_raises_type_error(lookback: object) -> None:
    # bool is an int subclass in Python; accepting True as lookback=1 would be absurd
    with pytest.raises(TypeError, match="lookback_days must be an int"):
        trend_signal(_frame(A=[1.0, 2.0, 3.0]), lookback, long_only=True)  # type: ignore[arg-type]


def test_numpy_integer_lookback_is_accepted() -> None:
    signal = trend_signal(_frame(A=[100.0, 110.0, 120.0]), np.int64(1), long_only=True)
    assert _col(signal, "A") == [0, 1, 1]


def test_long_only_must_be_passed_by_keyword() -> None:
    with pytest.raises(TypeError):
        trend_signal(_frame(A=[1.0, 2.0]), 1, True)  # type: ignore[misc]
