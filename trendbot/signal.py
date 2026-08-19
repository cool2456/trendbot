"""THE strategy.

This module contains the one and only definition of the trend signal. The backtest
engine and the live runner both import :func:`trend_signal` from here; there is no
second implementation anywhere in the repository, and ``tests/test_single_signal.py``
fails if one appears.

From PREREGISTRATION.md section 3::

    trend_i(t) = sign( P_i(t) / P_i(t-252) - 1 )

    - Lookback: 252 trading days.
    - Computed on the close of bar t using only data through bar t.
    - Undefined (insufficient history) -> position 0.
    - Long-only variant: trend_i in {0, 1}.

Two things this module deliberately does NOT do:

1. It does not lag the signal for execution. The value at index ``t`` is the signal
   *as known at the close of bar t*. Turning that into a tradeable position at the
   open of bar ``t+1`` is the engine's job and happens in exactly one place,
   :func:`trendbot.engine.backtest.lag_for_execution`. Keeping the shift out of here
   means a signal can never accidentally be used contemporaneously in one code path
   and lagged in another.

2. It does not size anything. Risk scaling lives in :mod:`trendbot.sizing`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["trend_signal", "validate_price_frame", "SIGNAL_ID"]

# Bumped only if the signal definition itself changes, which per PREREGISTRATION.md
# would make it a different strategy with a different version number.
SIGNAL_ID = "sign(P_t / P_{t-lookback} - 1), long-only clipped to {0,1}"


def validate_price_frame(prices: pd.DataFrame) -> None:
    """Reject a price frame whose index cannot support a 'trading days' row shift.

    The lookback is expressed in trading days and implemented as a row shift, which
    is only meaningful on a frame whose rows are consecutive trading days in order.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex of trading days")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be sorted ascending")
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()].unique().tolist()
        raise ValueError(f"price index has duplicate dates: {dupes[:5]}")
    if prices.columns.has_duplicates:
        raise ValueError("price frame has duplicate columns")


def trend_signal(
    prices: pd.DataFrame,
    lookback_days: int,
    *,
    long_only: bool,
) -> pd.DataFrame:
    """Time-series momentum sign, computed on the close of each bar.

    Parameters
    ----------
    prices:
        Close prices, dates on the index (ascending, unique trading days), one
        column per instrument.
    lookback_days:
        Number of trading days in the momentum lookback. From
        ``Config.lookback_days``; never a literal at the call site.
    long_only:
        From ``Config.long_only``. When true the signal is clipped to {0, 1}; when
        false it takes values in {-1, 0, +1}.

    Returns
    -------
    DataFrame of the same shape as ``prices``. The value at row ``t`` uses only
    prices at rows ``<= t``. Rows without ``lookback_days`` of history, and any
    instrument whose price is missing at either end of the lookback window, are 0.

    Notes
    -----
    ``shift(+lookback_days)`` looks *backwards*, which is what makes this function
    structurally incapable of leaking the future: there is no negative shift and no
    centred window anywhere in it.
    """
    validate_price_frame(prices)
    if not isinstance(lookback_days, (int, np.integer)) or isinstance(lookback_days, bool):
        raise TypeError(f"lookback_days must be an int, got {lookback_days!r}")
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")

    past = prices.shift(lookback_days)

    # Guard against non-positive prices, which make the ratio meaningless rather
    # than merely unknown. Treated the same as insufficient history: position 0.
    current_ok = prices.where(prices > 0)
    past_ok = past.where(past > 0)

    momentum = current_ok / past_ok - 1.0
    signal = np.sign(momentum)

    if long_only:
        signal = signal.clip(lower=0.0)

    return signal.fillna(0.0).astype(float)
