from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["trend_signal", "validate_price_frame", "SIGNAL_ID"]

SIGNAL_ID = "sign(P_t / P_{t-lookback} - 1), long-only clipped to {0,1}"


def validate_price_frame(prices: pd.DataFrame) -> None:
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
    validate_price_frame(prices)
    if not isinstance(lookback_days, (int, np.integer)) or isinstance(lookback_days, bool):
        raise TypeError(f"lookback_days must be an int, got {lookback_days!r}")
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")

    past = prices.shift(lookback_days)

    current_ok = prices.where(prices > 0)
    past_ok = past.where(past > 0)

    momentum = current_ok / past_ok - 1.0
    signal = np.sign(momentum)

    if long_only:
        signal = signal.clip(lower=0.0)

    return signal.fillna(0.0).astype(float)
