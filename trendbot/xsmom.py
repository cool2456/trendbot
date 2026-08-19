"""THE experiment 002 strategy: cross-sectional momentum.

This module holds the one and only definition of the rule. PREREG_002.md section 3::

    momentum_i(t) = P_i(t-21) / P_i(t-252) - 1

    Each rebalance date, rank all instruments by momentum_i. Position:
    - Top quintile (highest ~8 of 41): equal weight, long.
    - All others: zero.
    Long-only. Insufficient history for any instrument -> excluded from that ranking.

Three things this module deliberately does NOT do, mirroring :mod:`trendbot.signal`:

1. It does not lag anything for execution. The value at row ``t`` is the signal *as
   known at the close of bar t*. The single execution shift lives in
   :func:`trendbot.engine.backtest.lag_for_execution` and is applied once, by the
   engine.
2. It does not decide the rebalance calendar. It produces a value on every bar; the
   engine consumes the rows it trades on.
3. It reads no parameter from anywhere. Formation window, skip and the number of
   buckets all arrive as arguments, out of :class:`~trendbot.config_002.Config002`,
   which parses PREREG_002.md.

Bucket sizes when the count does not divide
-------------------------------------------
41 instruments do not split into five equal quintiles. Section 3 says "Top quintile
(highest ~8 of 41)", and 41 // 5 == 8, so the top bucket holds 8 and the leftover
name falls into the *bottom* bucket: sizes 8, 8, 8, 8, 9. This is the only partition
of all 41 in which the top bucket is exactly the 8 that section 3 names, and it is
the convention used for both the traded position (section 3) and the five-bucket
monotonicity gate (section 8), so the traded set and Q1 are the same set by
construction rather than by coincidence.

Where fewer than ``n_quantiles`` instruments have a defined momentum the bucket size
is zero, nothing is ranked, and no instrument is selected. That is the honest
extension of "insufficient history -> excluded from the ranking" to the case where
the ranking itself cannot exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "SIGNAL_ID",
    "cross_sectional_momentum",
    "quantile_labels",
    "top_quantile_weights",
    "quantile_sizes",
]

# Bumped only if the rule itself changes, which per PREREG_002.md section 6 would make
# it a different experiment with its own document, date and tag.
SIGNAL_ID = "P_t-skip / P_t-formation - 1, ranked cross-sectionally, top bucket equal-weight long"


def _validate(prices: pd.DataFrame, formation_days: int, skip_days: int) -> None:
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
    for name, value in (("formation_days", formation_days), ("skip_days", skip_days)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {value!r}")
    if skip_days < 0:
        raise ValueError(f"skip_days must be >= 0, got {skip_days}")
    if formation_days <= skip_days:
        raise ValueError(
            f"formation_days ({formation_days}) must exceed skip_days ({skip_days}); "
            "otherwise the ratio runs backwards in time"
        )


def cross_sectional_momentum(
    prices: pd.DataFrame,
    formation_days: int,
    skip_days: int,
) -> pd.DataFrame:
    """``momentum_i(t) = P_i(t-skip) / P_i(t-formation) - 1``, on the close of bar t.

    Returns a frame the same shape as ``prices``. The value at row ``t`` uses only
    prices at rows ``<= t - skip_days``; NaN means the instrument has no defined
    momentum on that date and must be **excluded from the ranking**, which is not the
    same as being assigned a momentum of zero.

    Both shifts are positive and therefore look strictly backwards; there is no
    negative shift and no centred window anywhere in this module.
    """
    _validate(prices, formation_days, skip_days)

    recent = prices.shift(skip_days)
    past = prices.shift(formation_days)

    # Non-positive prices make the ratio meaningless rather than merely unknown, so
    # they are treated exactly like insufficient history: excluded from the ranking.
    recent_ok = recent.where(recent > 0)
    past_ok = past.where(past > 0)

    return (recent_ok / past_ok - 1.0).astype(float)


def quantile_sizes(n_valid: int, n_quantiles: int) -> tuple[int, ...]:
    """Bucket sizes for ``n_valid`` instruments cut into ``n_quantiles`` buckets.

    Every bucket but the last holds ``n_valid // n_quantiles``; the remainder falls
    into the bottom bucket. Returned so the convention can be asserted in a test and
    printed in a report rather than inferred from behaviour.
    """
    if n_quantiles < 2:
        raise ValueError("a quantile sort needs at least two buckets")
    size = int(n_valid) // int(n_quantiles)
    if size == 0:
        return tuple([0] * n_quantiles)
    return tuple([size] * (n_quantiles - 1) + [int(n_valid) - size * (n_quantiles - 1)])


def quantile_labels(momentum: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """0-based bucket label per instrument per date; 0 is the highest momentum.

    NaN where the instrument has no defined momentum on that date, and NaN across a
    whole row when fewer than ``n_quantiles`` instruments do.
    """
    if n_quantiles < 2:
        raise ValueError("a quantile sort needs at least two buckets")

    # method="first" breaks ties by column order, which makes the label deterministic.
    # Exact ties in a twelve-month return on float prices are vanishingly rare; leaving
    # them to an averaged rank would put a name in a fractional bucket instead.
    ranks = momentum.rank(axis=1, ascending=False, method="first").to_numpy(dtype=float)
    valid = np.isfinite(ranks)
    size = valid.sum(axis=1) // int(n_quantiles)

    labels = np.full(ranks.shape, np.nan)
    rows = np.flatnonzero(size > 0)
    if len(rows):
        bucket = (ranks[rows] - 1.0) // size[rows, None]
        bucket = np.minimum(bucket, n_quantiles - 1)
        labels[rows] = np.where(valid[rows], bucket, np.nan)
    return pd.DataFrame(labels, index=momentum.index, columns=momentum.columns)


def top_quantile_weights(momentum: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """Section 3's position: equal weight across the top bucket, zero elsewhere.

    ``w_i = 1 / n_selected`` for the selected names, so gross exposure is exactly 1.0
    on any date where the ranking exists and exactly 0.0 - all cash - on any date
    where it does not.
    """
    labels = quantile_labels(momentum, n_quantiles)
    selected = labels == 0.0
    counts = selected.sum(axis=1)
    weights = selected.astype(float).div(counts.where(counts > 0), axis=0)
    return weights.fillna(0.0)
