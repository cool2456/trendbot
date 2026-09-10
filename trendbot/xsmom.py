from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "SIGNAL_ID",
    "BucketMethod",
    "cross_sectional_momentum",
    "quantile_labels",
    "top_quantile_weights",
    "quantile_sizes",
]

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
    _validate(prices, formation_days, skip_days)

    recent = prices.shift(skip_days)
    past = prices.shift(formation_days)

    recent_ok = recent.where(recent > 0)
    past_ok = past.where(past > 0)

    return (recent_ok / past_ok - 1.0).astype(float)


BucketMethod = Literal["floor", "even"]


def _check_method(method: str) -> None:
    if method not in ("floor", "even"):
        raise ValueError(
            f"unknown bucket method {method!r}; expected 'floor' (experiment 002's "
            "remainder-to-the-bottom rule) or 'even' (sizes differing by at most one)"
        )


def quantile_sizes(
    n_valid: int, n_quantiles: int, method: BucketMethod = "floor"
) -> tuple[int, ...]:
    if n_quantiles < 2:
        raise ValueError("a quantile sort needs at least two buckets")
    _check_method(method)
    n, k = int(n_valid), int(n_quantiles)
    if n < k:
        return tuple([0] * k)
    if method == "floor":
        size = n // k
        return tuple([size] * (k - 1) + [n - size * (k - 1)])
    edges = [-(-(b * n) // k) for b in range(k)] + [n]
    return tuple(edges[b + 1] - edges[b] for b in range(k))


def quantile_labels(
    momentum: pd.DataFrame, n_quantiles: int, method: BucketMethod = "floor"
) -> pd.DataFrame:
    if n_quantiles < 2:
        raise ValueError("a quantile sort needs at least two buckets")
    _check_method(method)

    ranks = momentum.rank(axis=1, ascending=False, method="first").to_numpy(dtype=float)
    valid = np.isfinite(ranks)
    n_valid = valid.sum(axis=1)
    k = int(n_quantiles)

    labels = np.full(ranks.shape, np.nan)
    rows = np.flatnonzero(n_valid >= k)
    if len(rows):
        rank0 = ranks[rows] - 1.0
        if method == "floor":
            bucket = rank0 // (n_valid[rows] // k)[:, None]
        else:
            bucket = (rank0 * k) // n_valid[rows][:, None]
        bucket = np.minimum(bucket, k - 1)
        labels[rows] = np.where(valid[rows], bucket, np.nan)
    return pd.DataFrame(labels, index=momentum.index, columns=momentum.columns)


def top_quantile_weights(
    momentum: pd.DataFrame, n_quantiles: int, method: BucketMethod = "floor"
) -> pd.DataFrame:
    labels = quantile_labels(momentum, n_quantiles, method)
    selected = labels == 0.0
    counts = selected.sum(axis=1)
    weights = selected.astype(float).div(counts.where(counts > 0), axis=0)
    return weights.fillna(0.0)
