from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "annualised_vol",
    "ewma_covariance",
    "TargetWeights",
    "target_weights",
    "apply_drift_band",
    "ShareAllocation",
    "whole_share_allocation",
]

CovarianceMethod = Literal["full", "diagonal"]
TRADING_DAYS_PER_YEAR = 252


def annualised_vol(returns: pd.DataFrame, halflife: int) -> pd.DataFrame:
    if halflife < 1:
        raise ValueError("halflife must be >= 1")
    return returns.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def ewma_covariance(returns: pd.DataFrame, halflife: int) -> pd.DataFrame:
    if halflife < 1:
        raise ValueError("halflife must be >= 1")
    return returns.ewm(halflife=halflife, min_periods=halflife).cov() * TRADING_DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class TargetWeights:
    date: pd.Timestamp
    weights: pd.Series
    raw: pd.Series
    k: float
    exante_vol_raw: float
    exante_vol_scaled: float
    exante_vol_final: float
    gross_before_caps: float
    gross_after_caps: float
    capped_instruments: tuple[str, ...]
    gross_cap_binding: bool
    n_active: int

    @property
    def gross(self) -> float:
        return float(self.weights.abs().sum())


def _exante_vol(
    weights: pd.Series,
    sigma: pd.Series,
    cov: pd.DataFrame | None,
    method: CovarianceMethod,
) -> float:
    w = weights.to_numpy(dtype=float)
    if method == "diagonal":
        s = sigma.reindex(weights.index).to_numpy(dtype=float)
        term = np.where(np.isnan(s), 0.0, w * np.nan_to_num(s))
        return float(np.sqrt(np.sum(term**2)))
    if method != "full":
        raise ValueError(f"unknown covariance method {method!r}")
    if cov is None:
        raise ValueError("covariance matrix required when method='full'")
    matrix = cov.reindex(index=weights.index, columns=weights.index).to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)
    variance = float(w @ matrix @ w)
    return float(np.sqrt(max(variance, 0.0)))


def target_weights(
    date: pd.Timestamp,
    trend: pd.Series,
    sigma: pd.Series,
    cfg: Config,
    *,
    cov: pd.DataFrame | None = None,
    covariance: CovarianceMethod = "full",
) -> TargetWeights:
    universe = list(cfg.universe)
    trend = trend.reindex(universe).fillna(0.0).astype(float)
    sigma = sigma.reindex(universe).astype(float)

    usable = sigma.notna() & (sigma > 0)
    x = pd.Series(0.0, index=universe)
    x[usable] = trend[usable] * (cfg.instrument_vol_target / sigma[usable])

    w_raw = x / cfg.n_universe

    exante_raw = _exante_vol(w_raw, sigma, cov, covariance)
    k = cfg.portfolio_vol_target / exante_raw if exante_raw > 0 else 0.0
    w = w_raw * k
    exante_scaled = _exante_vol(w, sigma, cov, covariance)

    capped = tuple(w.index[w.abs() > cfg.per_instrument_cap + 1e-12])
    w = w.clip(lower=-cfg.per_instrument_cap, upper=cfg.per_instrument_cap)

    gross_before = float(w_raw.abs().sum())
    gross = float(w.abs().sum())
    gross_binding = gross > cfg.gross_exposure_cap + 1e-12
    if gross_binding:
        w = w * (cfg.gross_exposure_cap / gross)

    return TargetWeights(
        date=pd.Timestamp(date),
        weights=w,
        raw=w_raw,
        k=float(k),
        exante_vol_raw=exante_raw,
        exante_vol_scaled=exante_scaled,
        exante_vol_final=_exante_vol(w, sigma, cov, covariance),
        gross_before_caps=gross_before,
        gross_after_caps=float(w.abs().sum()),
        capped_instruments=capped,
        gross_cap_binding=gross_binding,
        n_active=int((trend != 0).sum()),
    )


def apply_drift_band(
    target: pd.Series,
    current: pd.Series,
    band: float,
) -> pd.Series:
    if band < 0:
        raise ValueError("drift band cannot be negative")
    target = target.astype(float)
    current = current.reindex(target.index).fillna(0.0).astype(float)
    move = (target - current).abs()
    threshold = band * target.abs()
    trade = move > threshold
    return target.where(trade, current)


@dataclass(frozen=True, slots=True)
class ShareAllocation:
    equity: float
    prices: pd.Series
    target_weights: pd.Series
    target_dollars: pd.Series
    shares: pd.Series
    realised_dollars: pd.Series
    realised_weights: pd.Series
    holdable: pd.Series
    cash_left: float

    @property
    def drift(self) -> pd.Series:
        return self.realised_weights - self.target_weights

    @property
    def tracking_error(self) -> float:
        return float(np.sqrt((self.drift**2).sum()))

    @property
    def absolute_error(self) -> float:
        return float(self.drift.abs().sum())

    @property
    def n_holdable(self) -> int:
        return int(self.holdable.sum())

    @property
    def n_wanted(self) -> int:
        return int((self.target_weights.abs() > 0).sum())

    @property
    def risk_budget_deployed(self) -> float:
        wanted = float(self.target_weights.abs().sum())
        return float(self.realised_weights.abs().sum()) / wanted if wanted > 0 else float("nan")


def whole_share_allocation(
    weights: pd.Series,
    prices: pd.Series,
    equity: float,
) -> ShareAllocation:
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    weights = weights.astype(float)
    prices = prices.reindex(weights.index).astype(float)
    if prices.isna().any():
        raise ValueError(f"missing price for {list(prices.index[prices.isna()])}")
    if (prices <= 0).any():
        raise ValueError(f"non-positive price for {list(prices.index[prices <= 0])}")

    target_dollars = weights * equity
    shares = np.trunc(target_dollars / prices).astype(int)
    realised_dollars = shares * prices
    realised_weights = realised_dollars / equity
    holdable = (shares != 0) | (weights == 0)

    return ShareAllocation(
        equity=float(equity),
        prices=prices,
        target_weights=weights,
        target_dollars=target_dollars,
        shares=shares,
        realised_dollars=realised_dollars,
        realised_weights=realised_weights,
        holdable=holdable.rename("holdable"),
        cash_left=float(equity - realised_dollars.abs().sum()),
    )
