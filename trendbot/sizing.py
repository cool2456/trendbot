"""Risk scaling and position sizing.

Implements PREREGISTRATION.md section 4 and the whole-share translation that turns
weights into an order-able integer share count.

Section 4, verbatim::

    sigma_i(t) = ewm_std(daily_returns_i, halflife=30) * sqrt(252)
    x_i(t)     = trend_i(t) * (0.10 / sigma_i(t))
    w_i(t)     = x_i(t) / 12
    w(t)       = k(t) * w(t)      # k scales ex-ante portfolio vol to 10% annualised

    - Gross exposure cap: sum(|w_i|) <= 1.0
    - Per-instrument cap: |w_i| <= 0.25

Three things section 4 leaves open, and how they are settled here. All three are
disclosed in FINDINGS.md; none of them is a free parameter that has been tuned.

1. **How "ex-ante portfolio vol" is computed.** Taken to mean ``sqrt(w' S w)`` with
   ``S`` the EWMA covariance matrix at the same halflife of 30 days that section 4
   already specifies for the per-instrument vol. The alternative reading - assume
   zero correlation, so portfolio vol is ``sqrt(sum(w_i^2 sigma_i^2))`` - requires
   an assumption the document never makes, and would be a strange thing to call
   *portfolio* vol. Measured on the real sample this choice barely matters, because
   the 1.0 gross cap binds in 89-100% of rebalances either way and truncates
   whatever ``k`` produces; ``covariance="diagonal"`` is available so the size of
   that residual effect can be reported rather than assumed.

2. **Order of operations between k and the two caps.** ``k`` first, then the
   per-instrument cap, then the gross cap. This is the only order in which both
   caps are guaranteed to hold at the end: clipping to 0.25 and then scaling the
   whole vector down to gross 1.0 cannot push anything back above 0.25, whereas
   scaling first and clipping second leaves gross below its cap for no reason.

3. **Whether k may exceed 1.** Yes. Section 4 places no bound on k, and the "no
   leverage, cash account" requirement is expressed as the gross cap, which is
   applied afterwards and enforces it exactly.
"""

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
    """``sigma_i(t) = ewm_std(daily_returns_i, halflife) * sqrt(252)``.

    ``min_periods=halflife`` keeps the estimate NaN until there is enough history to
    make it meaningful; a NaN sigma yields a zero position downstream, matching the
    document's "insufficient history -> position 0".
    """
    if halflife < 1:
        raise ValueError("halflife must be >= 1")
    return returns.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def ewma_covariance(returns: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """Annualised EWMA covariance matrices, one per date, as a MultiIndex frame.

    Uses the same halflife as the per-instrument vol, because section 4 specifies
    exactly one halflife and inventing a second one would be inventing a parameter.
    """
    if halflife < 1:
        raise ValueError("halflife must be >= 1")
    return returns.ewm(halflife=halflife, min_periods=halflife).cov() * TRADING_DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class TargetWeights:
    """Target portfolio weights for one date, with the diagnostics behind them."""

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
    """Annualised ex-ante volatility of a weight vector."""
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
    # An instrument with no covariance estimate has no position either (its sigma is
    # NaN, so its weight is 0); zeroing the row keeps the quadratic form well defined.
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
    """Section 4, for one date.

    This is the single implementation of risk scaling. The backtest calls it once
    per rebalance date and the live runner calls it once per run, so a paper fill
    and a backtest bar are sized by literally the same code.

    Parameters
    ----------
    trend:
        Signal from :func:`trendbot.signal.trend_signal` as of this date's close.
    sigma:
        Annualised vol as of this date's close.
    cov:
        Annualised covariance matrix as of this date's close. Required when
        ``covariance="full"``.
    """
    universe = list(cfg.universe)
    trend = trend.reindex(universe).fillna(0.0).astype(float)
    sigma = sigma.reindex(universe).astype(float)

    # x_i = trend_i * (vol_target / sigma_i); an unusable sigma means position 0.
    usable = sigma.notna() & (sigma > 0)
    x = pd.Series(0.0, index=universe)
    x[usable] = trend[usable] * (cfg.instrument_vol_target / sigma[usable])

    # w_i = x_i / N. N is the fixed universe size from section 2, held at 12 even
    # when fewer instruments are active, because section 2 fixes the universe and
    # section 3 sets inactive instruments to 0 rather than reweighting the rest.
    w_raw = x / cfg.n_universe

    exante_raw = _exante_vol(w_raw, sigma, cov, covariance)
    k = cfg.portfolio_vol_target / exante_raw if exante_raw > 0 else 0.0
    w = w_raw * k
    exante_scaled = _exante_vol(w, sigma, cov, covariance)

    # per-instrument cap, then gross cap (see module docstring for why this order)
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
    """Section 5's no-trade band.

    ``only trade instrument i if |w_target - w_current| > 0.20 * |w_target|``

    Instruments inside the band keep their *current* weight, not their target. Note
    the behaviour when ``w_target`` is exactly zero: the threshold collapses to
    ``|0 - w_current| > 0``, so any residual position is always closed. That falls
    out of the rule as written and is the desirable reading - an instrument whose
    trend has turned off should not be held simply because the band scaled with it.
    """
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
    """Integer share counts and the portfolio you actually end up holding."""

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
        """Realised minus target weight, per instrument."""
        return self.realised_weights - self.target_weights

    @property
    def tracking_error(self) -> float:
        """L2 norm of the weight error introduced purely by whole-share rounding."""
        return float(np.sqrt((self.drift**2).sum()))

    @property
    def absolute_error(self) -> float:
        """L1 norm of the same error."""
        return float(self.drift.abs().sum())

    @property
    def n_holdable(self) -> int:
        return int(self.holdable.sum())

    @property
    def n_wanted(self) -> int:
        return int((self.target_weights.abs() > 0).sum())

    @property
    def risk_budget_deployed(self) -> float:
        """Fraction of the intended gross exposure that actually gets held."""
        wanted = float(self.target_weights.abs().sum())
        return float(self.realised_weights.abs().sum()) / wanted if wanted > 0 else float("nan")


def whole_share_allocation(
    weights: pd.Series,
    prices: pd.Series,
    equity: float,
) -> ShareAllocation:
    """Turn target weights into integer share counts.

    This is where a small account breaks. The target dollar allocation for an
    instrument is ``w_i * equity``; if that is less than the price of one share, the
    integer share count is zero, the position simply does not exist, and that sleeve
    is silently absent from the portfolio. Nothing warns you - the portfolio just
    quietly stops being the portfolio you designed.

    Rounding is toward zero, so the realised book never exceeds the target and the
    gross cap of section 4 continues to hold after rounding.
    """
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
