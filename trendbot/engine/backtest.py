from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from ..data import PriceData, daily_risk_free
from ..signal import trend_signal
from ..sizing import (
    CovarianceMethod,
    annualised_vol,
    apply_drift_band,
    ewma_covariance,
    target_weights,
)
from .metrics import PerformanceStats, summarise

__all__ = [
    "lag_for_execution",
    "rebalance_dates",
    "full_universe_start",
    "BacktestResult",
    "run_backtest",
    "buy_and_hold",
]

SignalFn = Callable[[pd.DataFrame, int], pd.DataFrame]


def lag_for_execution(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.shift(1)


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("rebalance_dates needs a DatetimeIndex")
    periods = index.to_period("M")
    is_first = np.r_[True, periods[1:] != periods[:-1]]
    dates = index[is_first]
    return dates[1:] if len(dates) else dates


def full_universe_start(prices: PriceData, cfg: Config) -> pd.Timestamp:
    close = prices.close[list(cfg.universe)]
    starts = []
    for ticker in cfg.universe:
        valid = close[ticker].dropna()
        if len(valid) <= cfg.lookback_days:
            raise ValueError(f"{ticker} has only {len(valid)} bars, fewer than the lookback")
        starts.append(valid.index[cfg.lookback_days])
    return max(starts)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    cfg: Config
    cost_bps: float
    covariance: CovarianceMethod
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.Series
    diagnostics: pd.DataFrame
    instrument_pnl: pd.DataFrame
    rf_daily: pd.Series

    @property
    def excess_returns(self) -> pd.Series:
        return (self.returns - self.rf_daily).rename("excess_return")

    @property
    def stats(self) -> PerformanceStats:
        return summarise(self.excess_returns, self.weights, trades=self.trades)

    @property
    def total_return_stats(self) -> PerformanceStats:
        return summarise(self.returns, self.weights, trades=self.trades)

    @property
    def gross_stats(self) -> PerformanceStats:
        return summarise((self.gross_returns - self.rf_daily), self.weights, trades=self.trades)

    def stats_from(self, start) -> PerformanceStats:
        returns = self.excess_returns.loc[start:]
        return summarise(
            returns,
            self.weights.loc[start:],
            trades=self.trades.loc[start:] if not self.trades.empty else None,
        )

    def instrument_sign_count(self) -> tuple[int, pd.Series]:
        total = self.instrument_pnl.sum()
        return int((total > 0).sum()), total


def _validate_inputs(prices: PriceData, cfg: Config) -> None:
    missing = [t for t in cfg.universe if t not in prices.close.columns]
    if missing:
        raise ValueError(f"price data is missing universe members {missing}")
    if len(prices.close) < cfg.lookback_days + 2:
        raise ValueError(
            f"need more than {cfg.lookback_days + 1} bars to produce any signal, "
            f"got {len(prices.close)}"
        )


def run_backtest(
    prices: PriceData,
    cfg: Config,
    *,
    cost_bps: float | None = None,
    covariance: CovarianceMethod = "full",
    signal: pd.DataFrame | None = None,
    risk_free: pd.Series | None = None,
    equity0: float = 1.0,
    _unsafe_disable_execution_lag: bool = False,
) -> BacktestResult:
    _validate_inputs(prices, cfg)
    universe = list(cfg.universe)
    close = prices.close[universe]
    open_ = prices.open[universe]
    rate = (cfg.cost_bps_per_side if cost_bps is None else float(cost_bps)) / 10_000.0
    if rate < 0:
        raise ValueError("cost cannot be negative")

    mark_close = close.ffill()
    mark_open = open_.where(open_.notna(), mark_close.shift(1)).ffill()

    returns_daily = close.pct_change(fill_method=None)
    sig = trend_signal(close, cfg.lookback_days, long_only=cfg.long_only) if signal is None else signal
    sig = sig.reindex(index=close.index, columns=universe).fillna(0.0)

    sigma = annualised_vol(returns_daily, cfg.ewma_halflife_days)
    cov = ewma_covariance(returns_daily, cfg.ewma_halflife_days) if covariance == "full" else None

    if _unsafe_disable_execution_lag:
        sig_x, sigma_x = sig, sigma
    else:
        sig_x, sigma_x = lag_for_execution(sig), lag_for_execution(sigma)
    cov_dates = lag_for_execution(pd.Series(close.index, index=close.index))

    if risk_free is None:
        rf_d = pd.Series(0.0, index=close.index)
    else:
        rf_d = daily_risk_free(risk_free, close.index)
    rf_v = rf_d.to_numpy(dtype=float)

    rebals = rebalance_dates(close.index)
    if len(rebals) == 0:
        raise ValueError("price history spans no complete month boundary")

    n = len(close.index)
    pos = {d: i for i, d in enumerate(close.index)}
    close_v = mark_close.to_numpy(dtype=float)
    open_v = mark_open.to_numpy(dtype=float)

    equity = np.full(n, np.nan)
    held_w = np.zeros((n, len(universe)))
    cost_arr = np.zeros(n)
    gross_equity = np.full(n, np.nan)

    start_i = pos[rebals[0]]
    pre = equity0 * np.cumprod(1.0 + rf_v[:start_i])
    equity[:start_i] = pre
    gross_equity[:start_i] = pre

    alloc = np.zeros(len(universe))
    cash = float(pre[-1]) if start_i > 0 else equity0
    g_alloc, g_cash = np.zeros(len(universe)), cash

    target_rows: list[pd.Series] = []
    trade_rows: list[pd.Series] = []
    diag_rows: list[dict] = []

    block_bounds = list(zip(rebals, list(rebals[1:]) + [None]))

    for reb, next_reb in block_bounds:
        i = pos[reb]
        prev_close = close_v[i - 1]
        o = open_v[i]

        ratio = np.where(np.isfinite(prev_close) & (prev_close > 0) & np.isfinite(o), o / prev_close, 1.0)
        alloc = alloc * ratio
        g_alloc = g_alloc * ratio
        e_open = float(alloc.sum() + cash)
        g_open = float(g_alloc.sum() + g_cash)
        if not np.isfinite(e_open) or e_open <= 0:
            raise RuntimeError(f"equity became non-positive at {reb.date()}; refusing to continue")

        w_current = pd.Series(alloc / e_open, index=universe)

        decision_date = cov_dates.loc[reb]
        cov_slice = None
        if cov is not None:
            cov_slice = cov.xs(decision_date, level=0) if pd.notna(decision_date) else None

        tw = target_weights(
            reb,
            sig_x.loc[reb],
            sigma_x.loc[reb],
            cfg,
            cov=cov_slice,
            covariance=covariance,
        )
        w_new = apply_drift_band(tw.weights, w_current, cfg.drift_band)

        instrument_cap_forced = bool((w_new.abs() > cfg.per_instrument_cap + 1e-12).any())
        w_new = w_new.clip(lower=-cfg.per_instrument_cap, upper=cfg.per_instrument_cap)
        band_gross = float(w_new.abs().sum())
        gross_cap_forced = band_gross > cfg.gross_exposure_cap + 1e-12
        if gross_cap_forced:
            w_new = w_new * (cfg.gross_exposure_cap / band_gross)

        turnover = float((w_new - w_current).abs().sum())
        cost = rate * turnover * e_open
        cost_arr[i] = cost

        e_invest = e_open - cost
        if e_invest <= 0:
            raise RuntimeError(f"costs exhausted the account at {reb.date()}")
        alloc = w_new.to_numpy() * e_invest
        cash = e_invest - float(alloc.sum())
        g_alloc = w_new.to_numpy() * g_open
        g_cash = g_open - float(g_alloc.sum())

        target_rows.append(tw.weights.rename(reb))
        trade_rows.append((w_new - w_current).rename(reb))
        diag_rows.append(
            {
                "date": reb,
                "decision_date": decision_date,
                "k": tw.k,
                "exante_vol_raw": tw.exante_vol_raw,
                "exante_vol_final": tw.exante_vol_final,
                "gross_before_caps": tw.gross_before_caps,
                "gross_after_caps": tw.gross_after_caps,
                "gross_traded": float(w_new.abs().sum()),
                "gross_after_band": band_gross,
                "gross_cap_forced": gross_cap_forced,
                "instrument_cap_forced": instrument_cap_forced,
                "n_active": tw.n_active,
                "n_capped": len(tw.capped_instruments),
                "capped": ",".join(tw.capped_instruments),
                "turnover": turnover,
                "cost": cost,
                "equity_open": e_open,
            }
        )

        end_i = pos[next_reb] if next_reb is not None else n
        seg = slice(i, end_i)
        px = close_v[seg]
        base = np.where(np.isfinite(o) & (o > 0), o, np.nan)
        growth = np.divide(px, base, out=np.ones_like(px), where=np.isfinite(px) & np.isfinite(base) & (base > 0))
        held = alloc[None, :] * growth
        g_held = g_alloc[None, :] * growth
        cash_path = cash * np.cumprod(1.0 + rf_v[seg])
        g_cash_path = g_cash * np.cumprod(1.0 + rf_v[seg])
        eq = held.sum(axis=1) + cash_path
        g_eq = g_held.sum(axis=1) + g_cash_path
        equity[seg] = eq
        gross_equity[seg] = g_eq
        with np.errstate(invalid="ignore", divide="ignore"):
            held_w[seg] = held / eq[:, None]

        alloc = held[-1]
        g_alloc = g_held[-1]
        cash = float(cash_path[-1])
        g_cash = float(g_cash_path[-1])

    equity_s = pd.Series(equity, index=close.index, name="equity")
    gross_s = pd.Series(gross_equity, index=close.index, name="gross_equity")
    rets = equity_s.pct_change().fillna(0.0)
    grets = gross_s.pct_change().fillna(0.0)
    weights_df = pd.DataFrame(held_w, index=close.index, columns=universe)

    lagged_w = weights_df.shift(1).fillna(0.0)
    instrument_pnl = (lagged_w * returns_daily.fillna(0.0)).astype(float)
    trades_df = pd.DataFrame(trade_rows)

    diagnostics = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()

    return BacktestResult(
        cfg=cfg,
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else float(cost_bps),
        covariance=covariance,
        equity=equity_s,
        returns=rets,
        gross_returns=grets,
        weights=weights_df,
        targets=pd.DataFrame(target_rows),
        trades=trades_df,
        costs=pd.Series(cost_arr, index=close.index, name="cost"),
        diagnostics=diagnostics,
        instrument_pnl=instrument_pnl,
        rf_daily=rf_d,
    )


def buy_and_hold(
    prices: PriceData,
    cfg: Config,
    *,
    rebalance: str = "none",
    start: pd.Timestamp | None = None,
) -> pd.Series:
    if rebalance not in ("none", "daily", "monthly"):
        raise ValueError(f"unknown rebalance mode {rebalance!r}")

    close = prices.close[list(cfg.universe)]
    if start is not None:
        close = close.loc[start:]
    marks = close.ffill()
    investable = marks.notna()
    has_return = close.notna() & close.shift(1).notna()

    if rebalance == "daily":
        rets = close.pct_change(fill_method=None)
        counts = has_return.sum(axis=1)
        w = has_return.div(counts.where(counts > 0), axis=0).fillna(0.0)
        return (w * rets.fillna(0.0)).sum(axis=1).fillna(0.0)

    index = close.index
    avail_v = investable.to_numpy()
    close_v = marks.to_numpy(dtype=float)
    n_days, n_assets = close_v.shape

    if rebalance == "monthly":
        reset_flags = np.zeros(n_days, dtype=bool)
        reset_flags[[index.get_loc(d) for d in rebalance_dates(index)]] = True
    else:
        reset_flags = np.r_[True, (avail_v[1:] != avail_v[:-1]).any(axis=1)]
    reset_flags[0] = True

    equity = np.empty(n_days)
    alloc = np.zeros(n_assets)
    cash = 1.0
    reset_positions = np.flatnonzero(reset_flags)

    for block, i in enumerate(reset_positions):
        end_i = reset_positions[block + 1] if block + 1 < len(reset_positions) else n_days
        live = avail_v[i]
        n_live = int(live.sum())
        base = float(alloc.sum() + cash)
        if n_live == 0:
            equity[i:end_i] = base
            alloc, cash = np.zeros(n_assets), base
            continue
        alloc = np.where(live, base / n_live, 0.0)
        cash = 0.0
        px = close_v[i:end_i]
        prior = close_v[i - 1] if i > 0 else close_v[i]
        ref = np.where(np.isfinite(prior) & (prior > 0), prior, close_v[i])
        ref = np.where(live & np.isfinite(ref) & (ref > 0), ref, np.nan)
        growth = np.divide(px, ref, out=np.ones_like(px), where=np.isfinite(px) & np.isfinite(ref))
        held = alloc[None, :] * np.where(live[None, :], growth, 0.0)
        equity[i:end_i] = held.sum(axis=1)
        alloc = held[-1]

    curve = pd.Series(equity, index=index)
    return curve.pct_change().fillna(0.0)
