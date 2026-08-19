"""The backtest engine.

Structure of the no-lookahead guarantee
---------------------------------------
There is exactly one *execution* shift in this package and it is
:func:`lag_for_execution` below: the single point at which a quantity known at the
close of bar t becomes something the engine may act on at bar t+1. Everything
downstream of it is, by construction, information known at or before the close of
the previous bar.

Other shifts exist and all of them look strictly backwards - the lookback in
:func:`trendbot.signal.trend_signal`, the previous-close reference in the return
calculation, the lagged weights used to attribute per-instrument P&L. None is a
negative shift; ``tests/test_no_lookahead.py`` asserts that no negative shift appears
anywhere in the package.

PREREGISTRATION.md section 5::

    Signal on close of bar t -> fill at open of bar t+1.
    Rebalance: first trading day of each month.

Those two lines fix the schedule completely once you notice they have to be
consistent with each other: the *trade* happens on the first trading day of the
month, so the *signal* behind it must be the one from the close of the last trading
day of the previous month. Rebalance date ``d`` therefore consumes
``lag_for_execution(signal).loc[d]``, which is ``signal.loc[d - 1 bar]``.

Accounting
----------
Positions are carried as dollar exposures rather than as a fixed weight vector,
because section 5's drift band explicitly compares a target weight against a
*current* weight - which only exists if weights are allowed to drift with prices
between monthly rebalances.

On a rebalance day the book is marked to that day's open, traded at that open, and
then marked on to the close. On every other day it simply drifts:

.. code-block:: text

    non-rebalance day d:   a_i(d) = a_i(d-1) * close_i(d) / close_i(d-1)
    rebalance day d:       a_i(open)  = a_i(d-1) * open_i(d) / close_i(d-1)
                           E_open     = sum_i a_i(open) + cash
                           w_current  = a_i(open) / E_open
                           w_target   = sizing on the PREVIOUS close's signal
                           w_new      = drift band applied to (w_target, w_current)
                           cost       = rate * sum_i |w_new - w_current| * E_open
                           a_i(open') = w_new_i * E_open
                           a_i(d)     = a_i(open') * close_i(d) / open_i(d)

Between rebalances the equity path is computed in one vectorised expression per
month rather than a per-day Python loop, so the engine iterates ~12 times a year,
not 252.
"""

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
    """THE shift. Information available at row ``t`` becomes actionable at row ``t+1``.

    Every quantity the engine uses to decide a trade passes through this function.
    There is no other ``.shift()`` in the execution path, and no negative shift
    anywhere in the package.
    """
    return frame.shift(1)


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each month present in ``index``.

    The first such date is dropped: there is no previous bar to take a signal from,
    and taking one from the same bar would be lookahead.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("rebalance_dates needs a DatetimeIndex")
    periods = index.to_period("M")
    is_first = np.r_[True, periods[1:] != periods[:-1]]
    dates = index[is_first]
    return dates[1:] if len(dates) else dates


def full_universe_start(prices: PriceData, cfg: Config) -> pd.Timestamp:
    """First date on which every instrument has a full lookback of history.

    The twelve ETFs list fourteen years apart, so a backtest that simply starts at
    the beginning of the data spends its first decade holding one or two names -
    which is a different strategy from the one section 2 describes. This marks the
    first bar on which the portfolio the pre-registration actually specifies can
    exist.
    """
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
    """Everything one backtest run produced."""

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
        """Return in excess of the risk-free rate.

        Idle cash already accrues at this rate inside the engine, so subtracting it
        here leaves the return of the risky book net of the cost of financing it -
        the quantity that is comparable to a fully invested benchmark measured the
        same way. When no risk-free series was supplied this is just ``returns``.
        """
        return (self.returns - self.rf_daily).rename("excess_return")

    @property
    def stats(self) -> PerformanceStats:
        """Headline statistics, computed on excess returns (see section 8)."""
        return summarise(self.excess_returns, self.weights, trades=self.trades)

    @property
    def total_return_stats(self) -> PerformanceStats:
        """Statistics on total return, i.e. including the cash yield."""
        return summarise(self.returns, self.weights, trades=self.trades)

    @property
    def gross_stats(self) -> PerformanceStats:
        """Excess-return statistics with transaction costs switched off."""
        return summarise((self.gross_returns - self.rf_daily), self.weights, trades=self.trades)

    def stats_from(self, start) -> PerformanceStats:
        """Statistics over a sub-window, with turnover counted only inside it."""
        returns = self.excess_returns.loc[start:]
        return summarise(
            returns,
            self.weights.loc[start:],
            trades=self.trades.loc[start:] if not self.trades.empty else None,
        )

    def instrument_sign_count(self) -> tuple[int, pd.Series]:
        """Per-instrument total P&L contribution, and how many were positive.

        Section 8 requires "the sign is positive in at least 9 of 12 instruments".
        """
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
    """Run the strategy over ``prices``.

    Parameters
    ----------
    cost_bps:
        Per-side cost in basis points. Defaults to the pre-registered
        ``Config.cost_bps_per_side``; overridden only to produce section 5's
        sensitivity ladder.
    risk_free:
        Annualised risk-free rate. Uninvested cash accrues at it, and
        ``BacktestResult.excess_returns`` is measured against it. Omitting it means
        cash earns nothing, which handicaps a book that is only ~70% invested
        against a fully invested benchmark.
    signal:
        Optional pre-computed signal frame, used by the validation and
        anti-lookahead tests to inject a deliberately pathological signal. When
        None - which is every production call - the engine computes it itself from
        :func:`trendbot.signal.trend_signal`, so backtest and live share one
        definition.
    _unsafe_disable_execution_lag:
        Test-only. Removes the execution shift so that a test can demonstrate the
        shift is load-bearing rather than decorative. Any non-test caller passing
        this is a bug; ``tests/test_no_lookahead.py`` asserts no production call
        site does.
    """
    _validate_inputs(prices, cfg)
    universe = list(cfg.universe)
    close = prices.close[universe]
    open_ = prices.open[universe]
    rate = (cfg.cost_bps_per_side if cost_bps is None else float(cost_bps)) / 10_000.0
    if rate < 0:
        raise ValueError("cost cannot be negative")

    # Prices used to VALUE a position, as distinct from prices used to form a signal.
    # A vendor hole is not a price of zero and not a price of whatever the position
    # was worth at the start of the month; the last observed price is the only
    # defensible mark. The signal and the vol estimate deliberately keep using the raw
    # series, so a hole produces no signal and no fake return rather than a stale one.
    mark_close = close.ffill()
    mark_open = open_.where(open_.notna(), mark_close.shift(1)).ffill()

    returns_daily = close.pct_change(fill_method=None)
    sig = trend_signal(close, cfg.lookback_days, long_only=cfg.long_only) if signal is None else signal
    sig = sig.reindex(index=close.index, columns=universe).fillna(0.0)

    sigma = annualised_vol(returns_daily, cfg.ewma_halflife_days)
    cov = ewma_covariance(returns_daily, cfg.ewma_halflife_days) if covariance == "full" else None

    # ---- THE shift -----------------------------------------------------------
    # Everything the sizing step consumes is lagged by exactly one bar here.
    if _unsafe_disable_execution_lag:
        sig_x, sigma_x = sig, sigma
    else:
        sig_x, sigma_x = lag_for_execution(sig), lag_for_execution(sigma)
    cov_dates = lag_for_execution(pd.Series(close.index, index=close.index))
    # --------------------------------------------------------------------------

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
    gross_equity = np.full(n, np.nan)  # same path with costs switched off

    # Before the first rebalance the account is entirely cash, and cash earns.
    start_i = pos[rebals[0]]
    pre = equity0 * np.cumprod(1.0 + rf_v[:start_i])
    equity[:start_i] = pre
    gross_equity[:start_i] = pre

    alloc = np.zeros(len(universe))  # dollar exposure per instrument
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

        # mark the existing book from the previous close to this open
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

        # Section 4's gross cap is stated as an inequality that must hold, and is
        # justified there by "no leverage, cash account". The drift band can leave a
        # book above it: instruments whose weight drifted up but stayed inside their
        # own band are not traded back, so the sum can exceed 1.0, which means
        # negative cash. Enforcing the cap on the weights actually held is applying
        # the document rather than adding to it. Without this the book reaches gross
        # 1.11 on 27% of days. The cost is that scaling touches instruments that were
        # inside their band; the alternative is breaching a constraint section 4
        # states unconditionally, which is worse.
        # Section 4 states BOTH caps unconditionally, so both are re-applied to the
        # weights actually held, in the same order the target pipeline uses: clip each
        # instrument, then scale the vector. Without the per-instrument clip here the
        # book reaches |w_i| = 0.279 on 15% of days, because an instrument that drifted
        # above 0.25 but stayed inside its own band is never traded back.
        instrument_cap_forced = bool((w_new.abs() > cfg.per_instrument_cap + 1e-12).any())
        w_new = w_new.clip(lower=-cfg.per_instrument_cap, upper=cfg.per_instrument_cap)
        band_gross = float(w_new.abs().sum())
        gross_cap_forced = band_gross > cfg.gross_exposure_cap + 1e-12
        if gross_cap_forced:
            w_new = w_new * (cfg.gross_exposure_cap / band_gross)

        turnover = float((w_new - w_current).abs().sum())
        cost = rate * turnover * e_open
        cost_arr[i] = cost

        # The cost is paid out of the account before the book is established, so a
        # fully invested target leaves cash at exactly zero rather than slightly
        # negative. Allocating first and then subtracting the cost would leave the
        # book financed by a few basis points of borrowing, which is both leverage
        # and, at gross exactly 1.0, invisible in the weights.
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

        # ---- vectorised drift to the end of the block --------------------------
        end_i = pos[next_reb] if next_reb is not None else n
        seg = slice(i, end_i)
        px = close_v[seg]
        base = np.where(np.isfinite(o) & (o > 0), o, np.nan)
        # Where a mark is genuinely unavailable - an instrument that has never
        # traded - the position is zero anyway, so a growth factor of 1 is harmless.
        # It is no longer reachable for a mid-series hole, because mark_close is
        # forward-filled above.
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

    # per-instrument P&L contribution, for the section 8 sign test
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
    """The section 8 benchmark: equal-weight buy-and-hold of the same twelve ETFs.

    "Buy-and-hold" is taken literally: equal dollars are placed once and then left
    alone, so the weights drift with prices. The only exception is that a newly
    listed instrument has to be bought at some point, so the basket is re-levelled
    on the days when the available set changes and on no others. Over the headline
    window all twelve already exist on day one, so that exception never fires and
    this is exactly "buy 1/12 of each and hold".

    The distinction is not cosmetic. Rebalancing back to equal weight harvests a
    rebalancing premium that a true buy-and-hold investor does not earn, and the
    section 8 rule turns on a margin of 0.15 - comfortably smaller than the gap
    between these two constructions. ``rebalance="daily"`` and ``rebalance="monthly"``
    are available so the size of that difference can be reported rather than assumed.
    """
    if rebalance not in ("none", "daily", "monthly"):
        raise ValueError(f"unknown rebalance mode {rebalance!r}")

    close = prices.close[list(cfg.universe)]
    if start is not None:
        close = close.loc[start:]
    # Two different notions of "available", and conflating them costs a day's return.
    # To *hold* an instrument you only need a price to buy at; to attribute a daily
    # return to it you additionally need the previous bar's price.
    # "Investable" means listed, not "the vendor published a price today". Treating a
    # one-day data hole as a delisting would liquidate the instrument and re-level the
    # whole basket across the survivors, which is the opposite of buy-and-hold.
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
        # re-level only when the investable set itself changes
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
        # The book is re-levelled using its value at the close of bar i-1, so bar i's
        # own market move still has to be applied. Referencing bar i's own close
        # instead would silently delete one day of market return on every re-levelling
        # day - on the full history that is 403 days, enough on its own to move the
        # benchmark's Sharpe by 0.16. An instrument with no price at i-1 is one being
        # bought for the first time at bar i's close, and correctly earns nothing that
        # day.
        prior = close_v[i - 1] if i > 0 else close_v[i]
        ref = np.where(np.isfinite(prior) & (prior > 0), prior, close_v[i])
        ref = np.where(live & np.isfinite(ref) & (ref > 0), ref, np.nan)
        growth = np.divide(px, ref, out=np.ones_like(px), where=np.isfinite(px) & np.isfinite(ref))
        held = alloc[None, :] * np.where(live[None, :], growth, 0.0)
        equity[i:end_i] = held.sum(axis=1)
        alloc = held[-1]

    curve = pd.Series(equity, index=index)
    return curve.pct_change().fillna(0.0)
