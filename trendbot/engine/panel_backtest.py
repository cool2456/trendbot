"""The backtest engine, generalised to a panel strategy.

This is experiment 001's engine with one substitution. Where that engine called
:func:`trendbot.sizing.target_weights` itself at each rebalance, this one takes the
whole frame of target weights from a :class:`~trendbot.engine.panel.PanelStrategy`
and reads the row it needs. Everything else - the single execution shift, the
rebalance calendar, the mark-to-open/trade/mark-to-close accounting, the drift band,
the re-application of the exposure caps to the weights actually held, the cost on
realised turnover, the cash accrual - is the same code doing the same arithmetic in
the same order.

That is deliberate and it is testable: ``tests/test_experiment_001_regression.py``
runs experiment 001 through this engine and asserts the equity curve agrees with the
original to floating-point tolerance, not merely to the three decimal places the
build gate asks for.

Everything imported from :mod:`trendbot.engine.backtest` below - the shift, the
calendar - is imported rather than re-implemented, so there is still exactly one
definition of each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import PriceData, daily_risk_free
from ..sizing import apply_drift_band
from .backtest import buy_and_hold, lag_for_execution, rebalance_dates
from .metrics import PerformanceStats, summarise
from .panel import Panel, PanelStrategy, price_panel

__all__ = [
    "UniverseView",
    "PanelBacktestResult",
    "run_panel_backtest",
    "panel_buy_and_hold",
    "panel_universe_start",
]


@dataclass(frozen=True, slots=True)
class UniverseView:
    """The only attribute :func:`trendbot.engine.backtest.buy_and_hold` reads.

    Experiment 002 needs the section 8 benchmark over a 41-instrument universe that
    has no :class:`~trendbot.config.Config`. Re-implementing the benchmark would risk
    two constructions that disagree by more than the 0.15 the decision rule turns on,
    so the original function is called with a view that carries only the universe.
    """

    universe: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanelBacktestResult:
    """Everything one panel backtest run produced."""

    strategy_name: str
    universe: tuple[str, ...]
    cost_bps: float
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
        the quantity comparable to a fully invested benchmark measured the same way.
        """
        return (self.returns - self.rf_daily).rename("excess_return")

    @property
    def stats(self) -> PerformanceStats:
        return summarise(self.excess_returns, self.weights, trades=self.trades)

    @property
    def gross_stats(self) -> PerformanceStats:
        return summarise((self.gross_returns - self.rf_daily), self.weights, trades=self.trades)

    def stats_from(self, start) -> PerformanceStats:
        """Statistics over a sub-window, with turnover counted only inside it."""
        return summarise(
            self.excess_returns.loc[start:],
            self.weights.loc[start:],
            trades=self.trades.loc[start:] if not self.trades.empty else None,
        )


def panel_universe_start(prices: PriceData, universe, lookback_days: int) -> pd.Timestamp:
    """First date on which every instrument has a full ``lookback_days`` of history."""
    starts = []
    for ticker in universe:
        valid = prices.close[ticker].dropna()
        if len(valid) <= lookback_days:
            raise ValueError(f"{ticker} has only {len(valid)} bars, fewer than the lookback")
        starts.append(valid.index[lookback_days])
    return max(starts)


def panel_buy_and_hold(
    prices: PriceData,
    universe,
    *,
    rebalance: str = "none",
    start: pd.Timestamp | None = None,
) -> pd.Series:
    """Equal-weight buy-and-hold over ``universe``, via experiment 001's own function."""
    return buy_and_hold(
        prices, UniverseView(tuple(universe)), rebalance=rebalance, start=start
    )


def _targets_from(
    strategy: PanelStrategy | None,
    targets: pd.DataFrame | None,
    panel: Panel,
    index: pd.DatetimeIndex,
    universe: list[str],
) -> tuple[pd.DataFrame, str]:
    if (strategy is None) == (targets is None):
        raise ValueError("supply exactly one of strategy= or targets=")
    if targets is None:
        name = getattr(strategy, "name", type(strategy).__name__)
        raw = strategy(panel)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"panel strategy {name!r} returned {type(raw).__name__}, not a DataFrame of "
                "target weights indexed by date and columned by symbol"
            )
    else:
        name, raw = "precomputed", targets
    if not raw.index.is_unique:
        raise ValueError("target weight frame has duplicate dates")
    unknown = [c for c in raw.columns if c not in universe]
    if unknown:
        raise ValueError(f"target weight frame names instruments outside the universe: {unknown}")
    return raw.reindex(index=index, columns=universe).fillna(0.0).astype(float), name


def run_panel_backtest(
    prices: PriceData,
    universe,
    strategy: PanelStrategy | None = None,
    *,
    targets: pd.DataFrame | None = None,
    cost_bps: float,
    drift_band: float | None = None,
    gross_cap: float | None = None,
    per_instrument_cap: float | None = None,
    risk_free: pd.Series | None = None,
    equity0: float = 1.0,
) -> PanelBacktestResult:
    """Run ``strategy`` over ``prices``.

    Parameters
    ----------
    strategy / targets:
        Exactly one. ``targets`` accepts a frame the caller already computed, which
        is how the cost-sensitivity ladder reruns five times without recomputing a
        signal that cannot depend on the cost.
    drift_band:
        ``None`` means no band, i.e. a full rebalance to the target every time -
        which is what PREREG_002.md section 5 states. A float applies experiment
        001's band.
    gross_cap / per_instrument_cap:
        Re-applied to the weights actually held, after the band, in the same order
        experiment 001's engine uses. ``None`` disables the corresponding cap.

    There is deliberately no equivalent of experiment 001's test-only lookahead
    escape hatch. That hatch exists so a test can switch the shift off and watch an
    edge appear; here the same job is done by ``tests/test_panel_engine.py``, which
    feeds a hand-built target frame and asserts the row consumed at rebalance ``d``
    is the row dated ``d - 1``, and by the causality sweep, which shows no future bar
    can move a past return. Neither needs a production code path that can be asked to
    look ahead.
    """
    universe = list(universe)
    missing = [t for t in universe if t not in prices.close.columns]
    if missing:
        raise ValueError(f"price data is missing universe members {missing}")
    rate = float(cost_bps) / 10_000.0
    if rate < 0:
        raise ValueError("cost cannot be negative")
    if drift_band is not None and drift_band < 0:
        raise ValueError("drift band cannot be negative")

    close = prices.close[universe]
    open_ = prices.open[universe]

    # Prices used to VALUE a position, as distinct from prices used to form a signal.
    # A vendor hole is not a price of zero; the last observed price is the only
    # defensible mark. The strategy keeps seeing the raw series, so a hole produces no
    # signal and no fake return rather than a stale one.
    mark_close = close.ffill()
    mark_open = open_.where(open_.notna(), mark_close.shift(1)).ffill()
    returns_daily = close.pct_change(fill_method=None)

    panel = price_panel(prices, universe)
    target_frame, strategy_name = _targets_from(strategy, targets, panel, close.index, universe)

    # ---- THE shift -----------------------------------------------------------
    # The one and only execution shift, imported from experiment 001's engine rather
    # than re-implemented, so both engines move information forward by the same code.
    targets_x = lag_for_execution(target_frame)
    decision_dates = lag_for_execution(pd.Series(close.index, index=close.index))
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
        ratio = np.where(
            np.isfinite(prev_close) & (prev_close > 0) & np.isfinite(o), o / prev_close, 1.0
        )
        alloc = alloc * ratio
        g_alloc = g_alloc * ratio
        e_open = float(alloc.sum() + cash)
        g_open = float(g_alloc.sum() + g_cash)
        if not np.isfinite(e_open) or e_open <= 0:
            raise RuntimeError(f"equity became non-positive at {reb.date()}; refusing to continue")

        w_current = pd.Series(alloc / e_open, index=universe)
        w_target = targets_x.loc[reb]

        if drift_band is None:
            w_new = w_target.astype(float)
        else:
            w_new = apply_drift_band(w_target, w_current, drift_band)

        # Both caps are re-applied to the weights actually held, in the same order the
        # target pipeline uses: clip each instrument, then scale the vector. Without
        # this a weight that drifted above its cap but stayed inside its band would
        # never be traded back, and a constraint stated unconditionally would be
        # quietly breached. With no drift band nothing can drift, so neither cap binds.
        instrument_cap_forced = False
        if per_instrument_cap is not None:
            instrument_cap_forced = bool((w_new.abs() > per_instrument_cap + 1e-12).any())
            w_new = w_new.clip(lower=-per_instrument_cap, upper=per_instrument_cap)
        band_gross = float(w_new.abs().sum())
        gross_cap_forced = gross_cap is not None and band_gross > gross_cap + 1e-12
        if gross_cap_forced:
            w_new = w_new * (gross_cap / band_gross)

        turnover = float((w_new - w_current).abs().sum())
        cost = rate * turnover * e_open
        cost_arr[i] = cost

        # The cost is paid out of the account before the book is established, so a
        # fully invested target leaves cash at exactly zero rather than slightly
        # negative.
        e_invest = e_open - cost
        if e_invest <= 0:
            raise RuntimeError(f"costs exhausted the account at {reb.date()}")
        alloc = w_new.to_numpy() * e_invest
        cash = e_invest - float(alloc.sum())
        g_alloc = w_new.to_numpy() * g_open
        g_cash = g_open - float(g_alloc.sum())

        target_rows.append(w_target.rename(reb))
        trade_rows.append((w_new - w_current).rename(reb))
        diag_rows.append(
            {
                "date": reb,
                "decision_date": decision_dates.loc[reb],
                "n_selected": int((w_target.abs() > 0).sum()),
                "gross_target": float(w_target.abs().sum()),
                "gross_traded": float(w_new.abs().sum()),
                "gross_after_band": band_gross,
                "gross_cap_forced": gross_cap_forced,
                "instrument_cap_forced": instrument_cap_forced,
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
        growth = np.divide(
            px, base, out=np.ones_like(px), where=np.isfinite(px) & np.isfinite(base) & (base > 0)
        )
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
    weights_df = pd.DataFrame(held_w, index=close.index, columns=universe)
    lagged_w = weights_df.shift(1).fillna(0.0)

    return PanelBacktestResult(
        strategy_name=strategy_name,
        universe=tuple(universe),
        cost_bps=float(cost_bps),
        equity=equity_s,
        returns=equity_s.pct_change().fillna(0.0),
        gross_returns=gross_s.pct_change().fillna(0.0),
        weights=weights_df,
        targets=pd.DataFrame(target_rows),
        trades=pd.DataFrame(trade_rows),
        costs=pd.Series(cost_arr, index=close.index, name="cost"),
        diagnostics=pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame(),
        instrument_pnl=(lagged_w * returns_daily.fillna(0.0)).astype(float),
        rf_daily=rf_d,
    )
