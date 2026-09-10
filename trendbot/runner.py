from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .brokers.base import Broker, Order, OrderSide, Position
from .config import Config, load_config
from .data import REPO_ROOT, PriceData, load_prices
from .engine.backtest import rebalance_dates
from .guards import (
    CONSERVATIVE,
    EquityHighWaterMark,
    GuardConfig,
    GuardViolation,
    HaltState,
    check_broker_is_paper,
    check_data_freshness,
    check_drawdown,
    check_no_open_orders,
    check_not_halted,
    check_notional,
    check_order_cap,
    fail_closed,
)
from .monitor.divergence import DivergenceLog, Prediction
from .signal import trend_signal
from .sizing import annualised_vol, apply_drift_band, ewma_covariance, target_weights, whole_share_allocation

__all__ = ["STATE_DIR", "order_id", "RunPlan", "RunOutcome", "plan_run", "execute_plan", "run_once"]

STATE_DIR = REPO_ROOT / "state"


def order_id(*, prereg_sha: str, session: dt.date, symbol: str, side: str, qty: int) -> str:
    material = f"{prereg_sha}|{session.isoformat()}|{symbol}|{side}|{int(qty)}"
    return f"tb1-{session.isoformat()}-{symbol}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    symbol: str
    side: OrderSide
    qty: int
    reference_price: float
    reference_date: dt.date
    notional: float
    current_shares: float
    target_shares: int
    client_order_id: str

    def __str__(self) -> str:
        return (
            f"{self.side.value.upper():4s} {self.qty:>6d} {self.symbol:<4s} "
            f"@ ~{self.reference_price:>8.2f} = {self.notional:>11,.2f}  "
            f"({self.current_shares:g} -> {self.target_shares})"
        )


@dataclass(frozen=True, slots=True)
class RunPlan:
    run_id: str
    session: dt.date
    is_rebalance_day: bool
    equity: float
    peak_equity: float
    broker_positions: dict[str, Position]
    reference_prices: pd.Series
    reference_dates: dict[str, dt.date]
    target_weights: pd.Series
    current_weights: pd.Series
    banded_weights: pd.Series
    target_shares: pd.Series
    orders: tuple[PlannedOrder, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def total_notional(self) -> float:
        return float(sum(o.notional for o in self.orders))

    def describe(self) -> str:
        lines = [
            f"run {self.run_id}  session {self.session}  "
            f"{'REBALANCE DAY' if self.is_rebalance_day else 'not a rebalance day'}",
            f"equity {self.equity:,.2f}   peak {self.peak_equity:,.2f}   "
            f"drawdown {1 - self.equity / self.peak_equity:.2%}" if self.peak_equity > 0 else "",
        ]
        if self.is_rebalance_day:
            table = pd.DataFrame(
                {
                    "target_w": self.target_weights,
                    "current_w": self.current_weights,
                    "banded_w": self.banded_weights,
                    "target_sh": self.target_shares,
                    "current_sh": pd.Series(
                        {s: (self.broker_positions[s].qty if s in self.broker_positions else 0.0)
                         for s in self.target_weights.index}
                    ),
                    "price": self.reference_prices,
                }
            )
            lines.append(table.round(4).to_string())
            lines.append(f"\n{len(self.orders)} order(s), {self.total_notional:,.2f} notional:")
            lines.extend(f"  {o}" for o in self.orders)
        return "\n".join(line for line in lines if line)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    plan: RunPlan
    submitted: tuple[Order, ...]
    reconciled: int
    dry_run: bool


def _is_rebalance_session(history_index: pd.DatetimeIndex, session: dt.date) -> bool:
    stamp = pd.Timestamp(session)
    extended = history_index.union(pd.DatetimeIndex([stamp]))
    return stamp in set(rebalance_dates(extended))


def plan_run(
    broker: Broker,
    cfg: Config,
    guard_cfg: GuardConfig,
    *,
    halt: HaltState,
    high_water: EquityHighWaterMark,
    history: PriceData,
    session: dt.date | None = None,
) -> RunPlan:
    check_not_halted(halt)
    check_broker_is_paper(broker)

    clock = broker.get_clock()
    session = session or clock.timestamp.date()
    run_id = f"{session.isoformat()}T{clock.timestamp.strftime('%H%M%S')}"

    account = broker.get_account()
    positions = broker.get_positions()
    open_orders = broker.get_open_orders()

    check_no_open_orders(open_orders)

    universe = list(cfg.universe)
    quotes = broker.get_last_close(universe)
    reference_prices = pd.Series({s: quotes[s][0] for s in universe}, name="price")
    reference_dates = {s: quotes[s][1] for s in universe}

    oldest_bar = min([*reference_dates.values(), history.close.index[-1].date()])
    calendar = broker.get_trading_sessions(oldest_bar - dt.timedelta(days=7), session)
    check_data_freshness(reference_dates, session, guard_cfg, trading_calendar=calendar)

    check_data_freshness(
        {"price history": history.close.index[-1].date()},
        session,
        guard_cfg,
        trading_calendar=calendar,
    )

    equity = account.equity
    peak = high_water.update(equity)
    check_drawdown(equity, peak, guard_cfg)

    current_shares = pd.Series(
        {s: (positions[s].qty if s in positions else 0.0) for s in universe}, dtype=float
    )
    current_weights = pd.Series(
        {s: (positions[s].market_value / equity if s in positions else 0.0) for s in universe},
        dtype=float,
    )

    is_rebalance = _is_rebalance_session(history.close.index, session)
    if not is_rebalance:
        empty = pd.Series(0.0, index=universe)
        return RunPlan(
            run_id=run_id,
            session=session,
            is_rebalance_day=False,
            equity=equity,
            peak_equity=peak,
            broker_positions=positions,
            reference_prices=reference_prices,
            reference_dates=reference_dates,
            target_weights=empty,
            current_weights=current_weights,
            banded_weights=current_weights,
            target_shares=current_shares.astype(int),
            diagnostics={"reason": "not the first trading day of the month"},
        )

    close = history.close[universe]
    returns = close.pct_change(fill_method=None)
    signal = trend_signal(close, cfg.lookback_days, long_only=cfg.long_only)
    sigma = annualised_vol(returns, cfg.ewma_halflife_days)
    cov = ewma_covariance(returns, cfg.ewma_halflife_days)

    decision_date = close.index[-1]
    if decision_date.date() >= session:
        raise GuardViolation(
            f"the newest bar in the history is {decision_date.date()}, which is not "
            f"strictly before the session being traded ({session}). Refusing to size a "
            "trade on a bar that has not closed."
        )

    targets = target_weights(
        decision_date,
        signal.loc[decision_date],
        sigma.loc[decision_date],
        cfg,
        cov=cov.xs(decision_date, level=0),
        covariance="full",
    )
    banded = apply_drift_band(targets.weights, current_weights, cfg.drift_band)
    banded = banded.clip(lower=-cfg.per_instrument_cap, upper=cfg.per_instrument_cap)
    band_gross = float(banded.abs().sum())
    if band_gross > cfg.gross_exposure_cap:
        banded = banded * (cfg.gross_exposure_cap / band_gross)

    allocation = whole_share_allocation(banded, reference_prices, equity)
    target_shares = allocation.shares

    orders: list[PlannedOrder] = []
    for symbol in universe:
        delta = int(target_shares[symbol]) - int(round(current_shares[symbol]))
        if delta == 0:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = abs(delta)
        price = float(reference_prices[symbol])
        orders.append(
            PlannedOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                reference_price=price,
                reference_date=reference_dates[symbol],
                notional=qty * price,
                current_shares=float(current_shares[symbol]),
                target_shares=int(target_shares[symbol]),
                client_order_id=order_id(
                    prereg_sha=cfg.source_sha256,
                    session=session,
                    symbol=symbol,
                    side=side.value,
                    qty=qty,
                ),
            )
        )

    return RunPlan(
        run_id=run_id,
        session=session,
        is_rebalance_day=True,
        equity=equity,
        peak_equity=peak,
        broker_positions=positions,
        reference_prices=reference_prices,
        reference_dates=reference_dates,
        target_weights=targets.weights,
        current_weights=current_weights,
        banded_weights=banded,
        target_shares=target_shares,
        orders=tuple(orders),
        diagnostics={
            "decision_date": str(decision_date.date()),
            "k": targets.k,
            "exante_vol_raw": targets.exante_vol_raw,
            "exante_vol_final": targets.exante_vol_final,
            "n_active": targets.n_active,
            "capped": list(targets.capped_instruments),
            "gross_target": float(targets.weights.abs().sum()),
            "gross_banded": float(banded.abs().sum()),
            "holdable": int(allocation.n_holdable),
            "wanted": int(allocation.n_wanted),
            "risk_budget_deployed": allocation.risk_budget_deployed,
            "tracking_error": allocation.tracking_error,
        },
    )


def execute_plan(
    plan: RunPlan,
    broker: Broker,
    cfg: Config,
    guard_cfg: GuardConfig,
    *,
    log: DivergenceLog,
    dry_run: bool,
) -> RunOutcome:
    reconciled = log.reconcile(broker, plan.run_id)

    if not plan.is_rebalance_day or not plan.orders:
        log.record_note(
            plan.run_id,
            "no orders",
            reason=plan.diagnostics.get("reason", "targets already match the broker's positions"),
            equity=plan.equity,
        )
        return RunOutcome(plan=plan, submitted=(), reconciled=len(reconciled), dry_run=dry_run)

    check_notional(plan.total_notional, guard_cfg)
    orders_today = _orders_already_today(broker, plan.session)
    check_order_cap(orders_today, len(plan.orders), guard_cfg)

    if dry_run:
        log.record_note(
            plan.run_id, "dry run", n_orders=len(plan.orders), notional=plan.total_notional
        )
        return RunOutcome(plan=plan, submitted=(), reconciled=len(reconciled), dry_run=True)

    submitted: list[Order] = []
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for planned in plan.orders:
        log.record_predictions(
            [
                Prediction(
                    run_id=plan.run_id,
                    submitted_at=now,
                    symbol=planned.symbol,
                    client_order_id=planned.client_order_id,
                    side=planned.side.value,
                    qty=planned.qty,
                    reference_price=planned.reference_price,
                    reference_date=planned.reference_date.isoformat(),
                    predicted_notional=planned.notional,
                    predicted_position_after=float(planned.target_shares),
                )
            ]
        )
        submitted.append(
            broker.submit(planned.symbol, planned.qty, planned.side, planned.client_order_id)
        )
    return RunOutcome(plan=plan, submitted=tuple(submitted), reconciled=len(reconciled), dry_run=False)


def _orders_already_today(broker: Broker, session: dt.date) -> int | None:
    midnight = dt.datetime.combine(session, dt.time.min, tzinfo=dt.timezone.utc)
    orders = broker.get_orders_since(midnight)
    return None if orders is None else len(orders)


def run_once(
    broker: Broker,
    *,
    cfg: Config | None = None,
    guard_cfg: GuardConfig = CONSERVATIVE,
    state_dir: Path | None = None,
    history: PriceData | None = None,
    dry_run: bool = True,
    session: dt.date | None = None,
) -> RunOutcome:
    state = Path(state_dir or STATE_DIR)
    halt = HaltState(state / "halt.json")

    with fail_closed(halt, context=f"run_once(dry_run={dry_run})"):
        cfg = cfg or load_config()
        high_water = EquityHighWaterMark(state / "high_water.json")
        log = DivergenceLog(state / "divergence.jsonl")
        prices = (
            history
            if history is not None
            else load_prices(cfg.universe, source="alpaca", start="2015-01-01", refresh=True)
        )
        plan = plan_run(
            broker,
            cfg,
            guard_cfg,
            halt=halt,
            high_water=high_water,
            history=prices,
            session=session,
        )
        return execute_plan(plan, broker, cfg, guard_cfg, log=log, dry_run=dry_run)
