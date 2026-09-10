from __future__ import annotations

import datetime as dt

from trendbot.brokers.base import Account, Broker, BrokerError, Clock, Order, OrderSide, Position


class MockBroker(Broker):
    is_paper = True

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
        bar_date: dt.date | None = None,
        now: dt.datetime | None = None,
        fail_on_submit: bool = False,
        sessions: set[dt.date] | None = None,
    ) -> None:
        self._sessions = sessions
        self._equity = equity
        self._prices = dict(prices or {})
        self._shares = dict(positions or {})
        self._bar_date = bar_date or dt.date(2026, 9, 1)
        self._now = now or dt.datetime(2026, 9, 1, 14, 0, tzinfo=dt.timezone.utc)
        self._orders: list[Order] = []
        self._fail_on_submit = fail_on_submit
        self.position_queries = 0

    @property
    def name(self) -> str:
        return "mock"

    def get_positions(self) -> dict[str, Position]:
        self.position_queries += 1
        return {
            symbol: Position(
                symbol=symbol,
                qty=qty,
                market_value=qty * self._prices[symbol],
                avg_entry_price=self._prices[symbol],
                current_price=self._prices[symbol],
            )
            for symbol, qty in self._shares.items()
            if qty != 0
        }

    def get_account(self) -> Account:
        return Account(
            equity=self._equity,
            cash=self._equity,
            buying_power=self._equity,
            status="ACTIVE",
            is_paper=True,
            account_number="MOCK",
        )

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders if o.is_open]

    def get_orders_since(self, after: dt.datetime) -> list[Order]:
        return [o for o in self._orders if o.submitted_at is None or o.submitted_at >= after]

    def submit(self, symbol: str, qty: int, side: OrderSide, client_order_id: str) -> Order:
        if self._fail_on_submit:
            raise BrokerError("simulated broker outage")
        if any(o.client_order_id == client_order_id for o in self._orders):
            raise BrokerError(f"duplicate client_order_id {client_order_id}")
        order = Order(
            symbol=symbol,
            qty=float(qty),
            side=side,
            client_order_id=client_order_id,
            broker_order_id=f"mock-{len(self._orders)}",
            status="filled",
            filled_qty=float(qty),
            filled_avg_price=self._prices[symbol],
            submitted_at=self._now,
        )
        self._orders.append(order)
        delta = qty if side is OrderSide.BUY else -qty
        self._shares[symbol] = self._shares.get(symbol, 0.0) + delta
        return order

    def get_clock(self) -> Clock:
        return Clock(timestamp=self._now, is_open=True)

    def get_trading_sessions(self, start: dt.date, end: dt.date):
        if self._sessions is None:
            return None
        return {d for d in self._sessions if start <= d <= end}

    def get_last_close(self, symbols: list[str]) -> dict[str, tuple[float, dt.date]]:
        missing = [s for s in symbols if s not in self._prices]
        if missing:
            raise BrokerError(f"no price for {missing}")
        return {s: (self._prices[s], self._bar_date) for s in symbols}
