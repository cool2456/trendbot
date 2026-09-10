from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

__all__ = ["OrderSide", "Position", "Order", "Account", "Clock", "Broker", "BrokerError"]


class BrokerError(RuntimeError):
    pass


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    qty: float
    side: OrderSide
    client_order_id: str
    broker_order_id: str | None = None
    status: str = "unknown"
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: dt.datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.status in {"new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding"}

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"


@dataclass(frozen=True, slots=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    status: str
    is_paper: bool
    account_number: str = ""


@dataclass(frozen=True, slots=True)
class Clock:
    timestamp: dt.datetime
    is_open: bool
    next_open: dt.datetime | None = None
    next_close: dt.datetime | None = None


class Broker(ABC):
    is_paper: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]: ...

    @abstractmethod
    def submit(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        client_order_id: str,
    ) -> Order: ...

    @abstractmethod
    def get_clock(self) -> Clock: ...

    @abstractmethod
    def get_last_close(self, symbols: list[str]) -> dict[str, tuple[float, dt.date]]: ...

    def get_orders_since(self, after: dt.datetime) -> list[Order] | None:
        return None

    def get_trading_sessions(self, start: dt.date, end: dt.date) -> set[dt.date] | None:
        return None

    def get_equity(self) -> float:
        return self.get_account().equity
