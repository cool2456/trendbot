"""The broker interface.

Hard invariant 3 of BUILD_PROMPT.md: *the broker is the only source of truth for
positions. Never infer from local state, logs, or believed fills. Query every run.*

That invariant is what shapes this interface. There is deliberately no
``set_position``, no local position cache, and no method that reports what the
runner *thinks* it holds. :meth:`Broker.get_positions` performs a live query every
time it is called, and it is the only way any code in this package learns what is
held.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

__all__ = ["OrderSide", "Position", "Order", "Account", "Clock", "Broker", "BrokerError"]


class BrokerError(RuntimeError):
    """Any failure to talk to, or make sense of, the broker.

    Deliberately not caught anywhere in the execution path: an unhandled broker
    failure must reach the fail-closed handler and set the halt flag rather than be
    swallowed and retried blindly.
    """


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Position:
    """A position as the broker reports it. Never constructed from local belief."""

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
    """Everything the runner is allowed to know about the outside world."""

    #: Adapters must set this. The runner refuses to operate against a False.
    is_paper: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Live query of currently held positions, keyed by symbol.

        Must hit the broker every call. An adapter that memoises this is a bug.
        """

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Every order not yet in a terminal state.

        The runner refuses to submit anything while this is non-empty, which is what
        prevents a duplicate book being built on top of an unfilled one.
        """

    @abstractmethod
    def submit(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        client_order_id: str,
    ) -> Order:
        """Submit a whole-share market order with a caller-supplied idempotency key."""

    @abstractmethod
    def get_clock(self) -> Clock: ...

    @abstractmethod
    def get_last_close(self, symbols: list[str]) -> dict[str, tuple[float, dt.date]]:
        """Last completed daily close per symbol, with the session date it belongs to.

        The date is what the stale-data guard checks; a price without a timestamp
        cannot be checked for freshness and so is not offered.
        """

    def get_orders_since(self, after: dt.datetime) -> list[Order] | None:
        """Every order submitted since ``after``, or None if the broker cannot say.

        Returning None is not the same as returning an empty list: the daily order cap
        and the divergence log both depend on this, and both refuse rather than
        proceed when it is unavailable.
        """
        return None

    def get_trading_sessions(self, start: dt.date, end: dt.date) -> set[dt.date] | None:
        """Authoritative trading sessions in ``[start, end]``, or None if unavailable.

        Staleness has to be measured in trading days, not calendar days: the strategy
        rebalances on the first trading day of the month, whose previous session is a
        Friday about half the time. An adapter that cannot supply a calendar returns
        None and the caller falls back to counting business days, which over-counts
        across market holidays and therefore errs towards refusing to trade.
        """
        return None

    def get_equity(self) -> float:
        """Convenience wrapper. Still a live query - it calls :meth:`get_account`."""
        return self.get_account().equity
