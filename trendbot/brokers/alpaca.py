from __future__ import annotations

import datetime as dt
from typing import Any

import requests

from ..data import alpaca_credentials
from .base import Account, Broker, BrokerError, Clock, Order, OrderSide, Position

__all__ = ["AlpacaPaperBroker", "PAPER_TRADING_URL", "MARKET_DATA_URL"]

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
MARKET_DATA_URL = "https://data.alpaca.markets"

_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}


class AlpacaPaperBroker(Broker):
    is_paper = True

    def __init__(self, *, timeout: float = 30.0, session: requests.Session | None = None) -> None:
        key, secret = alpaca_credentials()
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._timeout = timeout
        self._session = session or requests.Session()
        self._key_prefix = key[:4]

    @property
    def name(self) -> str:
        return f"alpaca-paper[{self._key_prefix}...]"

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self._session.request(
            method, url, headers=self._headers, timeout=self._timeout, **kwargs
        )
        if response.status_code >= 400:
            raise BrokerError(f"{method} {url} -> HTTP {response.status_code}: {response.text[:300]}")
        if not response.content:
            return None
        return response.json()

    def _trading(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, f"{PAPER_TRADING_URL}{path}", **kwargs)

    def get_account(self) -> Account:
        payload = self._trading("GET", "/v2/account")
        return Account(
            equity=float(payload["equity"]),
            cash=float(payload["cash"]),
            buying_power=float(payload["buying_power"]),
            status=str(payload["status"]),
            is_paper=True,
            account_number=str(payload.get("account_number", "")),
        )

    def get_positions(self) -> dict[str, Position]:
        payload = self._trading("GET", "/v2/positions") or []
        return {
            str(p["symbol"]): Position(
                symbol=str(p["symbol"]),
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
            )
            for p in payload
        }

    def get_open_orders(self) -> list[Order]:
        payload = self._trading("GET", "/v2/orders", params={"status": "open", "limit": 500}) or []
        return [self._to_order(o) for o in payload]

    def get_orders_since(self, after: dt.datetime) -> list[Order]:
        payload = (
            self._trading(
                "GET",
                "/v2/orders",
                params={
                    "status": "all",
                    "after": after.astimezone(dt.timezone.utc).isoformat(),
                    "limit": 500,
                },
            )
            or []
        )
        return [self._to_order(o) for o in payload]

    def submit(self, symbol: str, qty: int, side: OrderSide, client_order_id: str) -> Order:
        if int(qty) != qty or qty <= 0:
            raise BrokerError(f"qty must be a positive whole number of shares, got {qty!r}")
        payload = self._trading(
            "POST",
            "/v2/orders",
            json={
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": OrderSide(side).value,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        return self._to_order(payload)

    def get_clock(self) -> Clock:
        payload = self._trading("GET", "/v2/clock")
        return Clock(
            timestamp=_parse_ts(payload["timestamp"]),
            is_open=bool(payload["is_open"]),
            next_open=_parse_ts(payload.get("next_open")),
            next_close=_parse_ts(payload.get("next_close")),
        )

    def get_last_close(self, symbols: list[str]) -> dict[str, tuple[float, dt.date]]:
        end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)
        start = end - dt.timedelta(days=20)
        out: dict[str, tuple[float, dt.date]] = {}
        for symbol in symbols:
            payload = self._request(
                "GET",
                f"{MARKET_DATA_URL}/v2/stocks/{symbol}/bars",
                params={
                    "timeframe": "1Day",
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "adjustment": "raw",
                    "feed": "sip",
                    "limit": 100,
                },
            )
            bars = (payload or {}).get("bars") or []
            if not bars:
                raise BrokerError(f"no recent daily bar for {symbol}")
            last = bars[-1]
            out[symbol] = (float(last["c"]), dt.date.fromisoformat(last["t"][:10]))
        return out

    def get_trading_sessions(self, start: dt.date, end: dt.date) -> set[dt.date] | None:
        payload = self._trading(
            "GET",
            "/v2/calendar",
            params={"start": start.isoformat(), "end": end.isoformat()},
        ) or []
        return {dt.date.fromisoformat(day["date"]) for day in payload} or None

    @staticmethod
    def _to_order(payload: dict) -> Order:
        filled_price = payload.get("filled_avg_price")
        return Order(
            symbol=str(payload["symbol"]),
            qty=float(payload["qty"]),
            side=OrderSide(str(payload["side"])),
            client_order_id=str(payload.get("client_order_id", "")),
            broker_order_id=str(payload.get("id", "")) or None,
            status=str(payload.get("status", "unknown")),
            filled_qty=float(payload.get("filled_qty") or 0.0),
            filled_avg_price=float(filled_price) if filled_price else None,
            submitted_at=_parse_ts(payload.get("submitted_at")),
        )


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
