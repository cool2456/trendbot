"""Broker adapters, offline.

The Alpaca adapter is exercised against a fake ``requests.Session`` so that its URL
construction, paper-only guarantees and response parsing are all tested without a
network call.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

import pytest

from trendbot.brokers import alpaca as alpaca_module
from trendbot.brokers.alpaca import MARKET_DATA_URL, PAPER_TRADING_URL, AlpacaPaperBroker
from trendbot.brokers.base import Account, Broker, BrokerError, Clock, Order, OrderSide, Position
from trendbot.brokers.schwab import SchwabBroker

REPO = Path(__file__).resolve().parent.parent


# ---- hard invariant 4: paper only ----------------------------------------------------


def test_the_only_trading_endpoint_is_the_paper_endpoint():
    assert PAPER_TRADING_URL == "https://paper-api.alpaca.markets"
    source = (REPO / "trendbot" / "brokers" / "alpaca.py").read_text()
    # Every mention of the trading host must carry the paper- prefix.
    for index in range(len(source)):
        if source.startswith("api.alpaca.markets", index):
            assert source[max(0, index - 6) : index] == "paper-", (
                f"a non-paper trading endpoint appears at offset {index}: "
                f"{source[max(0, index - 40): index + 30]!r}"
            )


def test_no_module_in_the_package_references_a_live_trading_endpoint():
    for path in (REPO / "trendbot").rglob("*.py"):
        text = path.read_text()
        for index in range(len(text)):
            if text.startswith("api.alpaca.markets", index):
                prefix = text[max(0, index - 6) : index]
                assert prefix == "paper-", f"{path} references a live endpoint"


def test_market_data_host_is_distinct_from_the_trading_host():
    # data.alpaca.markets is read-only and cannot place an order.
    assert MARKET_DATA_URL == "https://data.alpaca.markets"
    assert "paper" not in MARKET_DATA_URL


def test_adapter_declares_itself_paper():
    assert AlpacaPaperBroker.is_paper is True
    # ... and it is a class attribute, so it cannot be turned off by a constructor arg.
    assert "is_paper" not in inspect.signature(AlpacaPaperBroker.__init__).parameters


def test_constructor_rejects_a_non_paper_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_KEY", "AKLIVEKEY123")
    monkeypatch.setenv("ALPACA_SECRET", "secret")
    monkeypatch.setattr(alpaca_module, "alpaca_credentials", _real_credentials_with_env)
    with pytest.raises(Exception, match="paper"):
        AlpacaPaperBroker()


def _real_credentials_with_env():
    import os

    from trendbot.data import DataError

    key, secret = os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"]
    if not key.startswith("PK"):
        raise DataError(f"ALPACA_KEY {key[:4]}... does not look like a paper key")
    return key, secret


# ---- Schwab is a documented refusal, not an oversight --------------------------------


def test_schwab_raises_with_both_reasons():
    with pytest.raises(NotImplementedError) as excinfo:
        SchwabBroker()
    message = str(excinfo.value)
    assert "paper" in message
    assert "7 days" in message or "7-day" in message


def test_schwab_implements_the_interface_but_every_method_raises():
    assert issubclass(SchwabBroker, Broker)
    instance = SchwabBroker.__new__(SchwabBroker)  # bypass __init__
    for name in ("get_positions", "get_account", "get_open_orders", "get_clock"):
        with pytest.raises(NotImplementedError):
            getattr(instance, name)()
    with pytest.raises(NotImplementedError):
        instance.submit("SPY", 1, OrderSide.BUY, "id")
    with pytest.raises(NotImplementedError):
        instance.get_last_close(["SPY"])


def test_schwab_docstring_records_why():
    doc = SchwabBroker.__module__ and __import__("trendbot.brokers.schwab", fromlist=["x"]).__doc__
    assert "no paper environment" in doc.lower() or "no sandbox" in doc.lower()
    assert "refresh token" in doc.lower()


# ---- the ABC forbids a local position cache ------------------------------------------


def test_the_interface_offers_no_way_to_set_a_position():
    names = {n for n, _ in inspect.getmembers(Broker, predicate=inspect.isfunction)}
    for forbidden in ("set_positions", "set_position", "cache_positions", "update_positions"):
        assert forbidden not in names
    for required in ("get_positions", "get_account", "get_open_orders", "submit", "get_clock"):
        assert required in names


def test_broker_is_abstract():
    with pytest.raises(TypeError):
        Broker()


# ---- the adapter, against a fake transport -------------------------------------------


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, int):
                    return FakeResponse({"message": "nope"}, status=payload)
                return FakeResponse(payload)
        raise AssertionError(f"unexpected request to {url}")


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setattr(alpaca_module, "alpaca_credentials", lambda: ("PKTEST", "SECRET"))

    def make(routes):
        return AlpacaPaperBroker(session=FakeSession(routes))

    return make


def test_get_account_parses_and_marks_paper(broker):
    b = broker(
        {
            "/v2/account": {
                "equity": "100000",
                "cash": "25000.5",
                "buying_power": "400000",
                "status": "ACTIVE",
                "account_number": "PA1",
            }
        }
    )
    account = b.get_account()
    assert account == Account(100000.0, 25000.5, 400000.0, "ACTIVE", True, "PA1")
    assert b.get_equity() == 100000.0


def test_get_positions_hits_the_paper_host_every_call(broker):
    b = broker(
        {
            "/v2/positions": [
                {
                    "symbol": "SPY",
                    "qty": "10",
                    "market_value": "7674.5",
                    "avg_entry_price": "760",
                    "current_price": "767.45",
                }
            ]
        }
    )
    for _ in range(3):
        positions = b.get_positions()
    assert positions["SPY"] == Position("SPY", 10.0, 7674.5, 760.0, 767.45)
    # three calls, every one of them a live request to the paper host
    assert len(b._session.calls) == 3
    assert all(url.startswith(PAPER_TRADING_URL) for _, url, _ in b._session.calls)


def test_submit_sends_a_whole_share_day_market_order_with_the_idempotency_key(broker):
    b = broker(
        {
            "/v2/orders": {
                "symbol": "SPY",
                "qty": "10",
                "side": "buy",
                "client_order_id": "tb1-x",
                "id": "abc",
                "status": "accepted",
                "filled_qty": "0",
                "submitted_at": "2026-09-01T13:31:00Z",
            }
        }
    )
    order = b.submit("SPY", 10, OrderSide.BUY, "tb1-x")
    _, url, kwargs = b._session.calls[-1]
    assert url == f"{PAPER_TRADING_URL}/v2/orders"
    assert kwargs["json"] == {
        "symbol": "SPY",
        "qty": "10",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": "tb1-x",
    }
    assert order.is_open and not order.is_filled


@pytest.mark.parametrize("qty", [0, -5, 1.5])
def test_submit_rejects_a_non_whole_positive_share_count(broker, qty):
    b = broker({"/v2/orders": {}})
    with pytest.raises(BrokerError, match="whole number"):
        b.submit("SPY", qty, OrderSide.BUY, "id")


def test_an_http_error_becomes_a_broker_error_and_is_not_swallowed(broker):
    b = broker({"/v2/account": 500})
    with pytest.raises(BrokerError, match="HTTP 500"):
        b.get_account()


def test_get_last_close_uses_raw_prices_and_returns_the_session_date(broker):
    b = broker(
        {
            "/v2/stocks/SPY/bars": {
                "bars": [
                    {"t": "2026-08-17T04:00:00Z", "c": 760.0},
                    {"t": "2026-08-18T04:00:00Z", "c": 767.45},
                ]
            }
        }
    )
    result = b.get_last_close(["SPY"])
    assert result == {"SPY": (767.45, dt.date(2026, 8, 18))}
    _, url, kwargs = b._session.calls[-1]
    assert url.startswith(MARKET_DATA_URL)
    # Feasibility and order sizing must use the price actually quoted, not a
    # back-adjusted research series.
    assert kwargs["params"]["adjustment"] == "raw"


def test_get_last_close_raises_when_a_symbol_has_no_bar(broker):
    b = broker({"/v2/stocks/SPY/bars": {"bars": []}})
    with pytest.raises(BrokerError, match="no recent daily bar"):
        b.get_last_close(["SPY"])


def test_order_open_and_filled_states():
    assert Order("SPY", 1, OrderSide.BUY, "i", status="new").is_open
    assert Order("SPY", 1, OrderSide.BUY, "i", status="partially_filled").is_open
    assert not Order("SPY", 1, OrderSide.BUY, "i", status="filled").is_open
    assert Order("SPY", 1, OrderSide.BUY, "i", status="filled").is_filled
    assert not Order("SPY", 1, OrderSide.BUY, "i", status="canceled").is_open


def test_clock_parses_z_suffixed_timestamps(broker):
    b = broker(
        {
            "/v2/clock": {
                "timestamp": "2026-09-01T13:31:00Z",
                "is_open": True,
                "next_open": "2026-09-02T13:30:00Z",
                "next_close": "2026-09-01T20:00:00Z",
            }
        }
    )
    clock = b.get_clock()
    assert isinstance(clock, Clock)
    assert clock.is_open
    assert clock.timestamp.tzinfo is not None
