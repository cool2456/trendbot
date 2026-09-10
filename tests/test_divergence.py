from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest

from tests.mock_broker import MockBroker
from trendbot.brokers.base import OrderSide
from trendbot.monitor.divergence import DivergenceLog, Prediction, summarise_divergence


def _prediction(symbol="SPY", side="buy", qty=10, ref=100.0, after=10.0, coid="c1"):
    return Prediction(
        run_id="r1",
        submitted_at="2026-09-01T13:31:00+00:00",
        symbol=symbol,
        client_order_id=coid,
        side=side,
        qty=qty,
        reference_price=ref,
        reference_date="2026-08-31",
        predicted_notional=qty * ref,
        predicted_position_after=after,
    )


@pytest.fixture
def log(tmp_path):
    return DivergenceLog(tmp_path / "divergence.jsonl")


def test_the_log_is_append_only_json_lines(log):
    log.record_predictions([_prediction(coid="a"), _prediction(coid="b")])
    log.record_note("r1", "hello", extra=1)
    lines = log.path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["kind"] for line in lines] == ["prediction", "prediction", "note"]
    log.record_predictions([_prediction(coid="c")])
    assert len(log.path.read_text().splitlines()) == 4


def test_entries_filters_by_kind_and_an_absent_file_is_empty(tmp_path):
    empty = DivergenceLog(tmp_path / "nothing.jsonl")
    assert empty.entries() == []
    empty.record_note("r", "x")
    assert len(empty.entries("note")) == 1
    assert empty.entries("reconciliation") == []


def test_unreconciled_tracks_what_is_still_outstanding(log):
    log.record_predictions([_prediction(coid="a"), _prediction(coid="b")])
    assert {p["client_order_id"] for p in log.unreconciled()} == {"a", "b"}


def test_reconcile_computes_slippage_signed_so_positive_is_worse(log):
    broker = MockBroker(prices={"SPY": 101.0})
    broker.submit("SPY", 10, OrderSide.BUY, "c1")
    log.record_predictions([_prediction(coid="c1", ref=100.0, after=10.0)])
    (result,) = log.reconcile(broker, "r2")
    assert result.fill_price == 101.0
    assert result.price_divergence_bps == pytest.approx(100.0)
    assert result.position_divergence == 0.0
    assert result.status == "filled"


def test_a_sell_filling_below_the_reference_is_also_positive_slippage(log):
    broker = MockBroker(prices={"SPY": 99.0}, positions={"SPY": 10.0})
    broker.submit("SPY", 10, OrderSide.SELL, "c1")
    log.record_predictions([_prediction(coid="c1", side="sell", ref=100.0, after=0.0)])
    (result,) = log.reconcile(broker, "r2")
    assert result.price_divergence_bps == pytest.approx(100.0)


def test_position_divergence_is_measured_against_the_broker_not_the_prediction(log):
    broker = MockBroker(prices={"SPY": 100.0})
    broker.submit("SPY", 10, OrderSide.BUY, "c1")
    broker._shares["SPY"] = 7.0
    log.record_predictions([_prediction(coid="c1", after=10.0)])
    (result,) = log.reconcile(broker, "r2")
    assert result.broker_position == 7.0
    assert result.predicted_position_after == 10.0
    assert result.position_divergence == -3.0


def test_an_order_still_open_stays_unreconciled(log):
    broker = MockBroker(prices={"SPY": 100.0})
    order = broker.submit("SPY", 10, OrderSide.BUY, "c1")
    broker._orders[-1] = dataclasses.replace(order, status="new", filled_qty=0.0)
    log.record_predictions([_prediction(coid="c1")])
    assert log.reconcile(broker, "r2") == []
    assert len(log.unreconciled()) == 1
    broker._orders[-1] = dataclasses.replace(order, status="filled", filled_qty=10.0)
    assert len(log.reconcile(broker, "r3")) == 1
    assert log.unreconciled() == []


def test_an_order_the_broker_has_never_heard_of_is_logged_not_raised(log):
    broker = MockBroker(prices={"SPY": 100.0})
    log.record_predictions([_prediction(coid="ghost")])
    assert log.reconcile(broker, "r2") == []
    notes = log.entries("note")
    assert any("not found at broker" in n["message"] for n in notes)
    assert len(log.unreconciled()) == 1


def test_a_rejected_order_reconciles_and_is_counted_as_unfilled(log):
    broker = MockBroker(prices={"SPY": 100.0})
    order = broker.submit("SPY", 10, OrderSide.BUY, "c1")
    broker._orders[-1] = dataclasses.replace(
        order, status="rejected", filled_qty=0.0, filled_avg_price=None
    )
    broker._shares["SPY"] = 0.0
    log.record_predictions([_prediction(coid="c1", after=10.0)])
    (result,) = log.reconcile(broker, "r2")
    assert result.status == "rejected"
    assert result.filled_qty == 0.0
    assert result.price_divergence_bps is None
    assert result.position_divergence == -10.0
    assert summarise_divergence(log)["n_not_filled"] == 1


def test_reconcile_is_idempotent(log):
    broker = MockBroker(prices={"SPY": 100.0})
    broker.submit("SPY", 10, OrderSide.BUY, "c1")
    log.record_predictions([_prediction(coid="c1")])
    assert len(log.reconcile(broker, "r2")) == 1
    assert log.reconcile(broker, "r3") == []
    assert len(log.entries("reconciliation")) == 1


def test_summarise_on_an_empty_log_and_on_a_populated_one(log):
    assert summarise_divergence(log) == {"n": 0}
    broker = MockBroker(prices={"SPY": 101.0, "TLT": 100.0})
    broker.submit("SPY", 10, OrderSide.BUY, "c1")
    broker.submit("TLT", 5, OrderSide.BUY, "c2")
    log.record_predictions(
        [
            _prediction(symbol="SPY", coid="c1", ref=100.0, after=10.0),
            _prediction(symbol="TLT", coid="c2", ref=100.0, after=5.0),
        ]
    )
    log.reconcile(broker, "r2")
    summary = summarise_divergence(log)
    assert summary["n"] == 2
    assert summary["worst_slippage_bps"] == pytest.approx(100.0)
    assert summary["mean_slippage_bps"] == pytest.approx(50.0)
    assert summary["n_position_mismatches"] == 0
    assert summary["n_not_filled"] == 0
