"""Predicted versus realised, recorded on every run.

PREREGISTRATION.md section 7 step 6 requires a divergence log on every run of the
six-month paper period. Two different divergences matter and both are recorded:

**Price divergence** - the price the sizing step assumed against the price the fill
actually happened at. Sizing uses the last completed close; the order fills at the
next open. The gap between them is the slippage the backtest's 5 bps cost assumption
is standing in for, and it is the number that tells you whether that assumption is
anywhere near right.

**Position divergence** - the position this bot predicted it would hold against the
position the broker reports on the following run. This should be identically zero.
Anything else means an order was rejected, partially filled, or filled at a size
nobody expected, and it is the check that makes the "broker is the only source of
truth" invariant observable rather than merely asserted.

Predictions are written when orders are submitted and reconciled on a later run,
because a market order submitted while the market is closed does not have a fill
price yet. Records are append-only JSON Lines.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..brokers.base import Broker, Position

__all__ = ["Prediction", "DivergenceLog", "summarise_divergence"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the bot expected, written at submission time."""

    run_id: str
    submitted_at: str
    symbol: str
    client_order_id: str
    side: str
    qty: int
    reference_price: float
    reference_date: str
    predicted_notional: float
    predicted_position_after: float


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What actually happened, written on a later run."""

    run_id: str
    reconciled_at: str
    client_order_id: str
    symbol: str
    status: str
    predicted_qty: int
    filled_qty: float
    reference_price: float
    fill_price: float | None
    price_divergence_bps: float | None
    predicted_position_after: float
    broker_position: float
    position_divergence: float


class DivergenceLog:
    """Append-only JSON Lines log of predictions and their reconciliations."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writing --------------------------------------------------------------

    def _append(self, kind: str, payload: dict) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({"kind": kind, **payload}) + "\n")

    def record_predictions(self, predictions: Iterable[Prediction]) -> int:
        count = 0
        for prediction in predictions:
            self._append("prediction", asdict(prediction))
            count += 1
        return count

    def record_reconciliation(self, reconciliation: Reconciliation) -> None:
        self._append("reconciliation", asdict(reconciliation))

    def record_note(self, run_id: str, message: str, **extra) -> None:
        self._append(
            "note",
            {
                "run_id": run_id,
                "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "message": message,
                **extra,
            },
        )

    # ---- reading --------------------------------------------------------------

    def entries(self, kind: str | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if kind is None or record.get("kind") == kind:
                out.append(record)
        return out

    def unreconciled(self) -> list[dict]:
        """Predictions with no matching reconciliation yet."""
        done = {r["client_order_id"] for r in self.entries("reconciliation")}
        return [p for p in self.entries("prediction") if p["client_order_id"] not in done]

    # ---- the reconciliation pass ----------------------------------------------

    def reconcile(self, broker: Broker, run_id: str) -> list[Reconciliation]:
        """Match every outstanding prediction against the broker's own record.

        Positions come from ``broker.get_positions()``, never from anything this
        process previously believed.
        """
        outstanding = self.unreconciled()
        if not outstanding:
            return []

        positions: dict[str, Position] = broker.get_positions()

        # The lookback reaches back to the OLDEST outstanding prediction, not a fixed
        # window. A fixed 45 days would make anything older permanently
        # unreconcilable and emit a false "not found at broker" note on every
        # subsequent run - reachable in normal operation, since the drawdown halt is
        # tighter than the strategy's own historical drawdown and a multi-week halt is
        # expected rather than exceptional.
        submitted = [p["submitted_at"] for p in outstanding if p.get("submitted_at")]
        oldest = min(
            (dt.datetime.fromisoformat(t) for t in submitted),
            default=dt.datetime.now(dt.timezone.utc),
        )
        lookback = oldest - dt.timedelta(days=1)

        by_client_id = {}
        broker_orders = broker.get_orders_since(lookback)
        if broker_orders is None:
            self.record_note(
                run_id,
                "broker cannot report order history; reconciliation skipped this run",
                outstanding=len(outstanding),
            )
            return []
        for order in broker_orders:
            if order.client_order_id:
                by_client_id[order.client_order_id] = order

        results: list[Reconciliation] = []
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        for prediction in outstanding:
            order = by_client_id.get(prediction["client_order_id"])
            if order is None:
                # The broker has no record of an order this bot believes it sent.
                # That is itself a divergence worth logging, not an error to skip.
                self.record_note(
                    run_id,
                    "predicted order not found at broker",
                    client_order_id=prediction["client_order_id"],
                    symbol=prediction["symbol"],
                )
                continue
            if order.is_open:
                continue  # not yet terminal; leave it outstanding

            held = positions.get(prediction["symbol"])
            broker_qty = float(held.qty) if held is not None else 0.0
            fill_price = order.filled_avg_price
            reference = float(prediction["reference_price"])
            divergence_bps = (
                (fill_price - reference) / reference * 10_000.0
                if fill_price is not None and reference > 0
                else None
            )
            # Signed so that positive always means "worse for us": paying above the
            # reference on a buy, or selling below it.
            if divergence_bps is not None and prediction["side"] == "sell":
                divergence_bps = -divergence_bps

            reconciliation = Reconciliation(
                run_id=run_id,
                reconciled_at=now,
                client_order_id=prediction["client_order_id"],
                symbol=prediction["symbol"],
                status=order.status,
                predicted_qty=int(prediction["qty"]),
                filled_qty=float(order.filled_qty),
                reference_price=reference,
                fill_price=fill_price,
                price_divergence_bps=divergence_bps,
                predicted_position_after=float(prediction["predicted_position_after"]),
                broker_position=broker_qty,
                position_divergence=broker_qty - float(prediction["predicted_position_after"]),
            )
            self.record_reconciliation(reconciliation)
            results.append(reconciliation)
        return results


def summarise_divergence(log: DivergenceLog) -> dict:
    """Aggregate the log into the handful of numbers worth looking at."""
    records = log.entries("reconciliation")
    if not records:
        return {"n": 0}
    slippage = [r["price_divergence_bps"] for r in records if r["price_divergence_bps"] is not None]
    position_errors = [r["position_divergence"] for r in records]
    unfilled = [r for r in records if r["status"] != "filled"]
    return {
        "n": len(records),
        "mean_slippage_bps": sum(slippage) / len(slippage) if slippage else None,
        "worst_slippage_bps": max(slippage) if slippage else None,
        "n_position_mismatches": sum(1 for e in position_errors if abs(e) > 1e-9),
        "max_abs_position_divergence": max((abs(e) for e in position_errors), default=0.0),
        "n_not_filled": len(unfilled),
    }
