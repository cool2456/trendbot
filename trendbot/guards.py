from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .brokers.base import Order

__all__ = [
    "GuardConfig",
    "CONSERVATIVE",
    "GuardViolation",
    "DrawdownBreach",
    "NotAPaperAccount",
    "HaltError",
    "HaltState",
    "fail_closed",
    "check_not_halted",
    "check_no_open_orders",
    "check_drawdown",
    "check_notional",
    "check_order_cap",
    "check_data_freshness",
    "check_broker_is_paper",
]


class GuardViolation(RuntimeError):
    sticky = False


class NotAPaperAccount(GuardViolation):
    sticky = True


class DrawdownBreach(GuardViolation):
    sticky = True


class HaltError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GuardConfig:
    max_drawdown: float

    max_gross_notional: float

    max_orders_per_day: int

    max_data_staleness_days: int

    def __post_init__(self) -> None:
        if not 0 < self.max_drawdown < 1:
            raise ValueError(f"max_drawdown must be in (0,1), got {self.max_drawdown}")
        if self.max_gross_notional <= 0:
            raise ValueError("max_gross_notional must be positive")
        if self.max_orders_per_day < 1:
            raise ValueError("max_orders_per_day must be at least 1")
        if self.max_data_staleness_days < 1:
            raise ValueError("max_data_staleness_days must be at least 1")


CONSERVATIVE = GuardConfig(
    max_drawdown=0.15,
    max_gross_notional=25_000.0,
    max_orders_per_day=12,
    max_data_staleness_days=1,
)


@dataclass(frozen=True, slots=True)
class HaltRecord:
    halted_at: str
    reason: str
    detail: str
    context: str = ""

    def __str__(self) -> str:
        return f"HALTED at {self.halted_at} during {self.context or 'unknown'}: {self.reason} - {self.detail}"


class HaltState:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def is_halted(self) -> bool:
        return self.path.is_file()

    def record(self) -> HaltRecord | None:
        if not self.is_halted():
            return None
        try:
            data = json.loads(self.path.read_text())
            return HaltRecord(**data)
        except (json.JSONDecodeError, TypeError, OSError, UnicodeDecodeError) as exc:
            return HaltRecord(
                halted_at="unknown",
                reason="unreadable halt flag",
                detail=f"{self.path} exists but could not be parsed ({type(exc).__name__}: {exc}). "
                "Treated as halted. Clear it with --reset-halt once you know why it is corrupt.",
                context="unknown",
            )

    def halt(self, reason: str, detail: str = "", context: str = "") -> HaltRecord:
        existing = self.record()
        if existing is not None:
            return existing
        record = HaltRecord(
            halted_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            reason=reason,
            detail=detail[:4000],
            context=context,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(record), handle, indent=2)
        os.replace(tmp, self.path)
        return record

    def clear(self, operator_note: str) -> None:
        if not operator_note or not operator_note.strip():
            raise ValueError("clearing the halt flag requires an operator note")
        if not self.is_halted():
            raise HaltError("nothing to clear: the halt flag is not set")
        archive = self.path.with_suffix(".cleared.jsonl")
        record = self.record()
        with archive.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "cleared_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "operator_note": operator_note.strip(),
                        "halt": asdict(record) if record else None,
                    }
                )
                + "\n"
            )
        self.path.unlink()


@contextmanager
def fail_closed(halt: HaltState, context: str) -> Iterator[None]:
    try:
        yield
    except GuardViolation as exc:
        if not exc.sticky:
            raise
        halt.halt(
            reason=f"{type(exc).__name__}",
            detail=f"{exc}",
            context=context,
        )
        raise
    except BaseException as exc:
        halt.halt(
            reason=f"unhandled {type(exc).__name__}",
            detail=f"{exc}",
            context=context,
        )
        raise


def check_not_halted(halt: HaltState) -> None:
    record = halt.record()
    if record is not None:
        raise HaltError(
            f"{record}\n"
            f"No orders will be submitted. To resume, investigate the cause and then run:\n"
            f"  python scripts/run_live.py --reset-halt --note 'what you found and fixed'"
        )


def check_broker_is_paper(broker) -> None:
    if not getattr(broker, "is_paper", False):
        raise NotAPaperAccount(
            f"broker {getattr(broker, 'name', broker)!r} does not declare itself as paper. "
            "This repository has no live trading path."
        )


def check_no_open_orders(open_orders: list[Order]) -> None:
    if open_orders:
        described = ", ".join(f"{o.side.value} {o.qty:g} {o.symbol} ({o.status})" for o in open_orders[:10])
        raise GuardViolation(
            f"{len(open_orders)} order(s) still open: {described}. "
            "Refusing to submit anything until they reach a terminal state."
        )


def check_drawdown(equity: float, peak_equity: float, cfg: GuardConfig) -> None:
    if peak_equity <= 0:
        raise GuardViolation(f"peak equity is not positive ({peak_equity}); cannot evaluate drawdown")
    drawdown = 1.0 - equity / peak_equity
    if drawdown > cfg.max_drawdown + 1e-12:
        raise DrawdownBreach(
            f"drawdown {drawdown:.2%} exceeds the {cfg.max_drawdown:.2%} limit "
            f"(equity {equity:,.2f} against a peak of {peak_equity:,.2f})"
        )


def check_notional(order_notional: float, cfg: GuardConfig) -> None:
    if order_notional > cfg.max_gross_notional:
        raise GuardViolation(
            f"orders total {order_notional:,.2f} of notional, above the "
            f"{cfg.max_gross_notional:,.2f} per-run limit"
        )


def check_order_cap(orders_already_today: int | None, n_new_orders: int, cfg: GuardConfig) -> None:
    if orders_already_today is None:
        raise GuardViolation(
            "the broker cannot report how many orders were already sent today, so the "
            f"daily cap of {cfg.max_orders_per_day} cannot be enforced. Refusing to submit."
        )
    total = orders_already_today + n_new_orders
    if total > cfg.max_orders_per_day:
        raise GuardViolation(
            f"{orders_already_today} order(s) already sent today plus {n_new_orders} new "
            f"would be {total}, above the daily cap of {cfg.max_orders_per_day}"
        )


def check_data_freshness(
    bar_dates: dict[str, dt.date],
    today: dt.date,
    cfg: GuardConfig,
    *,
    trading_calendar: set[dt.date] | None = None,
) -> None:
    if not bar_dates:
        raise GuardViolation("no price timestamps supplied; cannot verify data freshness")
    stale: dict[str, int] = {}
    for symbol, bar_date in bar_dates.items():
        if bar_date > today:
            raise GuardViolation(f"{symbol} has a bar dated {bar_date}, in the future relative to {today}")
        if trading_calendar is not None:
            age = sum(1 for d in trading_calendar if bar_date < d <= today)
        else:
            age = int(np.busday_count(bar_date, today))
        if age > cfg.max_data_staleness_days:
            stale[symbol] = age
    if stale:
        raise GuardViolation(
            f"stale price data (limit {cfg.max_data_staleness_days}d): "
            + ", ".join(f"{s} is {age}d old" for s, age in sorted(stale.items()))
        )


@dataclass
class EquityHighWaterMark:
    path: Path
    _data: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_file():
            self._data = json.loads(self.path.read_text())

    def update(self, equity: float) -> float:
        peak = max(float(self._data.get("peak_equity", 0.0)), float(equity))
        self._data = {
            "peak_equity": peak,
            "last_equity": float(equity),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(self._data, handle, indent=2)
        os.replace(tmp, self.path)
        return peak

    @property
    def peak(self) -> float:
        return float(self._data.get("peak_equity", 0.0))
