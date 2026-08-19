"""Pre-trade guards and the fail-closed halt flag.

BUILD_PROMPT.md hard invariant 5: *Fail closed. Unhandled exception in the
execution path sets a persistent halt flag and stops. Clearing requires an explicit
CLI command.*

Where the thresholds come from
------------------------------
Nowhere in PREREGISTRATION.md. That document specifies a strategy; it contains no
operational risk limits, and BUILD_PROMPT.md forbids inventing parameters. So
:class:`GuardConfig` has **no default values** - constructing one requires stating
every threshold explicitly - and the numbers actually in use live in the named
profile :data:`CONSERVATIVE`, which records that they were chosen by the operator
rather than derived from the pre-registration. They govern only the paper path and
cannot affect a backtest.

A guard whose threshold is missing must fail closed, never permissive. That is why
there is no default and no ``getattr(cfg, 'x', something)`` anywhere below.
"""

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
    """A pre-trade check refused this run. No orders are submitted.

    A guard violation is a *designed refusal*, not a failure, so by default it does
    not set the persistent halt flag: the run stops, and the next scheduled run
    re-evaluates from scratch. That distinction matters operationally. Stale data
    over a holiday weekend, or an order still open from an hour ago, are transient
    conditions; if they latched the halt flag, an operator would have to clear it by
    hand every few weeks and would quickly learn to do so reflexively - which is
    exactly how a halt flag stops meaning anything.

    Subclasses that set ``sticky = True`` do latch it. Those are the conditions where
    a human genuinely should look before the bot trades again.
    """

    #: Whether tripping this guard sets the persistent halt flag.
    sticky = False


class NotAPaperAccount(GuardViolation):
    """The broker handed to the runner does not declare itself a paper account.

    Sticky. BUILD_PROMPT hard invariant 4 is "paper only, no live code path", so this
    is not a transient condition that might resolve on the next scheduled run - it
    means something is wired up wrong, and retrying silently every hour until someone
    notices is the wrong response.
    """

    sticky = True


class DrawdownBreach(GuardViolation):
    """Equity has fallen further below its peak than the operator permitted.

    Sticky, unlike other guards: this is not a transient market-data condition that
    will resolve itself tomorrow, it is the risk limit the account was given doing
    the one thing it exists to do. Trading resumes only after a person has looked at
    it and cleared the flag with a note.
    """

    sticky = True


class HaltError(RuntimeError):
    """The persistent halt flag is set. Nothing runs until it is cleared."""


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """Operational limits for the paper path. Every field is required."""

    max_drawdown: float
    """Fraction below the high-water mark at which the bot halts, e.g. 0.15."""

    max_gross_notional: float
    """Largest total dollar value of orders permitted in a single run."""

    max_orders_per_day: int
    """Largest number of orders permitted in one calendar day."""

    max_data_staleness_days: int
    """Refuse to trade if the newest bar is older than this many calendar days."""

    def __post_init__(self) -> None:
        if not 0 < self.max_drawdown < 1:
            raise ValueError(f"max_drawdown must be in (0,1), got {self.max_drawdown}")
        if self.max_gross_notional <= 0:
            raise ValueError("max_gross_notional must be positive")
        if self.max_orders_per_day < 1:
            raise ValueError("max_orders_per_day must be at least 1")
        if self.max_data_staleness_days < 1:
            raise ValueError("max_data_staleness_days must be at least 1")


#: The operator's chosen limits. NOT from PREREGISTRATION.md - that document has no
#: operational thresholds - and deliberately tight: this is a paper account whose
#: purpose is to surface problems, so a guard that trips too often is a better
#: failure than one that never trips. Note that the backtest's worst drawdown was
#: 20.8%, which is wider than the 15% halt here: a faithful paper run of this
#: strategy is *expected* to halt at some point, and that halt is information, not
#: an error.
CONSERVATIVE = GuardConfig(
    max_drawdown=0.15,
    max_gross_notional=25_000.0,
    max_orders_per_day=12,
    max_data_staleness_days=1,
)


# --------------------------------------------------------------------------------------
# persistent halt flag
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HaltRecord:
    halted_at: str
    reason: str
    detail: str
    context: str = ""

    def __str__(self) -> str:
        return f"HALTED at {self.halted_at} during {self.context or 'unknown'}: {self.reason} - {self.detail}"


class HaltState:
    """A halt flag that survives process death.

    Stored as a file rather than in memory because the failure mode it exists to
    prevent is precisely the one where the process dies unexpectedly: an in-memory
    flag would be cleared by the very crash it is supposed to record.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def is_halted(self) -> bool:
        return self.path.is_file()

    def record(self) -> HaltRecord | None:
        """The stored halt, or None if the flag is not set.

        An unreadable flag file still counts as halted - the safe direction - but it
        must not raise, because the only way to clear a halt is through a CLI command
        that has to read it first. A flag that blocks trading *and* crashes the tool
        that clears it can only be recovered by deleting a file by hand, which is
        exactly the reflexive manual intervention this class exists to avoid.
        """
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
        """Set the flag. Idempotent: an existing halt is preserved, not overwritten.

        Preserving the first halt matters - the original cause is the interesting
        one, and a cascade of follow-on failures must not bury it.
        """
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
        # Written atomically so a crash mid-write cannot leave an unreadable flag
        # that neither halts nor clears.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(record), handle, indent=2)
        os.replace(tmp, self.path)
        return record

    def clear(self, operator_note: str) -> None:
        """Clear the flag. Reachable only from ``run_live.py --reset-halt``.

        Requires a non-empty note so that clearing is a deliberate act with a
        recorded reason rather than a reflex.
        """
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
    """Any exception escaping this block sets the halt flag, then propagates.

    Catches ``BaseException`` rather than ``Exception`` so that a KeyboardInterrupt
    or a SystemExit part-way through submitting orders also halts: an interrupted
    rebalance leaves the book in an unknown state, which is exactly when the next
    run must not start.

    The one exemption is a non-sticky :class:`GuardViolation`, which is a guard
    refusing this run on purpose rather than something going wrong. See that class
    for why latching on those would make the flag worse than useless.

    The exception is always re-raised, exemption or not. This is not error handling;
    it is a recorder.
    """
    try:
        yield
    except GuardViolation as exc:
        if not exc.sticky:
            raise  # a designed refusal; the run stops but the bot is not latched
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


# --------------------------------------------------------------------------------------
# pre-trade checks
# --------------------------------------------------------------------------------------


def check_not_halted(halt: HaltState) -> None:
    record = halt.record()
    if record is not None:
        raise HaltError(
            f"{record}\n"
            f"No orders will be submitted. To resume, investigate the cause and then run:\n"
            f"  python scripts/run_live.py --reset-halt --note 'what you found and fixed'"
        )


def check_broker_is_paper(broker) -> None:
    """Refuse to operate against anything that is not a paper account."""
    if not getattr(broker, "is_paper", False):
        raise NotAPaperAccount(
            f"broker {getattr(broker, 'name', broker)!r} does not declare itself as paper. "
            "This repository has no live trading path."
        )


def check_no_open_orders(open_orders: list[Order]) -> None:
    """An unfilled order from a previous run blocks this one.

    Submitting on top of an open order is how a position gets built twice: the
    second run sees the pre-order position, computes the same trade again, and sends
    it.
    """
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
    # The tolerance is for representation error, not slack: 1 - 85/100 evaluates to
    # 0.15000000000000002, which would otherwise trip a limit of exactly 0.15.
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
    """Refuse if today's orders would exceed the daily cap.

    ``orders_already_today`` of None means the broker cannot report its own order
    history. That is not zero: a guard whose input is missing must fail closed, or a
    re-run against an adapter that happens not to implement the query would submit a
    second full set of orders on top of a book that is already correct.
    """
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
    """Refuse to trade on stale prices.

    Staleness is measured in **trading days**, never calendar days. Counting calendar
    days here would refuse roughly half of all rebalances: the strategy trades on the
    first trading day of the month, which is a Monday about half the time, and a
    Monday's newest close is Friday's - three calendar days old but one trading day
    old.

    When ``trading_calendar`` is supplied (the broker's own, authoritative) it is
    used directly. Otherwise the count falls back to business days, which treats a
    market holiday as a trading day and so over-states staleness by one for the first
    session after each holiday. That errs towards refusing to trade, which is the
    safe direction, but it is why the live path passes the broker's calendar.
    """
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


# --------------------------------------------------------------------------------------
# high-water mark
# --------------------------------------------------------------------------------------


@dataclass
class EquityHighWaterMark:
    """Peak equity, persisted between runs.

    This is local state, which the broker-is-the-source-of-truth invariant is
    suspicious of - but a high-water mark is a *history* of equity, and the broker
    does not expose one. It is never used to infer a position; only to evaluate the
    drawdown guard. On a first run it seeds from the broker's current equity, which
    means a fresh install cannot trip the drawdown guard until it has seen a peak.
    """

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
        # Written atomically, like the halt flag: a crash mid-write would otherwise
        # leave a truncated file that the next run parses, and a high-water mark that
        # fails to load is a drawdown guard that cannot evaluate.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(self._data, handle, indent=2)
        os.replace(tmp, self.path)
        return peak

    @property
    def peak(self) -> float:
        return float(self._data.get("peak_equity", 0.0))
