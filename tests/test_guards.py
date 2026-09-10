from __future__ import annotations

import datetime as dt
import json

import pytest

from trendbot.brokers.base import Order, OrderSide
from trendbot.guards import (
    CONSERVATIVE,
    EquityHighWaterMark,
    GuardConfig,
    GuardViolation,
    HaltError,
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


@pytest.fixture
def halt(tmp_path):
    return HaltState(tmp_path / "halt.json")


def test_guard_config_requires_every_threshold():
    with pytest.raises(TypeError):
        GuardConfig()
    with pytest.raises(TypeError):
        GuardConfig(max_drawdown=0.15)
    with pytest.raises(TypeError):
        GuardConfig(max_drawdown=0.15, max_gross_notional=1.0, max_orders_per_day=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_drawdown": 0.0},
        {"max_drawdown": 1.0},
        {"max_drawdown": -0.1},
        {"max_gross_notional": 0.0},
        {"max_orders_per_day": 0},
        {"max_data_staleness_days": 0},
    ],
)


def test_guard_config_rejects_nonsense(kwargs):
    base = dict(
        max_drawdown=0.15, max_gross_notional=25_000.0, max_orders_per_day=12, max_data_staleness_days=1
    )
    with pytest.raises(ValueError):
        GuardConfig(**{**base, **kwargs})


def test_conservative_profile_matches_the_operator_decision():
    assert CONSERVATIVE.max_drawdown == 0.15
    assert CONSERVATIVE.max_gross_notional == 25_000.0
    assert CONSERVATIVE.max_orders_per_day == 12
    assert CONSERVATIVE.max_data_staleness_days == 1


def test_guard_config_is_frozen():
    with pytest.raises(Exception):
        CONSERVATIVE.max_drawdown = 0.9


def test_halt_flag_persists_across_objects(tmp_path):
    path = tmp_path / "halt.json"
    HaltState(path).halt("because", "detail", "ctx")
    assert HaltState(path).is_halted()
    assert HaltState(path).record().reason == "because"


def test_halt_is_idempotent_and_preserves_the_first_cause(halt):
    halt.halt("first", "original cause")
    halt.halt("second", "follow-on noise")
    record = halt.record()
    assert record.reason == "first"
    assert record.detail == "original cause"


def test_clear_requires_a_note_and_archives_the_halt(halt):
    halt.halt("boom", "detail")
    with pytest.raises(ValueError):
        halt.clear("")
    with pytest.raises(ValueError):
        halt.clear("   ")
    halt.clear("investigated; the feed was down")
    assert not halt.is_halted()
    archive = halt.path.with_suffix(".cleared.jsonl")
    entry = json.loads(archive.read_text().splitlines()[-1])
    assert entry["operator_note"] == "investigated; the feed was down"
    assert entry["halt"]["reason"] == "boom"


def test_clearing_an_unset_flag_raises(halt):
    with pytest.raises(HaltError):
        halt.clear("nothing to do")


def test_fail_closed_sets_the_flag_and_reraises(halt):
    with pytest.raises(ValueError, match="boom"):
        with fail_closed(halt, "unit test"):
            raise ValueError("boom")
    record = halt.record()
    assert record is not None
    assert record.reason == "unhandled ValueError"
    assert record.context == "unit test"


def test_fail_closed_also_catches_base_exceptions(halt):
    with pytest.raises(KeyboardInterrupt):
        with fail_closed(halt, "interrupted"):
            raise KeyboardInterrupt
    assert halt.is_halted()


def test_fail_closed_is_transparent_on_success(halt):
    with fail_closed(halt, "fine"):
        pass
    assert not halt.is_halted()


def test_check_not_halted_blocks_and_explains_how_to_recover(halt):
    halt.halt("bad thing", "details")
    with pytest.raises(HaltError, match="run_live.py --reset-halt"):
        check_not_halted(halt)


def test_check_broker_is_paper():
    class Live:
        is_paper = False
        name = "live"

    class Paper:
        is_paper = True
        name = "paper"

    check_broker_is_paper(Paper())
    with pytest.raises(GuardViolation, match="does not declare itself as paper"):
        check_broker_is_paper(Live())
    with pytest.raises(GuardViolation):
        check_broker_is_paper(object())


def _order(status="new"):
    return Order(symbol="SPY", qty=1, side=OrderSide.BUY, client_order_id="x", status=status)


def test_open_orders_block_new_ones():
    check_no_open_orders([])
    check_no_open_orders([_order("filled")] and [])
    with pytest.raises(GuardViolation, match="still open"):
        check_no_open_orders([_order("new")])


def test_drawdown_guard():
    check_drawdown(90.0, 100.0, CONSERVATIVE)
    with pytest.raises(GuardViolation, match="drawdown"):
        check_drawdown(84.0, 100.0, CONSERVATIVE)
    with pytest.raises(GuardViolation):
        check_drawdown(100.0, 0.0, CONSERVATIVE)


def test_drawdown_boundary_is_strict():
    check_drawdown(85.0, 100.0, CONSERVATIVE)


def test_notional_guard():
    check_notional(25_000.0, CONSERVATIVE)
    with pytest.raises(GuardViolation, match="notional"):
        check_notional(25_000.01, CONSERVATIVE)


def test_order_cap_counts_orders_already_sent_today():
    check_order_cap(0, 12, CONSERVATIVE)
    check_order_cap(6, 6, CONSERVATIVE)
    with pytest.raises(GuardViolation, match="daily cap"):
        check_order_cap(6, 7, CONSERVATIVE)


def test_a_friday_close_is_one_trading_day_old_on_monday():
    friday, monday = dt.date(2026, 8, 28), dt.date(2026, 8, 31)
    check_data_freshness({"SPY": friday}, monday, CONSERVATIVE)
    sessions = {dt.date(2026, 8, 27), friday, monday}
    check_data_freshness({"SPY": friday}, monday, CONSERVATIVE, trading_calendar=sessions)


def test_a_holiday_is_not_a_trading_day_when_a_calendar_is_supplied():
    sessions = {dt.date(2025, 12, 31), dt.date(2026, 1, 2), dt.date(2026, 1, 5)}
    check_data_freshness({"SPY": dt.date(2025, 12, 31)}, dt.date(2026, 1, 2), CONSERVATIVE, trading_calendar=sessions)
    with pytest.raises(GuardViolation, match="stale"):
        check_data_freshness({"SPY": dt.date(2025, 12, 31)}, dt.date(2026, 1, 2), CONSERVATIVE)


def test_a_genuinely_stale_feed_is_still_refused():
    for gap_days, in ((9,), (30,), (400,)):
        bar = dt.date(2026, 8, 31) - dt.timedelta(days=gap_days)
        with pytest.raises(GuardViolation, match="stale"):
            check_data_freshness({"SPY": bar}, dt.date(2026, 8, 31), CONSERVATIVE)


def test_data_freshness_calendar_days():
    today = dt.date(2026, 9, 1)
    check_data_freshness({"SPY": dt.date(2026, 8, 31)}, today, CONSERVATIVE)
    with pytest.raises(GuardViolation, match="stale"):
        check_data_freshness({"SPY": dt.date(2026, 8, 28)}, today, CONSERVATIVE)


def test_data_freshness_uses_a_trading_calendar_when_given():
    today = dt.date(2026, 9, 1)
    sessions = {dt.date(2026, 8, 28), dt.date(2026, 9, 1)}
    check_data_freshness(
        {"SPY": dt.date(2026, 8, 28)}, today, CONSERVATIVE, trading_calendar=sessions
    )


def test_data_freshness_rejects_a_future_bar_and_an_empty_map():
    today = dt.date(2026, 9, 1)
    with pytest.raises(GuardViolation, match="future"):
        check_data_freshness({"SPY": dt.date(2026, 9, 2)}, today, CONSERVATIVE)
    with pytest.raises(GuardViolation, match="no price timestamps"):
        check_data_freshness({}, today, CONSERVATIVE)


def test_high_water_mark_only_ratchets_up(tmp_path):
    hwm = EquityHighWaterMark(tmp_path / "hw.json")
    assert hwm.update(100.0) == 100.0
    assert hwm.update(120.0) == 120.0
    assert hwm.update(90.0) == 120.0
    assert EquityHighWaterMark(tmp_path / "hw.json").peak == 120.0


def test_a_corrupt_halt_flag_still_blocks_but_stays_clearable(tmp_path):
    path = tmp_path / "halt.json"
    path.write_text('{"halted_at": "2026-01-0')
    halt = HaltState(path)

    assert halt.is_halted()
    record = halt.record()
    assert record is not None
    assert record.reason == "unreadable halt flag"
    with pytest.raises(HaltError, match="unreadable halt flag"):
        check_not_halted(halt)

    halt.clear("investigated: truncated by a crash")
    assert not halt.is_halted()


def test_a_halt_flag_containing_unexpected_keys_does_not_crash(tmp_path):
    path = tmp_path / "halt.json"
    path.write_text('{"totally": "unexpected"}')
    assert HaltState(path).record().reason == "unreadable halt flag"


def test_an_unknown_order_count_refuses_rather_than_assuming_zero():
    with pytest.raises(GuardViolation, match="cannot report"):
        check_order_cap(None, 1, CONSERVATIVE)


def test_a_non_paper_broker_latches_the_halt_flag(tmp_path):
    from trendbot.guards import NotAPaperAccount

    class Live:
        is_paper = False
        name = "live"

    halt = HaltState(tmp_path / "halt.json")
    with pytest.raises(NotAPaperAccount):
        with fail_closed(halt, "paper check"):
            check_broker_is_paper(Live())
    assert halt.is_halted(), "a non-paper broker must not be retried silently for ever"
    assert issubclass(NotAPaperAccount, GuardViolation)
    assert NotAPaperAccount.sticky is True
