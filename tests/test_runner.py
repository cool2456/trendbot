"""The runner, against a mock broker. BUILD_PROMPT build step 7's acceptance gate:

    (a) a repeated run submits nothing
    (b) an exception sets the halt flag
    (c) a halted state blocks all subsequent runs
    (d) positions are read from the broker, never from local state
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
import pytest

from tests.mock_broker import MockBroker
from trendbot.brokers.base import BrokerError, OrderSide
from trendbot.data import PriceData
from trendbot.engine.backtest import rebalance_dates
from trendbot.engine.validation import synthetic_prices
from trendbot.guards import CONSERVATIVE, GuardConfig, GuardViolation, HaltError, HaltState
from trendbot.runner import order_id, plan_run, run_once

#: The operator's CONSERVATIVE profile caps a single run at $25,000 of notional, which
#: is smaller than a $100,000 account's own rebalance. Tests of the trading path
#: therefore use a profile sized for the scenario, so that they exercise the mechanism
#: rather than all failing on the same cap. The interaction itself is asserted
#: explicitly in test_conservative_notional_cap_blocks_a_hundred_thousand_dollar_account.
ROOMY = GuardConfig(
    max_drawdown=CONSERVATIVE.max_drawdown,
    max_gross_notional=1_000_000.0,
    max_orders_per_day=CONSERVATIVE.max_orders_per_day,
    max_data_staleness_days=CONSERVATIVE.max_data_staleness_days,
)


@pytest.fixture
def scenario(cfg):
    """A synthetic history ending the bar before a real first-trading-day-of-month."""
    prices = synthetic_prices(cfg.universe, seed=7, n_days=1500)
    rebals = rebalance_dates(prices.close.index)
    # Pick a rebalance date late enough that the 252-day lookback is satisfied and
    # whose previous bar is the previous calendar day, so the staleness guard sees a
    # one-day-old bar exactly as it would in production.
    session_ts = next(
        d
        for d in rebals
        if prices.close.index.get_loc(d) > 400
        and (d - prices.close.index[prices.close.index.get_loc(d) - 1]).days == 1
    )
    idx = prices.close.index.get_loc(session_ts)
    history = PriceData(
        open=prices.open.iloc[:idx],
        close=prices.close.iloc[:idx],
        source="synthetic",
        adjusted=True,
        fetched_at="",
    )
    session = session_ts.date()
    last_close = history.close.iloc[-1]
    broker = MockBroker(
        equity=100_000.0,
        prices={t: float(last_close[t]) for t in cfg.universe},
        bar_date=history.close.index[-1].date(),
        now=dt.datetime.combine(session, dt.time(14, 0), tzinfo=dt.timezone.utc),
    )
    return broker, history, session


@pytest.fixture
def full_month_scenario(cfg):
    """History running through a rebalance day, with the session on the day after."""
    prices = synthetic_prices(cfg.universe, seed=7, n_days=1500)
    rebals = rebalance_dates(prices.close.index)
    reb = next(d for d in rebals if prices.close.index.get_loc(d) > 400)
    idx = prices.close.index.get_loc(reb)
    session_ts = prices.close.index[idx + 1]
    history = PriceData(
        open=prices.open.iloc[: idx + 1],
        close=prices.close.iloc[: idx + 1],
        source="synthetic",
        adjusted=True,
        fetched_at="",
    )
    session = session_ts.date()
    last_close = history.close.iloc[-1]
    broker = MockBroker(
        equity=100_000.0,
        prices={t: float(last_close[t]) for t in cfg.universe},
        bar_date=history.close.index[-1].date(),
        now=dt.datetime.combine(session, dt.time(14, 0), tzinfo=dt.timezone.utc),
    )
    return broker, history, session


def _run(broker, history, session, tmp_path, cfg, dry_run=True, guard_cfg=ROOMY):
    return run_once(
        broker,
        cfg=cfg,
        guard_cfg=guard_cfg,
        state_dir=tmp_path,
        history=history,
        dry_run=dry_run,
        session=session,
    )


# ---- the gate ----------------------------------------------------------------------


def test_a_repeated_run_submits_nothing(scenario, tmp_path, cfg):
    broker, history, session = scenario
    first = _run(broker, history, session, tmp_path, cfg, dry_run=False)
    assert first.plan.is_rebalance_day
    assert len(first.submitted) > 0, "the scenario must actually trade, or this proves nothing"

    # The mock fills immediately, so the second run sees the new positions and has
    # nothing left to do. This is the primary defence: idempotency by construction.
    second = _run(broker, history, session, tmp_path, cfg, dry_run=False)
    assert second.submitted == ()
    assert second.plan.orders == ()


def test_a_repeated_run_is_blocked_by_open_orders_when_fills_are_pending(scenario, tmp_path, cfg, monkeypatch):
    broker, history, session = scenario
    # Make fills stay open, as a market order submitted outside market hours would.
    original = broker.submit

    def submit_but_leave_open(symbol, qty, side, client_order_id):
        order = original(symbol, qty, side, client_order_id)
        broker._orders[-1] = dataclasses.replace(order, status="new", filled_qty=0.0)
        broker._shares[symbol] -= qty if side is OrderSide.BUY else -qty
        return broker._orders[-1]

    monkeypatch.setattr(broker, "submit", submit_but_leave_open)
    _run(broker, history, session, tmp_path, cfg, dry_run=False)
    assert broker.get_open_orders()
    with pytest.raises(GuardViolation, match="still open"):
        _run(broker, history, session, tmp_path, cfg, dry_run=False)


def test_b_an_exception_sets_the_halt_flag(scenario, tmp_path, cfg):
    broker, history, session = scenario
    broker._fail_on_submit = True
    halt = HaltState(tmp_path / "halt.json")
    assert not halt.is_halted()
    with pytest.raises(BrokerError, match="simulated broker outage"):
        _run(broker, history, session, tmp_path, cfg, dry_run=False)
    assert halt.is_halted()
    record = halt.record()
    assert record.reason == "unhandled BrokerError"
    assert "run_once" in record.context


def test_c_a_halted_state_blocks_all_subsequent_runs(scenario, tmp_path, cfg):
    broker, history, session = scenario
    HaltState(tmp_path / "halt.json").halt("earlier failure", "detail")
    for dry in (True, False):
        with pytest.raises(HaltError, match="HALTED"):
            _run(broker, history, session, tmp_path, cfg, dry_run=dry)
    assert broker._orders == []

    # ... and only an explicit clear releases it.
    HaltState(tmp_path / "halt.json").clear("investigated")
    outcome = _run(broker, history, session, tmp_path, cfg, dry_run=True)
    assert outcome.plan.is_rebalance_day


def test_d_positions_come_from_the_broker_not_from_local_state(scenario, tmp_path, cfg):
    broker, history, session = scenario
    first = _run(broker, history, session, tmp_path, cfg, dry_run=False)
    queries_after_first = broker.position_queries
    assert queries_after_first >= 1

    # Someone flattens the account by hand, behind the bot's back. A bot that
    # remembered its own fills would see no work to do; one that asks the broker
    # rebuilds the book.
    broker._shares = {}
    second = _run(broker, history, session, tmp_path, cfg, dry_run=True)
    assert broker.position_queries > queries_after_first
    assert (second.plan.current_weights == 0).all()
    assert len(second.plan.orders) == len(first.submitted)
    assert all(o.current_shares == 0.0 for o in second.plan.orders)


# ---- deterministic order ids ---------------------------------------------------------


def test_order_id_is_deterministic_and_intent_specific():
    base = dict(prereg_sha="abc123", session=dt.date(2026, 9, 1), symbol="SPY", side="buy", qty=10)
    assert order_id(**base) == order_id(**base)
    assert order_id(**{**base, "qty": 11}) != order_id(**base)
    assert order_id(**{**base, "side": "sell"}) != order_id(**base)
    assert order_id(**{**base, "symbol": "TLT"}) != order_id(**base)
    assert order_id(**{**base, "session": dt.date(2026, 10, 1)}) != order_id(**base)
    # A different pre-registration is a different strategy and must not share ids.
    assert order_id(**{**base, "prereg_sha": "def456"}) != order_id(**base)
    assert len(order_id(**base)) <= 128


def test_order_ids_do_not_depend_on_wall_clock(scenario, tmp_path, cfg):
    broker, history, session = scenario
    plan_a = _run(broker, history, session, tmp_path, cfg, dry_run=True).plan
    broker._now = broker._now + dt.timedelta(hours=3)
    plan_b = _run(broker, history, session, tmp_path, cfg, dry_run=True).plan
    assert [o.client_order_id for o in plan_a.orders] == [o.client_order_id for o in plan_b.orders]


def test_the_broker_rejects_a_duplicate_client_order_id(scenario, tmp_path, cfg):
    broker, history, session = scenario
    plan = _run(broker, history, session, tmp_path, cfg, dry_run=True).plan
    first = plan.orders[0]
    broker.submit(first.symbol, first.qty, first.side, first.client_order_id)
    with pytest.raises(BrokerError, match="duplicate"):
        broker.submit(first.symbol, first.qty, first.side, first.client_order_id)


# ---- schedule and guards in the runner -----------------------------------------------


def test_nothing_trades_on_a_non_rebalance_day(full_month_scenario, tmp_path, cfg):
    """The day after a rebalance is not itself a rebalance day."""
    broker, history, session = full_month_scenario
    outcome = _run(broker, history, session, tmp_path, cfg, dry_run=False)
    assert not outcome.plan.is_rebalance_day
    assert outcome.submitted == ()
    assert "first trading day" in outcome.plan.diagnostics["reason"]


def test_a_stale_price_history_refuses_rather_than_rebalancing_daily(scenario, tmp_path, cfg):
    """A history that stops short must not make every day look like the 1st.

    The month containing the session has no earlier bar in a truncated history, so a
    naive first-of-month test would fire a rebalance every single day of that month.
    """
    broker, history, session = scenario
    later = session + dt.timedelta(days=8)
    broker._bar_date = later  # quotes are fresh; the history is not
    with pytest.raises(GuardViolation, match="stale"):
        _run(broker, history, later, tmp_path, cfg, dry_run=True)


def test_dry_run_submits_nothing_but_still_plans(scenario, tmp_path, cfg):
    broker, history, session = scenario
    outcome = _run(broker, history, session, tmp_path, cfg, dry_run=True)
    assert outcome.dry_run
    assert outcome.submitted == ()
    assert len(outcome.plan.orders) > 0
    assert broker._orders == []


def test_stale_data_refuses_to_trade(scenario, tmp_path, cfg):
    broker, history, session = scenario
    broker._bar_date = session - dt.timedelta(days=9)
    with pytest.raises(GuardViolation, match="stale"):
        _run(broker, history, session, tmp_path, cfg, dry_run=True)


def test_a_non_paper_broker_is_refused(scenario, tmp_path, cfg):
    broker, history, session = scenario
    broker.is_paper = False
    with pytest.raises(GuardViolation, match="paper"):
        _run(broker, history, session, tmp_path, cfg, dry_run=True)


def test_drawdown_halts_the_run(scenario, tmp_path, cfg):
    broker, history, session = scenario
    _run(broker, history, session, tmp_path, cfg, dry_run=True)  # sets the high-water mark
    broker._equity = 100_000.0 * 0.80  # a 20% drawdown, past the 15% limit
    with pytest.raises(GuardViolation, match="drawdown"):
        _run(broker, history, session, tmp_path, cfg, dry_run=True)


def test_the_plan_never_exceeds_the_gross_cap_or_the_instrument_cap(scenario, tmp_path, cfg):
    broker, history, session = scenario
    plan = _run(broker, history, session, tmp_path, cfg, dry_run=True).plan
    assert plan.target_weights.abs().max() <= cfg.per_instrument_cap + 1e-9
    assert plan.target_weights.abs().sum() <= cfg.gross_exposure_cap + 1e-9
    assert plan.banded_weights.abs().sum() <= cfg.gross_exposure_cap + 1e-9
    # whole-share rounding is toward zero, so the realised book cannot exceed it either
    realised = (plan.target_shares * plan.reference_prices).abs().sum() / plan.equity
    assert realised <= cfg.gross_exposure_cap + 1e-9


def test_the_runner_refuses_a_history_that_includes_the_session_being_traded(scenario, tmp_path, cfg):
    broker, history, session = scenario
    # Append a bar dated on the session itself: sizing on a bar that has not closed
    # would be lookahead, and the runner must refuse rather than quietly use it.
    stamp = pd.Timestamp(session)
    extended = PriceData(
        open=pd.concat([history.open, history.open.iloc[[-1]].rename(index={history.open.index[-1]: stamp})]),
        close=pd.concat([history.close, history.close.iloc[[-1]].rename(index={history.close.index[-1]: stamp})]),
        source="synthetic",
        adjusted=True,
        fetched_at="",
    )
    with pytest.raises(GuardViolation, match="not strictly before"):
        _run(broker, extended, session, tmp_path, cfg, dry_run=True)


# ---- the notional cap interacts badly with the account size --------------------------


def test_conservative_notional_cap_blocks_a_hundred_thousand_dollar_account(scenario, tmp_path, cfg):
    """The operator's chosen $25,000 per-run notional limit is smaller than the trade.

    Building a ~100% gross book on a $100,000 account is ~$100,000 of notional, and
    even a routine monthly rebalance is a median ~$23,000 and a 90th percentile
    ~$56,000. The guard is doing exactly what it was told to do; what it was told is
    incompatible with the account it is guarding. Asserted here so the conflict is a
    documented property rather than a surprise in production.
    """
    broker, history, session = scenario
    assert broker.get_account().equity == 100_000.0
    with pytest.raises(GuardViolation, match="notional"):
        _run(broker, history, session, tmp_path, cfg, dry_run=True, guard_cfg=CONSERVATIVE)

    # The same limit is comfortable on the account size it was evidently sized for.
    # A fresh state directory, because the high-water mark from the $100k run above
    # would otherwise register a 75% drawdown.
    broker._equity = 25_000.0
    outcome = _run(broker, history, session, tmp_path / "smaller", cfg, dry_run=True, guard_cfg=CONSERVATIVE)
    assert outcome.plan.total_notional <= CONSERVATIVE.max_gross_notional


def test_order_count_stays_within_the_daily_cap(scenario, tmp_path, cfg):
    broker, history, session = scenario
    plan = _run(broker, history, session, tmp_path, cfg, dry_run=True).plan
    # There are only 12 instruments, so a single rebalance can never exceed 12 orders.
    assert len(plan.orders) <= cfg.n_universe == CONSERVATIVE.max_orders_per_day


def test_a_monday_rebalance_after_a_friday_close_is_not_refused(cfg, tmp_path):
    """The staleness guard must not block the ordinary monthly rebalance.

    Regression for a bug that refused 48% of all monthly rebalances by counting
    Friday-to-Monday as three days stale.
    """
    prices = synthetic_prices(cfg.universe, seed=11, n_days=1500)
    rebals = rebalance_dates(prices.close.index)
    # a rebalance whose previous bar is three CALENDAR days earlier, i.e. a Monday
    monday = next(
        d
        for d in rebals
        if prices.close.index.get_loc(d) > 400
        and (d - prices.close.index[prices.close.index.get_loc(d) - 1]).days == 3
    )
    idx = prices.close.index.get_loc(monday)
    history = PriceData(
        open=prices.open.iloc[:idx], close=prices.close.iloc[:idx],
        source="synthetic", adjusted=True, fetched_at="",
    )
    session = monday.date()
    last = history.close.iloc[-1]
    broker = MockBroker(
        equity=100_000.0,
        prices={t: float(last[t]) for t in cfg.universe},
        bar_date=history.close.index[-1].date(),
        now=dt.datetime.combine(session, dt.time(14, 0), tzinfo=dt.timezone.utc),
    )
    outcome = _run(broker, history, session, tmp_path, cfg, dry_run=True)
    assert outcome.plan.is_rebalance_day
    assert len(outcome.plan.orders) > 0


def test_a_broker_failure_midway_still_records_the_orders_already_sent(scenario, tmp_path, cfg):
    """Predictions are durable per order, not per batch.

    If the broker dies on order five of ten, the first four are real fills sitting at
    the broker. Batching the log writes until the end would leave them unrecorded by
    the one mechanism whose job is to notice exactly that.
    """
    from trendbot.monitor.divergence import DivergenceLog

    broker, history, session = scenario
    original = broker.submit
    calls = {"n": 0}

    def flaky(symbol, qty, side, client_order_id):
        calls["n"] += 1
        if calls["n"] > 2:
            raise BrokerError("simulated broker outage")
        return original(symbol, qty, side, client_order_id)

    broker.submit = flaky
    with pytest.raises(BrokerError):
        _run(broker, history, session, tmp_path, cfg, dry_run=False)

    log = DivergenceLog(tmp_path / "divergence.jsonl")
    predictions = log.entries("prediction")
    assert len(predictions) == 3, "every attempted order must be on record, filled or not"
    assert len([o for o in broker._orders]) == 2


def test_a_broker_that_cannot_report_its_order_history_refuses_to_trade(scenario, tmp_path, cfg):
    """The daily order cap must fail closed when its input is unavailable.

    ``Broker.get_orders_since`` returns None on the base class, so an adapter that
    does not implement it would otherwise let the cap read as zero and permit a
    duplicate book on a re-run.
    """
    broker, history, session = scenario
    broker.get_orders_since = lambda after: None
    with pytest.raises(GuardViolation, match="cannot report"):
        _run(broker, history, session, tmp_path, cfg, dry_run=False)
