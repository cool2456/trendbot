#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trendbot.brokers.alpaca import AlpacaPaperBroker  # noqa: E402
from trendbot.config import load_config  # noqa: E402
from trendbot.guards import CONSERVATIVE, HaltState, HaltError  # noqa: E402
from trendbot.monitor.divergence import DivergenceLog, summarise_divergence  # noqa: E402
from trendbot.runner import STATE_DIR, run_once  # noqa: E402


def cmd_reset_halt(args) -> int:
    halt = HaltState(Path(args.state_dir) / "halt.json")
    record = halt.record()
    if record is None:
        print("The halt flag is not set. Nothing to clear.")
        return 1
    print(f"Clearing:\n  {record}")
    halt.clear(args.note)
    print(f"Cleared, with note: {args.note!r}")
    print("The previous halt has been appended to halt.cleared.jsonl.")
    return 0


def cmd_status(args) -> int:
    state = Path(args.state_dir)
    halt = HaltState(state / "halt.json")
    log = DivergenceLog(state / "divergence.jsonl")
    record = halt.record()
    print(f"halt flag : {'SET - ' + str(record) if record else 'clear'}")

    broker = AlpacaPaperBroker()
    account = broker.get_account()
    print(f"broker    : {broker.name} (paper={broker.is_paper})")
    print(f"account   : {account.status} equity {account.equity:,.2f} cash {account.cash:,.2f}")
    positions = broker.get_positions()
    if positions:
        for symbol, p in sorted(positions.items()):
            print(f"  {symbol:<5s} {p.qty:>8g} sh  {p.market_value:>12,.2f}  @ {p.current_price:,.2f}")
    else:
        print("  (no positions)")
    open_orders = broker.get_open_orders()
    print(f"open orders: {len(open_orders)}")
    for o in open_orders:
        print(f"  {o.side.value} {o.qty:g} {o.symbol} [{o.status}] {o.client_order_id}")
    print(f"divergence : {summarise_divergence(log)}")
    print(f"unreconciled predictions: {len(log.unreconciled())}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan and check everything, submit nothing")
    mode.add_argument("--submit", action="store_true", help="actually send orders to the paper account")
    mode.add_argument("--reset-halt", action="store_true", help="clear the persistent halt flag")
    mode.add_argument("--status", action="store_true", help="report broker and halt state, then exit")
    parser.add_argument("--note", default="", help="required with --reset-halt: why you are clearing it")
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    args = parser.parse_args(argv)

    if args.reset_halt:
        if not args.note.strip():
            parser.error("--reset-halt requires --note explaining what you found and fixed")
        return cmd_reset_halt(args)
    if args.status:
        return cmd_status(args)
    if not (args.dry_run or args.submit):
        parser.error("choose one of --dry-run, --submit, --status or --reset-halt")

    cfg = load_config()
    broker = AlpacaPaperBroker()
    print(cfg.describe())
    print(f"broker: {broker.name}  paper={broker.is_paper}")
    print(f"guards: {CONSERVATIVE}")
    print(
        "  (guard thresholds are the operator's, not the pre-registration's - that document\n"
        "   contains no operational limits)"
    )

    try:
        outcome = run_once(
            broker,
            cfg=cfg,
            guard_cfg=CONSERVATIVE,
            state_dir=Path(args.state_dir),
            dry_run=not args.submit,
        )
    except HaltError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    print(f"\n{outcome.plan.describe()}")
    if outcome.plan.diagnostics:
        print("\ndiagnostics:")
        for key, value in outcome.plan.diagnostics.items():
            print(f"  {key:24s} {value}")
    print(f"\nreconciled {outcome.reconciled} previous prediction(s)")
    if outcome.dry_run:
        print(f"DRY RUN - nothing submitted. {len(outcome.plan.orders)} order(s) would have been sent.")
    else:
        print(f"submitted {len(outcome.submitted)} order(s):")
        for order in outcome.submitted:
            print(f"  {order.side.value} {order.qty:g} {order.symbol} -> {order.status} [{order.client_order_id}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
