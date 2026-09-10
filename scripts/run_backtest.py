#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trendbot.config import load_config  # noqa: E402
from trendbot.data import daily_risk_free, load_prices, load_risk_free_rate  # noqa: E402
from trendbot.engine.backtest import (  # noqa: E402
    buy_and_hold,
    full_universe_start,
    run_backtest,
)
from trendbot.engine.metrics import sharpe, summarise  # noqa: E402
from trendbot.engine.panel_backtest import (  # noqa: E402
    panel_buy_and_hold,
    run_panel_backtest,
)
from trendbot.regression import (  # noqa: E402
    EXPERIMENT_001_BENCHMARK_SHARPE,
    EXPERIMENT_001_GATE_DECIMALS,
    EXPERIMENT_001_NET_SHARPE,
    EXPERIMENT_001_WINDOW,
)
from trendbot.strategies import TimeSeriesTrend  # noqa: E402
from trendbot.engine.validation import (  # noqa: E402
    deflated_sharpe_ratio,
    evaluate_decision_rule,
    in_sample_out_of_sample,
    noise_sweep_study,
    noise_test,
    standalone_instrument_sharpes,
    synthetic_prices,
    walk_forward,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def cmd_regression(cfg, args) -> int:
    rule("EXPERIMENT 001 REGRESSION GATE - THE GENERALISED ENGINE MUST REPRODUCE 001")
    prices = load_prices(cfg.universe, source=args.source)
    rf = load_risk_free_rate()
    rf_daily = daily_risk_free(rf, prices.close.index)
    start = full_universe_start(prices, cfg)
    end = prices.close.index[-1]

    print(cfg.describe())
    print(f"window: {start.date()} -> {end.date()}")
    print(f"expected window: {EXPERIMENT_001_WINDOW[0]} -> {EXPERIMENT_001_WINDOW[1]}")

    original = run_backtest(prices, cfg, risk_free=rf)
    generalised = run_panel_backtest(
        prices,
        cfg.universe,
        TimeSeriesTrend(cfg),
        cost_bps=cfg.cost_bps_per_side,
        drift_band=cfg.drift_band,
        gross_cap=cfg.gross_exposure_cap,
        per_instrument_cap=cfg.per_instrument_cap,
        risk_free=rf,
    )
    original_sharpe = original.stats_from(start).sharpe
    panel_sharpe = generalised.stats_from(start).sharpe
    bh_original = sharpe((buy_and_hold(prices, cfg, start=start) - rf_daily).loc[start:])
    bh_panel = sharpe((panel_buy_and_hold(prices, cfg.universe, start=start) - rf_daily).loc[start:])

    d = EXPERIMENT_001_GATE_DECIMALS
    checks = [
        (
            "window matches the one experiment 001 reported",
            (str(start.date()), str(end.date())) == EXPERIMENT_001_WINDOW,
            f"{start.date()} -> {end.date()}",
        ),
        (
            f"generalised engine net Sharpe == {round(EXPERIMENT_001_NET_SHARPE, d)} to {d} dp",
            round(panel_sharpe, d) == round(EXPERIMENT_001_NET_SHARPE, d),
            f"{panel_sharpe:.10f}",
        ),
        (
            f"benchmark Sharpe == {round(EXPERIMENT_001_BENCHMARK_SHARPE, d)} to {d} dp",
            round(bh_panel, d) == round(EXPERIMENT_001_BENCHMARK_SHARPE, d),
            f"{bh_panel:.10f}",
        ),
        (
            "original engine still produces its own recorded number",
            round(original_sharpe, d) == round(EXPERIMENT_001_NET_SHARPE, d),
            f"{original_sharpe:.10f}",
        ),
        (
            "benchmark is literally the same function in both paths",
            bh_original == bh_panel,
            f"{bh_original:.12f} vs {bh_panel:.12f}",
        ),
    ]

    delta = float((generalised.equity / original.equity - 1.0).abs().max())
    checks.append(
        (
            "equity curves agree bit for bit (max relative delta == 0)",
            delta == 0.0,
            f"{delta:.3e}",
        )
    )
    for field in ("weights", "targets", "trades"):
        a = getattr(original, field).to_numpy()
        b = getattr(generalised, field).to_numpy()
        checks.append(
            (
                f"{field} identical",
                bool(np.array_equal(a, b, equal_nan=True)),
                f"max |delta| {np.nanmax(np.abs(a - b)):.3e}",
            )
        )

    print()
    for name, ok, observed in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {observed}")

    passed = all(ok for _, ok, _ in checks)
    print(
        f"\nGATE: {'PASS' if passed else 'FAIL'}. "
        + (
            "The generalised engine reproduces experiment 001 exactly."
            if passed
            else "The refactor changed experiment 001's result. Find it before writing any "
            "new signal; do NOT edit trendbot/regression.py."
        )
    )
    return 0 if passed else 1


def cmd_synthetic(cfg, args) -> int:
    rule(f"SECTION 7 STEP 1 - NOISE TEST (seed {args.seed})")
    print(
        "The strategy is run on independent driftless random walks. A materially\n"
        "positive Sharpe here means the engine has a bug, not that the strategy works.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or 8)))
    result = noise_test(cfg, seeds=seeds, n_days=args.n_days)
    table = pd.DataFrame(
        {"strategy": result.strategy_sharpes, "buy_and_hold": result.buy_and_hold_sharpes},
        index=pd.Index(seeds, name="seed"),
    )
    print(table.round(4).to_string())
    print(f"\n{result}")
    passed = result.passes(tolerance=args.tolerance)
    print(
        f"\nGATE: mean |Sharpe| <= {args.tolerance} for both the strategy and buy-and-hold "
        f"-> {'PASS' if passed else 'FAIL'}"
    )

    drifting = noise_test(cfg, seeds=seeds[:3], n_days=args.n_days, annual_drift=0.08)
    print(
        f"\nCounterpart check on +8%/yr drifting random walks: strategy Sharpe "
        f"{drifting.mean_sharpe:+.3f} (should be clearly positive - a long-only rule on "
        f"trending data must make money, or the engine is dead rather than merely honest)"
    )
    return 0 if passed else 1


def cmd_validate(cfg, args) -> int:
    prices = load_prices(cfg.universe, source=args.source)
    rf = load_risk_free_rate()
    start = prices.close.index[0] if args.full_history else full_universe_start(prices, cfg)
    result = run_backtest(prices, cfg, risk_free=rf)
    returns = result.excess_returns.loc[start:]

    rule("SECTION 7 STEP 1 - NOISE TEST")
    noise = noise_test(cfg, seeds=(0, 1, 2, 3), n_days=4000)
    print(noise)
    print(f"  -> {'PASS' if noise.passes() else 'FAIL'}: the engine earns nothing on random walks")

    rule("SECTION 7 STEP 3 - IN-SAMPLE / OUT-OF-SAMPLE")
    print("Nothing is fitted, so both halves are out-of-sample by construction.")
    split = in_sample_out_of_sample(returns)
    print(f"  {split}")
    print(
        "  A large asymmetry would indicate a data problem rather than decay."
        if abs(split.asymmetry) < 0.5
        else "  NOTE: the halves differ materially. Investigate the data before the strategy."
    )

    rule("SECTION 7 STEP 4 - WALK-FORWARD")
    print(
        f"Rolling windows with nothing re-optimised. Window lengths are NOT pre-registered\n"
        f"(section 6's frozen list does not contain them) and cannot change a position;\n"
        f"they are reporting choices, stated here: {args.train_years:g}y in-window, "
        f"{args.test_years:g}y forward, stepping {args.step_years:g}y.\n"
    )
    wf = walk_forward(
        returns,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
    )
    print(wf)
    print()
    print(
        wf.windows.assign(
            train_start=lambda d: d.train_start.dt.date, test_end=lambda d: d.test_end.dt.date
        )[["train_start", "test_end", "train_sharpe", "test_sharpe", "test_return"]]
        .round(3)
        .to_string(index=False)
    )

    rule("SECTION 7 STEP 5 - DEFLATED SHARPE")
    dsr = deflated_sharpe_ratio(returns, cfg.configurations_tried)
    print(f"configurations tried, per PREREGISTRATION.md section 7: {cfg.configurations_tried}")
    print(f"  {dsr}")
    print(f"  {dsr.note}")

    rule("WHAT THIS MODULE IS FOR - MINING A FAKE EDGE OUT OF PURE NOISE")
    print(
        "One hundred configurations swept across lookback, vol-estimate halflife and\n"
        "long-only/long-short, on synthetic random walks containing no signal whatsoever,\n"
        "repeated on six independent datasets.\n"
    )
    study = noise_sweep_study(cfg)
    print(study.table.round(4).to_string(index=False))
    print(f"\n{study}")
    gate = study.max_best_sharpe > 1.0 and not study.any_significant
    print(
        f"\nGATE: mining manufactures an in-sample Sharpe above 1.0 while the deflated\n"
        f"Sharpe refuses to call it significant -> {'PASS' if gate else 'FAIL'}"
    )
    fooled = study.table[(study.table.psr_vs_zero > 0.95) & (~study.table.significant)]
    if len(fooled):
        print(
            f"\nOn {len(fooled)} of the six datasets an uncorrected probabilistic Sharpe would\n"
            f"have declared the mined result significant (PSR up to "
            f"{fooled.psr_vs_zero.max():.4f}) while the deflated Sharpe did not. That gap is\n"
            f"the entire value of this correction."
        )
    return 0 if gate and noise.passes() else 1


def cmd_backtest(cfg, args) -> int:
    prices = load_prices(cfg.universe, source=args.source)
    rf = load_risk_free_rate()
    rf_daily = daily_risk_free(rf, prices.close.index)
    start = prices.close.index[0] if args.full_history else full_universe_start(prices, cfg)
    label = "FULL AVAILABLE HISTORY" if args.full_history else "ALL-TWELVE WINDOW"

    rule(f"SECTION 7 STEP 2 - FULL-SAMPLE BACKTEST, ONE RUN  [{label}]")
    print(cfg.describe())
    print(f"data: {prices.describe()}")
    print(f"window: {start.date()} -> {prices.close.index[-1].date()}")
    print(
        "Sharpe is computed on returns in EXCESS of the 13-week T-bill, and uninvested\n"
        "cash accrues at that same rate. Both the strategy and the benchmark are treated\n"
        "identically. The strategy is only ~70-94% invested, so crediting the cash without\n"
        "charging its opportunity cost would flatter it against a fully invested benchmark."
    )

    result = run_backtest(prices, cfg, risk_free=rf)
    returns = result.excess_returns.loc[start:]
    stats = result.stats_from(start)

    bh_excess = (buy_and_hold(prices, cfg, start=start) - rf_daily).loc[start:]
    bh_stats = summarise(bh_excess)

    print(f"\nstrategy (net of {cfg.cost_bps_per_side:g} bps/side, excess of T-bill):")
    print(f"  {stats}")
    print(f"equal-weight buy-and-hold (excess of T-bill):")
    print(f"  {bh_stats}")
    print(f"\ngross of costs: Sharpe {sharpe(result.gross_returns.loc[start:] - rf_daily.loc[start:]):+.3f}")

    rule("SECTION 5 - COST SENSITIVITY")
    print("The 5 bps figure is the headline; the rest is the sensitivity the document asks for.")
    ladder = []
    for bps in cfg.cost_sensitivity_bps:
        r = run_backtest(prices, cfg, risk_free=rf, cost_bps=bps)
        ex = r.excess_returns.loc[start:]
        ladder.append(
            {
                "cost bps/side": bps,
                "Sharpe": sharpe(ex),
                "CAGR": summarise(ex).cagr,
                "vs B&H": sharpe(ex) - sharpe(bh_excess),
            }
        )
    ladder_df = pd.DataFrame(ladder).set_index("cost bps/side")
    print(ladder_df.round(4).to_string())

    rule("SECTION 8's BENCHMARK - CONSTRUCTION SENSITIVITY")
    print(
        'Section 8 says "equal-weight buy-and-hold" and does not say how often, if ever,\n'
        "it is rebalanced. Buy-and-hold read literally means never, which is the headline.\n"
        "The alternatives are shown because this single unstated detail moves the margin\n"
        "by more than the 0.15 the decision rule turns on:\n"
    )
    bench_rows = []
    for mode, label in (
        ("none", "buy once, hold (literal)"),
        ("monthly", "rebalanced monthly"),
        ("daily", "rebalanced daily"),
    ):
        series = (buy_and_hold(prices, cfg, rebalance=mode, start=start) - rf_daily).loc[start:]
        bench_rows.append(
            {
                "construction": label,
                "Sharpe": sharpe(series),
                "CAGR": summarise(series).cagr,
                "margin": stats.sharpe - sharpe(series),
                "beats it?": "no" if stats.sharpe <= sharpe(series) else "yes",
            }
        )
    print(pd.DataFrame(bench_rows).set_index("construction").round(4).to_string())

    rule("SECTION 8 - PRE-COMMITTED DECISION RULE")
    standalone = standalone_instrument_sharpes(prices.slice(str(start.date())), cfg, risk_free=rf)
    contribution = result.instrument_pnl.loc[start:].sum()
    n_positive_standalone = int((standalone > 0).sum())
    n_positive_contribution = int((contribution > 0).sum())

    print("Per-instrument breadth, both readings of \"the sign is positive in 9 of 12\":\n")
    breadth = pd.DataFrame(
        {
            "standalone Sharpe": standalone,
            "P&L contribution": contribution,
        }
    )
    print(breadth.round(4).to_string())
    print(
        f"\n  standalone trend rule positive in {n_positive_standalone}/12  <- used for the rule\n"
        f"  P&L contribution positive in     {n_positive_contribution}/12"
    )
    print(
        "\nThe standalone reading is used because section 1's hypothesis is a claim about the\n"
        "rule working across markets, and it is the reading in the cited Moskowitz, Ooi &\n"
        "Pedersen paper. The contribution reading is reported alongside it."
    )

    decision = evaluate_decision_rule(
        cfg,
        strategy_sharpe=stats.sharpe,
        benchmark_sharpe=bh_stats.sharpe,
        n_positive_instruments=n_positive_standalone,
    )
    print(f"\n{decision}")

    rule("SECTION 9 - EXPECTATIONS OF RECORD, CHECKED")
    checks = [
        (
            f"realistic net Sharpe {cfg.expected_sharpe_low}-{cfg.expected_sharpe_high}",
            cfg.expected_sharpe_low <= stats.sharpe <= cfg.expected_sharpe_high,
            f"{stats.sharpe:.3f}",
        ),
        (
            f"above {cfg.bug_threshold_sharpe} means a bug",
            stats.sharpe <= cfg.bug_threshold_sharpe,
            f"{stats.sharpe:.3f}",
        ),
        (
            "roughly one year in three is a losing year",
            0.15 <= stats.losing_years / max(stats.n_years_observed, 1) <= 0.55,
            f"{stats.losing_years}/{stats.n_years_observed} = "
            f"{stats.losing_years / max(stats.n_years_observed, 1):.0%}",
        ),
        (
            "underwater periods of 3-5 years are normal",
            True,
            f"longest underwater run {stats.longest_drawdown_days / 252:.1f}y",
        ),
    ]
    for name, ok, observed in checks:
        print(f"  [{'OK ' if ok else 'NO '}] {name:52s} observed: {observed}")

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {
            "prereg_sha256": cfg.source_sha256,
            "window": [str(start.date()), str(prices.close.index[-1].date())],
            "strategy": stats.as_dict(),
            "benchmark": bh_stats.as_dict(),
            "decision": decision.verdict,
            "cost_ladder": ladder_df.reset_index().to_dict("records"),
            "standalone_sharpes": standalone.to_dict(),
        }
        out = RESULTS_DIR / ("backtest_full_history.json" if args.full_history else "backtest_headline.json")
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="001",
        choices=["001", "002", "003", "004", "005", "006"],
        help="which pre-registration to run",
    )
    parser.add_argument("--regression", action="store_true", help="experiment 001 refactor gate")
    parser.add_argument("--noise", action="store_true", help="experiment 002-006 noise tests")
    parser.add_argument(
        "--sleeves", action="store_true", help="experiment 006 step 1, the sleeve reproduction gate"
    )
    parser.add_argument("--free-tier", action="store_true", help="experiment 004 step 1 pipeline gate")
    parser.add_argument("--paired", action="store_true", help="experiment 004 step 6 A/B diagnostic")
    parser.add_argument("--refresh", action="store_true", help="refetch vendor data, ignoring cache")
    parser.add_argument("--synthetic", action="store_true", help="section 7 step 1 only")
    parser.add_argument("--validate", action="store_true", help="section 7 steps 1, 3, 4 and 5")
    parser.add_argument("--full-history", action="store_true", help="start at the first bar, not the all-12 date")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-days", type=int, default=6000)
    parser.add_argument("--tolerance", type=float, default=0.2, help="noise-test Sharpe tolerance")
    parser.add_argument("--source", default="yahoo", choices=["yahoo", "alpaca"])
    parser.add_argument("--train-years", type=float, default=3.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--step-years", type=float, default=1.0)
    parser.add_argument("--save", action="store_true", help="write a JSON result artefact")
    args = parser.parse_args(argv)

    if args.experiment == "002":
        import run_experiment_002  # noqa: PLC0415

        return run_experiment_002.main(args)
    if args.experiment == "003":
        import run_experiment_003  # noqa: PLC0415

        return run_experiment_003.main(args)
    if args.experiment == "004":
        import run_experiment_004  # noqa: PLC0415

        return run_experiment_004.main(args)
    if args.experiment == "005":
        import run_experiment_005  # noqa: PLC0415

        return run_experiment_005.main(args)
    if args.experiment == "006":
        import run_experiment_006  # noqa: PLC0415

        return run_experiment_006.main(args)

    cfg = load_config()
    if args.regression:
        return cmd_regression(cfg, args)
    if args.synthetic:
        return cmd_synthetic(cfg, args)
    if args.validate:
        return cmd_validate(cfg, args)
    return cmd_backtest(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
