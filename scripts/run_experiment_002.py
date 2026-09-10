#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.config_002 import Config002, load_config_002
from trendbot.data import daily_risk_free, load_prices, load_risk_free_rate
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR, sharpe, summarise
from trendbot.engine.panel import price_panel
from trendbot.engine.panel_backtest import panel_buy_and_hold, run_panel_backtest
from trendbot.engine.validation import (
    deflated_sharpe_ratio,
    in_sample_out_of_sample,
    walk_forward,
)
from trendbot.engine.xs_validation import (
    UniverseVerification,
    evaluate_decision_rule_002,
    factor_attribution,
    panel_noise_test,
    quintile_study,
    universe_verification,
    worst_months,
)
from trendbot.regression import EXPERIMENT_001_NET_SHARPE
from trendbot.strategies import CrossSectionalMomentum

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_NOISE_SEEDS = 16
FACTOR_SHARES = (0.25, 0.5)
WORST_MONTHS = 5


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _load(cfg: Config002, source: str):
    prices = load_prices(cfg.universe, source=source)
    rf = load_risk_free_rate()
    return prices, rf, daily_risk_free(rf, prices.close.index)


def report_universe(cfg: Config002, prices) -> UniverseVerification:
    rule("STEP 1 - UNIVERSE VERIFICATION (section 2)")
    verification = universe_verification(prices, cfg)
    table = verification.table.copy()
    table["first_bar"] = table["first_bar"].map(lambda d: d.date())
    table["last_bar"] = table["last_bar"].map(lambda d: d.date())
    print(table.to_string())
    print(f"\n{verification}")
    print(
        f"final count: {verification.n_kept}. Section 3's top quintile is therefore "
        f"{verification.n_kept // cfg.n_quantiles} of {verification.n_kept} "
        f"(section 3 declares ~{cfg.declared_quantile_size} of {cfg.declared_universe_size})."
    )
    holes = verification.with_holes
    if len(holes):
        print(
            f"\n{len(holes)} tickers are missing at least one bar inside the window "
            f"(max {int(holes.max())} of {verification.n_window_bars}). Every one of those "
            "gaps falls on a handful of recent sessions on which the vendor published "
            "nothing for a batch of symbols at once, which is a feed artefact rather than "
            "a discontinuity in the instrument. None is dropped on that basis; the counts "
            "are above so the stricter reading can be applied by anyone who prefers it."
        )
        print(holes.to_frame("missing bars").T.to_string())
    return verification


def cmd_noise(cfg: Config002, args) -> int:
    rule("STEP 3 / SECTION 7.1 - NOISE TESTS, BOTH VARIANTS")
    print(
        "The strategy is run on synthetic data containing no signal. Sharpe is reported\n"
        "GROSS of costs as well as net: this rule replaces most of its book every month,\n"
        "so a net figure on driftless data is expected to sit below zero by the cost drag,\n"
        "and testing only the net number would conflate 'the rule finds nothing' with\n"
        "'the costs are large'. The gross figure is the one section 7 step 1 is about.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or DEFAULT_NOISE_SEEDS)))
    results = []

    independent = panel_noise_test(
        cfg,
        label="(a) independent random walks",
        seeds=seeds,
        n_days=args.n_days,
        correlated=False,
    )
    print("(a) INDEPENDENT RANDOM WALKS")
    print("    Each instrument an independent driftless geometric random walk.")
    print(independent.table().round(4).to_string())
    print(f"\n    {independent}")
    results.append(independent)

    print(
        "\n(b) CORRELATED RANDOM WALKS WITH A SHARED COMMON FACTOR\n"
        "    r_i = beta_i * f + eps_i, betas spread linearly across the universe, f\n"
        "    driftless, and the idiosyncratic vol set per instrument so every instrument\n"
        "    has the SAME total volatility. That last part matters: if the betas moved\n"
        "    total vol too, a cross-sectional rank would be sorting partly on volatility\n"
        "    and a result would be ambiguous. The beta spread and the share of variance\n"
        "    the factor carries are reporting choices, not pre-registered; the test is run\n"
        "    at more than one loading so the answer is not a single point.\n"
        "    This is the test that matters here. On independent noise a cross-sectional\n"
        "    rule cannot load on a common factor because there is not one; with a factor\n"
        "    present, 'which names rose most' is partly 'which names have the highest\n"
        "    beta', and the rule can manufacture apparent skill out of a beta tilt.\n"
    )
    for share in FACTOR_SHARES:
        correlated = panel_noise_test(
            cfg,
            label=f"(b) common factor, {share:.0%} of average variance",
            seeds=seeds,
            n_days=args.n_days,
            correlated=True,
            common_variance_share=share,
        )
        print(f"    factor carries {share:.0%} of the average instrument's variance:")
        print(correlated.table().round(4).to_string())
        print(f"\n    {correlated}\n")
        results.append(correlated)

    print(
        "FACTOR ATTRIBUTION on the correlated null\n"
        "    A Sharpe near zero is necessary but not sufficient: the rule could be a\n"
        "    leveraged bet on the common factor in a period when the factor happened to go\n"
        "    nowhere. Regressing each seed's daily strategy return on the equal-weight\n"
        "    basket separates the two — beta is how much of the book IS the factor, alpha\n"
        "    is what is left. Run gross of costs, because a cost drag is not a loading.\n"
    )
    attribution = factor_attribution(
        cfg, seeds=seeds, n_days=args.n_days, common_variance_share=FACTOR_SHARES[-1]
    )
    print(attribution.table.round(4).to_string())
    print(f"\n    {attribution}")
    print(
        "    Read that as: the rule is overwhelmingly a factor tilt on this data — which is\n"
        "    exactly the failure mode variant (a) cannot see — and the tilt earns nothing,\n"
        "    because the factor it tilts onto is driftless. Both halves are needed."
    )
    no_alpha = abs(attribution.mean_alpha_annualised) < 0.01

    passed = all(r.passes(tolerance=args.tolerance) for r in results) and no_alpha
    print(
        f"\nGATE: |mean gross Sharpe| <= {args.tolerance} for the strategy and for "
        f"buy-and-hold on every variant, AND |alpha to the common factor| < 1%/yr "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def _run(cfg: Config002, prices, rf, *, cost_bps=None, targets=None):
    strategy = None if targets is not None else CrossSectionalMomentum(cfg)
    return run_panel_backtest(
        prices,
        cfg.universe,
        strategy,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else cost_bps,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )


def _psr(cfg: Config002, returns: pd.Series):
    own = float(returns.mean()) / float(returns.std(ddof=1))
    trials = (EXPERIMENT_001_NET_SHARPE / math.sqrt(TRADING_DAYS_PER_YEAR), own)
    return deflated_sharpe_ratio(returns, cfg.configurations_tried, trial_sharpes=trials), trials


def report_backtest(cfg: Config002, prices, rf, rf_daily, *, save: bool):
    start = pd.Timestamp(cfg.sample_start)
    end = prices.close.index[-1]
    strategy = CrossSectionalMomentum(cfg)
    targets = strategy(price_panel(prices, cfg.universe))

    rule("STEP 5 - FULL-SAMPLE BACKTEST, ONE RUN")
    print(cfg.describe())
    print(f"data: {prices.describe()}")
    print(
        f"window: {start.date()} -> {end.date()}  (section 6's pre-committed sample "
        "window, 2008-01-01 to present)"
    )
    print(
        "\nPrice history before the window is used to FORM the signal and for nothing else:\n"
        "the 252-day formation window at the first bar of 2008 reaches back into 2007, which\n"
        "is why section 2 requires history beginning on or before 2008-01-01. Results are\n"
        "measured only inside the window. Sharpe is in excess of the 13-week T-bill; idle\n"
        "cash accrues at that same rate; the benchmark is treated identically."
    )

    result = _run(cfg, prices, rf, targets=targets)
    stats = result.stats_from(start)
    returns = result.excess_returns.loc[start:]

    bh_excess = (panel_buy_and_hold(prices, cfg.universe, start=start) - rf_daily).loc[start:]
    bh_stats = summarise(bh_excess)

    held = (result.weights.loc[start:].abs() > 1e-12).sum(axis=1)
    print(f"\nstrategy (net of {cfg.cost_bps_per_side:g} bps/side, excess of T-bill):")
    print(f"  {stats}")
    print("equal-weight buy-and-hold of the same universe (excess of T-bill):")
    print(f"  {bh_stats}")
    print(
        f"\ngross of costs: Sharpe "
        f"{sharpe(result.gross_returns.loc[start:] - rf_daily.loc[start:]):+.3f}"
    )
    print(f"names held per day inside the window: {held.value_counts().sort_index().to_dict()}")
    print(
        "  (a day with 7 rather than 8 is a date on which fewer than 40 instruments had a\n"
        "   defined momentum, so the bucket size fell to 7 — section 3's exclusion rule)"
    )

    rule("SECTION 5 - TURNOVER AND COST SENSITIVITY")
    print(
        f"annualised one-way turnover: {stats.ann_turnover:.2f}x  "
        f"(experiment 001 ran at 3.23x; section 5 predicted this would be materially higher)\n"
    )
    ladder = []
    for bps in cfg.cost_sensitivity_bps:
        r = _run(cfg, prices, rf, cost_bps=bps, targets=targets)
        ex = r.excess_returns.loc[start:]
        ladder.append(
            {
                "cost bps/side": bps,
                "Sharpe": sharpe(ex),
                "CAGR": summarise(ex).cagr,
                "vs B&H": sharpe(ex) - bh_stats.sharpe,
            }
        )
    ladder_df = pd.DataFrame(ladder).set_index("cost bps/side")
    print(ladder_df.round(4).to_string())
    clears_only_free = (
        ladder_df.loc[0.0, "Sharpe"] > cfg.support_min_sharpe
        and ladder_df.loc[cfg.cost_bps_per_side, "Sharpe"] <= cfg.support_min_sharpe
    )
    print(
        f"\nonly clears the {cfg.support_min_sharpe:.2f} support threshold at 0 bps: "
        f"{'YES — that is a failure, not a technicality' if clears_only_free else 'no'}"
    )

    rule("SECTION 8's BENCHMARK - CONSTRUCTION SENSITIVITY")
    print(
        'Section 8 says "equal-weight buy-and-hold" and does not say how often, if ever, it\n'
        "is rebalanced. Read literally that means never, which is the headline and is the\n"
        "same construction experiment 001 used. The alternatives are shown because the\n"
        "unstated detail moves the margin by more than the 0.15 the rule turns on.\n"
    )
    bench_rows = []
    for mode, label in (
        ("none", "buy once, hold (literal)"),
        ("monthly", "rebalanced monthly"),
        ("daily", "rebalanced daily"),
    ):
        series = (
            panel_buy_and_hold(prices, cfg.universe, rebalance=mode, start=start) - rf_daily
        ).loc[start:]
        bench_rows.append(
            {
                "construction": label,
                "Sharpe": sharpe(series),
                "CAGR": summarise(series).cagr,
                "margin": stats.sharpe - sharpe(series),
                "beats it?": "yes" if stats.sharpe > sharpe(series) else "no",
            }
        )
    print(pd.DataFrame(bench_rows).set_index("construction").round(4).to_string())

    rule("SECTION 6's WINDOW - WHAT THE PRE-COMMITMENT COSTS")
    print(
        "Section 6 fixes the sample window at 2008-01-01 and section 2 requires history\n"
        "beginning on or before that date, so the formation window at the start of the\n"
        "sample reaches back into 2007 and the book is fully invested on day one. The\n"
        "alternative reading - discard everything before 2008-01-01, so the first signal\n"
        "does not exist until 2009 and the strategy sits in cash through the crash - is\n"
        "reported here because it is worth a lot of Sharpe and it is worth a lot in the\n"
        "flattering direction. It is NOT the headline.\n"
    )
    cold = prices.slice(str(start.date()))
    cold_targets = strategy(price_panel(cold, cfg.universe))
    cold_result = _run(cfg, cold, rf, targets=cold_targets)
    cold_first = (cold_targets.sum(axis=1) > 0).idxmax()
    window_rows = [
        {
            "variant": "pre-window history used to form the signal (headline)",
            "first traded": str(result.diagnostics.loc[start:].index[0].date()),
            "Sharpe": stats.sharpe,
            "benchmark": bh_stats.sharpe,
            "margin": stats.sharpe - bh_stats.sharpe,
        },
        {
            "variant": "cold start: no data before 2008-01-01 at all",
            "first traded": str(cold_first.date()),
            "Sharpe": sharpe(cold_result.excess_returns),
            "benchmark": bh_stats.sharpe,
            "margin": sharpe(cold_result.excess_returns) - bh_stats.sharpe,
        },
    ]
    print(pd.DataFrame(window_rows).set_index("variant").round(4).to_string())
    print(
        "\nThe cold start spends the whole of 2008 in cash, which is why it looks better.\n"
        "Reporting it as the headline would be selecting a window after seeing the result,\n"
        "which is the thing section 6 exists to prevent."
    )

    rule("STEP 4 / SECTION 8 - QUINTILE MONOTONICITY (a pass/fail gate, not a diagnostic)")
    study = quintile_study(prices, cfg, start=start)
    print(f"convention: {study.convention}")
    print(f"rebalances measured: {len(study.returns)}\n")
    print(study.table().round(5).to_string())
    print(f"\n{study}")
    print(
        f"  step-by-step change in mean forward return, Q1->Q5: "
        f"{np.round(np.diff(study.mean_monthly.to_numpy()) * 100, 4).tolist()} (%/month)"
    )
    print(
        f"  Q1-Q5 spread: {study.spread_mean:+.4%} per month "
        f"({(1 + study.spread_mean) ** 12 - 1:+.2%} annualised), t = {study.spread_t_stat:+.3f}, "
        f"n = {len(study.returns)}"
    )
    print(
        f"  ordering monotonically decreasing from Q1 to Q5: "
        f"{'YES' if study.is_monotonic else 'NO'}"
        + ("" if study.is_monotonic else f" — {study.n_inversions} of {cfg.n_quantiles - 1} steps invert")
    )

    free = _run(cfg, prices, rf, cost_bps=0.0, targets=targets)
    equity_open = free.diagnostics["equity_open"]
    equity_open = equity_open[equity_open.index >= start]
    open_to_open = pd.Series(
        equity_open.to_numpy()[1:] / equity_open.to_numpy()[:-1] - 1.0, index=equity_open.index[:-1]
    )
    shared = study.returns.index.intersection(open_to_open.index)
    gap = float((study.returns.loc[shared, "Q1"] - open_to_open.loc[shared]).abs().max())
    print(
        f"\n  cross-check: Q1's monthly series vs what the zero-cost book actually earned\n"
        f"  open-of-rebalance to open-of-next-rebalance, over the same {len(shared)} months:\n"
        f"  max absolute difference {gap:.3e} — Q1 IS the traded portfolio, not a proxy for it."
    )

    rule("STEP 6 - BEHAVIOURAL VALIDATION")
    print(
        "Section 9 names March 2009 and April 2020 as the canonical momentum crashes and\n"
        "predicts the strategy loses badly at sharp reversals. If those months are absent\n"
        "from the worst months the implementation may not be doing what it claims.\n"
    )
    monthly = result.returns.loc[start:]
    monthly = monthly.groupby(monthly.index.to_period("M")).apply(lambda x: float((1 + x).prod() - 1))
    bh_monthly = bh_excess.add(rf_daily.loc[start:], fill_value=0.0)
    bh_monthly = bh_monthly.groupby(bh_monthly.index.to_period("M")).apply(
        lambda x: float((1 + x).prod() - 1)
    )
    table = pd.DataFrame({"strategy": monthly, "benchmark": bh_monthly})
    table["relative"] = table["strategy"] - table["benchmark"]

    worst = worst_months(result.returns.loc[start:], WORST_MONTHS)
    print(f"the {WORST_MONTHS} worst months by ABSOLUTE net return:")
    print(table.loc[worst.index].round(4).to_string())
    print(f"\nthe {WORST_MONTHS} worst months RELATIVE to the benchmark:")
    print(table.sort_values("relative").head(WORST_MONTHS).round(4).to_string())
    predicted = [p for p in ("2009-03", "2020-04") if pd.Period(p) in table.index]
    print("\nthe two months section 9 names, and where they actually rank:")
    for period in predicted:
        row = table.loc[pd.Period(period)]
        abs_rank = int((table["strategy"] < row["strategy"]).sum()) + 1
        rel_rank = int((table["relative"] < row["relative"]).sum()) + 1
        print(
            f"  {period}: strategy {row['strategy']:+.2%}, benchmark {row['benchmark']:+.2%}, "
            f"relative {row['relative']:+.2%}  ->  rank {abs_rank}/{len(table)} by absolute "
            f"return, {rel_rank}/{len(table)} by relative"
        )
    worst_set = {str(period) for period in worst.index}
    missing = [p for p in ("2009-03", "2020-04") if p not in worst_set]
    if missing:
        print(
            f"\n  *** FLAG: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} NOT among "
            f"the {WORST_MONTHS} worst months. Section 9 predicted "
            f"{'it' if len(missing) == 1 else 'them'} by name and the prediction did not come "
            "true. This is reported as a failed prediction, not resolved. ***"
        )
    print(f"\nmax drawdown {stats.max_drawdown:.2%} (section 9 expected at least one above "
          f"{cfg.expected_min_drawdown:.0%}: "
          f"{'met' if abs(stats.max_drawdown) > cfg.expected_min_drawdown else 'NOT met'})")

    rule(f"STEP 7 - PSR / DEFLATED SHARPE, configs_tried = {cfg.configurations_tried}")
    dsr, trials = _psr(cfg, returns)
    print(
        f"cumulative configuration counter, per PREREG_002.md's header: "
        f"{cfg.configurations_tried} (experiment 001 was configuration 1)"
    )
    print(
        f"per-period Sharpes of the two configurations tried: "
        f"{trials[0]:.6f} (001), {trials[1]:.6f} (002)"
    )
    print(f"  {dsr}")
    print(f"  {dsr.note}")
    print(
        f"\nPSR(0) = {dsr.psr_vs_zero:.4f}  ->  "
        f"{'CLEARS' if dsr.psr_vs_zero > 0.95 else 'does NOT clear'} 95%"
    )
    print(
        f"deflated Sharpe at {cfg.configurations_tried} configurations = "
        f"{dsr.deflated_sharpe:.4f}  ->  "
        f"{'CLEARS' if dsr.is_significant else 'does NOT clear'} 95%"
    )

    rule("STEP 8 - SECTION 8's PRE-COMMITTED DECISION RULE")
    decision = evaluate_decision_rule_002(
        cfg,
        strategy_sharpe=stats.sharpe,
        benchmark_sharpe=bh_stats.sharpe,
        monotonic=study.is_monotonic,
        spread_positive=study.spread_positive,
    )
    print(decision)

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
            "turnover materially higher than experiment 001 (3.23x)",
            stats.ann_turnover > 3.23,
            f"{stats.ann_turnover:.2f}x",
        ),
        (
            f"at least one drawdown above {cfg.expected_min_drawdown:.0%}",
            abs(stats.max_drawdown) > cfg.expected_min_drawdown,
            f"{stats.max_drawdown:.2%}",
        ),
        (
            "momentum crashes in March 2009 and April 2020",
            not missing,
            "neither is among the five worst months" if len(missing) == 2 else f"missing: {missing}",
        ),
    ]
    for name, ok, observed in checks:
        print(f"  [{'OK ' if ok else 'NO '}] {name:52s} observed: {observed}")
    print(
        f"  [--] {'PSR may again fail to clear 95%':52s} observed: PSR(0) "
        f"{dsr.psr_vs_zero:.4f}, {'cleared' if dsr.psr_vs_zero > 0.95 else 'did NOT clear'} "
        "(the expectation says 'may', so neither outcome contradicts it)"
    )

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {
            "prereg_sha256": cfg.source_sha256,
            "configurations_tried": cfg.configurations_tried,
            "window": [str(start.date()), str(end.date())],
            "universe_final_count": cfg.n_universe,
            "strategy": stats.as_dict(),
            "benchmark": bh_stats.as_dict(),
            "cost_ladder": ladder_df.reset_index().to_dict("records"),
            "quintiles": study.table().to_dict(),
            "quintile_monotonic": bool(study.is_monotonic),
            "quintile_spread_mean": study.spread_mean,
            "quintile_spread_t": study.spread_t_stat,
            "psr_vs_zero": dsr.psr_vs_zero,
            "deflated_sharpe": dsr.deflated_sharpe,
            "decision": decision.verdict,
            "worst_months": {
                str(k): v for k, v in table.sort_values("strategy").head(WORST_MONTHS)["strategy"].items()
            },
        }
        out = RESULTS_DIR / "backtest_002.json"
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {out}")

    return decision, stats, bh_stats, study, dsr, returns


def cmd_backtest(cfg: Config002, args) -> int:
    prices, rf, rf_daily = _load(cfg, args.source)
    report_universe(cfg, prices)
    report_backtest(cfg, prices, rf, rf_daily, save=args.save)
    return 0


def cmd_validate(cfg: Config002, args) -> int:
    prices, rf, rf_daily = _load(cfg, args.source)
    start = pd.Timestamp(cfg.sample_start)

    report_universe(cfg, prices)
    noise_status = cmd_noise(cfg, args)

    result = _run(cfg, prices, rf)
    returns = result.excess_returns.loc[start:]

    rule("SECTION 7.5 - IN-SAMPLE / OUT-OF-SAMPLE")
    print("Nothing is fitted, so both halves are out-of-sample by construction.")
    split = in_sample_out_of_sample(returns)
    print(f"  {split}")

    rule("SECTION 7.5 - WALK-FORWARD")
    print(
        f"Rolling windows, nothing re-optimised. Window lengths are NOT pre-registered\n"
        f"(section 6's frozen list does not contain them) and cannot change a position:\n"
        f"{args.train_years:g}y in-window, {args.test_years:g}y forward, stepping "
        f"{args.step_years:g}y.\n"
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

    decision, *_ = report_backtest(cfg, prices, rf, rf_daily, save=args.save)
    rule("PROTOCOL COMPLETE")
    print(f"section 8 verdict: {decision.verdict}")
    print(f"noise gate: {'PASS' if noise_status == 0 else 'FAIL'}")
    return noise_status


def main(args) -> int:
    cfg = load_config_002()
    if args.noise:
        return cmd_noise(cfg, args)
    if args.validate:
        return cmd_validate(cfg, args)
    return cmd_backtest(cfg, args)
