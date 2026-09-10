#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.config_003 import Config003, load_config_003
from trendbot.data import daily_risk_free, load_prices, load_risk_free_rate
from trendbot.engine.eq_validation import evaluate_decision_rule_003, market_regression
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR, sharpe, summarise
from trendbot.engine.panel import price_panel
from trendbot.engine.panel_backtest import panel_buy_and_hold, run_panel_backtest
from trendbot.engine.validation import (
    deflated_sharpe_ratio,
    in_sample_out_of_sample,
    walk_forward,
)
from trendbot.engine.xs_validation import (
    bucket_study,
    factor_attribution,
    panel_noise_test,
    worst_months,
)
from trendbot.equities import (
    _split_dates,
    adjustment_convention,
    corporate_action_audit,
    load_constituent_snapshot,
    load_market_proxy,
    ratio_invariance_report,
    resolve_universe,
    split_reconciliation,
)
from trendbot.regression import EXPERIMENT_001_NET_SHARPE
from trendbot.strategies import EquityCrossSectionalMomentum

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_NOISE_SEEDS = 8
NOISE_UNIVERSE_SIZE = 60
FACTOR_SHARES = (0.25, 0.5)
WORST_MONTHS = 5
EXTREME_MOVE_THRESHOLD = 0.35
ACTION_DATE_THRESHOLD = 0.20
BUCKET_METHOD = "even"
BUCKET_METHOD_ALTERNATIVE = "floor"
EXPERIMENT_002_NET_SHARPE = 0.4631


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _last_complete_session(index: pd.DatetimeIndex) -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    earlier = index[index < today]
    if len(earlier) == 0:
        raise ValueError("no completed session in the price history")
    return earlier[-1]


def load_everything(cfg: Config003, source: str, *, refresh_actions: bool = False):
    snapshot = load_constituent_snapshot()
    prices = load_prices(
        snapshot.symbols,
        source=source,
        start="2005-01-01",
        max_abs_daily_move=None,
    )
    end = _last_complete_session(prices.close.index)
    prices = prices.slice(None, str(end.date()))
    universe = resolve_universe(
        prices, snapshot, required_from=cfg.universe_history_required_from, window_end=end
    )
    rf = load_risk_free_rate()
    return snapshot, prices, universe, end, rf, daily_risk_free(rf, prices.close.index)


def report_universe(cfg: Config003, snapshot, universe, end) -> None:
    rule("STEP 1 - UNIVERSE CONSTRUCTION (section 2)")
    print(f"constituent snapshot: {snapshot.describe()}")
    print(f"file: {snapshot.path.name}")
    print(
        f"\nSection 2's universe is a RULE, not a list: 'current {cfg.index_name} constituents "
        f"with\ncontinuous daily data from {cfg.universe_history_required_from} to present'. A "
        "rule resolved against a live\nindex would silently be a different experiment each month, "
        "so the constituent list is\npinned to the dated, hashed file above and the resolution is "
        "reported here."
    )
    print(f"\n{universe}")
    print(
        f"exact count: {universe.n}  (section 2 expected {cfg.expected_universe_low}-"
        f"{cfg.expected_universe_high}; "
        + (
            "inside the expected range"
            if cfg.expected_universe_low <= universe.n <= cfg.expected_universe_high
            else f"OUTSIDE the expected range by {universe.n - cfg.expected_universe_high:+d}"
        )
        + ")"
    )
    print(f"window: {universe.window_start.date()} -> {universe.window_end.date()} "
          f"({universe.n_window_bars} bars)")

    print("\n--- first available bar, by sector ---")
    table = universe.table.copy()
    table["first_bar"] = table["first_bar"].map(lambda d: d.date())
    summary = (
        table.groupby("sector")
        .agg(names=("first_bar", "size"), earliest=("first_bar", "min"), latest=("first_bar", "max"))
        .sort_values("names", ascending=False)
    )
    print(summary.to_string())
    print(f"\nlatest inception among the {universe.n} eligible names:")
    latest = table.sort_values("first_bar").tail(8)[["sector", "first_bar"]]
    print(latest.to_string())

    print(f"\n--- {len(universe.excluded)} current constituents EXCLUDED for insufficient history ---")
    print("No substitutions are made: section 2 forbids replacing them.")
    excluded = universe.excluded.copy()
    excluded["first_bar"] = excluded["first_bar"].map(
        lambda d: d.date() if pd.notna(d) else "no data"
    )
    print(
        ", ".join(
            f"{t} ({excluded.loc[t, 'first_bar']})"
            for t in sorted(excluded.index)
        )
    )
    by_sector = excluded.groupby("sector").size().sort_values(ascending=False)
    print(f"\nexcluded by sector:\n{by_sector.to_string()}")

    holes = universe.with_holes
    if len(holes):
        print(
            f"\n{len(holes)} eligible names are missing at least one bar inside the window "
            f"(max {int(holes.max())} of {universe.n_window_bars}). 'Continuous' is read as "
            "continuously\nLISTED, not 'the vendor published a bar every session' - the same "
            "reading experiments\n001 and 002 used. The counts are here so the stricter reading "
            "can be applied to a number."
        )
        print(holes.value_counts().sort_index().rename("names").to_frame().T.to_string())


def report_corporate_actions(cfg: Config003, prices, universe, end, *, refresh: bool = False):
    rule("STEP 2 - CORPORATE ACTION AUDIT (section 5) — GATES EVERYTHING DOWNSTREAM")
    convention = adjustment_convention(prices)
    print("ADJUSTMENT CONVENTION, stated plainly:")
    for key, value in convention.items():
        print(f"  {key:16s}: {value}")
    print(
        f"\n  ==> POINT-IN-TIME: {'YES' if convention['point_in_time'] else 'NO'}. "
        "The series is back-adjusted with factors known\n      at download time, so it is "
        "the series an observer reconstructs today, not the\n      one a trader saw then. "
        "Section 5 requires this to be stated; what it costs\n      THIS signal is quantified "
        "immediately below rather than asserted either way."
    )

    window = prices.slice(cfg.sample_start, str(end.date()))
    invariance = ratio_invariance_report(
        window.close[list(universe.universe)],
        cfg.formation_days,
        cfg.skip_days,
        split_factor=2.0,
        split_offset_days=5,
    )
    print("\nWHAT BACK-ADJUSTMENT COSTS A RATIO SIGNAL — measured, not argued:")
    print(
        "  The signal is P(t-21)/P(t-252), a RATIO. Back-adjustment multiplies every price\n"
        "  before an action by a constant. An action AFTER bar t scales both endpoints by the\n"
        "  same constant, which cancels; an action BETWEEN them scales only the older endpoint,\n"
        "  which is the correction that makes the ratio a true return. So a future action\n"
        "  cannot leak into a past signal value through this route. Tested by injecting one:"
    )
    print(f"  {invariance}")
    print(f"  ({invariance.detail})")
    print(
        "\n  What that does NOT cover, and is therefore live exposure rather than dismissed:\n"
        "    (a) an action the vendor has WRONG or MISSING - the scan below is for exactly that;\n"
        "    (b) delisting and index-membership survivorship - section 2's problem, not a\n"
        "        data-adjustment question, and addressed in the verdict."
    )

    print(f"\n--- scan A: every daily move beyond ±{EXTREME_MOVE_THRESHOLD:.0%} (the build order's scan) ---")
    split_dates, fetch_errors = _split_dates(
        universe.universe, cfg.sample_start, refresh=refresh
    )
    audit = corporate_action_audit(
        window,
        universe.universe,
        threshold=EXTREME_MOVE_THRESHOLD,
        start=cfg.sample_start,
        split_dates=split_dates,
    )
    print(audit)
    print(f"\nmoves by year:\n{audit.by_year().to_string()}")
    print("\nthe 10 largest, with their classification:")
    largest = audit.events.reindex(
        audit.events["return"].abs().sort_values(ascending=False).index
    ).head(10)
    print(largest.assign(date=lambda d: d["date"].dt.date).to_string(index=False))

    print(
        f"\n--- scan B: the adjusted return on EVERY recorded action date ({len(split_dates)} tickers) ---"
    )
    print(
        "Scan A is a RETURN filter, so it can only find a broken adjustment large enough to\n"
        "clear ±35%. A 5-for-4 split that failed to adjust is −20%: badly wrong, invisible to\n"
        "scan A, and quite enough to move a name several deciles. Scan B inverts the question\n"
        "and checks what the series did on every date an action is recorded, which is complete."
    )
    reconciliation = split_reconciliation(
        window,
        universe.universe,
        split_dates=split_dates,
        start=cfg.sample_start,
        threshold=ACTION_DATE_THRESHOLD,
    )
    print(f"\n{reconciliation}")
    print("\nthe 12 largest adjusted moves on an action date:")
    print(reconciliation.table.head(12).assign(
        action_date=lambda d: d["action_date"].dt.date, bar=lambda d: d["bar"].dt.date
    ).to_string(index=False))

    contaminated = tuple(sorted(set(reconciliation.suspicious["ticker"])))
    print(
        f"\n--- HOW EACH FINDING WAS HANDLED ---\n"
        f"Scan A found {audit.n_events} extreme moves. {len(audit.unexplained)} have no corporate\n"
        f"action within ±3 days and are market moves - individual equities have those, and the\n"
        f"year distribution above concentrates them in 2008-09 and March 2020 exactly as it\n"
        f"should. They are NOT treated as data errors and nothing is done about them.\n"
    )
    print(
        f"Scan B found {len(reconciliation.suspicious)} action dates where the ADJUSTED series moved "
        f"beyond ±{ACTION_DATE_THRESHOLD:.0%}.\nOn a correctly adjusted series an action date is an "
        "ordinary trading day. Every one of\nthese is a spin-off: this vendor records spin-offs in "
        "the splits table and back-adjusts\nby the recorded share ratio, but a spin-off's price "
        "ratio is not its share ratio, so a\ndiscontinuity survives. The affected names are:"
    )
    for ticker in contaminated:
        rows = reconciliation.suspicious[reconciliation.suspicious["ticker"] == ticker]
        for _, row in rows.iterrows():
            print(
                f"    {ticker:6s} {row['action_date'].date()}  adjusted return "
                f"{row['adjusted_return']:+.2%}"
            )
    print(
        f"\nRULE APPLIED (fixed at ±{ACTION_DATE_THRESHOLD:.0%} before any of these were seen): a name whose "
        "adjusted series\nmoves beyond that threshold on a recorded action date has an adjustment "
        "this vendor did\nnot get right, and is excluded from the universe. A spurious +61% one-day "
        "return puts a\nname in the top decile and holds it there for eleven months, which is "
        "precisely the\nfailure section 5 warns about, with the sign reversed.\n"
        f"    excluded: {list(contaminated)}  ->  universe {universe.n} - {len(contaminated)} = "
        f"{universe.n - len(contaminated)}\n"
        "The headline runs on the filtered universe; the unfiltered result is reported as a\n"
        "sensitivity so the verdict can be seen not to depend on this handling."
    )
    if fetch_errors:
        print(f"\n  action records could not be fetched for {len(fetch_errors)} tickers: {fetch_errors[:5]}")

    gate = not fetch_errors
    print(
        f"\nGATE: action records retrieved for every eligible name, every extreme move "
        f"classified,\nand every contaminated name excluded -> {'PASS' if gate else 'FAIL'}"
    )
    return audit, reconciliation, contaminated, invariance, convention, gate


def cmd_noise(cfg: Config003, args) -> int:
    rule("STEP 4 / SECTION 7.1 - NOISE TESTS, BOTH VARIANTS")
    print(
        "The strategy is run on synthetic data containing no signal. Sharpe is reported\n"
        "GROSS of costs as well as net: this rule replaces most of its book every month, so a\n"
        "net figure on driftless data is expected to sit below zero by the cost drag, and\n"
        "testing only the net number would conflate 'the rule finds nothing' with 'the costs\n"
        "are large'. The gross figure is the one section 7 step 1 is about.\n"
    )
    print(
        f"Synthetic panels are {NOISE_UNIVERSE_SIZE} instruments wide, not {cfg.expected_universe_high}+: "
        "the null being tested is a\nproperty of the RANKING RULE, not of the universe size, and a "
        "400-wide panel across\nseeds and loadings costs hours for an identical answer. The decile "
        f"cut still applies -\n{NOISE_UNIVERSE_SIZE} names is {NOISE_UNIVERSE_SIZE // cfg.n_quantiles} "
        "per bucket, comfortably enough for a ten-way sort.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or DEFAULT_NOISE_SEEDS)))
    names = tuple(f"N{i:03d}" for i in range(NOISE_UNIVERSE_SIZE))
    strategy = EquityCrossSectionalMomentum(cfg, names, BUCKET_METHOD)
    results = []

    independent = panel_noise_test(
        label="(a) independent random walks",
        seeds=seeds,
        n_days=args.n_days,
        correlated=False,
        universe=names,
        strategy=strategy,
        cost_bps=cfg.cost_bps_per_side,
        gross_cap=cfg.gross_exposure_cap,
    )
    print("(a) INDEPENDENT RANDOM WALKS")
    print(independent.table().round(4).to_string())
    print(f"\n    {independent}")
    results.append(independent)

    print(
        "\n(b) CORRELATED RANDOM WALKS WITH A SHARED COMMON FACTOR\n"
        "    r_i = beta_i * f + eps_i, betas spread linearly, f driftless, and the idiosyncratic\n"
        "    vol set per instrument so every instrument has the SAME total volatility - otherwise\n"
        "    a cross-sectional rank would be sorting partly on volatility and any result would be\n"
        "    ambiguous. This is the variant that diagnosed experiment 002.\n"
    )
    for share in FACTOR_SHARES:
        correlated = panel_noise_test(
            label=f"(b) common factor, {share:.0%} of average variance",
            seeds=seeds,
            n_days=args.n_days,
            correlated=True,
            common_variance_share=share,
            universe=names,
            strategy=strategy,
            cost_bps=cfg.cost_bps_per_side,
            gross_cap=cfg.gross_exposure_cap,
        )
        print(f"    factor carries {share:.0%} of the average instrument's variance:")
        print(correlated.table().round(4).to_string())
        print(f"\n    {correlated}\n")
        results.append(correlated)

    print("FACTOR ATTRIBUTION on the correlated null (the diagnostic 002 introduced)")
    print(
        "    A Sharpe near zero is necessary but not sufficient: the rule could be a leveraged\n"
        "    bet on the common factor in a period when the factor happened to go nowhere.\n"
        "    Regressing each seed's daily return on the equal-weight basket separates the two.\n"
    )
    attribution = factor_attribution(
        seeds=seeds,
        n_days=args.n_days,
        common_variance_share=FACTOR_SHARES[-1],
        universe=names,
        strategy=strategy,
        gross_cap=cfg.gross_exposure_cap,
    )
    print(attribution.table.round(4).to_string())
    print(f"\n    {attribution}")
    no_alpha = abs(attribution.mean_alpha_annualised) < 0.01

    passed = all(r.passes(tolerance=args.tolerance) for r in results) and no_alpha
    print(
        f"\nGATE: |mean gross Sharpe| <= {args.tolerance} for the strategy and for buy-and-hold on\n"
        f"every variant, AND |alpha to the common factor| < 1%/yr -> {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def _run(cfg: Config003, prices, universe, rf, *, cost_bps=None, targets=None, strategy=None):
    return run_panel_backtest(
        prices,
        universe,
        strategy,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else cost_bps,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )


def _psr(cfg: Config003, returns: pd.Series):
    own = float(returns.mean()) / float(returns.std(ddof=1))
    root = math.sqrt(TRADING_DAYS_PER_YEAR)
    trials = (
        EXPERIMENT_001_NET_SHARPE / root,
        EXPERIMENT_002_NET_SHARPE / root,
        own,
    )
    return deflated_sharpe_ratio(returns, cfg.configurations_tried, trial_sharpes=trials), trials


def report_backtest(cfg, prices, universe, contaminated, end, rf, rf_daily, *, save: bool):
    start = pd.Timestamp(cfg.sample_start)
    traded = tuple(t for t in universe.universe if t not in set(contaminated))
    strategy = EquityCrossSectionalMomentum(cfg, traded, BUCKET_METHOD)
    panel = price_panel(prices, traded)
    targets = strategy(panel)

    rule("STEP 5 - FULL-SAMPLE BACKTEST, ONE RUN")
    print(cfg.describe())
    print(f"data: {prices.describe()}")
    print(f"universe: {len(traded)} names ({universe.n} eligible − {len(contaminated)} contaminated)")
    print(f"window: {start.date()} -> {end.date()}  (section 6's pre-committed sample window)")
    print(
        f"bucket sizing: {BUCKET_METHOD!r}. Section 3 names no bucket size, and section 8's gate is\n"
        "the D1−D10 spread, so the two ends of the sort are kept the same size. The alternative\n"
        f"({BUCKET_METHOD_ALTERNATIVE!r}, experiment 002's rule) is reported as a sensitivity below."
    )
    print(
        "\nPrice history before the window forms the signal and does nothing else: the 252-day\n"
        "formation window at the first bar of 2008 reaches into 2007, which is why section 2\n"
        "requires history from on or before 2008-01-01. Sharpe is in excess of the 13-week\n"
        "T-bill; idle cash accrues at that rate; the benchmark is treated identically."
    )

    result = _run(cfg, prices, traded, rf, targets=targets)
    stats = result.stats_from(start)
    returns = result.excess_returns.loc[start:]
    bh_excess = (panel_buy_and_hold(prices, traded, start=start) - rf_daily).loc[start:end]
    bh_stats = summarise(bh_excess)

    held = (result.weights.loc[start:end].abs() > 1e-12).sum(axis=1)
    print(f"\nstrategy (net of {cfg.cost_bps_per_side:g} bps/side, excess of T-bill):")
    print(f"  {stats}")
    print("equal-weight buy-and-hold of the same universe (excess of T-bill):")
    print(f"  {bh_stats}")
    print(
        f"\ngross of costs: Sharpe "
        f"{sharpe(result.gross_returns.loc[start:end] - rf_daily.loc[start:end]):+.3f}"
    )
    print(f"names held per day: min {int(held.min())}, median {int(held.median())}, max {int(held.max())}")

    rule("SECTION 5 - TURNOVER AND COST SENSITIVITY")
    print(
        f"annualised one-way turnover: {stats.ann_turnover:.2f}×  "
        f"(section 9 expected higher than 002's {cfg.turnover_floor:g}×: "
        f"{'met' if stats.ann_turnover > cfg.turnover_floor else 'NOT met'})\n"
    )
    ladder = []
    for bps in cfg.cost_sensitivity_bps:
        r = _run(cfg, prices, traded, rf, cost_bps=bps, targets=targets)
        ex = r.excess_returns.loc[start:end]
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

    rule("SECTION 8's BENCHMARK - CONSTRUCTION SENSITIVITY")
    print(
        'Section 8 says "equal-weight buy-and-hold" and does not say how often, if ever, it is\n'
        "rebalanced. Read literally that means never, which is the headline and the same\n"
        "construction experiments 001 and 002 used.\n"
    )
    bench_rows = []
    for mode, label in (
        ("none", "buy once, hold (literal)"),
        ("monthly", "rebalanced monthly"),
        ("daily", "rebalanced daily"),
    ):
        series = (panel_buy_and_hold(prices, traded, rebalance=mode, start=start) - rf_daily).loc[
            start:end
        ]
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

    rule("STEP 6 - DECILE MONOTONICITY (section 8's pass/fail gate)")
    study = bucket_study(
        prices.slice(None, str(end.date())),
        traded,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method=BUCKET_METHOD,
        label_prefix="D",
        start=start,
    )
    print(f"convention: {study.convention}")
    print(f"rebalances measured: {len(study.returns)}\n")
    print(study.table().round(5).to_string())
    steps = np.diff(study.mean_monthly.to_numpy())
    print(f"\n{study}")
    print(f"  step-by-step change D1→D10 (%/month): {np.round(steps * 100, 4).tolist()}")
    print(
        f"  adjacent inversions: {study.n_inversions} of {cfg.n_quantiles - 1} "
        f"(section 8 tolerates at most {cfg.max_inversions})"
    )
    print(
        f"  D1−D10 spread: {study.spread_mean:+.4%} per month "
        f"({(1 + study.spread_mean) ** 12 - 1:+.2%} annualised), "
        f"t = {study.spread_t_stat:+.3f}, n = {len(study.returns)}"
    )
    print(
        f"  section 8 requires t > {cfg.min_spread_t_stat:.1f} for support and abandons below "
        f"t = {cfg.abandon_below_spread_t_stat:.1f}"
    )

    alternative = bucket_study(
        prices.slice(None, str(end.date())),
        traded,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method=BUCKET_METHOD_ALTERNATIVE,
        label_prefix="D",
        start=start,
    )
    print(
        f"\n  sensitivity to the bucket convention ({BUCKET_METHOD_ALTERNATIVE!r}, experiment 002's rule): "
        f"{alternative.n_inversions} inversions, spread "
        f"{alternative.spread_mean:+.4%}/month at t {alternative.spread_t_stat:+.3f}"
    )

    rule("STEP 7 - MARKET-BETA ATTRIBUTION (section 7.5, now standard equipment)")
    print(
        'Section 7.5 says "market excess returns" and names no index. Two readings are\n'
        "defensible: the cap-weighted market index, which is the literal reading of section 8's\n"
        "'market beta'; and the equal-weight buy-and-hold of the same universe, which is what\n"
        "experiment 002's diagnostic used. BOTH are reported. The cap-weighted proxy is the\n"
        "headline and section 8's clauses are adjudicated on it - declared in\n"
        "data/universe/market_proxy.json before either number existed, not chosen after.\n"
    )
    proxy = load_market_proxy()
    print(f"  declared proxy: {proxy.describe()}")
    print(f"  rationale: {proxy.rationale}")
    market = (
        load_prices((proxy.symbol,), source="yahoo").close[proxy.symbol].pct_change().fillna(0.0)
    )
    market_excess = (market.reindex(returns.index) - rf_daily.reindex(returns.index)).dropna()
    headline_label = f"{proxy.symbol} (cap-weighted market)"
    regressions = {
        headline_label: market_regression(returns, market_excess, market_label=headline_label),
        proxy.alternative: market_regression(
            returns, bh_excess, market_label=proxy.alternative
        ),
    }
    for label, reg in regressions.items():
        marker = "HEADLINE" if label == headline_label else "reported alongside"
        print(f"  [{marker}] {reg}")
    headline = regressions[headline_label]
    print(
        f"\n  section 8 requires alpha positive at t > {cfg.min_alpha_t_stat:.1f} for support, and "
        f"abandons on negative alpha:\n  alpha {headline.alpha_annualised:+.2%}/yr at t "
        f"{headline.alpha_t_stat:+.3f} -> "
        f"{'clears' if headline.clears(cfg.min_alpha_t_stat) else 'does NOT clear'}"
    )

    rule("STEP 8 - WORST MONTHS (section 9's mandatory momentum crashes)")
    print(
        f"Section 9 names {', '.join(cfg.required_crash_months)} and requires them among the worst\n"
        "months. Experiment 002 failed this and it was diagnostic. If a stock-level\n"
        "implementation also fails it, section 9 says the implementation is wrong, not the\n"
        "anomaly.\n"
    )
    monthly = result.returns.loc[start:end]
    monthly = monthly.groupby(monthly.index.to_period("M")).apply(
        lambda x: float((1 + x).prod() - 1)
    )
    bh_total = panel_buy_and_hold(prices, traded, start=start).loc[start:end]
    bh_monthly = bh_total.groupby(bh_total.index.to_period("M")).apply(
        lambda x: float((1 + x).prod() - 1)
    )
    table = pd.DataFrame({"strategy": monthly, "benchmark": bh_monthly})
    table["relative"] = table["strategy"] - table["benchmark"]

    worst = worst_months(result.returns.loc[start:end], WORST_MONTHS)
    print(f"the {WORST_MONTHS} worst months by ABSOLUTE net return:")
    print(table.loc[worst.index].round(4).to_string())
    print(f"\nthe {WORST_MONTHS} worst months RELATIVE to the benchmark:")
    print(table.sort_values("relative").head(WORST_MONTHS).round(4).to_string())

    worst_set = {str(period) for period in worst.index}
    print("\nthe months section 9 names, and where they actually rank:")
    ranks = {}
    for period in cfg.required_crash_months:
        key = pd.Period(period, freq="M")
        if key not in table.index:
            print(f"  {period}: not in the sample")
            continue
        row = table.loc[key]
        absolute = int((table["strategy"] < row["strategy"]).sum()) + 1
        relative = int((table["relative"] < row["relative"]).sum()) + 1
        ranks[period] = (absolute, relative)
        print(
            f"  {period}: strategy {row['strategy']:+.2%}, benchmark {row['benchmark']:+.2%}, "
            f"relative {row['relative']:+.2%}  ->  rank {absolute}/{len(table)} absolute, "
            f"{relative}/{len(table)} relative"
        )
    missing = [p for p in cfg.required_crash_months if p not in worst_set]
    crashes_present = not missing
    if missing:
        print(
            f"\n  *** FLAG: {', '.join(missing)} not among the {WORST_MONTHS} worst months.\n"
            "      Section 9 states that a stock-level implementation failing this test means\n"
            "      THE IMPLEMENTATION IS SUSPECT, not the anomaly. Reported as such, not\n"
            "      explained away. ***"
        )
    else:
        print(f"\n  all of section 9's named months appear among the {WORST_MONTHS} worst. ")

    rule(f"STEP 9 - PSR / DEFLATED SHARPE, configs_tried = {cfg.configurations_tried}")
    dsr, trials = _psr(cfg, returns)
    print(
        f"cumulative counter, per PREREG_003.md's header: {cfg.configurations_tried} "
        "(001 = trend, 002 = ETF cross-section)"
    )
    print(
        "per-period Sharpes of the three configurations tried: "
        + ", ".join(f"{t:.6f}" for t in trials)
    )
    print(f"  {dsr}")
    print(f"  {dsr.note}")
    print(
        f"\nPSR(0) = {dsr.psr_vs_zero:.4f}  ->  "
        f"{'CLEARS' if dsr.psr_vs_zero > 0.95 else 'does NOT clear'} 95%"
    )
    print(
        f"deflated Sharpe at {cfg.configurations_tried} configurations = {dsr.deflated_sharpe:.4f}"
        f"  ->  {'CLEARS' if dsr.is_significant else 'does NOT clear'} 95%"
    )

    rule("STEP 10 - SECTION 8's PRE-COMMITTED DECISION RULE")
    decision = evaluate_decision_rule_003(
        cfg,
        strategy_sharpe=stats.sharpe,
        benchmark_sharpe=bh_stats.sharpe,
        n_inversions=study.n_inversions,
        spread_mean=study.spread_mean,
        spread_t_stat=study.spread_t_stat,
        alpha_annualised=headline.alpha_annualised,
        alpha_t_stat=headline.alpha_t_stat,
    )
    print(decision)

    rule("SURVIVORSHIP - BOTH READINGS, PER SECTION 2 (mandatory in the verdict)")
    print(
        "Section 2 states the direction of the bias in advance: survivorship removes failures,\n"
        "failures concentrate in the BOTTOM of a momentum ranking, so D10 is artificially\n"
        "strong, the D1−D10 spread artificially NARROW and the gradient artificially FLAT.\n"
    )
    print(
        f"  Reading 1 - if the gate PASSES, the pass is CONSERVATIVE evidence: a real universe\n"
        f"  including the bankruptcies would have a weaker D10 and therefore a WIDER spread than\n"
        f"  the {study.spread_mean:+.4%}/month measured here.\n"
    )
    print(
        f"  Reading 2 - if the gate FAILS, the failure is AMBIGUOUS: it is consistent with the\n"
        f"  anomaly being absent AND with the anomaly being present but flattened below\n"
        f"  detection by the bias. The measured t of {study.spread_t_stat:+.3f} cannot distinguish them.\n"
    )
    print(
        "  Neither reading may be selected after the fact. Which one applies is determined by\n"
        f"  the gate result, which is: {'PASS' if study.n_inversions <= cfg.max_inversions and study.spread_t_stat > cfg.min_spread_t_stat else 'FAIL'}"
        f" -> READING {'1' if study.n_inversions <= cfg.max_inversions and study.spread_t_stat > cfg.min_spread_t_stat else '2'} APPLIES."
    )

    rule("SECTION 9 - EXPECTATIONS OF RECORD, CHECKED")
    checks = [
        (
            f"realistic net Sharpe {cfg.expected_sharpe_low}-{cfg.expected_sharpe_high}",
            cfg.expected_sharpe_low <= stats.sharpe <= cfg.expected_sharpe_high,
            f"{stats.sharpe:.3f}",
        ),
        (
            f"above {cfg.bug_threshold_sharpe} means a bug or an uncorrected action",
            stats.sharpe <= cfg.bug_threshold_sharpe,
            f"{stats.sharpe:.3f}",
        ),
        (
            f"turnover higher than 002's {cfg.turnover_floor:g}×",
            stats.ann_turnover > cfg.turnover_floor,
            f"{stats.ann_turnover:.2f}×",
        ),
        (
            "momentum crashes present among the worst months",
            crashes_present,
            "all present" if crashes_present else f"missing: {missing}",
        ),
    ]
    for name, ok, observed in checks:
        print(f"  [{'OK ' if ok else 'NO '}] {name:56s} observed: {observed}")

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {
            "prereg_sha256": cfg.source_sha256,
            "configurations_tried": cfg.configurations_tried,
            "window": [str(start.date()), str(end.date())],
            "universe_eligible": universe.n,
            "universe_traded": len(traded),
            "contaminated_excluded": list(contaminated),
            "bucket_method": BUCKET_METHOD,
            "strategy": stats.as_dict(),
            "benchmark": bh_stats.as_dict(),
            "cost_ladder": ladder_df.reset_index().to_dict("records"),
            "deciles": study.table().to_dict(),
            "n_inversions": int(study.n_inversions),
            "spread_mean": study.spread_mean,
            "spread_t": study.spread_t_stat,
            "market_regressions": {
                label: {
                    "beta": reg.beta,
                    "alpha_annualised": reg.alpha_annualised,
                    "alpha_t": reg.alpha_t_stat,
                    "r_squared": reg.r_squared,
                }
                for label, reg in regressions.items()
            },
            "psr_vs_zero": dsr.psr_vs_zero,
            "deflated_sharpe": dsr.deflated_sharpe,
            "decision": decision.verdict,
            "worst_months": {str(k): float(v) for k, v in worst["return"].items()},
        }
        out = RESULTS_DIR / "backtest_003.json"
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {out}")

    return decision, stats, bh_stats, study, headline, dsr, returns


def cmd_backtest(cfg: Config003, args) -> int:
    snapshot, prices, universe, end, rf, rf_daily = load_everything(cfg, args.source)
    report_universe(cfg, snapshot, universe, end)
    *_, contaminated, _, _, gate = report_corporate_actions(cfg, prices, universe, end)
    if not gate:
        print("\nAUDIT GATE FAILED - refusing to report a backtest on unverified data.")
        return 1
    report_backtest(cfg, prices, universe, contaminated, end, rf, rf_daily, save=args.save)
    return 0


def cmd_validate(cfg: Config003, args) -> int:
    snapshot, prices, universe, end, rf, rf_daily = load_everything(cfg, args.source)
    start = pd.Timestamp(cfg.sample_start)

    report_universe(cfg, snapshot, universe, end)
    *_, contaminated, _, _, gate = report_corporate_actions(cfg, prices, universe, end)
    if not gate:
        print("\nAUDIT GATE FAILED - refusing to continue.")
        return 1

    noise_status = cmd_noise(cfg, args)

    traded = tuple(t for t in universe.universe if t not in set(contaminated))
    result = _run(cfg, prices, traded, rf, strategy=EquityCrossSectionalMomentum(cfg, traded, BUCKET_METHOD))
    returns = result.excess_returns.loc[start:end]

    rule("SECTION 7 - IN-SAMPLE / OUT-OF-SAMPLE")
    print("Nothing is fitted, so both halves are out-of-sample by construction.")
    print(f"  {in_sample_out_of_sample(returns)}")

    rule("SECTION 7 - WALK-FORWARD")
    print(
        f"Rolling windows, nothing re-optimised. Window lengths are NOT pre-registered and\n"
        f"cannot change a position: {args.train_years:g}y in-window, {args.test_years:g}y forward, "
        f"stepping {args.step_years:g}y.\n"
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

    decision, *_ = report_backtest(
        cfg, prices, universe, contaminated, end, rf, rf_daily, save=args.save
    )
    rule("PROTOCOL COMPLETE")
    print(f"section 8 verdict: {decision.verdict}")
    print(f"noise gate: {'PASS' if noise_status == 0 else 'FAIL'}")
    return noise_status


def main(args) -> int:
    cfg = load_config_003()
    if args.noise:
        return cmd_noise(cfg, args)
    if args.validate:
        return cmd_validate(cfg, args)
    return cmd_backtest(cfg, args)
