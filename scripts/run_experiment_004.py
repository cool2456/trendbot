#!/usr/bin/env python3
"""PREREG_004.md section 7's protocol, point-in-time cross-sectional momentum.

Driven from ``scripts/run_backtest.py --experiment 004``.

Every reporting choice this file makes that PREREG_004.md does not fix is stated in the
output as it is made, not left to the reader to reverse-engineer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.config_004 import ACQUISITION, BANKRUPTCY, UNKNOWN, Config004, load_config_004
from trendbot.data import daily_risk_free, load_prices, load_risk_free_rate
from trendbot.engine.backtest import rebalance_dates
from trendbot.engine.eq_validation import evaluate_decision_rule_003, market_regression
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR, summarise
from trendbot.engine.panel import price_panel
from trendbot.engine.panel_backtest import run_panel_backtest
from trendbot.engine.validation import deflated_sharpe_ratio
from trendbot.engine.xs_validation import bucket_study, factor_attribution, panel_noise_test, worst_months
from trendbot.equities import load_market_proxy
from trendbot.pit_universe import (
    assemble_wide_panels,
    build_panels,
    build_pit_universe,
    build_security_master,
    resolve_permatickers,
    verify_delisting_returns,
)
from trendbot.regression import EXPERIMENT_001_NET_SHARPE
from trendbot.sharadar import COMMON_STOCK_CATEGORIES, SharadarClient, SharadarError
from trendbot.strategies import PointInTimeMomentum

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Reporting choices, not strategy parameters. Section 6's frozen list contains none of
# them and none can change a position.
DEFAULT_NOISE_SEEDS = 8
NOISE_UNIVERSE_SIZE = 60
FACTOR_SHARES = (0.25, 0.5)
WORST_MONTHS = 5
BUCKET_METHOD = "even"
# Per-period Sharpes of the three configurations already tried, for the four-trial
# deflation term. Outputs of 002 and 003, recorded in their findings documents.
EXPERIMENT_002_NET_SHARPE = 0.4631
EXPERIMENT_003_NET_SHARPE = 0.6988


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------------------
# the equal-weight point-in-time benchmark
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointInTimeEqualWeight:
    """Section 8's benchmark: equal weight across the whole point-in-time universe.

    Experiments 001-003 read "equal-weight buy-and-hold" literally - buy once, never
    trade - because their universes were fixed lists. That construction has no meaning
    here: a universe defined by a rule at each date has no single basket to buy once,
    and its members delist. The only coherent reading of "buy-and-hold of the same
    point-in-time universe" is *hold all of it, equally weighted, as the rule defines
    it each month*, which is what this does. Its turnover is forced by the universe
    rule rather than chosen by a strategy, so it is run at zero cost, exactly as the
    zero-turnover benchmarks of 001-003 were. The cost-paying variant is reported as a
    sensitivity so the choice is visible.
    """

    membership: pd.DataFrame

    @property
    def name(self) -> str:
        return "equal-weight point-in-time universe"

    def __call__(self, panel) -> pd.DataFrame:
        from trendbot.engine.panel import panel_field, validate_panel

        validate_panel(panel, require=("close",))
        close = panel_field(panel, "close")
        mask = self.membership.reindex(index=close.index, columns=close.columns)
        mask = mask.fillna(False).astype(bool)
        counts = mask.sum(axis=1)
        return mask.astype(float).div(counts.where(counts > 0), axis=0).fillna(0.0)


# --------------------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Everything one arm of section 10's paired diagnostic needs."""

    arm: str
    master: object
    panels_raw: dict
    universe: object
    prices: object
    applied_delistings: pd.DataFrame
    decision_bars: pd.DatetimeIndex
    calendar: pd.DatetimeIndex


def _decision_bars(calendar: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """The bar each rebalance decides on, and the rebalance bars themselves.

    Section 2 evaluates its rule "at each monthly rebalance date t" and section 5 fills
    at the open of the bar after the signal. Those reconcile only one way: rule and
    signal are both read off the close before the rebalance, and the trade happens at
    the next open. Reading the rule off the rebalance bar's own close while filling at
    that bar's open would need the close before the open.
    """
    rebalances = rebalance_dates(calendar)
    positions = {d: i for i, d in enumerate(calendar)}
    decisions = pd.DatetimeIndex([calendar[positions[d] - 1] for d in rebalances])
    return decisions, rebalances


def load_pipeline(
    cfg: Config004,
    *,
    arm: str,
    client: SharadarClient,
    unknown_override: float | None = None,
    rf: pd.Series | None = None,
    survivors_asof: pd.Timestamp | None = None,
) -> Pipeline:
    """Build one arm: (A) point-in-time with delistings, (B) survivors only.

    ``rf`` is the *annualised* risk-free series. It is aligned to the trading calendar
    here rather than passed in already aligned, because the calendar is only known once
    the price table has been read - and requiring an aligned series would force every
    caller to build the whole pipeline twice just to learn the dates.
    """
    tickers_table = client.security_master()
    actions_table = client.actions()
    master = build_security_master(tickers_table, actions_table)

    sep = client.bulk_prices()
    resolved = resolve_permatickers(sep, master)
    wide = assemble_wide_panels(resolved)
    calendar = wide["closeadj"].index
    rf_daily = None if rf is None else daily_risk_free(rf, calendar)

    eligible_ids = [int(p) for p in master.table.index if master.is_common_stock(int(p))]
    restrict = None
    if arm == "B":
        asof = survivors_asof or calendar[-1]
        restrict = sorted(master.survives_to(pd.Timestamp(asof)))

    decisions, _ = _decision_bars(calendar)
    universe = build_pit_universe(
        cfg,
        closeunadj=wide["closeunadj"],
        closeadj=wide["closeadj"],
        dollar_volume=wide["dollar_volume"],
        master=master,
        rebalances=decisions,
        eligible_ids=eligible_ids,
        restrict_to=restrict,
    )

    members = [str(p) for p in universe.all_members()]
    trimmed = {k: v[members] for k, v in wide.items()}
    prices, applied = build_panels(
        cfg,
        closeadj=trimmed["closeadj"],
        openadj=trimmed["openadj"],
        master=master,
        rf_daily=rf_daily,
        unknown_override=unknown_override,
    )
    return Pipeline(
        arm=arm,
        master=master,
        panels_raw=trimmed,
        universe=universe,
        prices=prices,
        applied_delistings=applied,
        decision_bars=decisions,
        calendar=calendar,
    )


# --------------------------------------------------------------------------------------
# step 1 - free-tier validation
# --------------------------------------------------------------------------------------


def cmd_free_tier(cfg: Config004, args) -> int:
    rule("STEP 1 - FREE-TIER PIPELINE VALIDATION (section 7.2)")
    print(
        "Section 7.2 requires ingestion, delisting handling and universe construction to be\n"
        "built and verified before any paid data is used. Two things are checked here: the\n"
        "offline fixture gate, which is the part that does not need a vendor at all, and a\n"
        "live round-trip against whatever the key can actually reach.\n"
    )
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_pit_universe.py"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    print("offline fixture gate (tests/test_pit_universe.py):")
    print("  " + proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "  no output")
    fixture_ok = proc.returncode == 0
    print(
        f"  -> {'PASS' if fixture_ok else 'FAIL'}: a synthetic delisting, a ticker change and a "
        "reused ticker all resolve by hand"
    )

    print("\nlive vendor round-trip:")
    try:
        client = SharadarClient.from_env()
        table = client.security_master(refresh=args.refresh)
        print(f"  security master: {len(table):,} securities")
        categories = table["category"].value_counts()
        common = int(table["category"].isin(COMMON_STOCK_CATEGORIES).sum())
        print(f"  common stock (section 2's inclusion list): {common:,}")
        print(f"  distinct categories: {len(categories)}")
        print(f"  delisted flag set on: {int(table['isdelisted'].sum()):,}")
        live_ok = True
    except SharadarError as exc:
        print(f"  UNAVAILABLE: {exc}")
        live_ok = False

    print(f"\nGATE: offline fixture {'PASS' if fixture_ok else 'FAIL'}; "
          f"live vendor {'reachable' if live_ok else 'UNREACHABLE'}")
    return 0 if fixture_ok else 1


# --------------------------------------------------------------------------------------
# steps 2 and 3 - universe construction and delisting audits
# --------------------------------------------------------------------------------------


def report_universe(cfg: Config004, pipeline: Pipeline) -> None:
    rule(f"STEP 2 - UNIVERSE CONSTRUCTION, ARM {pipeline.arm} (section 2)")
    universe = pipeline.universe
    master = pipeline.master
    print(
        f"rule: top {cfg.universe_size} US common stocks by median dollar volume over the "
        f"trailing\n{cfg.liquidity_window_days} trading days, unadjusted close >= "
        f"${cfg.price_floor:.2f}, at least {cfg.min_history_days} trading days of history,\n"
        "and not yet delisted. Evaluated on the decision bar before each rebalance."
    )
    print(f"\nrebalances evaluated: {len(universe.counts)}")
    print(f"securities ever in the universe: {len(universe.all_members()):,}")
    print(
        f"\nSECTION 6's REALISED START DATE: "
        f"{universe.start_date.date() if universe.start_date is not None else 'NEVER REACHED'}"
    )
    print(
        f"  (section 6 starts the sample at the first month the rule yields "
        f"{cfg.start_rule_min_names} qualifying names;\n   the date is whatever it is and is "
        "reported, not chosen)"
    )
    flow = universe.entries_and_exits()
    if universe.start_date is not None:
        inside = flow.loc[universe.start_date :]
        print(f"\nnames per rebalance inside the sample: min {int(inside['n'].min())}, "
              f"median {int(inside['n'].median())}, max {int(inside['n'].max())}")
        print(f"entering per rebalance: mean {inside['entered'].mean():.1f}, max {int(inside['entered'].max())}")
        print(f"leaving  per rebalance: mean {inside['left'].mean():.1f}, max {int(inside['left'].max())}")
        print("\nfirst and last few rebalances:")
        print(pd.concat([inside.head(4), inside.tail(4)]).to_string())

    reused = master.reused_tickers()
    print(f"\nticker strings carried by more than one security: {len(reused)}")
    if len(reused):
        print("  (reported, never merged - the panel is keyed on the permanent ID)")
        print(reused.head(8).to_string(index=False))

    # THE STEP 2 GATE
    violations = []
    for date in universe.membership.index:
        for permaticker in universe.members(date):
            delist = master.delist_date(permaticker)
            if delist is not None and date >= pd.Timestamp(delist):
                violations.append((date, permaticker))
    print(
        f"\nGATE: no security appears in the universe on or after its delisting date -> "
        f"{'PASS' if not violations else f'FAIL ({len(violations)} violations)'}"
    )
    if violations:
        for date, permaticker in violations[:10]:
            print(f"  {date.date()} {master.label(permaticker)} ({permaticker})")


def report_delistings(cfg: Config004, pipeline: Pipeline) -> pd.DataFrame:
    rule(f"STEP 3 - DELISTING TREATMENT, ARM {pipeline.arm} (section 3)")
    applied = pipeline.applied_delistings
    print(f"section 3's table: {cfg.delisting.describe()}\n")
    if applied.empty:
        print("no delistings among the securities that ever entered the universe.")
        return applied

    counts = applied.groupby("bucket").size().reindex([ACQUISITION, BANKRUPTCY, UNKNOWN]).fillna(0).astype(int)
    print("delistings by section 3 bucket, among universe members:")
    for bucket, n in counts.items():
        assigned = cfg.delisting.assigned_return(bucket)
        print(f"  {bucket:12s} {n:5d}   assigned return {assigned:+.0%}")
    print(f"  {'total':12s} {int(counts.sum()):5d}")

    print("\nvendor reasons behind them:")
    print(applied.groupby(["bucket", "raw_reason"]).size().rename("n").to_frame().to_string())

    print("\nby year:")
    print(applied.groupby(applied["delist_date"].dt.year).size().rename("n").to_frame().T.to_string())

    checked = verify_delisting_returns(pipeline.prices, applied)
    disagree = checked[~checked["agrees"]]
    print(
        f"\nGATE: the realised return on every delist bar equals section 3's assigned return "
        f"-> {'PASS' if disagree.empty else f'FAIL ({len(disagree)})'}"
    )
    if not disagree.empty:
        print(disagree.head(8).to_string(index=False))
    return checked


# --------------------------------------------------------------------------------------
# step 4 - noise tests
# --------------------------------------------------------------------------------------


def cmd_noise(cfg: Config004, args) -> int:
    rule("STEP 4 / SECTION 7.1 - NOISE TESTS, BOTH VARIANTS")
    print(
        "The strategy is run on synthetic data containing no signal. Sharpe is reported\n"
        "GROSS of costs as well as net: this rule replaces most of its book every month, so a\n"
        "net figure on driftless data is expected to sit below zero by the cost drag.\n"
    )
    print(
        f"Synthetic panels are {NOISE_UNIVERSE_SIZE} instruments wide with no delistings: the null\n"
        "being tested is a property of the RANKING RULE, and the point-in-time machinery has\n"
        "its own gates in steps 2 and 3. Every instrument is in the universe on every date,\n"
        f"so the decile cut still applies at {NOISE_UNIVERSE_SIZE // cfg.n_quantiles} names per bucket.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or DEFAULT_NOISE_SEEDS)))
    names = tuple(f"N{i:03d}" for i in range(NOISE_UNIVERSE_SIZE))

    from trendbot.engine.validation import synthetic_prices

    reference = synthetic_prices(names, seed=0, n_days=args.n_days)
    membership = pd.DataFrame(True, index=reference.close.index, columns=list(names))
    strategy = PointInTimeMomentum(cfg, membership, BUCKET_METHOD)

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
        "    r_i = beta_i * f + eps_i, betas spread linearly, f driftless, total volatility\n"
        "    held equal across instruments. This is the variant that diagnosed 002.\n"
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

    print("FACTOR ATTRIBUTION on the correlated null")
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


# --------------------------------------------------------------------------------------
# steps 5, 7, 8 - the backtest for one arm
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    stats: object
    benchmark_stats: object
    returns: pd.Series
    study: object
    regressions: dict
    turnover: float
    start: pd.Timestamp
    end: pd.Timestamp
    n_names: int


def run_arm(
    cfg: Config004,
    pipeline: Pipeline,
    *,
    rf,
    rf_daily,
    cost_bps: float | None = None,
    label: str = "",
) -> ArmResult:
    """Steps 5, 7 and 8 for one arm of section 10's pair."""
    universe = pipeline.universe
    if universe.start_date is None:
        raise ValueError("section 6's start rule was never satisfied; nothing to backtest")
    start = universe.start_date
    end = pipeline.prices.close.index[-1]
    members = [str(p) for p in universe.all_members()]

    membership = universe.membership.reindex(columns=members).fillna(False)
    # Membership is defined on decision bars only; every other bar is False, which is
    # correct - the engine reads the decision bar and nothing else.
    full = pd.DataFrame(False, index=pipeline.prices.close.index, columns=members)
    full.loc[membership.index, membership.columns] = membership.to_numpy()

    strategy = PointInTimeMomentum(cfg, full, BUCKET_METHOD)
    targets = strategy(price_panel(pipeline.prices, members))
    result = run_panel_backtest(
        pipeline.prices,
        members,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else cost_bps,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )
    stats = summarise(
        result.excess_returns.loc[start:end],
        result.weights.loc[start:end],
        trades=result.trades.loc[start:end] if not result.trades.empty else None,
    )

    benchmark = run_panel_backtest(
        pipeline.prices,
        members,
        PointInTimeEqualWeight(full),
        cost_bps=0.0,
        drift_band=None,
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )
    benchmark_stats = summarise(benchmark.excess_returns.loc[start:end])

    study = bucket_study(
        pipeline.prices,
        members,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method=BUCKET_METHOD,
        label_prefix="D",
        start=start,
        end=end,
    )

    returns = result.excess_returns.loc[start:end]
    proxy = load_market_proxy()
    if proxy.symbol != cfg.market_proxy_symbol:
        raise ValueError(
            f"section 8 measures alpha against {cfg.market_proxy_symbol} but the declared "
            f"market proxy is {proxy.symbol}. The document decides; the declaration only "
            "records the choice for the secondary regression."
        )
    market = load_prices((proxy.symbol,), source="yahoo").close[proxy.symbol].pct_change().fillna(0.0)
    market_excess = (market.reindex(returns.index) - rf_daily.reindex(returns.index)).dropna()
    regressions = {
        f"{proxy.symbol} (cap-weighted market)": market_regression(
            returns, market_excess, market_label=f"{proxy.symbol} (cap-weighted market)"
        ),
        "equal-weight point-in-time universe": market_regression(
            returns,
            benchmark.excess_returns.loc[start:end],
            market_label="equal-weight point-in-time universe",
        ),
    }
    return ArmResult(
        arm=pipeline.arm,
        stats=stats,
        benchmark_stats=benchmark_stats,
        returns=returns,
        study=study,
        regressions=regressions,
        turnover=stats.ann_turnover,
        start=start,
        end=end,
        n_names=len(members),
    )


# --------------------------------------------------------------------------------------
# steps 5-11 - the full protocol, both arms
# --------------------------------------------------------------------------------------


def _psr(cfg: Config004, returns: pd.Series):
    """PSR(0) and the deflated Sharpe at the cumulative counter of 4 configurations."""
    own = float(returns.mean()) / float(returns.std(ddof=1))
    root = math.sqrt(TRADING_DAYS_PER_YEAR)
    trials = (
        EXPERIMENT_001_NET_SHARPE / root,
        EXPERIMENT_002_NET_SHARPE / root,
        EXPERIMENT_003_NET_SHARPE / root,
        own,
    )
    return deflated_sharpe_ratio(returns, cfg.configurations_tried, trial_sharpes=trials), trials


def report_arm(cfg: Config004, arm: ArmResult, *, header: str) -> None:
    rule(header)
    print(f"window: {arm.start.date()} -> {arm.end.date()}   securities ever held: {arm.n_names:,}")
    print(f"\nstrategy (net of {cfg.cost_bps_per_side:g} bps/side, excess of T-bill):")
    print(f"  {arm.stats}")
    print("equal-weight point-in-time universe (excess of T-bill, no cost):")
    print(f"  {arm.benchmark_stats}")
    print(f"\nannualised one-way turnover: {arm.turnover:.2f}x")

    study = arm.study
    print(f"\n--- decile monotonicity ({study.n_quantiles} buckets, {len(study.returns)} rebalances) ---")
    print(study.table().round(5).to_string())
    steps = np.diff(study.mean_monthly.to_numpy())
    print(f"\n{study}")
    print(f"  step-by-step change D1->D10 (%/month): {np.round(steps * 100, 4).tolist()}")
    print(f"  adjacent inversions: {study.n_inversions} of {cfg.n_quantiles - 1} "
          f"(section 8 tolerates at most {cfg.max_inversions})")
    print(f"  D1-D10 spread: {study.spread_mean:+.4%} per month, t = {study.spread_t_stat:+.3f}, "
          f"n = {len(study.returns)}")

    print("\n--- market-beta attribution ---")
    for label, regression in arm.regressions.items():
        marker = "HEADLINE" if "cap-weighted" in label else "secondary"
        print(f"  [{marker}] {regression}")


def cmd_paired(cfg: Config004, args) -> int:
    """STEP 6 / SECTION 10 - the paired diagnostic. Required regardless of verdict."""
    client = SharadarClient.from_env()
    rf = load_risk_free_rate()

    arms: dict[str, ArmResult] = {}
    pipelines: dict[str, Pipeline] = {}
    for arm in ("A", "B"):
        pipeline = load_pipeline(cfg, arm=arm, client=client, rf=rf)
        rf_daily = daily_risk_free(rf, pipeline.prices.close.index)
        pipelines[arm] = pipeline
        report_universe(cfg, pipeline)
        report_delistings(cfg, pipeline)
        arms[arm] = run_arm(cfg, pipeline, rf=rf, rf_daily=rf_daily)

    for arm, label in (("A", "point-in-time, WITH delisted securities"),
                       ("B", "survivors-only, same rule")):
        report_arm(cfg, arms[arm], header=f"ARM {arm} - {label}")

    rule("STEP 6 / SECTION 10 - THE PAIRED DIAGNOSTIC, A MINUS B")
    print(
        "Identical strategy, identical code, identical dates. The A-B gap is the measured\n"
        "size of survivorship bias, and section 10 makes it a finding of this experiment\n"
        "whether or not the strategy passes.\n"
    )
    a, b = arms["A"], arms["B"]
    rows = [
        ("net Sharpe", a.stats.sharpe, b.stats.sharpe),
        ("benchmark Sharpe", a.benchmark_stats.sharpe, b.benchmark_stats.sharpe),
        ("margin over benchmark", a.stats.sharpe - a.benchmark_stats.sharpe,
         b.stats.sharpe - b.benchmark_stats.sharpe),
        ("CAGR (excess)", a.stats.cagr, b.stats.cagr),
        ("volatility", a.stats.ann_vol, b.stats.ann_vol),
        ("max drawdown", a.stats.max_drawdown, b.stats.max_drawdown),
        ("annual turnover", a.turnover, b.turnover),
        ("decile inversions", float(a.study.n_inversions), float(b.study.n_inversions)),
        ("D1-D10 spread /month", a.study.spread_mean, b.study.spread_mean),
        ("D1-D10 t-statistic", a.study.spread_t_stat, b.study.spread_t_stat),
    ]
    for label, regressions in (("alpha to market /yr", "cap-weighted"),):
        key_a = next(k for k in a.regressions if regressions in k)
        key_b = next(k for k in b.regressions if regressions in k)
        rows.append((label, a.regressions[key_a].alpha_annualised, b.regressions[key_b].alpha_annualised))
        rows.append(("alpha t-statistic", a.regressions[key_a].alpha_t_stat,
                     b.regressions[key_b].alpha_t_stat))

    table = pd.DataFrame(
        [{"metric": m, "A (point-in-time)": av, "B (survivors)": bv, "A - B": av - bv}
         for m, av, bv in rows]
    ).set_index("metric")
    print(table.round(4).to_string())
    print(
        f"\n  ==> SURVIVORSHIP BIAS, MEASURED: arm B's net Sharpe exceeds arm A's by "
        f"{b.stats.sharpe - a.stats.sharpe:+.3f}."
    )
    print(
        "      Section 10: if (B) reproduces something close to 003's inverted gradient while\n"
        f"      (A) does not, that confirms the 003 diagnosis directly. A: {a.study.n_inversions} "
        f"inversions, spread t {a.study.spread_t_stat:+.2f}; "
        f"B: {b.study.n_inversions} inversions, spread t {b.study.spread_t_stat:+.2f}."
    )
    return 0


def cmd_validate(cfg: Config004, args) -> int:
    """The whole of section 7, in order."""
    status = cmd_free_tier(cfg, args)
    if status != 0:
        print("\nSTEP 1 GATE FAILED - refusing to continue.")
        return status
    noise = cmd_noise(cfg, args)

    client = SharadarClient.from_env()
    rf = load_risk_free_rate()
    pipeline = load_pipeline(cfg, arm="A", client=client, rf=rf)
    rf_daily = daily_risk_free(rf, pipeline.prices.close.index)

    report_universe(cfg, pipeline)
    report_delistings(cfg, pipeline)
    arm = run_arm(cfg, pipeline, rf=rf, rf_daily=rf_daily)
    report_arm(cfg, arm, header="STEPS 5, 7, 8 - ARM A (point-in-time, with delistings)")

    rule("SECTION 3 - DELISTING SENSITIVITY LADDER (mandatory)")
    print("The unknown/OTC bucket moved across section 3's three points; nothing else moves.\n")
    ladder = []
    for override in cfg.delisting.sensitivity_returns:
        alt = load_pipeline(cfg, arm="A", client=client, rf=rf, unknown_override=override)
        result = run_arm(cfg, alt, rf=rf, rf_daily=rf_daily)
        ladder.append({
            "unknown bucket": f"{override:+.0%}",
            "Sharpe": result.stats.sharpe,
            "CAGR": result.stats.cagr,
            "vs benchmark": result.stats.sharpe - result.benchmark_stats.sharpe,
        })
    print(pd.DataFrame(ladder).set_index("unknown bucket").round(4).to_string())

    rule("SECTION 5 - COST SENSITIVITY LADDER")
    costs = []
    for bps in cfg.cost_sensitivity_bps:
        result = run_arm(cfg, pipeline, rf=rf, rf_daily=rf_daily, cost_bps=bps)
        costs.append({
            "cost bps/side": bps,
            "Sharpe": result.stats.sharpe,
            "CAGR": result.stats.cagr,
            "vs benchmark": result.stats.sharpe - result.benchmark_stats.sharpe,
        })
    print(pd.DataFrame(costs).set_index("cost bps/side").round(4).to_string())

    rule("STEP 9 - WORST MONTHS (section 9's mandatory momentum crashes)")
    monthly = arm.returns.groupby(arm.returns.index.to_period("M")).apply(
        lambda x: float((1 + x).prod() - 1)
    )
    worst = worst_months(arm.returns, WORST_MONTHS)
    print(f"the {WORST_MONTHS} worst months by net excess return:")
    print(worst.round(4).to_string())
    worst_set = {str(p) for p in worst.index}
    missing = [m for m in cfg.required_crash_months if m not in worst_set]
    print("\nthe months section 9 names, and where they rank:")
    for month in cfg.required_crash_months:
        key = pd.Period(month, freq="M")
        if key not in monthly.index:
            print(f"  {month}: not in the sample")
            continue
        value = monthly.loc[key]
        rank = int((monthly < value).sum()) + 1
        print(f"  {month}: {value:+.2%}  ->  rank {rank}/{len(monthly)}")

    valid = not missing
    if missing:
        print(
            f"\n  *** SECTION 9 VALIDITY DETERMINATION: {', '.join(missing)} absent from the\n"
            f"      {WORST_MONTHS} worst months. This check has now failed a THIRD time, on a\n"
            "      point-in-time universe. Section 9: 'the implementation is wrong, not the\n"
            "      anomaly, and the result must be reported as INVALID rather than as a\n"
            "      verdict.' No verdict is issued below. ***"
        )
    else:
        print(f"\n  section 9's named months all appear among the {WORST_MONTHS} worst: check PASSES.")

    rule(f"STEP 10 - PSR / DEFLATED SHARPE, configs_tried = {cfg.configurations_tried}")
    dsr, trials = _psr(cfg, arm.returns)
    print("per-period Sharpes of the four configurations tried: "
          + ", ".join(f"{t:.6f}" for t in trials))
    print(f"  {dsr}")
    print(f"\nPSR(0) = {dsr.psr_vs_zero:.4f}  ->  "
          f"{'CLEARS' if dsr.psr_vs_zero > 0.95 else 'does NOT clear'} 95%")

    rule("STEP 11 - SECTION 8's PRE-COMMITTED DECISION RULE")
    if not valid:
        print(
            "NOT ISSUED. Section 9 makes a third failure of the momentum-crash check a\n"
            "statement about the implementation rather than about the strategy, and a verdict\n"
            "computed on a faulty implementation is not a verdict."
        )
    else:
        headline = next(r for k, r in arm.regressions.items() if "cap-weighted" in k)
        decision = evaluate_decision_rule_003(
            cfg,
            strategy_sharpe=arm.stats.sharpe,
            benchmark_sharpe=arm.benchmark_stats.sharpe,
            n_inversions=arm.study.n_inversions,
            spread_mean=arm.study.spread_mean,
            spread_t_stat=arm.study.spread_t_stat,
            alpha_annualised=headline.alpha_annualised,
            alpha_t_stat=headline.alpha_t_stat,
        )
        print(decision)
    return noise


def main(args) -> int:
    cfg = load_config_004()
    if args.free_tier:
        return cmd_free_tier(cfg, args)
    if args.noise:
        return cmd_noise(cfg, args)
    try:
        if args.paired:
            return cmd_paired(cfg, args)
        return cmd_validate(cfg, args)
    except SharadarError as exc:
        rule("BLOCKED - VENDOR DATA UNAVAILABLE")
        print(
            "Steps 2, 3 and 5-11 read the Sharadar SEP price table, which this key cannot\n"
            f"currently reach:\n\n  {exc}\n\n"
            "Nothing is estimated or substituted in its place. Step 1 (--free-tier) and step 4\n"
            "(--noise) do not need it and are unaffected."
        )
        return 2
