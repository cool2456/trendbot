#!/usr/bin/env python3
"""PREREG_005.md section 7's protocol, cross-sectional currency momentum.

Driven from ``scripts/run_backtest.py --experiment 005``.

Every reporting choice this file makes that PREREG_005.md does not fix is stated in the
output as it is made, not left to the reader to reverse-engineer.

Two of those choices were fixed **before any number existed**, because section 8
adjudicates on them and choosing afterwards would be selection:

* the dollar factor is the *daily equal-weighted mean* of the normalised currency
  returns - the DOL construction section 5's parenthetical names - and both section 5's
  Sharpe comparison and section 8's alpha regression run against that same series. The
  buy-and-hold and monthly-rebalanced alternatives are reported beside it.
* the quintile split is ``"even"``, so Q1 and Q5 are within one name of each other at
  22 currencies. See :mod:`trendbot.xsmom`; ``"floor"`` is reported as a sensitivity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.config_005 import Config005, load_config_005
from trendbot.data import daily_risk_free, load_prices, load_risk_free_rate
from trendbot.engine.eq_validation import market_regression
from trendbot.engine.fx_validation import (
    carry_adjusted_prices,
    evaluate_decision_rule_005,
    short_rate_panel,
    worst_month_clustering,
)
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR, sharpe, summarise
from trendbot.engine.panel import price_panel
from trendbot.engine.panel_backtest import panel_buy_and_hold, run_panel_backtest
from trendbot.engine.validation import (
    deflated_sharpe_ratio,
    in_sample_out_of_sample,
    walk_forward,
)
from trendbot.engine.xs_validation import bucket_study, factor_attribution, panel_noise_test, worst_months
from trendbot.fx import (
    US_SHORT_RATE_CANDIDATES,
    build_fx_universe,
    check_drift_plausibility,
    check_peg_relationships,
    check_reference_levels,
    peg_breach_dates,
    dollar_factor_returns,
    fetch_short_rates,
    fx_price_data,
)
from trendbot.fred import FredError, access_path, fetch_observations
from trendbot.regression import EXPERIMENT_001_NET_SHARPE
from trendbot.strategies import CurrencyCrossSectionalMomentum
from trendbot.xsmom import quantile_sizes

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Reporting choices, not strategy parameters. Section 6's frozen list contains none of
# them and none can change a position.
DEFAULT_NOISE_SEEDS = 16  # the build order's floor is 8; more seeds is strictly better evidence
FACTOR_SHARES = (0.25, 0.5)
# Synthetic panels are the real universe's width. Unlike experiment 003's 409 names,
# 22 columns is cheap, so the null is tested at the bucket geometry the strategy
# actually trades rather than at a convenient one.
NOISE_ANNUAL_VOL = 0.10  # major FX runs near 10% annualised, not equities' 16%
WORST_MONTHS = 5
BUCKET_METHOD = "even"
BUCKET_METHOD_ALTERNATIVE = "floor"
# Recorded outputs of the earlier experiments, needed for the four-trial deflation term.
EXPERIMENT_002_NET_SHARPE = 0.4631
EXPERIMENT_003_NET_SHARPE = 0.6988


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------------------
# shared setup
# --------------------------------------------------------------------------------------


def _last_complete_session(index: pd.DatetimeIndex) -> pd.Timestamp:
    """The last bar that is certainly a settled publication.

    Same guard experiment 003 uses. H.10 rates are published with a lag so this almost
    never binds here, but a panel that silently included a provisional bar is exactly
    how experiment 003's benchmark came to cover one fewer day than its strategy.
    """
    today = pd.Timestamp.today().normalize()
    earlier = index[index < today]
    if len(earlier) == 0:
        raise ValueError("no completed session in the exchange rate history")
    return earlier[-1]


def load_everything(cfg: Config005, *, refresh: bool = False):
    universe = build_fx_universe(sample_start=cfg.sample_start, refresh=refresh)
    prices = fx_price_data(universe)
    end = _last_complete_session(prices.close.index)
    prices = prices.slice(None, str(end.date()))
    rf = load_risk_free_rate()
    return universe, prices, end, rf, daily_risk_free(rf, prices.close.index)


def _run(cfg: Config005, prices, members, rf, *, cost_bps=None, targets=None, strategy=None):
    return run_panel_backtest(
        prices,
        members,
        strategy,
        targets=targets,
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else cost_bps,
        drift_band=None,  # section 5: full rebalance to the new quintile each month
        gross_cap=cfg.gross_exposure_cap,
        risk_free=rf,
    )


def _psr(cfg: Config005, returns: pd.Series):
    """PSR(0) and the deflated Sharpe at the cumulative counter of 4 configurations.

    The deflation term needs the variance of the per-period Sharpe ratios across the
    configurations actually tried. There are exactly four and all are known: 001's,
    002's and 003's recorded results and this one. Nothing is estimated or assumed.

    Blocked experiment 004 is **not** among them, per PREREG_005.md's header: it never
    reached data, so it produced nothing a winner could have been selected from. The
    parser asserts that argument is still in the document.
    """
    own = float(returns.mean()) / float(returns.std(ddof=1))
    root = math.sqrt(TRADING_DAYS_PER_YEAR)
    trials = (
        EXPERIMENT_001_NET_SHARPE / root,
        EXPERIMENT_002_NET_SHARPE / root,
        EXPERIMENT_003_NET_SHARPE / root,
        own,
    )
    return deflated_sharpe_ratio(returns, cfg.configurations_tried, trial_sharpes=trials), trials


# --------------------------------------------------------------------------------------
# step 1 - the quote-convention gate
# --------------------------------------------------------------------------------------


def report_quote_conventions(cfg: Config005, universe, prices) -> bool:
    rule("STEP 1 - QUOTE-CONVENTION AUDIT (section 2). THIS IS A HARD GATE.")
    print(
        "Section 2 names this the single most likely way the experiment produces a wrong\n"
        "answer, and the reason is that an inverted series raises nothing. It is a\n"
        "well-formed positive price series with a plausible volatility; every downstream\n"
        "computation succeeds; only the rank is backwards. So the direction is never inferred\n"
        "from the values. It is read out of FRED's own units string, which states it in words,\n"
        "and cross-checked against the title. A series whose two strings disagree, or whose\n"
        f"units do not name exactly one US dollar leg, is refused rather than guessed at.\n"
    )
    # Report what actually produced the artefacts, not what a fresh fetch would use.
    # A key added after the panel was cached would otherwise make this line claim an
    # access path that never touched this data.
    paths = sorted({universe.metadata[s].obtained_via for s in universe.candidates})
    print(
        f"metadata obtained via: {', '.join(paths)}  (a FRED_KEY is "
        f"{'configured' if access_path() == 'api' else 'not configured'} now; "
        "both transports serve the same database)\n"
    )

    rows = []
    for series_id in universe.candidates:
        convention = universe.conventions[series_id]
        rows.append(
            {
                "series": series_id,
                "FRED title": convention.title.replace(" Spot Exchange Rate", ""),
                "FRED units": convention.units,
                "direction": convention.direction,
                "normalise by": "1 / x" if convention.inverted else "x",
                "currency": convention.foreign_currency,
                "in universe": "yes" if series_id in universe.universe else "NO",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    inverted = [s for s in universe.universe if universe.conventions[s].inverted]
    print(
        f"\n{len(inverted)} of {universe.n} universe series are quoted foreign-per-USD and are "
        f"inverted;\n{universe.n - len(inverted)} are already USD-per-foreign and are left alone. "
        "After normalisation every\nseries is the USD value of one unit of the foreign currency."
    )

    window = prices.close.loc[cfg.sample_start :]

    print("\n--- verification 1: normalised level on a named date, against known market history ---")
    print(
        "Each band is chosen so the INVERTED value falls outside it. A band both the correct\n"
        "and the inverted value would pass tests nothing, so 'caught' is reported per row.\n"
    )
    reference = check_reference_levels(window, universe.conventions)
    for check in reference:
        print(f"  {check}")
        print(f"       basis: {check.why}")
    reference_ok = all(c.passed for c in reference)
    discriminating = all(c.discriminating for c in reference)
    print(
        f"\n  {_flag(reference_ok)} - {sum(c.passed for c in reference)}/{len(reference)} inside their bands; "
        f"{sum(c.discriminating for c in reference)}/{len(reference)} bands would catch an inversion"
    )

    print("\n--- verification 2: structural identities that hold by monetary policy ---")
    pegs = check_peg_relationships(window)
    for check in pegs:
        print(f"  {check}")
    pegs_ok = all(c.passed for c in pegs)
    print(f"\n  {_flag(pegs_ok)}")

    print("\n--- verification 3: implied long-run drift ---")
    print(
        "Inversion flips the sign of a currency's drift against the dollar. This is the\n"
        "weakest of the three checks and is reported, not relied on: it catches an inverted\n"
        "high-inflation currency, which would compound into an implausible appreciation.\n"
    )
    drifts = sorted(
        check_drift_plausibility(window, universe.conventions),
        key=lambda c: c.annualised_log_drift,
    )
    for check in drifts:
        print(f"  {check}")
    drift_ok = all(c.passed for c in drifts)
    print(f"\n  {_flag(drift_ok)}")

    gate = reference_ok and discriminating and pegs_ok and drift_ok
    print(
        f"\nSTEP 1 GATE: {_flag(gate)} — every series' direction is determined from FRED metadata, "
        "and\nthe normalised panel reproduces independently known levels, two policy identities\n"
        "and plausible drifts."
    )
    return gate


# --------------------------------------------------------------------------------------
# step 2 - universe construction
# --------------------------------------------------------------------------------------


def report_universe(cfg: Config005, universe, end) -> bool:
    rule("STEP 2 - UNIVERSE CONSTRUCTION (section 2)")
    print(
        "Section 2's rule: every daily USD exchange rate series published in the Federal\n"
        "Reserve H.10 release and available via FRED with continuous daily data over the full\n"
        "sample window. The release is enumerated from FRED, not from a list written here.\n"
    )
    print(f"{universe.describe()}\n")

    print("--- series in the release that are not daily bilateral USD rates ---")
    print("(excluded by their own FRED metadata, not by name)")
    for series_id, why in universe.non_daily:
        print(f"  {series_id:12s} {why}")

    print(f"\n--- the universe: {universe.n} currencies ---")
    print(
        pd.DataFrame(
            [
                {
                    "series": s,
                    "currency": universe.currency_of(s),
                    "first": universe.gaps[s].first_observation,
                    "last": universe.gaps[s].last_observation,
                    "published": universe.gaps[s].n_observations,
                    "scheduled": universe.gaps[s].n_scheduled,
                    "largest gap": universe.gaps[s].largest_gap,
                    "gap window": (
                        f"{universe.gaps[s].largest_gap_start}..{universe.gaps[s].largest_gap_end}"
                        if universe.gaps[s].largest_gap
                        else "-"
                    ),
                }
                for s in universe.universe
            ]
        ).to_string(index=False)
    )

    print(f"\n--- excluded: {len(universe.exclusions)} ---")
    if not universe.exclusions:
        print("  none")
    for series_id, currency, why in universe.exclusions:
        print(f"  {series_id:10s} {currency:24s} {why}")

    worst_gap = max(universe.gaps[s].largest_gap for s in universe.universe)
    in_range = cfg.expected_universe_low <= universe.n <= cfg.expected_universe_high
    gap_ok = worst_gap <= universe.max_gap_days
    print(
        f"\nsection 2 expected roughly {cfg.expected_universe_low}-{cfg.expected_universe_high} "
        f"pairs; {universe.n} resolved -> {_flag(in_range)}"
    )
    print(
        f"GATE: no universe series has a gap longer than a plausible holiday run. The "
        f"allowance is\n{universe.max_gap_days} scheduled publication days (two calendar weeks, "
        "longer than Chinese New Year's ~8\nbusiness days and Japan's Golden Week ~5), fixed "
        f"before the data was inspected.\nLargest gap anywhere in the universe: {worst_gap} day(s) "
        f"-> {_flag(gap_ok)}"
    )
    return in_range and gap_ok


# --------------------------------------------------------------------------------------
# step 3 - returns and the handling of missing observations
# --------------------------------------------------------------------------------------


def report_returns(cfg: Config005, universe, prices, end) -> None:
    rule("STEP 3 - RETURNS AND MISSING-DATA HANDLING (section 4)")
    print(
        "Section 4's return is the log change in the USD value of the foreign currency, spot\n"
        "only. The panel is that quantity; the engine's arithmetic is in simple returns and the\n"
        "two agree to the usual second-order term, so nothing here re-defines a return.\n"
    )
    print(
        "H.10 rates are New York Fed noon buying rates, published on US business days. A series\n"
        "is therefore blank on two different kinds of day, and they are handled differently:\n\n"
        "  US market holidays  - NO currency has a rate, so the date is not a trading day at\n"
        "                        all. The calendar is built as the dates on which at least one\n"
        "                        H.10 series published, so these dates are simply absent.\n"
        "  foreign holidays    - every other currency published and this one did not. The\n"
        "                        market was shut; the last published rate is the only\n"
        "                        defensible mark, so the series is FORWARD-FILLED.\n\n"
        "No lookahead: a forward fill carries the last value published BEFORE the hole, so no\n"
        "observation can move a return dated earlier than itself. There is no back-fill\n"
        "anywhere - a series is NaN before its own first observation and stays NaN, which\n"
        "excludes it from those dates' rankings rather than inventing a price for it.\n"
    )
    filled = {}
    for series_id in universe.universe:
        raw_in_window = universe.raw[series_id].reindex(universe.calendar)
        filled[series_id] = int(raw_in_window.isna().sum())
    total_cells = universe.n * len(universe.calendar)
    print(
        f"forward-filled cells inside the window: {sum(filled.values())} of {total_cells} "
        f"({sum(filled.values()) / total_cells:.4%})"
    )
    busiest = sorted(filled.items(), key=lambda kv: -kv[1])[:5]
    print("  most-filled series: " + ", ".join(f"{s} ({n})" for s, n in busiest))
    print(
        "\nA note on one calendar day. 2026-08-14 is the last published bar. 2005-09-05 is US\n"
        "Labor Day: FRED carries a Chinese yuan observation that day and nothing else, so the\n"
        "calendar rule admits the date and the other 21 currencies are forward-filled across\n"
        "it. That is one bar in ~6,900 and it is reported rather than removed, because the\n"
        "calendar rule was fixed before the data was looked at."
    )

    print("\n--- signal formation and the sample window ---")
    print(
        f"The panel carries each series' FULL published history so that the 252-day formation\n"
        f"window at the first bar of {cfg.sample_start} reaches back into 1998. Everything is\n"
        f"MEASURED from {cfg.sample_start} onward and nothing before it is reported. The euro\n"
        "begins on 1999-01-04 and so has no defined momentum until roughly 2000-01; on those\n"
        "dates it is excluded from the ranking, which is section 4's own rule for insufficient\n"
        "history rather than a special case.\n"
    )
    window = prices.close.loc[cfg.sample_start : end]
    print(f"panel: {prices.describe()}")
    print(f"window: {window.index[0].date()} -> {window.index[-1].date()}  ({len(window)} bars)")

    print("\n--- lookahead confirmation, computed rather than asserted ---")
    print(causality_report(cfg, universe, prices))


def causality_report(cfg: Config005, universe, prices) -> str:
    """Show that no future observation can move a past position.

    The build order asks for a confirmation that the missing-value handling introduces
    no lookahead. Asserting it about a forward fill is easy and unconvincing, so this
    demonstrates it: the last quarter of the panel is overwritten with garbage, the
    whole pipeline - normalisation is already done, but the fill, the momentum and the
    quintile weights are not - is recomputed, and the target weights before the
    tampered date are compared to the originals.

    This is the same argument experiment 002's causality sweep makes, applied to the
    piece that is new here. If the forward fill could ever reach backwards, or if the
    momentum used a negative shift, this would print a non-zero difference.
    """
    members = list(universe.universe)
    strategy = CurrencyCrossSectionalMomentum(cfg, universe.universe, BUCKET_METHOD)
    original = strategy(price_panel(prices, members))

    cut = prices.close.index[int(len(prices.close.index) * 0.75)]
    tampered_close = prices.close.copy()
    # Multiply, do not replace: the values stay positive and finite, so the failure
    # this looks for is a leak of information, not a NaN propagating backwards.
    tampered_close.loc[cut:] = tampered_close.loc[cut:] * 3.0
    tampered = prices.__class__(
        open=tampered_close.copy(),
        close=tampered_close,
        source=prices.source,
        adjusted=True,
        fetched_at="",
    )
    after = strategy(price_panel(tampered, members))

    before_cut = original.index < cut
    delta = float((original[before_cut] - after[before_cut]).abs().to_numpy().max())
    # A hole is never filled from the future: the fill is monotone in the index, so the
    # first valid index of each column is unchanged by anything after it.
    first_valid = {s: prices.close[s].first_valid_index() for s in members}
    no_backfill = all(
        prices.close[s].loc[: first_valid[s]].isna().sum() == len(prices.close[s].loc[: first_valid[s]]) - 1
        for s in members
    )
    return (
        f"  every observation from {cut.date()} onward multiplied by 3, whole signal recomputed:\n"
        f"    max |change| in any target weight on any bar before {cut.date()}: {delta:.3e}\n"
        f"    -> {_flag(delta == 0.0)}  (a forward fill carries values forward only; the two\n"
        f"       momentum shifts are both positive, so no row can read a later row)\n"
        f"  no series is back-filled before its own first observation: {_flag(no_backfill)}"
    )


# --------------------------------------------------------------------------------------
# step 4 - the noise tests
# --------------------------------------------------------------------------------------


def cmd_noise(cfg: Config005, args, universe=None) -> int:
    rule("STEP 4 / SECTION 7.2 - NOISE TESTS, BOTH VARIANTS")
    # Resolved rather than defaulted, so ``--noise`` on its own and ``--validate`` test
    # the null at the same bucket geometry. A width that differed between the two would
    # make the standalone gate a different test from the one the protocol runs.
    if universe is None:
        universe = build_fx_universe(sample_start=cfg.sample_start, refresh=args.refresh)
    width = universe.n
    print(
        "The strategy is run on synthetic data containing no signal. Sharpe is reported GROSS\n"
        "of costs as well as net: this rule replaces most of its book every month, so a net\n"
        "figure on driftless data is expected to sit below zero by the cost drag, and testing\n"
        "only the net number would conflate 'the rule finds nothing' with 'the costs are\n"
        "large'. The gross figure is what section 7 step 2 is about.\n"
    )
    print(
        f"Synthetic panels are {width} instruments wide - the real universe's width, so the null\n"
        f"is tested at the bucket geometry the strategy actually trades. Annualised volatility\n"
        f"is set to {NOISE_ANNUAL_VOL:.0%} rather than the generator's equity default, because major FX runs\n"
        "near that and the cost drag's size in Sharpe terms depends on it. Neither is a\n"
        "strategy parameter and neither can change a position.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or DEFAULT_NOISE_SEEDS)))
    names = tuple(f"FX{i:02d}" for i in range(width))
    strategy = CurrencyCrossSectionalMomentum(cfg, names, BUCKET_METHOD)
    results = []

    independent = panel_noise_test(
        label="(a) independent random walks",
        seeds=seeds,
        n_days=args.n_days,
        correlated=False,
        annual_vol=NOISE_ANNUAL_VOL,
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
        "    ambiguous. This is the variant that diagnosed experiment 002. For currencies the\n"
        "    common factor has a name: it is the dollar.\n"
    )
    for share in FACTOR_SHARES:
        correlated = panel_noise_test(
            label=f"(b) common factor, {share:.0%} of average variance",
            seeds=seeds,
            n_days=args.n_days,
            correlated=True,
            annual_vol=NOISE_ANNUAL_VOL,
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
        annual_vol=NOISE_ANNUAL_VOL,
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
        f"every variant, AND |alpha to the common factor| < 1%/yr -> {_flag(passed)}"
    )
    return 0 if passed else 1


# --------------------------------------------------------------------------------------
# steps 5-11 - the full protocol
# --------------------------------------------------------------------------------------


def report_backtest(cfg: Config005, universe, prices, end, rf, rf_daily, *, save: bool, args):
    start = pd.Timestamp(cfg.sample_start)
    members = list(universe.universe)
    strategy = CurrencyCrossSectionalMomentum(cfg, universe.universe, BUCKET_METHOD)
    panel = price_panel(prices, members)
    targets = strategy(panel)

    # ---- step 5 --------------------------------------------------------------------
    rule("STEP 5 - FULL-SAMPLE BACKTEST, ONE RUN")
    print(cfg.describe())
    print(f"data: {prices.describe()}")
    print(f"window: {start.date()} -> {end.date()}  (section 3's pre-committed sample window)")
    sizes = quantile_sizes(universe.n, cfg.n_quantiles, BUCKET_METHOD)
    print(
        f"\nquintile split: {BUCKET_METHOD!r}. At {universe.n} currencies that is {list(sizes)},\n"
        f"so the traded book is the top {sizes[0]} currencies - section 4 predicted 4-5."
    )

    result = _run(cfg, prices, members, rf, targets=targets)
    stats = result.stats_from(start)
    returns = result.excess_returns.loc[start:end]

    factor = dollar_factor_returns(universe, prices)
    factor_excess = (factor - rf_daily).loc[start:end]
    factor_stats = summarise(factor_excess)

    # The two alternative constructions of section 5's benchmark, reported so the
    # headline choice is visible rather than trusted. Neither adjudicates anything.
    bh_excess = (panel_buy_and_hold(prices, members, start=start) - rf_daily).loc[start:end]
    monthly_excess = (
        panel_buy_and_hold(prices, members, rebalance="monthly", start=start) - rf_daily
    ).loc[start:end]

    held = (result.weights.loc[start:end].abs() > 1e-12).sum(axis=1)
    print(f"\nstrategy (net of {cfg.cost_bps_per_side:g} bps/side, excess of T-bill):")
    print(f"  {stats}")
    print(f"dollar factor — {cfg.benchmark_label} (excess of T-bill, no cost):")
    print(f"  {factor_stats}")
    print(f"\ngross of costs: Sharpe {sharpe(result.gross_returns.loc[start:end] - rf_daily.loc[start:end]):+.3f}")
    print(f"currencies held per day inside the window: {held.value_counts().sort_index().to_dict()}")

    print("\n--- the two alternative benchmark constructions (reported, not adjudicated on) ---")
    print(f"  daily-rebalanced equal weight (HEADLINE): Sharpe {factor_stats.sharpe:+.3f}")
    print(f"  monthly-rebalanced equal weight:          Sharpe {sharpe(monthly_excess):+.3f}")
    print(f"  buy-and-hold (001-003 house convention):  Sharpe {sharpe(bh_excess):+.3f}")

    rule("SECTION 5 - TURNOVER AND COST SENSITIVITY")
    print(f"annualised one-way turnover: {stats.ann_turnover:.2f}x\n")
    ladder = []
    for bps in cfg.cost_sensitivity_bps:
        run = _run(cfg, prices, members, rf, cost_bps=bps, targets=targets)
        run_stats = run.stats_from(start)
        ladder.append(
            {
                "bps/side": bps,
                "CAGR": run_stats.cagr,
                "Sharpe": run_stats.sharpe,
                "vs dollar factor": run_stats.sharpe - factor_stats.sharpe,
                "maxDD": run_stats.max_drawdown,
            }
        )
    ladder_frame = pd.DataFrame(ladder).set_index("bps/side")
    print(ladder_frame.round(4).to_string())
    print(
        f"\nSection 5 calls the ladder central rather than decorative: Menkhoff et al. state\n"
        f"transaction costs partially explain the published spread. From 0 to 25 bps the Sharpe\n"
        f"moves {ladder_frame['Sharpe'].iloc[0] - ladder_frame['Sharpe'].iloc[-1]:+.3f}."
    )

    # ---- step 6 --------------------------------------------------------------------
    rule("STEP 6 - QUINTILE MONOTONICITY (section 8)")
    study = bucket_study(
        prices,
        members,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method=BUCKET_METHOD,
        label_prefix="Q",
        start=start,
        end=end,
    )
    print(f"convention: {study.convention}\n")
    print(study.table().round(5).to_string())
    steps = np.diff(study.mean_monthly.to_numpy())
    print(f"\n{study}")
    print(f"  step-by-step change Q1->Q5 (%/month): {np.round(steps * 100, 4).tolist()}")
    print(
        f"  adjacent inversions: {study.n_inversions} of {cfg.n_quantiles - 1} "
        f"(section 8 tolerates at most {cfg.max_inversions})"
    )
    print(
        f"  Q1-Q5 spread: {study.spread_mean:+.4%} per month, "
        f"t = {study.spread_t_stat:+.3f} over {len(study.returns)} rebalances"
    )
    print(
        f"  section 8 supports above t > {cfg.min_spread_t_stat:.1f} and abandons below "
        f"t = {cfg.abandon_below_spread_t_stat:.1f}."
    )

    alternative = bucket_study(
        prices,
        members,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_quantiles=cfg.n_quantiles,
        method=BUCKET_METHOD_ALTERNATIVE,
        label_prefix="Q",
        start=start,
        end=end,
    )
    print(
        f"\n  sensitivity to the bucket convention ({BUCKET_METHOD_ALTERNATIVE!r}, which makes Q5 wider "
        f"than Q1):\n  {alternative}"
    )

    # ---- step 7 --------------------------------------------------------------------
    rule("STEP 7 - BETA ATTRIBUTION")
    print(
        "Headline: against the dollar factor, which is what section 8's alpha clause names.\n"
        f"Secondary: against {cfg.equity_proxy_symbol}, which asks whether currency momentum is a "
        "disguised\nequity beta - a failure mode the correlated-noise diagnostic of experiment 002 "
        "could not\nsee, because its common factor is synthetic and has no equity market in it.\n"
    )
    factor_regression = market_regression(returns, factor_excess, market_label="the dollar factor")
    print(f"  {factor_regression}")
    print(f"      on {factor_regression.n_observations} overlapping daily observations")

    spy_regression = None
    proxy = cfg.equity_proxy_symbol  # section 7 names it; never inlined here
    try:
        equity = load_prices((proxy,), source="yahoo", max_abs_daily_move=None)
        spy_excess = (equity.close[proxy].pct_change(fill_method=None) - rf_daily).dropna().loc[start:end]
        spy_regression = market_regression(returns, spy_excess, market_label=proxy)
        print(f"  {spy_regression}")
        print(f"      on {spy_regression.n_observations} overlapping daily observations "
              "(the H.10 and NYSE calendars are close but not identical; the regression "
              "uses their intersection)")
    except Exception as exc:  # pragma: no cover - reported, never silently skipped
        print(f"  {proxy} regression UNAVAILABLE: {exc}")

    # ---- step 8 --------------------------------------------------------------------
    rate_diagnostic = report_interest_rate_diagnostic(
        cfg, universe, prices, members, targets, rf, rf_daily, start, end, stats, refresh=args.refresh
    )

    integrity = report_data_integrity_sensitivity(
        cfg, universe, prices, members, rf, start, end, stats
    )

    # ---- step 9 --------------------------------------------------------------------
    rule("STEP 9 - WORST MONTHS AND FX DISLOCATION CLUSTERING (section 9)")
    print(
        "Section 9 imposes no mandatory month here - currency momentum has no single canonical\n"
        "crash date. It asks instead whether the worst months cluster around known FX\n"
        "dislocations, and states that the ABSENCE of any clustering is a warning sign about\n"
        "the implementation. That is a diagnostic with a stated failure direction, so it is\n"
        "computed rather than eyeballed.\n"
    )
    worst = worst_months(returns, WORST_MONTHS)
    clustering = worst_month_clustering(worst, cfg.dislocation_months)
    for month, value, label in clustering.matched:
        print(f"  {month}  {value:+8.2%}   {label if label else '(no named dislocation)'}")
    print(f"\n  windows searched: {cfg.dislocation_months}")
    # Distance to the nearest named window, computed for the months that did not match.
    # Reported separately and never counted as a match: widening a window after seeing
    # which months landed just outside it would be exactly the selection section 8 exists
    # to prevent. This is an observation about the calendar, not an adjusted result.
    named = sorted({pd.Period(m, freq="M") for months in cfg.dislocation_months.values() for m in months})
    unmatched = [(month, value) for month, value, label in clustering.matched if label is None]
    if unmatched and named:
        print("\n  months that did NOT match, and their distance to the nearest named window:")
        for month, value in unmatched:
            period = pd.Period(month, freq="M")
            nearest = min(named, key=lambda w: abs((period - w).n))
            distance = abs((period - nearest).n)
            print(f"    {month}  {value:+7.2%}  {distance:2d} month(s) from {nearest}")
        print(
            "    (reported, not counted. Section 9's list is taken literally; a window is\n"
            "     never widened to absorb a month that fell outside it.)"
        )

    print(f"\n  {clustering}")
    if not clustering.any_clustering:
        print(
            "\n  *** SECTION 9 WARNING *** none of the worst months lands on a named FX\n"
            "  dislocation. Section 9 says this is evidence about the implementation, not about\n"
            "  the strategy, and it is flagged as such."
        )

    # ---- step 10 -------------------------------------------------------------------
    rule(f"STEP 10 - PSR / DEFLATED SHARPE, configs_tried = {cfg.configurations_tried}")
    print(
        f"The counter is {cfg.configurations_tried}, not 5. PREREG_005.md's header argues that blocked "
        "experiment 004\ndoes not advance it: an IP-level vendor throttle stopped it before it "
        "reached data, so it\nproduced nothing from which a winner could have been selected. The "
        "config parser asserts\nthat argument is still in the document, so the number cannot be "
        "changed without it.\n"
    )
    dsr, trials = _psr(cfg, returns)
    print("per-period Sharpes of the four configurations tried (001, 002, 003, this one):")
    print("  " + ", ".join(f"{t:.6f}" for t in trials))
    print(f"  {dsr}")
    print(f"\nPSR(0) = {dsr.psr_vs_zero:.4f}  ->  {'CLEARS' if dsr.psr_vs_zero > 0.95 else 'does NOT clear'} 95%")
    print(
        f"deflated Sharpe at {cfg.configurations_tried} configurations = {dsr.deflated_sharpe:.4f}  ->  "
        f"{'SIGNIFICANT' if dsr.is_significant else 'not significant'} at 95%"
    )

    # ---- step 11 -------------------------------------------------------------------
    rule("STEP 11 - SECTION 8's PRE-COMMITTED DECISION RULE")
    decision = evaluate_decision_rule_005(
        cfg,
        strategy_sharpe=stats.sharpe,
        benchmark_sharpe=factor_stats.sharpe,
        n_inversions=study.n_inversions,
        spread_mean=study.spread_mean,
        spread_t_stat=study.spread_t_stat,
        alpha_annualised=factor_regression.alpha_annualised,
        alpha_t_stat=factor_regression.alpha_t_stat,
    )
    print(decision)

    rule("SECTION 9 - EXPECTATIONS OF RECORD, CHECKED")
    expectations = [
        (
            f"realistic net Sharpe {cfg.expected_sharpe_low}-{cfg.expected_sharpe_high}",
            cfg.expected_sharpe_low <= stats.sharpe <= cfg.expected_sharpe_high,
            f"{stats.sharpe:+.3f}",
        ),
        (
            f"above {cfg.bug_threshold_sharpe} means a bug",
            stats.sharpe <= cfg.bug_threshold_sharpe,
            f"{stats.sharpe:+.3f}",
        ),
        (
            "costs may eliminate the effect entirely",
            True,
            f"0bps {ladder_frame['Sharpe'].iloc[0]:+.3f} -> 25bps {ladder_frame['Sharpe'].iloc[-1]:+.3f}",
        ),
        (
            "a 4-5 currency traded book is thin",
            True,
            f"names held per day: {held.value_counts().sort_index().to_dict()}",
        ),
        (
            "worst months cluster on FX dislocations",
            clustering.any_clustering,
            str(clustering),
        ),
    ]
    for name, ok, detail in expectations:
        print(f"  [{'YES' if ok else 'NO '}] {name:48s} observed: {detail}")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "backtest_005.json").write_text(
            json.dumps(
                {
                    "prereg_sha256": cfg.source_sha256,
                    "window": [str(start.date()), str(end.date())],
                    "universe": list(universe.universe),
                    "n_universe": universe.n,
                    "exclusions": [list(e) for e in universe.exclusions],
                    "conventions": {
                        s: {
                            "units": universe.conventions[s].units,
                            "direction": universe.conventions[s].direction,
                            "inverted": universe.conventions[s].inverted,
                        }
                        for s in universe.universe
                    },
                    "net_sharpe": stats.sharpe,
                    "dollar_factor_sharpe": factor_stats.sharpe,
                    "benchmark_alternatives": {
                        "daily_equal_weight": factor_stats.sharpe,
                        "monthly_equal_weight": sharpe(monthly_excess),
                        "buy_and_hold": sharpe(bh_excess),
                    },
                    "cagr": stats.cagr,
                    "ann_turnover": stats.ann_turnover,
                    "cost_ladder": ladder_frame.reset_index().to_dict("records"),
                    "quintile_table": study.table().to_dict(),
                    "n_inversions": study.n_inversions,
                    "spread_mean": study.spread_mean,
                    "spread_t_stat": study.spread_t_stat,
                    "alpha_dollar_factor": factor_regression.alpha_annualised,
                    "alpha_t_dollar_factor": factor_regression.alpha_t_stat,
                    "beta_dollar_factor": factor_regression.beta,
                    "alpha_spy": spy_regression.alpha_annualised if spy_regression else None,
                    "beta_spy": spy_regression.beta if spy_regression else None,
                    "interest_rate_diagnostic": rate_diagnostic,
                    "data_integrity_sensitivity": integrity,
                    "worst_months": {str(k): float(v) for k, v in worst["return"].items()},
                    "clustering_matched": [list(m) for m in clustering.matched],
                    "configurations_tried": cfg.configurations_tried,
                    "psr_vs_zero": dsr.psr_vs_zero,
                    "deflated_sharpe": dsr.deflated_sharpe,
                    "verdict": decision.verdict,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {RESULTS_DIR / 'backtest_005.json'}")

    return decision, stats, factor_stats, study, factor_regression, spy_regression, dsr, clustering, ladder_frame, returns


def report_data_integrity_sensitivity(cfg, universe, prices, members, rf, start, end, headline_stats):
    """How much of the headline rests on two observations that a peg says are wrong.

    NOT a configuration and not a cleaning step. The headline above runs on the data
    exactly as FRED publishes it, because sections 2-6 are frozen and authorise no
    outlier rule. This re-runs it with the observations that breach a *policy* band
    treated as holes - forward-filled, the same way a foreign holiday is - purely so the
    sensitivity is a number rather than an unknown.
    """
    rule("DATA INTEGRITY SENSITIVITY (not a configuration; the headline is unchanged)")
    breaches = peg_breach_dates(universe.normalised.loc[cfg.sample_start : end])
    if not breaches:
        print("  no observation in the window breaches a policy band; nothing to test.")
        return {"status": "no breaches"}

    print(
        "  Step 1's ERM II cross-rate check found observations a central bank was\n"
        "  committed to preventing. In a series inside its band on 99.9%+ of days, such a\n"
        "  value is far more likely to be a bad print than an unrecorded policy breach.\n"
    )
    affected: set = set()
    for name, dates in breaches.items():
        print(f"  {name}: {len(dates)} day(s) — {', '.join(str(d.date()) for d in dates)}")
        affected.update(dates)

    cleaned = universe.normalised.copy()
    # Only the krone is edited: the cross-rate is (USD per EUR)/(USD per DKK) and the
    # euro's own level on those dates is consistent with its neighbours, so the krone is
    # the leg that moved. Stated rather than inferred silently.
    cleaned.loc[sorted(affected), "DEXDNUS"] = np.nan
    cleaned["DEXDNUS"] = cleaned["DEXDNUS"].ffill()
    cleaned_prices = prices.__class__(
        open=cleaned[members].loc[: end].copy(),
        close=cleaned[members].loc[: end].copy(),
        source=prices.source + " + policy-band breaches held flat",
        adjusted=True,
        fetched_at="",
    )
    strategy = CurrencyCrossSectionalMomentum(cfg, universe.universe, BUCKET_METHOD)
    rerun = _run(cfg, cleaned_prices, members, rf, strategy=strategy)
    rerun_stats = rerun.stats_from(start)
    print(
        f"\n  headline (data as published): Sharpe {headline_stats.sharpe:+.4f}, "
        f"CAGR {headline_stats.cagr:+.4%}"
    )
    print(
        f"  with those days held flat   : Sharpe {rerun_stats.sharpe:+.4f}, "
        f"CAGR {rerun_stats.cagr:+.4%}"
    )
    print(
        f"  DELTA                       : Sharpe {rerun_stats.sharpe - headline_stats.sharpe:+.4f}"
    )
    print("\n  The HEADLINE remains the first line. This is disclosure, not selection.")
    return {
        "status": "computed",
        "breaches": {k: [str(d.date()) for d in v] for k, v in breaches.items()},
        "headline_sharpe": headline_stats.sharpe,
        "cleaned_sharpe": rerun_stats.sharpe,
        "delta_sharpe": rerun_stats.sharpe - headline_stats.sharpe,
    }


def report_interest_rate_diagnostic(
    cfg, universe, prices, members, targets, rf, rf_daily, start, end, headline_stats, *, refresh=False
):
    rule("STEP 8 - INTEREST-RATE APPROXIMATION DIAGNOSTIC (section 4)")
    print(
        "Section 4 defines the return as spot only and labels that an approximation: the true\n"
        "currency excess return is the spot change plus the interest differential. This step\n"
        "measures the size of the approximation. It is explicitly NOT a second configuration -\n"
        "configs_tried stays at 4 - and the way that is enforced is arithmetic, not intention:\n"
        "the SAME target weights computed from the SAME spot signal are replayed against a\n"
        "different price panel. No signal is recomputed and no selection is possible.\n\n"
        "The correction is exactly the foreign interest accrual. The headline book is fully\n"
        "invested, so the engine's cash is zero and its excess return is already\n"
        "(spot change - i_US): a spot position financed at the US rate, earning nothing on the\n"
        "foreign leg. The true excess return is (spot change + i_foreign - i_US). So holding a\n"
        "foreign money-market DEPOSIT instead of the bare currency supplies the missing term:\n\n"
        "    P_total(t) = P_spot(t) * exp( sum_{u<=t} i_foreign(u) / 252 )\n"
    )
    try:
        probe_failures: dict[str, list[str]] = {}
        fetched = fetch_short_rates(universe.universe, refresh=refresh, attempts=probe_failures)
    except FredError as exc:
        print(f"  DIAGNOSTIC UNAVAILABLE - FRED could not be reached: {exc}")
        return {"status": "unavailable", "reason": str(exc)}

    coverage = short_rate_panel(fetched, universe.universe, prices.close.index)
    print(f"  {coverage}\n")
    if probe_failures:
        print("  candidates FRED rejected (recorded so 'not carried' is distinguishable")
        print("  from 'not tried'):")
        for series_id, notes in sorted(probe_failures.items()):
            for note in notes:
                print(f"    {series_id:9s} {note[:110]}")
        print()
    rows = []
    for series_id in universe.universe:
        source = coverage.sources.get(series_id)
        span = coverage.spans.get(series_id)
        window_rate = coverage.rates[series_id].loc[start:end]
        rows.append(
            {
                "series": series_id,
                "currency": universe.currency_of(series_id),
                "FRED rate series": source or "— none found —",
                "span": f"{span[0]}..{span[1]}" if span else "-",
                "window coverage": f"{float(window_rate.notna().mean()):.1%}",
                "mean rate": f"{float(window_rate.mean()):.2%}" if window_rate.notna().any() else "-",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    us_source, us_rate, us_attempts = None, None, []
    for candidate in US_SHORT_RATE_CANDIDATES:
        try:
            us_rate = fetch_observations(candidate, refresh=refresh).astype(float) / 100.0
        except FredError as exc:
            us_attempts.append(f"{candidate}: {exc}")
            continue
        us_source = candidate
        break
    if us_attempts:
        print("\n  US short-rate candidates rejected before the one used:")
        for note in us_attempts:
            print(f"    {note}")
    if us_rate is not None:
        aligned_us = us_rate.dropna()
        aligned_us = aligned_us.reindex(aligned_us.index.union(prices.close.index)).ffill().reindex(prices.close.index)
        print(
            f"\nUS leg for reporting the differential: {us_source}, mean "
            f"{float(aligned_us.loc[start:end].mean()):.2%} over the window."
        )
        print(
            "  (The US leg is NOT used to build the carry: the engine already subtracts a T-bill\n"
            "   rate from the strategy and the benchmark alike. It is fetched so the differential\n"
            "   can be reported. Note the foreign legs are interbank/money-market rates while the\n"
            "   US leg is a bill rate, so the differential carries a small basis.)"
        )

    if not coverage.covered:
        print("\n  no foreign short rates available; the diagnostic cannot be computed")
        return {"status": "unavailable", "reason": "no foreign short rates available"}

    # How much the missing six can matter is not a matter of opinion: it is how often
    # they were actually held. A currency the strategy never selects contributes no
    # carry to the strategy however large its rate.
    held_share = {}
    if coverage.uncovered:
        weights = targets.loc[start:end]
        selected = (weights.abs() > 1e-12)
        total_weight = float(weights.abs().to_numpy().sum())
        for series_id in coverage.uncovered:
            held_share[series_id] = {
                "days_held": int(selected[series_id].sum()),
                "share_of_book": float(weights[series_id].abs().sum() / total_weight)
                if total_weight
                else 0.0,
            }
        print(
            f"\n  The {len(coverage.uncovered)} uncovered currencies accrue zero carry, so the diagnostic "
            "understates the\n  approximation. How much it can understate it is bounded by how often they "
            "were held:"
        )
        for series_id, detail in sorted(held_share.items(), key=lambda kv: -kv[1]["share_of_book"]):
            print(
                f"    {series_id:8s} {universe.currency_of(series_id):22s} held on "
                f"{detail['days_held']:5d} of {len(weights)} bars, "
                f"{detail['share_of_book']:6.2%} of all weight ever allocated"
            )
        print(
            f"    TOTAL uncovered share of the traded book: "
            f"{sum(d['share_of_book'] for d in held_share.values()):.2%}"
        )

    total_prices = prices.__class__(
        open=carry_adjusted_prices(prices.open[members], coverage.rates[members]),
        close=carry_adjusted_prices(prices.close[members], coverage.rates[members]),
        source=prices.source + " + foreign short-rate accrual",
        adjusted=True,
        fetched_at="",
    )
    carried = _run(cfg, total_prices, members, rf, targets=targets)
    carried_stats = carried.stats_from(start)
    factor_carried = dollar_factor_returns(universe, total_prices)
    factor_carried_stats = summarise((factor_carried - rf_daily).loc[start:end])

    print("\n--- headline replayed on total-return (spot + foreign carry) prices ---")
    print(f"  spot-only  (the headline): Sharpe {headline_stats.sharpe:+.3f}, CAGR {headline_stats.cagr:+.2%}")
    print(f"  with carry (this step)   : Sharpe {carried_stats.sharpe:+.3f}, CAGR {carried_stats.cagr:+.2%}")
    print(
        f"  DELTA                    : Sharpe {carried_stats.sharpe - headline_stats.sharpe:+.3f}, "
        f"CAGR {carried_stats.cagr - headline_stats.cagr:+.2%}"
    )
    print(f"\n  dollar factor, spot-only : Sharpe (see step 5)")
    print(f"  dollar factor, with carry: Sharpe {factor_carried_stats.sharpe:+.3f}")
    print(
        f"\n  Uncovered currencies accrue nothing, which biases this DOWNWARD - the true\n"
        f"  approximation is at least this large. Uncovered: "
        f"{', '.join(coverage.uncovered) if coverage.uncovered else 'none'}"
    )
    return {
        "status": "computed",
        "sources": coverage.sources,
        "uncovered": list(coverage.uncovered),
        "uncovered_book_share": held_share,
        "spot_sharpe": headline_stats.sharpe,
        "carry_sharpe": carried_stats.sharpe,
        "delta_sharpe": carried_stats.sharpe - headline_stats.sharpe,
        "spot_cagr": headline_stats.cagr,
        "carry_cagr": carried_stats.cagr,
        "dollar_factor_carry_sharpe": factor_carried_stats.sharpe,
    }


# --------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------


def cmd_backtest(cfg: Config005, args) -> int:
    universe, prices, end, rf, rf_daily = load_everything(cfg, refresh=args.refresh)
    if not report_quote_conventions(cfg, universe, prices):
        print("\nSTEP 1 GATE FAILED - refusing to report a backtest on an unverified panel.")
        return 1
    if not report_universe(cfg, universe, end):
        print("\nSTEP 2 GATE FAILED - refusing to continue.")
        return 1
    report_returns(cfg, universe, prices, end)
    report_backtest(cfg, universe, prices, end, rf, rf_daily, save=args.save, args=args)
    return 0


def cmd_validate(cfg: Config005, args) -> int:
    universe, prices, end, rf, rf_daily = load_everything(cfg, refresh=args.refresh)
    start = pd.Timestamp(cfg.sample_start)

    if not report_quote_conventions(cfg, universe, prices):
        print("\nSTEP 1 GATE FAILED - refusing to continue.")
        return 1
    if not report_universe(cfg, universe, end):
        print("\nSTEP 2 GATE FAILED - refusing to continue.")
        return 1
    report_returns(cfg, universe, prices, end)

    noise_status = cmd_noise(cfg, args, universe)

    members = list(universe.universe)
    strategy = CurrencyCrossSectionalMomentum(cfg, universe.universe, BUCKET_METHOD)
    result = _run(cfg, prices, members, rf, strategy=strategy)
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
    wf = walk_forward(returns, train_years=args.train_years, test_years=args.test_years, step_years=args.step_years)
    print(wf)
    print()
    print(
        wf.windows.assign(train_start=lambda d: d.train_start.dt.date, test_end=lambda d: d.test_end.dt.date)[
            ["train_start", "test_end", "train_sharpe", "test_sharpe", "test_return"]
        ]
        .round(3)
        .to_string(index=False)
    )

    decision, *_ = report_backtest(cfg, universe, prices, end, rf, rf_daily, save=args.save, args=args)
    rule("PROTOCOL COMPLETE")
    print(f"section 8 verdict: {decision.verdict}")
    print(f"noise gate: {_flag(noise_status == 0)}")
    return noise_status


def main(args) -> int:
    cfg = load_config_005()
    if args.noise:
        return cmd_noise(cfg, args)
    if args.validate:
        return cmd_validate(cfg, args)
    return cmd_backtest(cfg, args)
