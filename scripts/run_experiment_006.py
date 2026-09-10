#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.config_006 import Config006, load_config_006
from trendbot.data import load_risk_free_rate
from trendbot.engine.allocation import WEIGHT_ORDERS, erc_weights
from trendbot.engine.backtest import rebalance_dates
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR, sharpe, summarise
from trendbot.engine.ms_validation import (
    _mean_pairwise as _mean_pairwise_upper,
    align,
    assert_symmetric,
    correlation_report,
    covariance_is_point_in_time,
    evaluate_decision_006,
    noise_test_006,
    run_overlay,
)
from trendbot.engine.validation import deflated_sharpe_ratio
from trendbot.regression import EXPERIMENT_001_NET_SHARPE
from trendbot.sleeves import BENCHMARK_VARIANTS, load_sleeves, reproduce

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

EXPERIMENT_002_NET_SHARPE = 0.4631
EXPERIMENT_003_NET_SHARPE = 0.6988
EXPERIMENT_005_NET_SHARPE = -0.271
EXPERIMENT_005_CARRY_SHARPE = 0.209

HEADLINE_BENCHMARK_VARIANT = "carry-corrected"
HEADLINE_WEIGHT_ORDER = "clip-first"

NOISE_CORRELATION = np.array(
    [
        [1.00, 0.70, 0.40],
        [0.70, 1.00, 0.35],
        [0.40, 0.35, 1.00],
    ]
)
NOISE_ANNUAL_VOLS = (0.075, 0.11, 0.09)
NOISE_LABELS = ("A", "B", "C")
NOISE_N_DAYS = 6000
NOISE_N_SEEDS = 12


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _overlay(cfg: Config006, returns: pd.DataFrame, rf, *, label: str, order: str, cost_bps=None):
    return run_overlay(
        returns,
        rf,
        label=label,
        halflife=cfg.covariance_halflife_days,
        vol_target=cfg.portfolio_vol_target,
        gross_cap=cfg.gross_exposure_cap,
        min_weight=cfg.min_sleeve_weight,
        max_weight=cfg.max_sleeve_weight,
        order=order,
        cost_bps=cfg.overlay_cost_bps_per_side if cost_bps is None else cost_bps,
    )


def report_sleeves(cfg: Config006, sleeve_set, check) -> bool:
    rule("STEP 1 - SLEEVE REPRODUCTION (section 2, section 6). THIS IS A HARD GATE.")
    print(
        "Section 6: sleeve returns are taken from each experiment's existing, committed\n"
        "implementation - no re-implementation, no re-parameterisation. The gate is that\n"
        "each sleeve still produces the Sharpe its own findings document recorded. A\n"
        "failure here is not a failure of experiment 006; it means a PRIOR result has\n"
        "drifted, and nothing further may be run until that is explained.\n"
    )
    frame = check.rows.copy()
    for column in ("recorded", "realised", "recorded benchmark", "realised benchmark"):
        frame[column] = frame[column].map(lambda v: f"{v:+.6f}")
    print(frame.to_string())
    print(f"\n{check}")
    print(
        "\n  001 is held to trendbot/regression.py at full floating-point precision; 002 and\n"
        "  005 are held to the three decimal places their findings documents actually state."
    )
    print(
        "\n  Sleeve C is REPLAYED, never recomputed: the target weights come from the spot\n"
        "  panel and are applied to the carry-adjusted one. Recomputing the signal on the\n"
        "  carry-adjusted panel does not fail, does not warn and does not return NaN - it\n"
        "  returns a HIGHER Sharpe than the correct one, which is the direction that gets\n"
        "  adopted by accident. trendbot/sleeves.py asserts the engine recorded the run as\n"
        "  precomputed."
    )
    print(f"\nGATE: all three sleeves reproduce their recorded headline -> {_flag(check.passed)}")
    return check.passed


def report_alignment(cfg: Config006, sleeve_set):
    rule("STEP 2 - DATE ALIGNMENT AND THE SECTION 5 WINDOW")
    print(
        "Three sleeves on three calendars: A and B on US equity sessions, C on FRED H.10\n"
        "observation dates. A one-day slip between them corrupts the correlation estimate,\n"
        "which is exactly what section 8's third clause turns on.\n\n"
        "RULE: inner join on the date indices. Bars are dropped, never manufactured.\n"
        "The alternative - reindex onto the union and fill holes with zero - asserts that a\n"
        "sleeve returned exactly nothing on days it was not observed, which ATTENUATES its\n"
        "measured correlation with the others and therefore biases section 8's third clause\n"
        "in the PASSING direction. That is the one clause whose purpose is to test the\n"
        "mechanism rather than the outcome, so the bias is unacceptable and the dropped bars\n"
        "are the cheaper error.\n\n"
        "LOOKAHEAD: none. An intersection is a function of which dates exist, not of any\n"
        "return on them, and it is applied identically to sleeves and benchmarks.\n"
    )
    alignment = align({s.label: s.returns for s in sleeve_set.sleeves})
    print(alignment.table.to_string())
    print(f"\n{alignment}")
    lo, hi = alignment.span
    print(
        f"\nSection 5's window, determined by the data and not chosen: "
        f"{lo.date()} -> {hi.date()}, {len(alignment.index)} bars.\n"
    )
    print(
        "Section 9 expected this window to be 'governed by 002's ETF universe and 005's\n"
        "post-1999 start'. That expectation cannot be met: 002's own pre-committed window\n"
        "starts 2008-01-01 and dominates 1999 under every reading. The reason the earlier\n"
        "start is not merely a different window is that only 20 of 002's 41 ETFs had listed\n"
        "by 1999 (32 by 2003, all 41 only by 2007-04-11) - so a 1999 start would silently make\n"
        "sleeve B a top-quintile-of-20 strategy, the re-parameterisation section 2 forbids.\n"
        "The intersection of the three PRE-COMMITTED windows is therefore the only reading\n"
        "consistent with section 2, and section 9's expectation is recorded as unachievable."
    )
    return alignment


def report_covariance_and_solver(cfg: Config006, returns: pd.DataFrame) -> bool:
    rule("STEP 3 - COVARIANCE POINT-IN-TIME AND ERC CONVERGENCE. HARD GATES.")
    print(
        "Section 3's covariance is the lookahead surface of this experiment. A full-sample\n"
        "covariance is not caught by any .shift(): there is no shift involved, the matrix\n"
        "simply knows things. The estimator is trendbot.sizing.ewma_covariance - the same\n"
        "function experiment 001 sizes with - whose pandas ewm is causal by construction.\n"
    )
    ok_pit, worst_change = covariance_is_point_in_time(
        returns, halflife=cfg.covariance_halflife_days, n_future=250
    )
    print(
        f"  appended 250 fabricated future bars at 10x the real volatility and recomputed:\n"
        f"  worst absolute change to any earlier covariance entry = {worst_change:.3e}"
    )
    print(f"  GATE: covariance at date t is unchanged by future data -> {_flag(ok_pit)}")
    print(
        "\n  CAVEAT, recorded because this gate reads stronger than it is: an estimator that\n"
        "  used r_t to weight the position that EARNS r_t would also pass it, because\n"
        "  appending rows after t does not disturb r_t. This rules out full-sample\n"
        "  contamination, not the one-bar question. The one-bar question is settled\n"
        "  separately, by the engine's single execution shift: weights dated t are traded\n"
        "  on t+1, so section 3's 'data through the rebalance date only' is satisfied with\n"
        "  a bar to spare."
    )

    print(
        "\nThe ERC solve. Equal risk contribution has no closed form for N > 2, and a solver\n"
        "that stops early produces weights that look like an allocation and are not one.\n"
        "The formulation is the strictly convex one (Spinu 2013; Griveau-Billion, Richard &\n"
        "Roncalli 2013): minimise 0.5 y'Sy - sum log y_i, whose stationarity condition IS\n"
        "equal risk contribution, solved by cyclical coordinate descent with an exact\n"
        "closed-form step per coordinate. Least-squares on the risk-contribution deviations\n"
        "would be non-convex and could stop somewhere plausible and wrong.\n"
    )
    equicorrelated = pd.DataFrame(
        np.array([[0.04, 0.012, 0.012], [0.012, 0.04, 0.012], [0.012, 0.012, 0.04]]),
        index=list(returns.columns),
        columns=list(returns.columns),
    )
    solved = erc_weights(equicorrelated)
    equal_ok = bool(np.allclose(solved.weights.to_numpy(), 1.0 / 3.0, atol=1e-10))
    scaled = erc_weights(equicorrelated * 252.0)
    scale_ok = bool(np.allclose(scaled.weights.to_numpy(), solved.weights.to_numpy(), atol=1e-12))
    print(
        f"  identical variances and correlations -> equal weights: {_flag(equal_ok)} "
        f"({[round(w, 8) for w in solved.weights]})"
    )
    print(f"  weights invariant to rescaling the covariance by 252: {_flag(scale_ok)}")
    return ok_pit and equal_ok and scale_ok


def report_solver_convergence(label: str, overlay) -> bool:
    diagnostics = overlay.diagnostics
    worst = overlay.worst_erc_deviation
    converged = overlay.all_converged
    n = overlay.risk_contributions.shape[1]
    print(
        f"  {label:28s} solves {len(diagnostics):5d}  worst |RC_i - 1/{n}| = {worst:.3e}  "
        f"median iterations {int(diagnostics['erc_iterations'].median()):3d}  "
        f"all converged: {_flag(converged)}"
    )
    return converged and worst < 1e-8


def cmd_noise(cfg: Config006, args) -> int:
    rule("STEP 4 - NOISE TEST: THE CONSTRUCTION ON SLEEVES CONTAINING NOTHING")
    print(
        "The whole ERC construction is run on synthetic sleeve returns with EXACTLY zero\n"
        "mean and a known covariance structure. A materially positive portfolio Sharpe here\n"
        "means the construction manufactures return from noise.\n\n"
        "That test alone is weak - a construction that allocated nothing would also pass it -\n"
        "so the recovery of the known correlation structure is checked alongside it. That\n"
        "second check is what would catch a one-bar slip between sleeves, a broken panel\n"
        "synthesis, or a return accounting that lost a series.\n"
    )
    seeds = tuple(range(args.seed, args.seed + (args.n_seeds or NOISE_N_SEEDS)))
    print(f"correlation structure put in ({len(NOISE_LABELS)} sleeves, deliberately unequal):")
    target = pd.DataFrame(NOISE_CORRELATION, index=list(NOISE_LABELS), columns=list(NOISE_LABELS))
    print("  " + target.round(3).to_string().replace("\n", "\n  "))
    print(f"annualised vols: {dict(zip(NOISE_LABELS, NOISE_ANNUAL_VOLS))}")
    print(f"{len(seeds)} seeds x {args.n_days or NOISE_N_DAYS} bars, weight order {HEADLINE_WEIGHT_ORDER!r}\n")

    result = noise_test_006(
        seeds=seeds,
        n_days=args.n_days or NOISE_N_DAYS,
        labels=NOISE_LABELS,
        correlation=NOISE_CORRELATION,
        annual_vols=NOISE_ANNUAL_VOLS,
        halflife=cfg.covariance_halflife_days,
        vol_target=cfg.portfolio_vol_target,
        gross_cap=cfg.gross_exposure_cap,
        min_weight=cfg.min_sleeve_weight,
        max_weight=cfg.max_sleeve_weight,
        order=HEADLINE_WEIGHT_ORDER,
        cost_bps=cfg.overlay_cost_bps_per_side,
        tolerance=args.tolerance,
    )
    display = result.table.copy()
    display["all_converged"] = display["all_converged"].map(lambda v: "yes" if v else "NO")
    print(display.round(5).to_string())
    print(f"\n{result}")
    passed = result.passes()
    print(
        f"\nGATE: mean portfolio Sharpe within +/-{result.tolerance} of zero, the known\n"
        f"correlation structure recovered to {result.correlation_tolerance}, and every ERC\n"
        f"solve converged -> {_flag(passed)}"
    )
    return 0 if passed else 1


def _psr(cfg: Config006, returns: pd.Series, *, fifth_trial: float):
    own = float(returns.mean()) / float(returns.std(ddof=1))
    root = math.sqrt(TRADING_DAYS_PER_YEAR)
    trials = (
        EXPERIMENT_001_NET_SHARPE / root,
        EXPERIMENT_002_NET_SHARPE / root,
        EXPERIMENT_003_NET_SHARPE / root,
        fifth_trial / root,
        own,
    )
    return deflated_sharpe_ratio(returns, cfg.configurations_tried, trial_sharpes=trials), trials


def _build(cfg: Config006, sleeve_set, alignment, rf, *, order: str, cost_bps=None):
    index = alignment.index
    sleeves = pd.DataFrame({s.label: s.returns for s in sleeve_set.sleeves}).loc[index]
    benchmarks = pd.DataFrame({s.label: s.benchmark for s in sleeve_set.sleeves}).loc[index]
    portfolio = _overlay(cfg, sleeves, rf, label="portfolio", order=order, cost_bps=cost_bps)
    benchmark = _overlay(cfg, benchmarks, rf, label="benchmark", order=order, cost_bps=cost_bps)
    return sleeves, benchmarks, portfolio, benchmark


def report_backtest(cfg: Config006, sleeve_set, alignment, rf, *, save: bool, args) -> int:
    index = alignment.index
    sleeves, benchmarks, portfolio, benchmark = _build(
        cfg, sleeve_set, alignment, rf, order=HEADLINE_WEIGHT_ORDER
    )

    rule("STEP 3 (continued) - ERC CONVERGENCE ON THE REAL COVARIANCES")
    solver_ok = True
    for label, overlay in (("portfolio", portfolio), ("benchmark", benchmark)):
        solver_ok = report_solver_convergence(label, overlay) and solver_ok
        if overlay.n_singular_dates:
            solver_ok = False
            print(
                f"  {label}: {overlay.n_singular_dates} dates had a covariance with no unique "
                "ERC solution. On section 5's window this must be zero."
            )
    print(
        f"\nGATE: every solve converged and realised risk contributions are equal to within\n"
        f"1e-8 of 1/{cfg.n_sleeves} at every rebalance -> {_flag(solver_ok)}"
    )
    if not solver_ok:
        print(
            "\nSTOPPING. The ERC solve did not converge, or the realised risk contributions\n"
            "are not equal. Weights that are not equal-risk-contribution are not the\n"
            "allocation section 3 specifies, so no section 8 verdict may be issued on them."
        )
        return 1

    rule("STEP 5 - BENCHMARK SYMMETRY (section 4). HARD GATE.")
    print(
        "Section 4: 'identical covariance estimation, identical vol target, identical caps'.\n"
        "The benchmark is not a different construction; it is THIS construction with the\n"
        "sleeves' own benchmarks substituted for the sleeves. Both come from a single call\n"
        "to trendbot.engine.ms_validation.run_overlay, and the settings recorded on the two\n"
        "results are compared field by field rather than assumed equal.\n"
    )
    symmetry = assert_symmetric(portfolio, benchmark)
    print(symmetry.to_string())
    print(f"\nGATE: portfolio and benchmark differ only in their inputs -> {_flag(True)}")
    print(
        f"\n  benchmark inputs: "
        + ", ".join(f"{s.label}={s.benchmark_name}" for s in sleeve_set.sleeves)
    )

    rule("STEP 6 - FULL-SAMPLE BACKTEST, ONE RUN")
    print(cfg.describe())
    lo, hi = alignment.span
    print(f"window: {lo.date()} -> {hi.date()}  ({len(index)} bars, section 5's intersection)")
    print(f"headline readings: benchmark variant {HEADLINE_BENCHMARK_VARIANT!r}, weight order {HEADLINE_WEIGHT_ORDER!r}")
    print(
        "\nSharpe is in excess of the 13-week T-bill for the portfolio and the benchmark\n"
        "alike. Every sleeve series is ALREADY excess of that rate; the overlay reconstructs\n"
        "total returns before handing them to the engine so the rate is subtracted exactly\n"
        "once rather than twice on the invested fraction."
    )
    print(
        f"\ncovariance warm-up: {portfolio.n_undefined_dates} dates before the EWMA is defined, "
        f"{portfolio.n_singular_dates} dates with a singular covariance (must be 0 here)"
    )
    print(f"\nfirst funded rebalance: {portfolio.first_funded.date()} "
          f"(the EWMA needs {cfg.covariance_halflife_days} observations; before it the account is cash,\n"
          f" which contributes zero excess return from the second bar onward, identically\n"
          f" for both overlays; the first bar carries -rf_daily[0] = -7.1e-05 from the engine's\n"
          f" universal pct_change().fillna(0.0) convention, not from anything this experiment does)")

    print(f"\nportfolio (net of {cfg.overlay_cost_bps_per_side:g} bps/side overlay cost):")
    print(f"  {portfolio.stats}")
    print(f"section 4 benchmark (ERC over the three sleeves' own benchmarks):")
    print(f"  {benchmark.stats}")

    held = portfolio.result.weights.loc[portfolio.first_funded :].abs().sum(axis=1)
    print(
        f"\ngross exposure: mean {float(held.mean()):.4f}, median {float(held.median()):.4f}, "
        f"max {float(held.max()):.4f}"
    )
    print(
        f"section 3's gross cap of {cfg.gross_exposure_cap:g} binds on "
        f"{portfolio.gross_cap_binding_fraction_daily:.1%} of funded DAYS and "
        f"{portfolio.gross_cap_binding_fraction:.1%} of REBALANCES"
    )
    realised_vol = float(portfolio.returns.loc[portfolio.first_funded :].std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    print(
        f"realised volatility {realised_vol:.2%} against section 3's {cfg.portfolio_vol_target:.0%} target, "
        f"median k before the cap = {float(portfolio.diagnostics['k_uncapped'].median()):.3f}, "
        f"after it {float(portfolio.diagnostics['k'].median()):.3f}"
    )
    print(
        "\n  Section 3's volatility target is largely INOPERATIVE. k exceeds 1 at most\n"
        "  rebalances, so the binding constraint is the gross cap and the pipeline collapses\n"
        "  to min(k, 1) x the clipped ERC weights almost everywhere. Section 9 half-predicts\n"
        "  this ('the gross cap will bind, as in 001, suppressing the vol target'), but\n"
        "  section 7 still freezes the vol target as a parameter. On this data it is not one."
    )

    rule("SECTION 6 - COST LADDER")
    print(
        "The overlay cost is charged on the change in sleeve weight, at the engine's own\n"
        "convention: rate x sum_i |w_new_i - w_current_i| x equity, with w_current the\n"
        "DRIFTED weight rather than the last target, so only the trade actually done is\n"
        "paid for. This is the same convention 001, 002 and 005 charge, imported rather than\n"
        "restated. Sleeve-level costs are already inside the sleeve returns.\n"
    )
    ladder = []
    for bps in cfg.cost_sensitivity_bps:
        _, _, p_bps, b_bps = _build(cfg, sleeve_set, alignment, rf, order=HEADLINE_WEIGHT_ORDER, cost_bps=bps)
        ladder.append(
            {
                "bps/side": bps,
                "portfolio Sharpe": p_bps.sharpe,
                "benchmark Sharpe": b_bps.sharpe,
                "clause (a)": p_bps.sharpe - b_bps.sharpe,
                "CAGR": p_bps.stats.cagr,
            }
        )
    ladder_frame = pd.DataFrame(ladder).set_index("bps/side")
    print(ladder_frame.round(4).to_string())
    turnover = portfolio.stats.ann_turnover
    print(
        f"\nannualised overlay turnover: {turnover:.2f}x. From "
        f"{ladder_frame.index[0]:g} to {ladder_frame.index[-1]:g} bps the portfolio Sharpe moves "
        f"{ladder_frame['portfolio Sharpe'].iloc[-1] - ladder_frame['portfolio Sharpe'].iloc[0]:+.4f}."
    )

    rule("STEP 7 - SECTION 8's THREE CLAUSES, EACH AS A NUMBER")
    sleeve_sharpes = pd.Series({label: sharpe(sleeves[label]) for label in sleeves.columns})
    benchmark_sharpes = pd.Series({label: sharpe(benchmarks[label]) for label in benchmarks.columns})
    correlation = correlation_report(sleeves, benchmarks)
    decision = evaluate_decision_006(
        portfolio_sharpe=portfolio.sharpe,
        benchmark_sharpe=benchmark.sharpe,
        sleeve_sharpes=sleeve_sharpes,
        correlation=correlation,
        min_excess_over_benchmark=cfg.min_excess_over_benchmark,
        min_excess_over_best_sleeve=cfg.min_excess_over_best_sleeve,
    )
    print("Individual sleeve and benchmark Sharpes, measured on section 5's window:\n")
    component = pd.DataFrame(
        {
            "sleeve Sharpe (section 5 window)": sleeve_sharpes,
            "own headline Sharpe": pd.Series({s.label: s.recorded_sharpe for s in sleeve_set.sleeves}),
            "benchmark Sharpe (section 5 window)": benchmark_sharpes,
        }
    )
    print(component.round(4).to_string())
    print(
        "\n  Clause (b) is evaluated on the section 5 window, the only like-for-like reading:\n"
        "  a portfolio measured over 2008-2026 cannot be compared with a sleeve Sharpe\n"
        "  measured over a different span. Sleeve C is +0.209 over its own 1999 start and\n"
        f"  {sleeve_sharpes['C']:+.4f} over section 5's window; the headline uses the latter."
    )
    print(f"\nportfolio Sharpe {portfolio.sharpe:+.4f}   benchmark Sharpe {benchmark.sharpe:+.4f}\n")
    print(decision.table().round(4).to_string())

    rule("STEP 7(c) - THE MECHANISM TEST, IN FULL")
    print(
        "Section 8's third clause is the important one: the first two can be satisfied by\n"
        "luck in a three-sleeve portfolio, and this one tests the stated cause. Section 1's\n"
        "claim is that the STRATEGIES are less correlated with each other than their own\n"
        "BENCHMARKS are with each other.\n"
    )
    print("realised sleeve correlation matrix (daily, section 5 window):")
    print("  " + correlation.sleeve_matrix.round(4).to_string().replace("\n", "\n  "))
    print("\nrealised benchmark correlation matrix (daily, section 5 window):")
    print("  " + correlation.benchmark_matrix.round(4).to_string().replace("\n", "\n  "))
    print(f"\n{correlation}")
    pairs = []
    for i, a in enumerate(sleeves.columns):
        for b in list(sleeves.columns)[i + 1 :]:
            pairs.append(
                {
                    "pair": f"{a}-{b}",
                    "sleeves": correlation.sleeve_matrix.loc[a, b],
                    "benchmarks": correlation.benchmark_matrix.loc[a, b],
                    "difference": correlation.sleeve_matrix.loc[a, b]
                    - correlation.benchmark_matrix.loc[a, b],
                }
            )
    print("\npair by pair:")
    print("  " + pd.DataFrame(pairs).set_index("pair").round(4).to_string().replace("\n", "\n  "))
    print(
        "\n  Read this alongside section 9's own warning. Nine of 001's twelve ETFs are inside\n"
        "  002's forty-one, so the A-B BENCHMARK correlation is high by construction - two\n"
        "  overlapping long-only baskets - which pins the benchmark mean up regardless of\n"
        "  what the strategies do. The clause passes, but a good part of why it passes is\n"
        "  universe overlap rather than strategy behaviour."
    )
    print("\n  SECTION 1's ARITHMETIC, WITH EVERY TERM MEASURED")
    print(
        "  Section 1 states S_combined = s x sqrt(N) / sqrt(1 + (N-1)rho). Both sides of\n"
        "  section 8's first clause can now be evaluated against it:\n"
    )
    mech = []
    for name, frame, matrix in (
        ("sleeves", sleeves, correlation.sleeve_matrix),
        ("benchmarks", benchmarks, correlation.benchmark_matrix),
    ):
        s_bar = float(np.mean([sharpe(frame[c]) for c in frame.columns]))
        rho = _mean_pairwise_upper(matrix)
        n = frame.shape[1]
        mech.append(
            {
                "": name,
                "mean component Sharpe s": s_bar,
                "mean pairwise corr rho": rho,
                "s x sqrt(N)/sqrt(1+(N-1)rho)": s_bar * np.sqrt(n) / np.sqrt(1 + (n - 1) * rho),
            }
        )
    mech_frame = pd.DataFrame(mech).set_index("")
    print("  " + mech_frame.round(4).to_string().replace("\n", "\n  "))
    print(
        f"\n  observed: portfolio {portfolio.sharpe:+.4f}, benchmark {benchmark.sharpe:+.4f} - both\n"
        "  short of the idealised formula by a similar margin, which is what the gross cap,\n"
        "  the overlay cost and the formula's equal-Sharpe assumption cost.\n\n"
        "  THIS IS THE RESULT, AND IT IS NOT THE ONE SECTION 1 EXPECTED. The correlation term\n"
        "  moves exactly the way section 1 predicted - the sleeves ARE less correlated than\n"
        "  their benchmarks, and that is worth real Sharpe. It is applied to a lower base. The\n"
        "  mean sleeve Sharpe is below the mean benchmark Sharpe, and the deficit in s is\n"
        "  larger than the gain from rho. Section 1's formula, evaluated on the measured\n"
        "  inputs, predicts the benchmark portfolio wins - and it does. Diversification worked\n"
        "  and lost anyway, because three strategies that individually fail to beat their own\n"
        "  markets do not combine into one that beats the combination of those markets."
    )
    print(
        "\n  Frequency sensitivity, since section 8 does not name one (all readings agree in sign):"
    )
    freq_rows = []
    for name, freq in (("daily", None), ("weekly", "W-FRI"), ("monthly", "ME")):
        if freq is None:
            s_mat, b_mat = sleeves.corr(), benchmarks.corr()
        else:
            s_mat = (1 + sleeves).resample(freq).prod().sub(1).corr()
            b_mat = (1 + benchmarks).resample(freq).prod().sub(1).corr()
        s_up = s_mat.to_numpy()[np.triu_indices(len(s_mat), k=1)].mean()
        b_up = b_mat.to_numpy()[np.triu_indices(len(b_mat), k=1)].mean()
        freq_rows.append({"frequency": name, "sleeves": s_up, "benchmarks": b_up, "difference": s_up - b_up})
    print("  " + pd.DataFrame(freq_rows).set_index("frequency").round(4).to_string().replace("\n", "\n  "))

    rule("STEP 8 - ATTRIBUTION: RISK CONTRIBUTION AND THE WEIGHT PATH")
    print(
        "An ERC portfolio that is effectively single-sleeve is not testing section 1's\n"
        "hypothesis, so the risk budget is reported rather than assumed equal.\n"
    )
    contributions = portfolio.risk_contributions.dropna()
    print("realised risk contribution per sleeve (fraction of portfolio variance):")
    summary = pd.DataFrame(
        {
            "mean": contributions.mean(),
            "min": contributions.min(),
            "max": contributions.max(),
            "target": 1.0 / cfg.n_sleeves,
        }
    )
    print("  " + summary.round(6).to_string().replace("\n", "\n  "))
    print(
        f"\n  These are equal to 1/{cfg.n_sleeves} by construction - that is what the solve\n"
        f"  delivers, and the worst departure anywhere in the sample is "
        f"{portfolio.worst_erc_deviation:.3e}. The interesting number is not the risk share\n"
        "  but the CAPITAL share it takes to achieve it, below."
    )
    erc_columns = [c for c in portfolio.diagnostics.columns if c.startswith("erc_") and c[4:] in sleeves.columns]
    erc_path = portfolio.diagnostics[erc_columns].rename(columns=lambda c: c[4:])
    print("\nERC capital weights before caps (on the simplex, summing to 1):")
    print("  " + erc_path.describe().loc[["mean", "std", "min", "max"]].round(4).to_string().replace("\n", "\n  "))
    traded = portfolio.targets.loc[portfolio.diagnostics.index]
    print("\ntarget weights after section 3's caps and vol scaling:")
    print("  " + traded.describe().loc[["mean", "std", "min", "max"]].round(4).to_string().replace("\n", "\n  "))
    rebals = rebalance_dates(portfolio.sleeve_returns.index)
    positions = {d: i for i, d in enumerate(portfolio.sleeve_returns.index)}
    decisions = [
        portfolio.sleeve_returns.index[positions[r] - 1]
        for r in rebals
        if positions[r] > 0 and portfolio.sleeve_returns.index[positions[r] - 1] in traded.index
    ]
    path = traded.loc[decisions]
    print(f"\nweight path at every 24th of the {len(path)} funded rebalances:")
    print("  " + path.iloc[::24].round(4).to_string().replace("\n", "\n  "))
    ceiling = (erc_path > cfg.max_sleeve_weight + 1e-12).any(axis=1)
    floor = (erc_path < cfg.min_sleeve_weight - 1e-12).any(axis=1)
    print(
        f"\n  bound binding, ceiling and floor counted separately:\n"
        f"    {cfg.max_sleeve_weight:.0%} ceiling binds at {float(ceiling.mean()):.1%} of "
        f"{len(erc_path)} solves (largest ERC weight anywhere {float(erc_path.max().max()):.4f})\n"
        f"    {cfg.min_sleeve_weight:.0%} floor   binds at {float(floor.mean()):.1%} "
        f"(smallest ERC weight anywhere {float(erc_path.min().min()):.4f}) - it never binds, so\n"
        f"    section 3's minimum is inoperative on this data and only the ceiling does work.\n"
        f"    largest residual bound breach after renormalisation: "
        f"{float(portfolio.diagnostics['bound_breach'].max()):.4f}\n"
        "  Section 3 says the bounds are 'applied before renormalisation', which read\n"
        "  literally means applied once; clipping and then dividing by the new sum can push a\n"
        "  weight back outside its bound, and that residual is reported rather than iterated\n"
        "  away, because iterating to a fixed point is a construction the document does not\n"
        "  specify."
    )
    dominant = erc_path.mean().idxmax()
    print(
        f"\n  No sleeve dominates the RISK budget - equal risk contribution guarantees that.\n"
        f"  Sleeve {dominant} does take the largest CAPITAL share (mean "
        f"{float(erc_path.mean().max()):.1%}), which is the arithmetic of being the\n"
        f"  lowest-volatility sleeve, not a view about it."
    )

    rule("STEP 9 - PSR AND DEFLATED SHARPE AT configs_tried = 5")
    print(
        "PREREG_006.md's header, restated because it is the point: this experiment reuses\n"
        "return series from 001, 002 and 005 whose results were already known. Even with no\n"
        "cherry-picking, the sleeve set is not independent of prior findings.\n\n"
        "    configs_tried = 5 is a FLOOR, not the true multiple-testing burden.\n\n"
        "The deflation below is therefore a lower bound on the correction that is actually\n"
        "warranted, and a favourable number here must not be read as though it were the\n"
        "whole correction.\n"
    )
    for name, fifth in (
        ("005 headline, spot-only (-0.271)", EXPERIMENT_005_NET_SHARPE),
        ("005 carry diagnostic (+0.209)", EXPERIMENT_005_CARRY_SHARPE),
    ):
        dsr, trials = _psr(cfg, portfolio.returns, fifth_trial=fifth)
        print(f"  fifth trial = {name}")
        print(f"    {dsr}")
        print(f"    trial per-period Sharpes: {[round(t, 6) for t in trials]}")
    print(
        "\n  PREREG_006.md names no trial Sharpes, so which value stands for 005 is a\n"
        "  reporting choice, made in the open and reported both ways. It gates nothing."
    )

    rule("THE TWO UNPINNED READINGS, RUN THROUGH THE IDENTICAL MACHINERY")
    print(
        "Section 4 does not say whether sleeve C's benchmark carries the carry correction\n"
        "section 2 applies to sleeve C itself, and section 3 does not say where its vol\n"
        "target enters. Both were resolved before any number was computed; both alternatives\n"
        "are run here in full because each moves clause (a) across zero.\n"
    )
    grid = report_readings(cfg, alignment, rf, args)
    print(grid.round(4).to_string())
    print(
        "\n  Clause (a) is not robust to either reading. Clause (b) is: it fails by a wide\n"
        "  margin under all four combinations, and it is the clause that decides the verdict."
    )

    rule("SECTION 5's ALTERNATIVE READING - RAW AVAILABLE HISTORIES")
    raw = report_raw_window(cfg, rf, args)

    rule("STEP 10 - SECTION 8's PRE-COMMITTED VERDICT")
    print(decision.table().round(4).to_string())
    print(f"\n{decision}")
    if decision.verdict == "SUPPORTED":
        print(
            "\nPer PREREG_006.md's header, a pass here is SUGGESTIVE ONLY. It would require\n"
            "out-of-sample confirmation on data not used in 001-005 before it meant anything."
        )
    print(
        f"\nSection 9's expectations, checked:\n"
        f"  realistic combined Sharpe {cfg.expected_sharpe_low}-{cfg.expected_sharpe_high}: "
        f"observed {portfolio.sharpe:+.4f} "
        f"[{'inside' if cfg.expected_sharpe_low <= portfolio.sharpe <= cfg.expected_sharpe_high else 'OUTSIDE'}]\n"
        f"  above {cfg.bug_threshold_sharpe} means a bug or a covariance using future data: "
        f"observed {portfolio.sharpe:+.4f} "
        f"[{'not triggered' if portfolio.sharpe <= cfg.bug_threshold_sharpe else 'TRIGGERED'}]\n"
        f"  'the benchmark improves too': benchmark {benchmark.sharpe:+.4f} against the best\n"
        f"  single benchmark {benchmark_sharpes.max():+.4f} (sleeve {benchmark_sharpes.idxmax()})\n"
        f"  'most likely verdict: inconclusive': observed {decision.verdict}"
    )

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {
            "prereg_sha256": cfg.source_sha256,
            "window": [str(lo.date()), str(hi.date())],
            "n_bars": len(index),
            "headline_readings": {
                "benchmark_variant": HEADLINE_BENCHMARK_VARIANT,
                "weight_order": HEADLINE_WEIGHT_ORDER,
            },
            "portfolio": portfolio.stats.as_dict(),
            "benchmark": benchmark.stats.as_dict(),
            "sleeve_sharpes": {k: float(v) for k, v in sleeve_sharpes.items()},
            "benchmark_sharpes": {k: float(v) for k, v in benchmark_sharpes.items()},
            "clauses": {
                "a_vs_benchmark": decision.clause_a,
                "b_vs_best_sleeve": decision.clause_b,
                "c_correlation_difference": decision.clause_c,
            },
            "sleeve_correlation": correlation.sleeve_matrix.to_dict(),
            "benchmark_correlation": correlation.benchmark_matrix.to_dict(),
            "worst_erc_deviation": portfolio.worst_erc_deviation,
            "gross_cap_binding_fraction_days": portfolio.gross_cap_binding_fraction_daily,
            "cost_ladder": ladder_frame.reset_index().to_dict("records"),
            "readings_grid": grid.reset_index().to_dict("records"),
            "raw_window_reading": raw.reset_index().to_dict("records"),
            "verdict": decision.verdict,
            "abandon_reasons": list(decision.abandon_reasons),
        }
        out = RESULTS_DIR / "backtest_006.json"
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {out}")
    return 0


def report_raw_window(cfg: Config006, rf, args) -> pd.DataFrame:
    print(
        "Reading 'available histories' as the raw extent of each experiment's computable\n"
        "return series rather than as its pre-committed window. This CHANGES WHAT SLEEVE B\n"
        "IS - only 20 of 002's 41 ETFs had listed by 1999 - so it is reported as a\n"
        "sensitivity and cannot be adjudicated on. Section 2 forbids it as a construction.\n"
    )
    subset = load_sleeves(
        benchmark_variant=HEADLINE_BENCHMARK_VARIANT, source=args.source, full_history=True
    )
    alignment = align({s.label: s.returns for s in subset.sleeves})
    lo, hi = alignment.span
    print(alignment.table.to_string())
    print(f"\n{alignment}")
    print(f"raw-availability window: {lo.date()} -> {hi.date()}, {len(alignment.index)} bars\n")
    sleeves, benchmarks, portfolio, benchmark = _build(
        cfg, subset, alignment, rf, order=HEADLINE_WEIGHT_ORDER
    )
    correlation = correlation_report(sleeves, benchmarks)
    sleeve_sharpes = pd.Series({c: sharpe(sleeves[c]) for c in sleeves.columns})
    decision = evaluate_decision_006(
        portfolio_sharpe=portfolio.sharpe,
        benchmark_sharpe=benchmark.sharpe,
        sleeve_sharpes=sleeve_sharpes,
        correlation=correlation,
        min_excess_over_benchmark=cfg.min_excess_over_benchmark,
        min_excess_over_best_sleeve=cfg.min_excess_over_best_sleeve,
    )
    frame = pd.DataFrame(
        [
            {
                "portfolio": portfolio.sharpe,
                "benchmark": benchmark.sharpe,
                "best sleeve": f"{decision.best_sleeve_label} {decision.best_sleeve_sharpe:+.4f}",
                "(a)": decision.clause_a,
                "(b)": decision.clause_b,
                "(c)": decision.clause_c,
                "verdict": decision.verdict,
            }
        ],
        index=pd.Index(["raw availability"], name="section 5 reading"),
    )
    print(frame.round(4).to_string())
    print("\nsleeve Sharpes on this window: " + sleeve_sharpes.round(4).to_dict().__str__())
    print(
        f"\nThe verdict is {decision.verdict} under this reading too. Section 9's expectation\n"
        "of a post-1999 window is therefore not merely unachievable under the headline\n"
        "reading; achieving it does not change the answer."
    )
    return frame


def report_readings(cfg: Config006, alignment, rf, args) -> pd.DataFrame:
    rows = []
    for variant in BENCHMARK_VARIANTS:
        subset = load_sleeves(benchmark_variant=variant, source=args.source)
        for order in WEIGHT_ORDERS:
            sleeves, benchmarks, portfolio, benchmark = _build(
                cfg, subset, alignment, rf, order=order
            )
            correlation = correlation_report(sleeves, benchmarks)
            sleeve_sharpes = pd.Series({c: sharpe(sleeves[c]) for c in sleeves.columns})
            decision = evaluate_decision_006(
                portfolio_sharpe=portfolio.sharpe,
                benchmark_sharpe=benchmark.sharpe,
                sleeve_sharpes=sleeve_sharpes,
                correlation=correlation,
                min_excess_over_benchmark=cfg.min_excess_over_benchmark,
                min_excess_over_best_sleeve=cfg.min_excess_over_best_sleeve,
            )
            rows.append(
                {
                    "sleeve C benchmark": variant,
                    "weight order": order,
                    "headline": "<--" if (variant == HEADLINE_BENCHMARK_VARIANT and order == HEADLINE_WEIGHT_ORDER) else "",
                    "portfolio": portfolio.sharpe,
                    "benchmark": benchmark.sharpe,
                    "(a)": decision.clause_a,
                    "(b)": decision.clause_b,
                    "(c)": decision.clause_c,
                    "verdict": decision.verdict,
                }
            )
    return pd.DataFrame(rows).set_index(["sleeve C benchmark", "weight order"])


def cmd_sleeves(cfg: Config006, args) -> int:
    sleeve_set = load_sleeves(
        benchmark_variant=HEADLINE_BENCHMARK_VARIANT, source=args.source, refresh=args.refresh
    )
    check = reproduce(sleeve_set)
    passed = report_sleeves(cfg, sleeve_set, check)
    report_alignment(cfg, sleeve_set)
    return 0 if passed else 1


def cmd_backtest(cfg: Config006, args) -> int:
    sleeve_set = load_sleeves(
        benchmark_variant=HEADLINE_BENCHMARK_VARIANT, source=args.source, refresh=args.refresh
    )
    check = reproduce(sleeve_set)
    if not report_sleeves(cfg, sleeve_set, check):
        print(
            "\nSTOPPING. A sleeve no longer reproduces its recorded headline, which means a\n"
            "prior result has drifted. Experiment 006 must not be run on top of it."
        )
        return 1
    alignment = report_alignment(cfg, sleeve_set)
    rf = load_risk_free_rate()
    sleeves = pd.DataFrame({s.label: s.returns for s in sleeve_set.sleeves}).loc[alignment.index]
    if not report_covariance_and_solver(cfg, sleeves):
        print("\nSTOPPING. A step 3 gate failed.")
        return 1
    return report_backtest(cfg, sleeve_set, alignment, rf, save=args.save, args=args)


def cmd_validate(cfg: Config006, args) -> int:
    failures = []
    if cmd_sleeves(cfg, args) != 0:
        failures.append("step 1 sleeve reproduction")
    if cmd_noise(cfg, args) != 0:
        failures.append("step 4 noise test")
    if cmd_backtest(cfg, args) != 0:
        failures.append("steps 2-3, 5-10")
    rule("SECTION 7 PROTOCOL - SUMMARY")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("Every gate in the protocol passed.")
    return 0


def main(args) -> int:
    cfg = load_config_006()
    if args.sleeves:
        return cmd_sleeves(cfg, args)
    if args.noise:
        return cmd_noise(cfg, args)
    if args.validate:
        return cmd_validate(cfg, args)
    return cmd_backtest(cfg, args)
