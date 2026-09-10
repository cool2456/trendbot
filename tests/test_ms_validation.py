from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.engine.metrics import sharpe
from trendbot.engine.ms_validation import (
    align,
    assert_symmetric,
    correlation_report,
    covariance_is_point_in_time,
    evaluate_decision_006,
    noise_test_006,
    overlay_targets,
    run_overlay,
    synthetic_panel,
    synthetic_sleeve_returns,
)
from trendbot.engine.panel_backtest import run_panel_backtest

LABELS = ("A", "B", "C")
CORRELATION = np.array([[1.0, 0.70, 0.40], [0.70, 1.0, 0.35], [0.40, 0.35, 1.0]])
VOLS = (0.075, 0.11, 0.09)

SETTINGS = dict(
    halflife=60,
    vol_target=0.10,
    gross_cap=1.0,
    min_weight=0.05,
    max_weight=0.60,
    order="clip-first",
    cost_bps=5.0,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    return synthetic_sleeve_returns(
        0, n_days=1500, labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )


@pytest.fixture
def zero_rf(returns) -> pd.Series:
    return pd.Series(0.0, index=returns.index)


def test_alignment_intersects_and_reports_what_it_dropped():
    index = pd.bdate_range("2010-01-04", periods=100)
    a = pd.Series(0.0, index=index)
    b = pd.Series(0.0, index=index.delete([10, 20, 30]))
    c = pd.Series(0.0, index=index.delete([10, 40]))
    alignment = align({"A": a, "B": b, "C": c})
    assert len(alignment.index) == 96
    assert alignment.table.loc["A", "dropped"] == 4
    assert alignment.table.loc["B", "dropped"] == 1
    assert alignment.table.loc["C", "dropped"] == 2
    assert alignment.rule == "inner join on date index"


def test_alignment_manufactures_no_dates():
    index = pd.bdate_range("2010-01-04", periods=50)
    a = pd.Series(0.0, index=index)
    b = pd.Series(0.0, index=pd.bdate_range("2010-02-01", periods=50))
    alignment = align({"A": a, "B": b})
    assert alignment.index.isin(a.index).all()
    assert alignment.index.isin(b.index).all()


def test_alignment_refuses_a_disjoint_set():
    a = pd.Series(0.0, index=pd.bdate_range("2010-01-04", periods=10))
    b = pd.Series(0.0, index=pd.bdate_range("2020-01-06", periods=10))
    with pytest.raises(ValueError, match="share no dates"):
        align({"A": a, "B": b})


def test_alignment_refuses_an_empty_set():
    with pytest.raises(ValueError, match="empty set"):
        align({})


@pytest.mark.parametrize("held", ["A", "B", "C"])
def test_a_single_sleeve_at_full_weight_reproduces_its_own_return(returns, zero_rf, held):
    prices = synthetic_panel(returns, zero_rf)
    targets = pd.DataFrame(0.0, index=returns.index, columns=list(LABELS))
    targets[held] = 1.0
    result = run_panel_backtest(
        prices, list(LABELS), None, targets=targets, cost_bps=0.0, gross_cap=1.0, risk_free=zero_rf
    )
    first = result.diagnostics.index[0]
    produced = result.excess_returns.loc[first:]
    expected = returns[held].loc[first:]
    assert np.allclose(produced.to_numpy(), expected.to_numpy(), atol=1e-12, rtol=0.0)
    assert sharpe(produced) == pytest.approx(sharpe(expected), abs=1e-10)


def test_the_round_trip_holds_with_a_non_zero_risk_free_rate(returns):
    rf = pd.Series(0.03, index=returns.index)
    prices = synthetic_panel(returns, rf)
    targets = pd.DataFrame(0.0, index=returns.index, columns=list(LABELS))
    targets["A"] = 1.0
    result = run_panel_backtest(
        prices, list(LABELS), None, targets=targets, cost_bps=0.0, gross_cap=1.0, risk_free=rf
    )
    first = result.diagnostics.index[0]
    produced = result.excess_returns.loc[first:]
    expected = returns["A"].loc[first:]
    assert np.allclose(produced.to_numpy(), expected.to_numpy(), atol=1e-12, rtol=0.0)


def test_synthetic_panel_prices_have_no_intraday_leg(returns, zero_rf):
    prices = synthetic_panel(returns, zero_rf)
    assert np.allclose(
        prices.open.to_numpy()[1:], prices.close.to_numpy()[:-1], atol=0.0, rtol=0.0
    )
    assert (prices.open.iloc[0] == 1.0).all()
    assert prices.open.notna().to_numpy().all()


def test_synthetic_panel_refuses_unaligned_input(returns, zero_rf):
    holed = returns.copy()
    holed.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="align"):
        synthetic_panel(holed, zero_rf)


def test_covariance_is_unchanged_by_appending_future_data(returns):
    passed, worst = covariance_is_point_in_time(returns, halflife=60, n_future=200)
    assert passed
    assert worst == 0.0


def test_weights_at_a_date_do_not_move_when_later_returns_change(returns):
    targets, _, _ = overlay_targets(returns, **{k: v for k, v in SETTINGS.items() if k != "cost_bps"})
    cut = returns.index[len(returns) * 2 // 3]
    perturbed = returns.copy()
    perturbed.loc[cut:] = perturbed.loc[cut:] * -3.0
    later, _, _ = overlay_targets(
        perturbed, **{k: v for k, v in SETTINGS.items() if k != "cost_bps"}
    )
    before = targets.loc[: returns.index[len(returns) * 2 // 3 - 1]]
    assert np.allclose(before.to_numpy(), later.loc[before.index].to_numpy(), atol=0.0, rtol=0.0)


def test_the_warm_up_period_is_cash_not_an_invented_allocation(returns):
    targets, diagnostics, _ = overlay_targets(
        returns, **{k: v for k, v in SETTINGS.items() if k != "cost_bps"}
    )
    warm_up = targets.loc[: diagnostics.index[0]].iloc[:-1]
    assert (warm_up.to_numpy() == 0.0).all()
    assert diagnostics.attrs["n_undefined"] == len(warm_up)
    assert diagnostics.attrs["n_singular"] == 0


def test_every_solve_on_realistic_covariance_converges(returns):
    _, diagnostics, contributions = overlay_targets(
        returns, **{k: v for k, v in SETTINGS.items() if k != "cost_bps"}
    )
    assert diagnostics["erc_converged"].all()
    assert float(diagnostics["erc_worst_deviation"].max()) < 1e-9
    live = contributions.dropna()
    assert np.allclose(live.to_numpy(), 1.0 / 3.0, atol=1e-9)


def test_symmetry_gate_accepts_identical_settings(returns, zero_rf):
    other = synthetic_sleeve_returns(
        1, n_days=len(returns), labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )
    other.index = returns.index
    portfolio = run_overlay(returns, zero_rf, label="portfolio", **SETTINGS)
    benchmark = run_overlay(other, zero_rf, label="benchmark", **SETTINGS)
    table = assert_symmetric(portfolio, benchmark)
    assert bool(table["same"].all())


@pytest.mark.parametrize(
    "changed", [{"halflife": 30}, {"vol_target": 0.15}, {"gross_cap": 2.0}, {"max_weight": 0.9}]
)


def test_symmetry_gate_refuses_any_asymmetry(returns, zero_rf, changed):
    portfolio = run_overlay(returns, zero_rf, label="portfolio", **SETTINGS)
    benchmark = run_overlay(returns, zero_rf, label="benchmark", **{**SETTINGS, **changed})
    with pytest.raises(RuntimeError, match="different settings"):
        assert_symmetric(portfolio, benchmark)


def test_identical_inputs_give_identical_overlays(returns, zero_rf):
    left = run_overlay(returns, zero_rf, label="left", **SETTINGS)
    right = run_overlay(returns, zero_rf, label="right", **SETTINGS)
    assert np.array_equal(left.returns.to_numpy(), right.returns.to_numpy())
    assert left.sharpe == right.sharpe


def _correlation(sleeve_rho: float, benchmark_rho: float):
    index = pd.bdate_range("2010-01-04", periods=600)
    rng = np.random.default_rng(4)

    def frame(rho):
        cov = np.full((3, 3), rho) + np.eye(3) * (1 - rho)
        draws = rng.multivariate_normal(np.zeros(3), cov, size=len(index), method="cholesky")
        return pd.DataFrame(draws, index=index, columns=list(LABELS))

    return correlation_report(frame(sleeve_rho), frame(benchmark_rho))


def test_correlation_clause_holds_when_sleeves_are_less_correlated():
    report = _correlation(0.1, 0.8)
    assert report.holds
    assert report.difference < 0
    assert report.sleeve_matrix.shape == (3, 3)
    assert np.allclose(np.diag(report.sleeve_matrix.to_numpy()), 1.0)


def test_correlation_clause_fails_when_sleeves_are_more_correlated():
    assert not _correlation(0.8, 0.1).holds


def test_correlation_report_refuses_mismatched_calendars():
    left = pd.DataFrame(0.0, index=pd.bdate_range("2010-01-04", periods=10), columns=list(LABELS))
    right = pd.DataFrame(0.0, index=pd.bdate_range("2011-01-03", periods=10), columns=list(LABELS))
    with pytest.raises(ValueError, match="different calendars"):
        correlation_report(left, right)


def _decision(portfolio, benchmark, sleeves, correlation):
    return evaluate_decision_006(
        portfolio_sharpe=portfolio,
        benchmark_sharpe=benchmark,
        sleeve_sharpes=pd.Series(sleeves),
        correlation=correlation,
        min_excess_over_benchmark=0.15,
        min_excess_over_best_sleeve=0.10,
    )


def test_all_three_clauses_must_hold_for_support():
    good = _correlation(0.1, 0.8)
    decision = _decision(0.80, 0.60, {"A": 0.3, "B": 0.5, "C": 0.1}, good)
    assert decision.clause_a == pytest.approx(0.20)
    assert decision.clause_b == pytest.approx(0.30)
    assert decision.verdict == "SUPPORTED"


def test_failing_the_best_sleeve_triggers_abandon():
    decision = _decision(0.30, 0.10, {"A": 0.3, "B": 0.9, "C": 0.1}, _correlation(0.1, 0.8))
    assert decision.best_sleeve_label == "B"
    assert decision.clause_b < 0
    assert decision.verdict == "ABANDON"
    assert any("best individual sleeve" in r for r in decision.abandon_reasons)


def test_failing_the_benchmark_triggers_abandon():
    decision = _decision(0.30, 0.40, {"A": 0.1, "B": 0.1, "C": 0.1}, _correlation(0.1, 0.8))
    assert decision.verdict == "ABANDON"
    assert any("section 4 benchmark" in r for r in decision.abandon_reasons)


def test_failing_the_correlation_clause_triggers_abandon():
    decision = _decision(0.90, 0.60, {"A": 0.1, "B": 0.1, "C": 0.1}, _correlation(0.8, 0.1))
    assert decision.clause_a >= 0.15 and decision.clause_b >= 0.10
    assert decision.verdict == "ABANDON"
    assert any("correlations" in r for r in decision.abandon_reasons)


def test_a_positive_but_insufficient_margin_is_inconclusive():
    decision = _decision(0.62, 0.60, {"A": 0.3, "B": 0.61, "C": 0.1}, _correlation(0.1, 0.8))
    assert decision.clause_a > 0 and decision.clause_b > 0
    assert decision.clause_a < 0.15 and decision.clause_b < 0.10
    assert decision.verdict == "INCONCLUSIVE"


def test_decision_rule_refuses_an_empty_sleeve_set():
    with pytest.raises(ValueError, match="empty set"):
        _decision(0.5, 0.4, {}, _correlation(0.1, 0.8))


def test_synthetic_sleeves_have_exactly_zero_mean():
    frame = synthetic_sleeve_returns(
        3, n_days=800, labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )
    assert np.allclose(frame.mean().to_numpy(), 0.0, atol=1e-18)


def test_synthetic_sleeves_carry_the_requested_correlation():
    frame = synthetic_sleeve_returns(
        3, n_days=20_000, labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )
    assert np.allclose(frame.corr().to_numpy(), CORRELATION, atol=0.03)


def test_synthetic_sleeves_carry_the_requested_volatility():
    frame = synthetic_sleeve_returns(
        5, n_days=20_000, labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )
    assert np.allclose(frame.std().to_numpy() * np.sqrt(252), np.array(VOLS), rtol=0.05)


def test_the_construction_earns_nothing_on_sleeves_containing_nothing():
    result = noise_test_006(
        seeds=(0, 1, 2, 3, 4, 5, 6, 7),
        n_days=1500,
        labels=LABELS,
        correlation=CORRELATION,
        annual_vols=VOLS,
        **SETTINGS,
    )
    assert result.passes()
    assert abs(result.mean_sharpe) < 0.2
    assert result.all_converged
    assert result.worst_erc_deviation < 1e-9


def test_the_correlation_recovery_check_is_not_vacuous():
    frame = synthetic_sleeve_returns(
        0, n_days=1500, labels=LABELS, correlation=CORRELATION, annual_vols=VOLS
    )
    slipped = frame.copy()
    slipped["B"] = slipped["B"].shift(1).bfill()
    target = pd.DataFrame(CORRELATION, index=list(LABELS), columns=list(LABELS))
    intact_error = float(np.max(np.abs(frame.corr().to_numpy() - target.to_numpy())))
    slipped_error = float(np.max(np.abs(slipped.corr().to_numpy() - target.to_numpy())))
    tolerance = 5.0 / np.sqrt(1500 - 3)
    assert intact_error <= tolerance
    assert slipped_error > tolerance
    assert slipped_error > 0.5
    assert slipped_error > 4 * tolerance


def test_synthetic_sleeve_shapes_are_validated():
    with pytest.raises(ValueError, match="3x3"):
        synthetic_sleeve_returns(
            0, n_days=10, labels=LABELS, correlation=np.eye(2), annual_vols=VOLS
        )
    with pytest.raises(ValueError, match="one annualised volatility"):
        synthetic_sleeve_returns(
            0, n_days=10, labels=LABELS, correlation=CORRELATION, annual_vols=(0.1, 0.1)
        )
