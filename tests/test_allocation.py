from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.engine.allocation import (
    WEIGHT_ORDERS,
    ErcSolution,
    allocate,
    erc_weights,
    risk_contributions,
)

LABELS = ["A", "B", "C"]


def _frame(matrix, labels=None) -> pd.DataFrame:
    labels = labels or LABELS[: len(matrix)]
    return pd.DataFrame(np.asarray(matrix, dtype=float), index=labels, columns=labels)


def _random_pd(rng: np.random.Generator, n: int) -> np.ndarray:
    factor = rng.normal(size=(n, n + 4))
    matrix = factor @ factor.T / (n + 4)
    scale = np.diag(rng.uniform(0.05, 0.30, size=n) / np.sqrt(np.diag(matrix)))
    return scale @ matrix @ scale


def test_equal_variances_and_correlations_give_equal_weights():
    cov = _frame([[0.04, 0.012, 0.012], [0.012, 0.04, 0.012], [0.012, 0.012, 0.04]])
    solution = erc_weights(cov)
    assert np.allclose(solution.weights.to_numpy(), 1.0 / 3.0, atol=1e-12)
    assert solution.converged


def test_diagonal_covariance_gives_inverse_volatility():
    vols = np.array([0.06, 0.12, 0.24])
    cov = _frame(np.diag(vols**2))
    expected = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(erc_weights(cov).weights.to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("rho", [-0.9, -0.4, 0.0, 0.35, 0.8, 0.99])
def test_two_assets_are_inverse_volatility_whatever_the_correlation(rho):
    vols = np.array([0.08, 0.20])
    cov = _frame(np.array([[1.0, rho], [rho, 1.0]]) * np.outer(vols, vols), labels=["A", "B"])
    expected = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(erc_weights(cov).weights.to_numpy(), expected, atol=1e-10)


def test_inverse_volatility_is_not_the_answer_for_three_correlated_assets():
    vols = np.array([0.08, 0.10, 0.25])
    correlation = np.array([[1.0, 0.85, 0.05], [0.85, 1.0, 0.10], [0.05, 0.10, 1.0]])
    cov = _frame(correlation * np.outer(vols, vols))
    inverse_vol = (1.0 / vols) / (1.0 / vols).sum()
    assert not np.allclose(erc_weights(cov).weights.to_numpy(), inverse_vol, atol=1e-3)


@pytest.mark.parametrize("n", [2, 3, 4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_realised_risk_contributions_are_equal_on_random_matrices(n, seed):
    rng = np.random.default_rng(seed * 100 + n)
    cov = _random_pd(rng, n)
    solution = erc_weights(_frame(cov, labels=[f"s{i}" for i in range(n)]))
    realised = risk_contributions(solution.weights.to_numpy(), cov)
    assert np.allclose(realised, 1.0 / n, atol=1e-10), realised
    assert solution.worst_deviation < 1e-10
    assert solution.converged


def test_worst_deviation_is_measured_from_the_returned_weights():
    cov = _frame([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
    solution = erc_weights(cov)
    recomputed = risk_contributions(solution.weights.to_numpy(), cov.to_numpy())
    expected = float(np.max(np.abs(recomputed - 1.0 / 3.0)))
    assert solution.worst_deviation == pytest.approx(expected, abs=1e-15)


def test_weights_are_positive_and_sum_to_one():
    rng = np.random.default_rng(7)
    solution = erc_weights(_frame(_random_pd(rng, 3)))
    assert (solution.weights > 0).all()
    assert float(solution.weights.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("factor", [1e-6, 1.0, 252.0, 1e6])
def test_weights_are_invariant_to_the_scale_of_the_covariance(factor):
    rng = np.random.default_rng(11)
    cov = _frame(_random_pd(rng, 3))
    base = erc_weights(cov).weights.to_numpy()
    scaled = erc_weights(cov * factor).weights.to_numpy()
    assert np.allclose(base, scaled, atol=1e-11)


@pytest.mark.parametrize("factor", [1e-20, 1e-8, 1.0, 252.0, 1e8, 1e20, 1e28])
def test_convergence_holds_at_extreme_covariance_scales(factor):
    cov = _frame([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]]) * factor
    solution = erc_weights(cov)
    assert solution.converged, f"did not converge at scale {factor:g}"
    assert solution.worst_deviation < 1e-11, solution.worst_deviation
    realised = risk_contributions(solution.weights.to_numpy(), cov.to_numpy())
    assert np.allclose(realised, 1.0 / 3.0, atol=1e-10)


def test_convergence_flag_is_honest_when_the_budget_runs_out():
    cov = _frame([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
    starved = erc_weights(cov, max_iter=1)
    assert not starved.converged
    assert starved.iterations == 1
    recomputed = risk_contributions(starved.weights.to_numpy(), cov.to_numpy())
    assert starved.worst_deviation == pytest.approx(
        float(np.max(np.abs(recomputed - 1.0 / 3.0))), abs=1e-15
    )


def test_solution_carries_labels_from_the_frame():
    cov = _frame([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]])
    solution = erc_weights(cov)
    assert list(solution.weights.index) == LABELS
    assert list(solution.risk_contributions.index) == LABELS
    assert isinstance(solution, ErcSolution)


def test_risk_contributions_sum_to_one_for_arbitrary_weights():
    rng = np.random.default_rng(3)
    cov = _random_pd(rng, 4)
    weights = rng.uniform(0.05, 1.0, size=4)
    assert float(risk_contributions(weights, cov).sum()) == pytest.approx(1.0, abs=1e-12)


def test_singular_covariance_is_refused_rather_than_regularised():
    cov = _frame([[0.04, 0.04, 0.0], [0.04, 0.04, 0.0], [0.0, 0.0, 0.09]])
    with pytest.raises(ValueError, match="positive definite"):
        erc_weights(cov)


def test_zero_variance_sleeve_is_refused():
    cov = _frame([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.09]])
    with pytest.raises(ValueError, match="non-positive variance"):
        erc_weights(cov)


def test_asymmetric_matrix_is_refused():
    cov = _frame([[0.04, 0.01, 0.0], [0.02, 0.09, 0.0], [0.0, 0.0, 0.16]])
    with pytest.raises(ValueError, match="not symmetric"):
        erc_weights(cov)


@pytest.mark.parametrize("factor", [1e-8, 1.0, 252.0, 1e8])
def test_symmetry_tolerance_scales_with_the_matrix(factor):
    base = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])

    rounding_sized = base.copy()
    rounding_sized[0, 1] += 1e-15 * base.max()
    erc_weights(_frame(rounding_sized) * factor)

    genuinely_asymmetric = base.copy()
    genuinely_asymmetric[0, 1] += 1e-6 * base.max()
    with pytest.raises(ValueError, match="not symmetric"):
        erc_weights(_frame(genuinely_asymmetric) * factor)


def test_non_finite_matrix_is_refused():
    cov = _frame([[0.04, np.nan, 0.0], [np.nan, 0.09, 0.0], [0.0, 0.0, 0.16]])
    with pytest.raises(ValueError, match="non-finite"):
        erc_weights(cov)


def test_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        erc_weights(np.zeros((2, 3)))


def test_risk_contributions_refuse_a_degenerate_portfolio():
    with pytest.raises(ValueError, match="undefined"):
        risk_contributions(np.zeros(3), np.eye(3))


DATE = pd.Timestamp("2015-06-01")
PIPELINE = dict(date=DATE, vol_target=0.10, gross_cap=1.0, min_weight=0.05, max_weight=0.60)


def test_both_orders_agree_when_no_constraint_binds():
    cov = _frame(np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]) * 0.15**2)
    results = {order: allocate(cov, order=order, **PIPELINE) for order in WEIGHT_ORDERS}
    a, b = (results[order].weights.to_numpy() for order in WEIGHT_ORDERS)
    assert np.allclose(a, b, atol=1e-12)
    assert not results["clip-first"].gross_cap_binding
    assert not results["clip-first"].min_max_binding


def test_the_two_orders_diverge_when_the_ceiling_binds():
    vols = np.array([0.03, 0.30, 0.30])
    cov = _frame(np.eye(3) * vols**2)
    results = {order: allocate(cov, order=order, **PIPELINE) for order in WEIGHT_ORDERS}
    a, b = (results[order].weights.to_numpy() for order in WEIGHT_ORDERS)
    assert not np.allclose(a, b, atol=1e-6)


def test_clip_first_applies_the_bounds_before_renormalising():
    vols = np.array([0.02, 0.30, 0.30])
    cov = _frame(np.eye(3) * vols**2)
    solved = allocate(cov, order="clip-first", **PIPELINE)
    assert solved.erc.iloc[0] > PIPELINE["max_weight"]
    assert solved.min_max_binding
    assert solved.bound_breach > 0
    assert float(solved.normalised.sum()) == pytest.approx(1.0, abs=1e-12)


def test_gross_cap_is_never_exceeded_under_either_order():
    rng = np.random.default_rng(5)
    for seed in range(25):
        cov = _frame(_random_pd(np.random.default_rng(seed), 3))
        for order in WEIGHT_ORDERS:
            solved = allocate(cov, order=order, **PIPELINE)
            assert float(solved.weights.abs().sum()) <= PIPELINE["gross_cap"] + 1e-9
    del rng


def test_vol_target_is_hit_exactly_when_the_gross_cap_does_not_bind():
    vols = np.array([0.30, 0.35, 0.40])
    cov = _frame(np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]]) * np.outer(vols, vols))
    solved = allocate(cov, order="clip-first", **PIPELINE)
    assert not solved.gross_cap_binding
    assert solved.final_vol == pytest.approx(PIPELINE["vol_target"], rel=1e-10)


def test_gross_cap_truncates_k_when_it_binds():
    vols = np.array([0.03, 0.04, 0.05])
    cov = _frame(np.eye(3) * vols**2)
    solved = allocate(cov, order="clip-first", **PIPELINE)
    assert solved.gross_cap_binding
    assert solved.k < solved.k_uncapped
    assert float(solved.weights.sum()) == pytest.approx(PIPELINE["gross_cap"], abs=1e-12)
    assert solved.final_vol < PIPELINE["vol_target"]


def test_unknown_weight_order_is_refused_and_has_no_default():
    cov = _frame(np.eye(3) * 0.04)
    with pytest.raises(ValueError, match="unknown weight order"):
        allocate(cov, order="whatever-looks-best", **PIPELINE)
    with pytest.raises(TypeError):
        allocate(cov, **PIPELINE)


def test_infeasible_bounds_are_refused():
    cov = _frame(np.eye(3) * 0.04)
    settings = dict(PIPELINE, min_weight=0.40, max_weight=0.60)
    with pytest.raises(ValueError, match="infeasible"):
        allocate(cov, order="clip-first", **settings)


def test_bounds_must_be_ordered():
    cov = _frame(np.eye(3) * 0.04)
    with pytest.raises(ValueError, match="not ordered"):
        allocate(cov, order="clip-first", **dict(PIPELINE, min_weight=0.7, max_weight=0.6))


def test_non_positive_vol_target_and_gross_cap_are_refused():
    cov = _frame(np.eye(3) * 0.04)
    with pytest.raises(ValueError, match="volatility target"):
        allocate(cov, order="clip-first", **dict(PIPELINE, vol_target=0.0))
    with pytest.raises(ValueError, match="gross exposure cap"):
        allocate(cov, order="clip-first", **dict(PIPELINE, gross_cap=0.0))


def test_pipeline_preserves_equal_risk_contribution_before_the_caps():
    rng = np.random.default_rng(21)
    cov = _frame(_random_pd(rng, 3))
    solved = allocate(cov, order="clip-first", **PIPELINE)
    if solved.min_max_binding:
        pytest.skip("a bound bound; the ERC property is not preserved through a clip")
    realised = risk_contributions(solved.weights.to_numpy(), cov.to_numpy())
    assert np.allclose(realised, 1.0 / 3.0, atol=1e-10)
