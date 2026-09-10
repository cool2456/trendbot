"""Equal-risk-contribution allocation — PREREG_006.md section 3.

The solver is the one piece of genuinely new arithmetic experiment 006 contains, and
the failure mode it has is silent: a solver that stops early returns weights that look
like an allocation and are not one. So it is checked against cases whose answer is known
in closed form before it is trusted on a case whose answer is not.

Three such cases exist and all three are used:

* identical variances and identical correlations -> equal weights;
* a diagonal covariance -> inverse volatility;
* **two assets -> inverse volatility whatever the correlation**, which is the sharpest
  of the three because it is the one case where "ERC" and "inverse vol" agree for a
  non-trivial reason, and a solver that had quietly implemented inverse-vol for N > 2
  would pass the other two and fail nothing here.

The fourth check has no closed form and is therefore the definition itself: on random
positive-definite matrices the realised risk contributions must be equal, computed from
the returned weights rather than from anything the solver reports about itself.
"""

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
    """A positive-definite covariance with realistic dispersion of variances."""
    factor = rng.normal(size=(n, n + 4))
    matrix = factor @ factor.T / (n + 4)
    scale = np.diag(rng.uniform(0.05, 0.30, size=n) / np.sqrt(np.diag(matrix)))
    return scale @ matrix @ scale


# --------------------------------------------------------------------------------------
# the three closed-form cases
# --------------------------------------------------------------------------------------


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
    """The case that separates equal risk contribution from anything resembling it.

    For N = 2 the ERC weights are exactly inverse-volatility for *every* correlation.
    A solver that had silently implemented inverse-variance, or that ignored the
    off-diagonal, would still pass the equicorrelated and diagonal cases above.
    """
    vols = np.array([0.08, 0.20])
    cov = _frame(np.array([[1.0, rho], [rho, 1.0]]) * np.outer(vols, vols), labels=["A", "B"])
    expected = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(erc_weights(cov).weights.to_numpy(), expected, atol=1e-10)


def test_inverse_volatility_is_not_the_answer_for_three_correlated_assets():
    """Guards the guard: the closed-form cases must not be the only ones that exist.

    If inverse-vol coincided with ERC here too, the tests above would be vacuous.
    """
    vols = np.array([0.08, 0.10, 0.25])
    correlation = np.array([[1.0, 0.85, 0.05], [0.85, 1.0, 0.10], [0.05, 0.10, 1.0]])
    cov = _frame(correlation * np.outer(vols, vols))
    inverse_vol = (1.0 / vols) / (1.0 / vols).sum()
    assert not np.allclose(erc_weights(cov).weights.to_numpy(), inverse_vol, atol=1e-3)


# --------------------------------------------------------------------------------------
# the definition itself
# --------------------------------------------------------------------------------------


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
    """The reported deviation must not be the solver's opinion of itself."""
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
    """Annualised and daily covariance must give the same allocation."""
    rng = np.random.default_rng(11)
    cov = _frame(_random_pd(rng, 3))
    base = erc_weights(cov).weights.to_numpy()
    scaled = erc_weights(cov * factor).weights.to_numpy()
    assert np.allclose(base, scaled, atol=1e-11)


@pytest.mark.parametrize("factor", [1e-20, 1e-8, 1.0, 252.0, 1e8, 1e20, 1e28])
def test_convergence_holds_at_extreme_covariance_scales(factor):
    """The stopping rule must mean the same thing at every scale.

    ``y`` shrinks like ``1/sqrt(c)`` when the covariance is multiplied by ``c``, so a
    tolerance on the change in ``y`` with an absolute floor stops being scale-invariant
    at extreme magnitudes and reports convergence on an unconverged solve. The criterion
    is therefore the risk-contribution deviation itself, which is a fraction of variance
    and means the same thing at any scale.
    """
    cov = _frame([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]]) * factor
    solution = erc_weights(cov)
    assert solution.converged, f"did not converge at scale {factor:g}"
    assert solution.worst_deviation < 1e-11, solution.worst_deviation
    realised = risk_contributions(solution.weights.to_numpy(), cov.to_numpy())
    assert np.allclose(realised, 1.0 / 3.0, atol=1e-10)


def test_convergence_flag_is_honest_when_the_budget_runs_out():
    """A solver that stops early must say so rather than claim success."""
    cov = _frame([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
    starved = erc_weights(cov, max_iter=1)
    assert not starved.converged
    assert starved.iterations == 1
    # The reported deviation is still measured from the returned weights, so the caller
    # can see exactly how far off the unconverged answer is.
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
    """The identity the 'equal' in ERC is relative to."""
    rng = np.random.default_rng(3)
    cov = _random_pd(rng, 4)
    weights = rng.uniform(0.05, 1.0, size=4)
    assert float(risk_contributions(weights, cov).sum()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


def test_singular_covariance_is_refused_rather_than_regularised():
    """Two perfectly collinear sleeves have no unique ERC solution."""
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
    """The same verdict at every scale, for rounding-sized and for real asymmetry.

    A fixed absolute tolerance would accept a daily matrix and reject the same matrix
    multiplied by 252, whose rounding error is 252 times larger in absolute terms -
    which would contradict the documented scale invariance. Both halves are checked,
    so a tolerance that had simply been loosened to nothing would fail the second.
    """
    base = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])

    rounding_sized = base.copy()
    rounding_sized[0, 1] += 1e-15 * base.max()  # far inside any sane tolerance
    erc_weights(_frame(rounding_sized) * factor)  # must not raise, at any scale

    genuinely_asymmetric = base.copy()
    genuinely_asymmetric[0, 1] += 1e-6 * base.max()  # a real disagreement
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


# --------------------------------------------------------------------------------------
# section 3's weight pipeline
# --------------------------------------------------------------------------------------

DATE = pd.Timestamp("2015-06-01")
PIPELINE = dict(date=DATE, vol_target=0.10, gross_cap=1.0, min_weight=0.05, max_weight=0.60)


def test_both_orders_agree_when_no_constraint_binds():
    """The two readings of section 3 differ only where a bound or the cap bites."""
    # An equicorrelated matrix gives weights of 1/3 each - inside [5%, 60%] - and a
    # volatility HIGH enough that k stays under 1, so the gross cap does not bind
    # either. At 15% vol and rho 0.3 the equal-weight portfolio runs at 10.95%, just
    # above the 10% target, so k is 0.91.
    cov = _frame(np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]) * 0.15**2)
    results = {order: allocate(cov, order=order, **PIPELINE) for order in WEIGHT_ORDERS}
    a, b = (results[order].weights.to_numpy() for order in WEIGHT_ORDERS)
    assert np.allclose(a, b, atol=1e-12)
    assert not results["clip-first"].gross_cap_binding
    assert not results["clip-first"].min_max_binding


def test_the_two_orders_diverge_when_the_ceiling_binds():
    """If they never diverged, reporting both would be theatre."""
    vols = np.array([0.03, 0.30, 0.30])
    cov = _frame(np.eye(3) * vols**2)
    results = {order: allocate(cov, order=order, **PIPELINE) for order in WEIGHT_ORDERS}
    a, b = (results[order].weights.to_numpy() for order in WEIGHT_ORDERS)
    assert not np.allclose(a, b, atol=1e-6)


def test_clip_first_applies_the_bounds_before_renormalising():
    """Section 3's literal order, including the breach it can leave behind."""
    vols = np.array([0.02, 0.30, 0.30])
    cov = _frame(np.eye(3) * vols**2)
    solved = allocate(cov, order="clip-first", **PIPELINE)
    # The raw ERC weight on the low-vol sleeve exceeds the 60% ceiling ...
    assert solved.erc.iloc[0] > PIPELINE["max_weight"]
    # ... it is clipped to it, and renormalising a vector that then sums to less than
    # one pushes it back above the ceiling. Section 3 says the bounds are applied
    # before renormalisation, so this residual is real and is reported.
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
    # Volatile sleeves make k < 1, so the cap is slack and the target must be met.
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
        allocate(cov, **PIPELINE)  # `order` is keyword-only and required


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
    """Scaling and renormalising are both uniform, so the ERC property survives them."""
    rng = np.random.default_rng(21)
    cov = _frame(_random_pd(rng, 3))
    solved = allocate(cov, order="clip-first", **PIPELINE)
    if solved.min_max_binding:
        pytest.skip("a bound bound; the ERC property is not preserved through a clip")
    realised = risk_contributions(solved.weights.to_numpy(), cov.to_numpy())
    assert np.allclose(realised, 1.0 / 3.0, atol=1e-10)
