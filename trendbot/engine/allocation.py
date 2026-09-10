"""Equal-risk-contribution allocation — PREREG_006.md section 3.

This is the only new machinery experiment 006 needs. Everything else it does is a
call into code experiments 001, 002 and 005 already validated: the covariance comes
from :func:`trendbot.sizing.ewma_covariance`, the backtest from
:func:`trendbot.engine.panel_backtest.run_panel_backtest`, the metrics from
:mod:`trendbot.engine.metrics`.

Why a solver at all
-------------------
Section 3 asks for weights satisfying ``w_i * (Sw)_i`` equal for all ``i``. For two
assets that has a closed form. For three it does not, and a formula that looks like
one — inverse-vol, or inverse-variance — is *not* equal risk contribution unless the
correlations happen to be identical. Getting this wrong produces weights that look
plausible and are not what the document specifies, which is why
:class:`ErcSolution` carries the realised risk contributions and the worst deviation
from equality rather than leaving the caller to trust the solve.

The formulation
---------------
Spinu (2013) and Griveau-Billion, Richard & Roncalli (2013) show the ERC portfolio is
the unique solution of a *strictly convex* problem::

    min_{y > 0}  f(y) = 0.5 * y' S y - sum_i log(y_i)

whose stationarity condition ``(Sy)_i = 1 / y_i`` is exactly ``y_i (Sy)_i = 1``, i.e.
equal risk contributions; the ERC weights are then ``w = y / sum(y)``. Solving the
convex problem rather than least-squares on the risk-contribution deviations matters:
the least-squares surface is non-convex and has stationary points that are not ERC,
so a solver can stop somewhere plausible and wrong. This one cannot: the objective is
strictly convex on the positive orthant for any positive-definite ``S``, so there is
one stationary point and it is the answer.

Cyclical coordinate descent minimises it. Fixing every coordinate but ``i`` leaves a
quadratic in ``y_i`` with one positive root::

    y_i = (-b + sqrt(b^2 + 4 * S_ii)) / (2 * S_ii),   b = sum_{j != i} S_ij y_j

which is a closed-form exact minimisation along that coordinate, so the objective
decreases monotonically and the iteration cannot oscillate.

Scale invariance
----------------
``f`` is minimised at a ``y`` whose scale depends on the units of ``S``, but ``w``
does not: multiplying ``S`` by a constant rescales ``y`` and leaves ``y / sum(y)``
alone. Annualised and daily covariance therefore give identical weights, and
:func:`erc_weights` is tested for exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ErcSolution",
    "SleeveWeights",
    "erc_weights",
    "risk_contributions",
    "allocate",
]


# --------------------------------------------------------------------------------------
# risk contributions
# --------------------------------------------------------------------------------------


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fractional risk contribution of each position, summing to one.

    ``RC_i = w_i (S w)_i / (w' S w)``. The denominator is the portfolio variance, so
    these are variance shares; because ``sum_i w_i (Sw)_i = w' S w`` identically, they
    sum to one for any weights whatsoever and the "equal" in equal risk contribution
    means every entry equals ``1 / N``.
    """
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    marginal = cov @ weights
    variance = float(weights @ marginal)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(
            f"portfolio variance is {variance!r}; risk contributions are undefined. "
            "A non-positive variance means the covariance matrix is not positive definite."
        )
    return (weights * marginal) / variance


@dataclass(frozen=True, slots=True)
class ErcSolution:
    """One equal-risk-contribution solve, with the evidence that it converged."""

    weights: pd.Series
    risk_contributions: pd.Series
    worst_deviation: float
    iterations: int
    converged: bool

    @property
    def n(self) -> int:
        return len(self.weights)

    def __str__(self) -> str:
        return (
            f"ERC on {self.n} sleeves: weights "
            f"{[round(w, 4) for w in self.weights]}, worst risk-contribution deviation "
            f"from 1/{self.n} = {self.worst_deviation:.3e} in "
            f"{self.iterations} iterations "
            f"({'converged' if self.converged else 'DID NOT CONVERGE'})"
        )


def erc_weights(
    cov: pd.DataFrame | np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> ErcSolution:
    """Solve for equal risk contribution across the columns of ``cov``.

    Parameters
    ----------
    cov:
        Positive-definite covariance matrix. Units are irrelevant — see the module
        docstring on scale invariance.
    tol:
        Convergence tolerance on the largest ``|RC_i - 1/N|``, i.e. on the property
        actually being solved for. Convergence is *not* tested on the change in ``y``
        between sweeps: the scale of ``y`` depends on the units of ``S`` — it shrinks
        like ``1/sqrt(c)`` when ``S`` is multiplied by ``c`` — so any tolerance on ``y``
        with an absolute floor stops being scale-invariant at extreme magnitudes and
        would report convergence on an unconverged solve. Risk contributions are
        fractions of variance summing to one, so a tolerance on them means the same
        thing at every scale.

    Returns
    -------
    ErcSolution
        ``weights`` sum to one and are strictly positive. ``worst_deviation`` is the
        largest ``|RC_i - 1/N|`` actually realised, computed from the returned weights
        rather than asserted from the solver's internal state — a solver that lies
        about convergence still cannot fake this number.
    """
    labels = list(cov.columns) if isinstance(cov, pd.DataFrame) else None
    matrix = np.asarray(cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 1:
        raise ValueError("cannot allocate across an empty universe")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance matrix contains non-finite entries")
    # The tolerance has to scale with the matrix, because this function is documented
    # as scale-invariant and is called with both daily and annualised covariance. A
    # fixed absolute tolerance would accept a daily matrix and reject the same matrix
    # multiplied by 252, whose rounding error is 252 times larger in absolute terms.
    magnitude = float(np.max(np.abs(matrix)))
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > 1e-12 * max(magnitude, 1e-300):
        raise ValueError(
            f"covariance matrix is not symmetric (worst |S - S'| = {asymmetry:.3e} "
            f"against a scale of {magnitude:.3e})"
        )
    diagonal = np.diag(matrix)
    if (diagonal <= 0).any():
        raise ValueError(
            f"covariance has a non-positive variance on the diagonal: {diagonal.tolist()}. "
            "A sleeve with zero variance has no risk to equalise."
        )
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 0:
        raise ValueError(
            f"covariance is not positive definite (smallest eigenvalue "
            f"{eigenvalues.min():.3e}); the ERC problem is not convex on it and the "
            "solution would not be unique"
        )

    # Start from the inverse-volatility portfolio. It is the exact answer when every
    # correlation is equal, so on realistic inputs the sweep begins close and the
    # iteration count reported below is a meaningful diagnostic rather than noise.
    y = 1.0 / np.sqrt(diagonal)
    y = y / np.sqrt(float(y @ matrix @ y))

    target = 1.0 / n
    converged = False
    iterations = 0
    worst = float("inf")
    for iterations in range(1, max_iter + 1):
        previous = y.copy()
        for i in range(n):
            # b is the covariance of coordinate i with everything else, holding the
            # others fixed. Excluding i by subtracting its own term avoids building a
            # mask and keeps the sweep O(n) per coordinate.
            b = float(matrix[i] @ y) - matrix[i, i] * y[i]
            y[i] = (-b + np.sqrt(b * b + 4.0 * matrix[i, i])) / (2.0 * matrix[i, i])
        worst = float(np.max(np.abs(risk_contributions(y / y.sum(), matrix) - target)))
        if worst <= tol:
            converged = True
            break
        if np.array_equal(y, previous):
            # A fixed point in floating point: no further sweep can improve it, so stop
            # and report honestly whether the tolerance was actually met.
            break

    weights = y / y.sum()
    contributions = risk_contributions(weights, matrix)
    worst = float(np.max(np.abs(contributions - target)))
    index = pd.Index(labels, name="sleeve") if labels is not None else pd.RangeIndex(n)
    return ErcSolution(
        weights=pd.Series(weights, index=index, name="weight"),
        risk_contributions=pd.Series(contributions, index=index, name="risk_contribution"),
        worst_deviation=worst,
        iterations=iterations,
        converged=converged,
    )


# --------------------------------------------------------------------------------------
# section 3's weight pipeline
# --------------------------------------------------------------------------------------


# Section 3 states four operations - solve for equal risk contribution, apply a 5%/60%
# per-sleeve bound "applied before renormalisation", renormalise, and scale the vector so
# ex-ante volatility hits 10% subject to a gross cap of 1.0 - and never says where the vol
# target enters. Two orders are defensible and they do not agree, so both are implemented
# and both are reported. Nothing downstream may pick one silently: `order` has no default.
#
# "clip-first"  erc -> clip[min,max] -> renormalise to sum 1 -> x k -> gross cap
#     The textual reading. "Renormalisation" then unambiguously names the divide-by-sum
#     that restores sum-to-one after clipping, which is the only operation in the pipeline
#     that word can denote, and the 5%/60% bounds are bounds on the *allocation* - a
#     vector summing to one - where a 5% floor and a 60% ceiling are coherent and feasible.
#
# "scale-first" erc -> x k -> clip[min,max] -> gross cap
#     The precedent reading: :func:`trendbot.sizing.target_weights` settles the identical
#     question for experiment 001 in this order, and PREREG_006.md section 6 says to follow
#     committed implementations. It is a weaker precedent than it looks, because 001 has
#     only a per-instrument *maximum*; nothing in 001 corresponds to section 3's 5% floor,
#     and a floor applied to already-vol-scaled weights inflates the book above the vol
#     target whenever k is small, which is why this is not the headline.
WEIGHT_ORDERS = ("clip-first", "scale-first")


@dataclass(frozen=True, slots=True)
class SleeveWeights:
    """Section 3's whole weight pipeline for one rebalance date, stage by stage.

    Every intermediate is kept so the ordering is auditable in the output rather than
    buried in the code, which matters because the order is not pinned by the document.
    """

    date: pd.Timestamp
    order: str
    erc: pd.Series
    normalised: pd.Series
    weights: pd.Series
    k: float
    k_uncapped: float
    exante_vol: float
    realised_risk_contributions: pd.Series
    erc_worst_deviation: float
    erc_iterations: int
    erc_converged: bool
    min_max_binding: bool
    gross_cap_binding: bool
    bound_breach: float
    final_vol: float

    @property
    def gross(self) -> float:
        return float(self.weights.abs().sum())


def allocate(
    cov: pd.DataFrame,
    *,
    date: pd.Timestamp,
    vol_target: float,
    gross_cap: float,
    min_weight: float,
    max_weight: float,
    order: str,
    tol: float = 1e-12,
) -> SleeveWeights:
    """Section 3, for one rebalance date, under one of the two readings of its order.

    ``cov`` must be *annualised*, because ``vol_target`` is: this function does no
    annualising of its own and would silently target a daily volatility if handed a
    daily matrix.

    Under ``"clip-first"``, clipping and then renormalising can push a weight back
    outside its own bound — clipping to a sum above one and dividing through moves
    every weight down. Section 3 says the bounds are "applied before renormalisation",
    which read literally means applied once, so that is what happens here; the
    resulting breach is measured into ``bound_breach`` and reported rather than
    iterated away, because iterating to a fixed point would be a construction the
    document does not specify.
    """
    if order not in WEIGHT_ORDERS:
        raise ValueError(f"unknown weight order {order!r}; expected one of {WEIGHT_ORDERS}")
    if not vol_target > 0:
        raise ValueError("volatility target must be positive")
    if not gross_cap > 0:
        raise ValueError("gross exposure cap must be positive")
    if not 0.0 <= min_weight <= max_weight:
        raise ValueError(f"weight bounds [{min_weight}, {max_weight}] are not ordered")
    n = cov.shape[0]
    if min_weight * n > 1.0 + 1e-12 or max_weight * n < 1.0 - 1e-12:
        raise ValueError(
            f"weight bounds [{min_weight}, {max_weight}] cannot produce {n} weights "
            "summing to one; section 3's constraints are infeasible on this universe"
        )

    solution = erc_weights(cov, tol=tol)
    matrix = cov.reindex(index=solution.weights.index, columns=solution.weights.index).to_numpy(
        dtype=float
    )

    def exante_of(vector: pd.Series) -> float:
        v = vector.to_numpy(dtype=float)
        return float(np.sqrt(max(float(v @ matrix @ v), 0.0)))

    def breach_of(vector: pd.Series, lower: float, upper: float) -> float:
        v = vector.to_numpy(dtype=float)
        return float(np.max(np.maximum(lower - v, v - upper).clip(min=0.0)))

    if order == "clip-first":
        clipped = solution.weights.clip(lower=min_weight, upper=max_weight)
        min_max_binding = not np.allclose(
            clipped.to_numpy(), solution.weights.to_numpy(), atol=1e-12, rtol=0.0
        )
        normalised = clipped / clipped.sum()
        bound_breach = breach_of(normalised, min_weight, max_weight)
        exante = exante_of(normalised)
        if not exante > 0:
            raise ValueError(
                f"ex-ante portfolio volatility is {exante!r} at {date}; the vol target "
                "cannot be hit and section 3 gives no fallback"
            )
        k_uncapped = vol_target / exante
        k = k_uncapped
        gross = k * float(normalised.abs().sum())
        gross_cap_binding = gross > gross_cap + 1e-12
        if gross_cap_binding:
            k = gross_cap / float(normalised.abs().sum())
        weights = normalised * k
    else:  # "scale-first"
        normalised = solution.weights  # the ERC solve already sums to one
        exante = exante_of(normalised)
        if not exante > 0:
            raise ValueError(
                f"ex-ante portfolio volatility is {exante!r} at {date}; the vol target "
                "cannot be hit and section 3 gives no fallback"
            )
        k_uncapped = vol_target / exante
        k = k_uncapped
        scaled = normalised * k
        weights = scaled.clip(lower=min_weight, upper=max_weight)
        min_max_binding = not np.allclose(
            weights.to_numpy(), scaled.to_numpy(), atol=1e-12, rtol=0.0
        )
        gross = float(weights.abs().sum())
        gross_cap_binding = gross > gross_cap + 1e-12
        if gross_cap_binding:
            weights = weights * (gross_cap / gross)
        bound_breach = breach_of(weights, min_weight, max_weight)

    return SleeveWeights(
        date=date,
        order=order,
        erc=solution.weights,
        normalised=normalised,
        weights=weights,
        k=float(k),
        k_uncapped=float(k_uncapped),
        exante_vol=exante,
        realised_risk_contributions=solution.risk_contributions,
        erc_worst_deviation=solution.worst_deviation,
        erc_iterations=solution.iterations,
        erc_converged=solution.converged,
        min_max_binding=bool(min_max_binding),
        gross_cap_binding=bool(gross_cap_binding),
        bound_breach=bound_breach,
        final_vol=exante_of(weights),
    )
