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


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
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
    labels = list(cov.columns) if isinstance(cov, pd.DataFrame) else None
    matrix = np.asarray(cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 1:
        raise ValueError("cannot allocate across an empty universe")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance matrix contains non-finite entries")
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

    y = 1.0 / np.sqrt(diagonal)
    y = y / np.sqrt(float(y @ matrix @ y))

    target = 1.0 / n
    converged = False
    iterations = 0
    worst = float("inf")
    for iterations in range(1, max_iter + 1):
        previous = y.copy()
        for i in range(n):
            b = float(matrix[i] @ y) - matrix[i, i] * y[i]
            y[i] = (-b + np.sqrt(b * b + 4.0 * matrix[i, i])) / (2.0 * matrix[i, i])
        worst = float(np.max(np.abs(risk_contributions(y / y.sum(), matrix) - target)))
        if worst <= tol:
            converged = True
            break
        if np.array_equal(y, previous):
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


WEIGHT_ORDERS = ("clip-first", "scale-first")


@dataclass(frozen=True, slots=True)
class SleeveWeights:
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
    else:
        normalised = solution.weights
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
