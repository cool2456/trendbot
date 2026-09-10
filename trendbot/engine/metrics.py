from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PerformanceStats",
    "equity_curve",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "drawdown_series",
    "time_underwater",
    "calmar",
    "annual_turnover",
    "annual_weight_churn",
    "average_gross_exposure",
    "calendar_year_returns",
    "longest_drawdown_days",
    "average_net_exposure",
    "invested_fraction",
    "hit_rate",
    "summarise",
]

TRADING_DAYS_PER_YEAR = 252


def _as_series(returns: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(returns, pd.Series):
        out = returns.astype(float)
    else:
        out = pd.Series(np.asarray(returns, dtype=float))
    if out.isna().any():
        raise ValueError(
            "return series contains NaN; the engine must emit 0.0 for a flat day, "
            "never NaN, so a NaN here means a real bug upstream"
        )
    return out


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + _as_series(returns)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    total_growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0:
        return float("nan")
    if total_growth <= 0:
        return -1.0
    return total_growth ** (1.0 / years) - 1.0


def _is_degenerate(r: pd.Series, sd: float) -> bool:
    if not np.isfinite(sd):
        return True
    scale = float(np.max(np.abs(r.to_numpy()))) if len(r) else 0.0
    return sd <= max(scale, 1.0) * 1e-12


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if _is_degenerate(r, sd):
        return float("nan")
    return float(r.mean()) / sd * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if _is_degenerate(r, dd):
        return float("nan")
    return float(r.mean()) / dd * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def time_underwater(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float((dd < 0).mean()) if len(dd) else float("nan")


def longest_drawdown_days(returns: pd.Series) -> int:
    dd = drawdown_series(returns)
    if not len(dd):
        return 0
    under = (dd < 0).to_numpy()
    best = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0.0:
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def annual_weight_churn(weights: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if weights.empty:
        return float("nan")
    dw = weights.diff()
    dw.iloc[0] = weights.iloc[0]
    total = float(dw.abs().sum(axis=1).sum())
    years = len(weights) / periods_per_year
    return total / years if years > 0 else float("nan")


def annual_turnover(
    trades: pd.DataFrame,
    n_periods: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if trades is None or trades.empty or n_periods <= 0:
        return float("nan")
    total = float(trades.abs().sum(axis=1).sum())
    years = n_periods / periods_per_year
    return total / years if years > 0 else float("nan")


def average_gross_exposure(weights: pd.DataFrame) -> float:
    return float(weights.abs().sum(axis=1).mean()) if not weights.empty else float("nan")


def average_net_exposure(weights: pd.DataFrame) -> float:
    return float(weights.sum(axis=1).mean()) if not weights.empty else float("nan")


def invested_fraction(weights: pd.DataFrame) -> float:
    if weights.empty:
        return float("nan")
    return float((weights.abs().sum(axis=1) > 0).mean())


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    r = _as_series(returns)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("calendar_year_returns needs a DatetimeIndex")
    return r.groupby(r.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))


def hit_rate(returns: pd.Series) -> float:
    r = _as_series(returns)
    active = r[r != 0.0]
    return float((active > 0).mean()) if len(active) else float("nan")


@dataclass(frozen=True, slots=True)
class PerformanceStats:
    n_periods: int
    years: float
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    time_underwater: float
    longest_drawdown_days: int
    hit_rate: float
    skew: float
    excess_kurtosis: float
    ann_turnover: float
    ann_weight_churn: float
    avg_gross_exposure: float
    avg_net_exposure: float
    invested_fraction: float
    losing_years: int
    n_years_observed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"CAGR {self.cagr:7.2%} | vol {self.ann_vol:6.2%} | Sharpe {self.sharpe:6.2f} | "
            f"Sortino {self.sortino:6.2f} | maxDD {self.max_drawdown:7.2%} | "
            f"turnover {self.ann_turnover:5.2f}x | gross {self.avg_gross_exposure:5.2%} | "
            f"{self.years:.1f}y"
        )


def summarise(
    returns: pd.Series,
    weights: pd.DataFrame | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    trades: pd.DataFrame | None = None,
) -> PerformanceStats:
    r = _as_series(returns)
    empty_weights = pd.DataFrame(index=r.index) if weights is None else weights

    if isinstance(r.index, pd.DatetimeIndex) and len(r):
        yearly = calendar_year_returns(r)
        losing = int((yearly < 0).sum())
        n_years = int(len(yearly))
    else:
        losing, n_years = 0, 0

    return PerformanceStats(
        n_periods=len(r),
        years=len(r) / periods_per_year,
        total_return=float((1.0 + r).prod() - 1.0) if len(r) else float("nan"),
        cagr=cagr(r, periods_per_year),
        ann_vol=float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan"),
        sharpe=sharpe(r, periods_per_year),
        sortino=sortino(r, periods_per_year),
        max_drawdown=max_drawdown(r),
        calmar=calmar(r, periods_per_year),
        time_underwater=time_underwater(r),
        longest_drawdown_days=longest_drawdown_days(r),
        hit_rate=hit_rate(r),
        skew=float(r.skew()) if len(r) > 2 else float("nan"),
        excess_kurtosis=float(r.kurtosis()) if len(r) > 3 else float("nan"),
        ann_turnover=annual_turnover(trades, len(r), periods_per_year)
        if trades is not None and not trades.empty
        else 0.0,
        ann_weight_churn=annual_weight_churn(empty_weights, periods_per_year)
        if not empty_weights.empty
        else 0.0,
        avg_gross_exposure=average_gross_exposure(empty_weights)
        if not empty_weights.empty
        else 0.0,
        avg_net_exposure=average_net_exposure(empty_weights) if not empty_weights.empty else 0.0,
        invested_fraction=invested_fraction(empty_weights) if not empty_weights.empty else 0.0,
        losing_years=losing,
        n_years_observed=n_years,
    )
