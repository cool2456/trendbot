from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .config_002 import Config002
from .config_003 import Config003
from .config_004 import Config004
from .config_005 import Config005
from .engine.panel import Panel, panel_field, validate_panel
from .signal import trend_signal
from .sizing import CovarianceMethod, annualised_vol, ewma_covariance, target_weights
from .xsmom import BucketMethod, cross_sectional_momentum, top_quantile_weights

__all__ = [
    "TimeSeriesTrend",
    "CrossSectionalMomentum",
    "EquityCrossSectionalMomentum",
    "PointInTimeMomentum",
    "CurrencyCrossSectionalMomentum",
]


def _covariance_blocks(cov: pd.DataFrame, dates: pd.Index, universe: list[str]):
    n_dates, n_assets = len(dates), len(universe)
    if len(cov) != n_dates * n_assets:
        raise ValueError(
            f"covariance frame has {len(cov)} rows, expected {n_dates * n_assets} "
            f"({n_dates} dates x {n_assets} instruments)"
        )
    outer = cov.index.get_level_values(0)[:: n_assets]
    if not pd.Index(outer).equals(pd.Index(dates)):
        raise ValueError("covariance blocks are not one per date, in date order")
    inner = list(cov.index.get_level_values(1)[:n_assets])
    if inner != universe or list(cov.columns) != universe:
        raise ValueError("covariance block axes are not in universe order")
    return cov.to_numpy(dtype=float).reshape(n_dates, n_assets, n_assets)


@dataclass(frozen=True, slots=True)
class TimeSeriesTrend:
    cfg: Config
    covariance: CovarianceMethod = "full"

    @property
    def name(self) -> str:
        return f"time-series trend v{self.cfg.version} ({self.covariance} covariance)"

    def __call__(self, panel: Panel) -> pd.DataFrame:
        validate_panel(panel, require=("close",))
        universe = list(self.cfg.universe)
        close = panel_field(panel, "close")
        missing = [t for t in universe if t not in close.columns]
        if missing:
            raise ValueError(f"panel is missing universe members {missing}")
        close = close[universe]

        returns_daily = close.pct_change(fill_method=None)
        sig = trend_signal(close, self.cfg.lookback_days, long_only=self.cfg.long_only)
        sig = sig.reindex(index=close.index, columns=universe).fillna(0.0)
        sigma = annualised_vol(returns_daily, self.cfg.ewma_halflife_days)

        blocks = None
        if self.covariance == "full":
            blocks = _covariance_blocks(
                ewma_covariance(returns_daily, self.cfg.ewma_halflife_days), close.index, universe
            )

        rows = []
        for i, date in enumerate(close.index):
            cov_slice = (
                pd.DataFrame(blocks[i], index=universe, columns=universe)
                if blocks is not None
                else None
            )
            rows.append(
                target_weights(
                    date,
                    sig.iloc[i],
                    sigma.iloc[i],
                    self.cfg,
                    cov=cov_slice,
                    covariance=self.covariance,
                ).weights.rename(date)
            )
        return pd.DataFrame(rows, index=close.index, columns=universe)


@dataclass(frozen=True, slots=True)
class CrossSectionalMomentum:
    cfg: Config002

    @property
    def name(self) -> str:
        return (
            f"cross-sectional momentum (formation {self.cfg.formation_days}d, "
            f"skip {self.cfg.skip_days}d, top 1/{self.cfg.n_quantiles})"
        )

    def momentum(self, panel: Panel) -> pd.DataFrame:
        validate_panel(panel, require=("close",))
        universe = list(self.cfg.universe)
        close = panel_field(panel, "close")
        missing = [t for t in universe if t not in close.columns]
        if missing:
            raise ValueError(f"panel is missing universe members {missing}")
        return cross_sectional_momentum(
            close[universe], self.cfg.formation_days, self.cfg.skip_days
        )

    def __call__(self, panel: Panel) -> pd.DataFrame:
        return top_quantile_weights(self.momentum(panel), self.cfg.n_quantiles)


@dataclass(frozen=True, slots=True)
class EquityCrossSectionalMomentum:
    cfg: Config003
    universe: tuple[str, ...]
    bucket_method: BucketMethod = "even"

    @property
    def name(self) -> str:
        return (
            f"equity cross-sectional momentum (formation {self.cfg.formation_days}d, "
            f"skip {self.cfg.skip_days}d, top 1/{self.cfg.n_quantiles} of "
            f"{len(self.universe)}, {self.bucket_method} buckets)"
        )

    def momentum(self, panel: Panel) -> pd.DataFrame:
        validate_panel(panel, require=("close",))
        universe = list(self.universe)
        close = panel_field(panel, "close")
        missing = [t for t in universe if t not in close.columns]
        if missing:
            raise ValueError(f"panel is missing universe members {missing[:10]}")
        return cross_sectional_momentum(
            close[universe], self.cfg.formation_days, self.cfg.skip_days
        )

    def __call__(self, panel: Panel) -> pd.DataFrame:
        return top_quantile_weights(
            self.momentum(panel), self.cfg.n_quantiles, self.bucket_method
        )


@dataclass(frozen=True, slots=True)
class PointInTimeMomentum:
    cfg: Config004
    membership: pd.DataFrame
    bucket_method: BucketMethod = "even"

    @property
    def name(self) -> str:
        return (
            f"point-in-time cross-sectional momentum (formation "
            f"{self.cfg.formation_days}d, skip {self.cfg.skip_days}d, top "
            f"1/{self.cfg.n_quantiles} of {self.cfg.universe_size})"
        )

    def eligible_momentum(self, panel: Panel) -> pd.DataFrame:
        validate_panel(panel, require=("close",))
        close = panel_field(panel, "close")
        momentum = cross_sectional_momentum(
            close, self.cfg.formation_days, self.cfg.skip_days
        )
        mask = self.membership.reindex(index=momentum.index, columns=momentum.columns)
        mask = mask.fillna(False).astype(bool)
        return momentum.where(mask)

    def __call__(self, panel: Panel) -> pd.DataFrame:
        return top_quantile_weights(
            self.eligible_momentum(panel), self.cfg.n_quantiles, self.bucket_method
        )


@dataclass(frozen=True, slots=True)
class CurrencyCrossSectionalMomentum:
    cfg: Config005
    universe: tuple[str, ...]
    bucket_method: BucketMethod = "even"

    @property
    def name(self) -> str:
        return (
            f"currency cross-sectional momentum (formation {self.cfg.formation_days}d, "
            f"skip {self.cfg.skip_days}d, top 1/{self.cfg.n_quantiles} of "
            f"{len(self.universe)}, {self.bucket_method} buckets)"
        )

    def momentum(self, panel: Panel) -> pd.DataFrame:
        validate_panel(panel, require=("close",))
        universe = list(self.universe)
        close = panel_field(panel, "close")
        missing = [t for t in universe if t not in close.columns]
        if missing:
            raise ValueError(f"panel is missing universe members {missing}")
        return cross_sectional_momentum(
            close[universe], self.cfg.formation_days, self.cfg.skip_days
        )

    def __call__(self, panel: Panel) -> pd.DataFrame:
        return top_quantile_weights(
            self.momentum(panel), self.cfg.n_quantiles, self.bucket_method
        )
