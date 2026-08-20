"""The two strategies, expressed in the generalised panel protocol.

Both are ``dict[str, DataFrame] -> DataFrame`` of target weights, and both are pure
functions of the panel: nothing here knows what the engine is holding, and nothing
here lags its own output.

:class:`TimeSeriesTrend` is experiment 001 re-expressed, not re-implemented. It calls
:func:`trendbot.signal.trend_signal` and :func:`trendbot.sizing.target_weights` - the
same objects the original engine calls - once per bar instead of once per rebalance,
which is the whole of the change. ``tests/test_experiment_001_regression.py`` asserts
the resulting equity curve is the original one.

:class:`CrossSectionalMomentum` is experiment 002, and is the reason the protocol had
to widen: its weight for instrument *i* on date *t* is a function of where *i* ranks
against every other instrument on that date, which no per-instrument signature can
express.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .config_002 import Config002
from .config_003 import Config003
from .engine.panel import Panel, panel_field, validate_panel
from .signal import trend_signal
from .sizing import CovarianceMethod, annualised_vol, ewma_covariance, target_weights
from .xsmom import BucketMethod, cross_sectional_momentum, top_quantile_weights

__all__ = ["TimeSeriesTrend", "CrossSectionalMomentum", "EquityCrossSectionalMomentum"]


def _covariance_blocks(cov: pd.DataFrame, dates: pd.Index, universe: list[str]):
    """Reshape a stacked EWMA covariance frame into one (n_dates, n, n) array.

    ``DataFrame.ewm().cov()`` returns one square block per date, stacked in date order
    with the inner index in column order. That layout is asserted rather than assumed,
    because silently transposing a covariance matrix would change every weight without
    changing any shape.
    """
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
    """Experiment 001's rule, as a panel strategy.

    Section 4's sizing pipeline - vol scaling, ``k``, the per-instrument cap, the
    gross cap - is already a pure function of the signal and the vol estimate on one
    date, so it lifts into this protocol unchanged. The drift band is not, and stays
    in the engine where experiment 001 put it.
    """

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
    """Experiment 002's rule, as a panel strategy.

    PREREG_002.md sections 3 and 4 in full: rank on ``P(t-skip)/P(t-formation) - 1``,
    hold the top bucket at ``1/n_selected`` each, hold nothing else. There is no vol
    scaling to apply (section 4 removes it) and no cap that can bind, because equal
    weights over the top bucket sum to exactly the gross exposure of 1.0 that section
    4 states.
    """

    cfg: Config002

    @property
    def name(self) -> str:
        return (
            f"cross-sectional momentum (formation {self.cfg.formation_days}d, "
            f"skip {self.cfg.skip_days}d, top 1/{self.cfg.n_quantiles})"
        )

    def momentum(self, panel: Panel) -> pd.DataFrame:
        """The raw ranking variable, exposed for the section 8 quintile study."""
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
    """Experiment 003's rule, as a panel strategy.

    Structurally identical to :class:`CrossSectionalMomentum` - PREREG_003.md section 3
    keeps 002's formula unchanged on purpose, "so that the universe is the only
    variable" - and differs in exactly three places, all of them parameters rather than
    logic:

    * the universe is resolved from a dated index snapshot rather than listed in the
      document, so it arrives as an argument instead of off the config;
    * the sort is into ten buckets rather than five;
    * the bucket-size convention is ``"even"`` rather than 002's ``"floor"``. Section 3
      names no bucket size, and section 8's gate is the D1-D10 spread, so the two ends
      of the sort are kept the same size. See :mod:`trendbot.xsmom` for the full
      argument and the sensitivity that is reported alongside the headline.

    Because the rule is the same object with different arguments, a disagreement
    between 002's and 003's implementations is impossible rather than merely unlikely.
    """

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
        """The raw ranking variable, exposed for the section 8 decile study."""
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
