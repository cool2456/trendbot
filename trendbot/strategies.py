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
from .config_004 import Config004
from .engine.panel import Panel, panel_field, validate_panel
from .signal import trend_signal
from .sizing import CovarianceMethod, annualised_vol, ewma_covariance, target_weights
from .xsmom import BucketMethod, cross_sectional_momentum, top_quantile_weights

__all__ = [
    "TimeSeriesTrend",
    "CrossSectionalMomentum",
    "EquityCrossSectionalMomentum",
    "PointInTimeMomentum",
]


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


@dataclass(frozen=True, slots=True)
class PointInTimeMomentum:
    """Experiment 004's rule: the same signal over a universe that changes every month.

    The formula, the decile cut, the weighting and the schedule are byte-identical to
    experiment 003 - PREREG_004.md section 1 says so explicitly, "so that the data is
    the only variable". The single difference is that the set of names eligible to be
    ranked is not a constant tuple but a per-date membership matrix computed by
    :func:`trendbot.pit_universe.build_pit_universe`.

    **Which bar the universe is evaluated on.** Section 2 says "at each monthly
    rebalance date t", and section 5 says the fill is at the open of the bar after the
    signal. Those two can only be made consistent one way: the universe rule and the
    momentum signal are both evaluated on the *decision bar* - the close before the
    rebalance - and the trade happens at the next open. Evaluating the rule on the
    rebalance bar's own close while filling at that bar's open would require knowing
    the close before the open, which is the lookahead the build order forbids. So
    ``membership`` is indexed by decision bar, and the engine's single execution shift
    carries it to the rebalance.

    **Names not in the universe on a date are excluded from that date's ranking**, not
    assigned a momentum of zero - the same distinction sections 2 and 3 of the previous
    two experiments turned on, applied here to a set that moves.
    """

    cfg: Config004
    membership: pd.DataFrame  # decision bar x permaticker-as-string, boolean
    bucket_method: BucketMethod = "even"

    @property
    def name(self) -> str:
        return (
            f"point-in-time cross-sectional momentum (formation "
            f"{self.cfg.formation_days}d, skip {self.cfg.skip_days}d, top "
            f"1/{self.cfg.n_quantiles} of {self.cfg.universe_size})"
        )

    def eligible_momentum(self, panel: Panel) -> pd.DataFrame:
        """Momentum, masked to the point-in-time universe on each decision bar.

        NaN means "not ranked on this date", which covers both "no momentum yet" and
        "not in the universe today". Both are exclusions from the sort, which is what
        section 2 and section 4 each ask for.
        """
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
