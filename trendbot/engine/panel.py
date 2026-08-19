"""The generalised strategy protocol.

Experiment 001's strategy is a per-instrument function: a ``DataFrame`` of one
instrument's prices in, a ``Series`` of that instrument's target position out. That
shape cannot express experiment 002, because a cross-sectional rule's position for
instrument *i* on date *t* depends on where *i* ranks against every other instrument
on that same date. No amount of per-instrument information determines it.

So the protocol widens by exactly one step - from an instrument to a panel:

.. code-block:: text

    001:  DataFrame                  -> Series      (one instrument, its position)
    002:  dict[str, DataFrame]       -> DataFrame   (the panel, target weights)

The input is keyed by symbol, one frame of that symbol's bars per key, which is the
literal generalisation of "a DataFrame of prices". The output is indexed by date and
columned by symbol.

What the returned frame means
-----------------------------
**Target weights as of the close of each row's own bar, before execution.** Two
consequences, both load-bearing:

* A strategy must never lag its own output. The single execution shift stays where
  experiment 001 put it - :func:`trendbot.engine.backtest.lag_for_execution`, called
  once, by the engine. A strategy that shifted its own frame would be lagged twice.
* The frame is a *pure function of the panel*. It is not allowed to depend on the
  book the engine happens to be holding. Everything path-dependent - the drift band,
  the re-application of the exposure caps to weights that drifted, the cost charged
  on the trade actually done - stays in the engine, exactly as in experiment 001,
  where :func:`trendbot.sizing.target_weights` is likewise a pure function of the
  signal and the vol estimate and ``apply_drift_band`` is applied afterwards by the
  engine.

That split is what makes the 001 regression possible at all: 001's sizing pipeline
up to and including the two caps is already pure, so it lifts into this protocol
unchanged and the engine's remaining work is identical to what it always did.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = [
    "PanelStrategy",
    "Panel",
    "price_panel",
    "panel_field",
    "panel_symbols",
    "validate_panel",
]

Panel = Mapping[str, pd.DataFrame]


@runtime_checkable
class PanelStrategy(Protocol):
    """A strategy: the whole panel in, a frame of target weights out.

    ``name`` travels with results so an artefact can say which rule produced it.
    """

    name: str

    def __call__(self, panel: Panel) -> pd.DataFrame:  # pragma: no cover - protocol
        ...


def price_panel(prices, universe) -> dict[str, pd.DataFrame]:
    """Turn a :class:`~trendbot.data.PriceData` into the panel the protocol takes.

    One frame per symbol with an ``open`` and a ``close`` column, all sharing the one
    trading-day calendar. Missing bars stay missing: a hole is not a price, and it is
    the engine's job - not a strategy's - to decide what to mark a position at.
    """
    universe = list(universe)
    missing = [t for t in universe if t not in prices.close.columns]
    if missing:
        raise ValueError(f"price data is missing panel members {missing}")
    return {
        symbol: pd.DataFrame(
            {"open": prices.open[symbol], "close": prices.close[symbol]},
            index=prices.close.index,
        )
        for symbol in universe
    }


def panel_symbols(panel: Panel) -> list[str]:
    """Panel members in insertion order, which is the column order of every frame."""
    return list(panel.keys())


def panel_field(panel: Panel, field: str) -> pd.DataFrame:
    """Collapse the panel to one wide frame of ``field``, dates x symbols."""
    validate_panel(panel, require=(field,))
    return pd.DataFrame(
        {symbol: frame[field] for symbol, frame in panel.items()},
        columns=panel_symbols(panel),
    )


def validate_panel(panel: Panel, *, require: tuple[str, ...] = ("close",)) -> None:
    """Reject a panel that cannot support a trading-day row shift or a cross-section.

    The lookbacks are expressed in trading days and implemented as row shifts, and a
    cross-sectional rank is only meaningful if every symbol's row ``t`` is the same
    calendar date. Both need one shared, sorted, duplicate-free index.
    """
    if not isinstance(panel, Mapping):
        raise TypeError(f"panel must be a mapping of symbol -> frame, got {type(panel).__name__}")
    if not panel:
        raise ValueError("panel is empty")

    reference: pd.Index | None = None
    for symbol, frame in panel.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"panel[{symbol!r}] must be a DataFrame, got {type(frame).__name__}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"panel[{symbol!r}] must be indexed by a DatetimeIndex of trading days")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"panel[{symbol!r}] index must be sorted ascending")
        if frame.index.has_duplicates:
            dupes = frame.index[frame.index.duplicated()].unique().tolist()
            raise ValueError(f"panel[{symbol!r}] index has duplicate dates: {dupes[:5]}")
        for field in require:
            if field not in frame.columns:
                raise ValueError(f"panel[{symbol!r}] has no {field!r} column")
        if reference is None:
            reference = frame.index
        elif not frame.index.equals(reference):
            raise ValueError(
                f"panel[{symbol!r}] is on a different trading calendar from the first "
                "member; a cross-sectional rank needs one shared calendar"
            )
