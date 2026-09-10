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
    name: str

    def __call__(self, panel: Panel) -> pd.DataFrame:  # pragma: no cover
        ...


def price_panel(prices, universe) -> dict[str, pd.DataFrame]:
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
    return list(panel.keys())


def panel_field(panel: Panel, field: str) -> pd.DataFrame:
    validate_panel(panel, require=(field,))
    return pd.DataFrame(
        {symbol: frame[field] for symbol, frame in panel.items()},
        columns=panel_symbols(panel),
    )


def validate_panel(panel: Panel, *, require: tuple[str, ...] = ("close",)) -> None:
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
