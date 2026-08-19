"""Parse PREREGISTRATION.md into a frozen configuration object.

The pre-registration document is the sole source of truth for every strategy
parameter. This module reads it directly rather than duplicating the numbers into
Python, so that changing a parameter means editing a signed document and changing
its content hash.

Design rules enforced here:

* No strategy parameter has a default. Every value is *pulled* out of the document
  and a missing or unparseable value raises :class:`ConfigParseError`. A silent
  fallback would defeat the entire point of pre-registering.
* The parsed object is frozen. Nothing downstream may mutate a parameter.
* The sha256 of the document is carried on the config so that any artefact produced
  by a run can be tied back to the exact text that produced it.
"""

from __future__ import annotations

import hashlib
import re
from types import MappingProxyType
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ConfigParseError",
    "Config",
    "load_config",
    "find_preregistration",
]

# Location of the document relative to the installed package (repo root).
_PREREG_NAME = "PREREGISTRATION.md"


class ConfigParseError(ValueError):
    """Raised when the pre-registration cannot be parsed into a complete config.

    This is deliberately fatal. If a parameter the strategy needs is absent from the
    document, the correct behaviour is to stop, not to guess.
    """


def find_preregistration(start: Path | None = None) -> Path:
    """Locate PREREGISTRATION.md by walking up from ``start`` (default: this file)."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. The strategy cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


# --------------------------------------------------------------------------------------
# primitive extractors
# --------------------------------------------------------------------------------------


def _require(pattern: str, text: str, what: str, flags: int = 0) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    if match is None:
        raise ConfigParseError(
            f"could not find {what} in {_PREREG_NAME} (pattern: {pattern!r}). "
            "Refusing to substitute a default value."
        )
    return match


def _require_unique(pattern: str, text: str, what: str, flags: int = 0) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags))
    if not matches:
        raise ConfigParseError(
            f"could not find {what} in {_PREREG_NAME} (pattern: {pattern!r}). "
            "Refusing to substitute a default value."
        )
    if len(matches) > 1:
        raise ConfigParseError(
            f"{what} is ambiguous in {_PREREG_NAME}: {len(matches)} matches for "
            f"{pattern!r} -> {[m.group(0) for m in matches]}. Refusing to guess which is meant."
        )
    return matches[0]


def _section(text: str, number: int) -> str:
    """Return the body of markdown section ``## {number}. ...``."""
    match = re.search(
        rf"^##\s+{number}\.\s+.*?$(.*?)(?=^##\s+\d+\.|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ConfigParseError(f"section {number} not found in {_PREREG_NAME}")
    return match.group(1)


def _pct(raw: str) -> float:
    return float(raw) / 100.0


# --------------------------------------------------------------------------------------
# config object
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """Every frozen parameter of diversified trend following v1.0.

    Field names mirror the language of the document. Attribute access is the only
    supported way to read a parameter; nothing in this package may inline a literal.
    """

    # provenance
    source_path: Path
    source_sha256: str
    version: str
    signed_by: str
    signed_date: str

    # section 2 - universe
    sleeves: Mapping[str, tuple[str, ...]]
    universe: tuple[str, ...]

    # section 3 - signal
    lookback_days: int
    variant: str  # "long-only" or "long-short"

    # section 4 - risk scaling
    instrument_vol_target: float
    ewma_halflife_days: int
    portfolio_vol_target: float
    gross_exposure_cap: float
    per_instrument_cap: float

    # section 5 - execution
    rebalance: str
    drift_band: float
    cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]

    # section 7 - protocol
    configurations_tried: int

    # section 8 - pre-committed decision rule
    support_min_sharpe: float
    support_min_positive_instruments: int
    support_min_sharpe_excess_over_buy_and_hold: float
    abandon_below_sharpe: float
    abandon_if_not_beating_buy_and_hold: bool

    # section 9 - expectations of record
    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float

    # derived, not a free parameter: the divisor in w_i = x_i / N is the fixed
    # universe size stated in section 2, held constant even when instruments are
    # inactive for want of history.
    n_universe: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_universe", len(self.universe))
        # A frozen dataclass freezes the binding, not the object bound. A plain dict
        # here would let a caller rewrite sleeve membership in place, which is
        # precisely the kind of quiet edit this whole module exists to prevent.
        object.__setattr__(self, "sleeves", MappingProxyType(dict(self.sleeves)))
        self._validate()

    def _validate(self) -> None:
        if self.variant not in ("long-only", "long-short"):
            raise ConfigParseError(f"unrecognised signal variant {self.variant!r}")
        if self.lookback_days <= 0:
            raise ConfigParseError("lookback must be positive")
        if self.n_universe == 0:
            raise ConfigParseError("universe is empty")
        flat = [t for tickers in self.sleeves.values() for t in tickers]
        if sorted(flat) != sorted(self.universe):
            raise ConfigParseError("sleeve membership does not reconcile with the universe")
        if len(set(self.universe)) != len(self.universe):
            raise ConfigParseError("duplicate ticker in universe")
        for name, value in (
            ("instrument_vol_target", self.instrument_vol_target),
            ("portfolio_vol_target", self.portfolio_vol_target),
            ("gross_exposure_cap", self.gross_exposure_cap),
            ("per_instrument_cap", self.per_instrument_cap),
            ("drift_band", self.drift_band),
        ):
            if not value > 0:
                raise ConfigParseError(f"{name} must be positive, got {value}")
        if self.cost_bps_per_side < 0:
            raise ConfigParseError("cost cannot be negative")
        if self.configurations_tried < 1:
            raise ConfigParseError("configurations tried must be at least 1")

    @property
    def long_only(self) -> bool:
        return self.variant == "long-only"

    @property
    def cost_rate_per_side(self) -> float:
        """Cost per unit of one-way turnover, as a decimal fraction."""
        return self.cost_bps_per_side / 10_000.0

    def sleeve_of(self, ticker: str) -> str:
        for sleeve, tickers in self.sleeves.items():
            if ticker in tickers:
                return sleeve
        raise KeyError(ticker)

    def describe(self) -> str:
        return (
            f"diversified trend following v{self.version} "
            f"[{self.variant}, lookback={self.lookback_days}d, "
            f"vol_target={self.portfolio_vol_target:.0%}, "
            f"cost={self.cost_bps_per_side:g}bps/side] "
            f"sha256={self.source_sha256[:12]}"
        )


# --------------------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------------------


def _parse_universe(text: str) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    body = _section(text, 2)
    sleeves: dict[str, tuple[str, ...]] = {}
    order: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        sleeve, tickers = cells
        if sleeve.lower() == "sleeve" or set(sleeve) <= set("- :"):
            continue  # header or separator row
        symbols = tuple(t.strip() for t in tickers.split(",") if t.strip())
        if not symbols:
            continue
        if not all(re.fullmatch(r"[A-Z]{1,5}", s) for s in symbols):
            raise ConfigParseError(f"unparseable ticker list in universe row: {line!r}")
        if sleeve in sleeves:
            raise ConfigParseError(f"duplicate sleeve {sleeve!r} in universe table")
        sleeves[sleeve] = symbols
        order.extend(symbols)
    if not sleeves:
        raise ConfigParseError("universe table in section 2 produced no instruments")
    declared = _require(
        r"##\s+2\.\s+Universe\s+[-–—]\s+(\d+)\s+instruments", text, "declared universe size"
    )
    if int(declared.group(1)) != len(order):
        raise ConfigParseError(
            f"section 2 heading declares {declared.group(1)} instruments but the table "
            f"lists {len(order)}: {order}"
        )
    return sleeves, tuple(order)


def _parse_text(text: str, source_path: Path) -> Config:
    sleeves, universe = _parse_universe(text)

    s3, s4, s5, s7, s8, s9 = (_section(text, n) for n in (3, 4, 5, 7, 8, 9))

    lookback = int(
        _require_unique(r"Lookback:\s*\*\*(\d+)\s+trading days\*\*", s3, "lookback").group(1)
    )
    variant = _require_unique(
        r"\*\*Variant in use:\*\*\s*(long-only|long-short)", s3, "signal variant"
    ).group(1)

    instrument_vol = _pct(
        _require_unique(
            r"Per-instrument vol target:\s*\*\*(\d+(?:\.\d+)?)%\*\*", s4, "per-instrument vol target"
        ).group(1)
    )
    halflife = int(
        _require_unique(r"EWMA halflife:\s*\*\*(\d+)\s+days\*\*", s4, "EWMA halflife").group(1)
    )
    portfolio_vol = _pct(
        _require_unique(
            r"Portfolio ex-ante vol target:\s*\*\*(\d+(?:\.\d+)?)%\*\*", s4, "portfolio vol target"
        ).group(1)
    )
    gross_cap = float(
        _require_unique(
            r"Gross exposure cap:\s*\*\*sum\(\|w_i\|\)\s*<=\s*(\d+(?:\.\d+)?)\*\*", s4, "gross exposure cap"
        ).group(1)
    )
    per_instrument_cap = float(
        _require_unique(
            r"Per-instrument cap:\s*\*\*\|w_i\|\s*<=\s*(\d+(?:\.\d+)?)\*\*", s4, "per-instrument cap"
        ).group(1)
    )

    rebalance = _require_unique(r"Rebalance:\s*(.+?)\.?\s*$", s5, "rebalance schedule", re.MULTILINE).group(1).strip()
    drift_band = float(
        _require_unique(
            r"only trade instrument i if `\|w_target\s*[-−]\s*w_current\|\s*>\s*(\d+(?:\.\d+)?)\s*\*\s*\|w_target\|`",
            s5,
            "drift band",
        ).group(1)
    )
    cost_bps = float(
        _require_unique(
            r"Cost assumption in backtest:\s*\*\*(\d+(?:\.\d+)?)\s*bps per side\*\*", s5, "cost assumption"
        ).group(1)
    )
    ladder_raw = _require_unique(
        r"Sensitivity reported at\s*([\d,\s]+?)\s*bps", s5, "cost sensitivity ladder"
    ).group(1)
    ladder = tuple(float(x) for x in re.findall(r"[\d.]+", ladder_raw))
    if cost_bps not in ladder:
        raise ConfigParseError(
            f"headline cost {cost_bps} bps is not present in the sensitivity ladder {ladder}"
        )

    configurations_tried = int(
        _require_unique(
            r"number of configurations tried\s*=\s*(\d+)", s7, "number of configurations tried"
        ).group(1)
    )

    support_sharpe = float(
        _require_unique(
            r"net Sharpe over the full sample\s*\n?\s*exceeds\s*(\d+(?:\.\d+)?)", s8, "support threshold Sharpe"
        ).group(1)
    )
    support_positive = _require_unique(
        r"positive in at least\s*(\d+)\s*of\s*(\d+)\s*instruments", s8, "sign-consistency threshold"
    )
    if int(support_positive.group(2)) != len(universe):
        raise ConfigParseError(
            f"section 8 refers to {support_positive.group(2)} instruments but the universe has "
            f"{len(universe)}"
        )
    support_excess = float(
        _require_unique(r"by at least\s*(\d+(?:\.\d+)?)", s8, "excess-over-buy-and-hold threshold").group(1)
    )
    abandon = float(
        _require_unique(r"net Sharpe is below\s*(\d+(?:\.\d+)?)", s8, "abandonment threshold").group(1)
    )
    # Section 8's abandonment rule has two limbs joined by "or". The second carries no
    # number, so it is easy to parse only the first and silently apply half the rule.
    # Its presence is asserted here so that deleting it from the document is a parse
    # error rather than a quiet loosening of the pre-committed criterion.
    _require_unique(
        r"fails to beat\s*\n?\s*equal-weight buy-and-hold at all",
        s8,
        "the second abandonment clause (failure to beat buy-and-hold)",
    )

    exp = _require_unique(
        r"Realistic net Sharpe:\s*\*\*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\*\*", s9, "expected Sharpe range"
    )
    bug_threshold = float(
        _require_unique(r"Above\s*(\d+(?:\.\d+)?)\s*means a bug", s9, "bug threshold Sharpe").group(1)
    )

    version = _require_unique(r"^#\s+Pre-registration.*?v(\d+(?:\.\d+)?)\s*$", text, "version", re.MULTILINE).group(1)
    signature = _require_unique(r"Signed:\s*(.+?)\s{2,}Date:\s*(.+?)\s*$", text, "signature", re.MULTILINE)

    return Config(
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=version,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2).strip(),
        sleeves=sleeves,
        universe=universe,
        lookback_days=lookback,
        variant=variant,
        instrument_vol_target=instrument_vol,
        ewma_halflife_days=halflife,
        portfolio_vol_target=portfolio_vol,
        gross_exposure_cap=gross_cap,
        per_instrument_cap=per_instrument_cap,
        rebalance=rebalance,
        drift_band=drift_band,
        cost_bps_per_side=cost_bps,
        cost_sensitivity_bps=ladder,
        configurations_tried=configurations_tried,
        support_min_sharpe=support_sharpe,
        support_min_positive_instruments=int(support_positive.group(1)),
        support_min_sharpe_excess_over_buy_and_hold=support_excess,
        abandon_below_sharpe=abandon,
        abandon_if_not_beating_buy_and_hold=True,
        expected_sharpe_low=float(exp.group(1)),
        expected_sharpe_high=float(exp.group(2)),
        bug_threshold_sharpe=bug_threshold,
    )


_CACHE: dict[Path, Config] = {}


def load_config(path: Path | str | None = None, *, use_cache: bool = True) -> Config:
    """Parse the pre-registration document into a frozen :class:`Config`."""
    resolved = Path(path).resolve() if path is not None else find_preregistration()
    if use_cache and resolved in _CACHE:
        return _CACHE[resolved]
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigParseError(f"cannot read {resolved}: {exc}") from exc
    cfg = _parse_text(text, resolved)
    if use_cache:
        _CACHE[resolved] = cfg
    return cfg
