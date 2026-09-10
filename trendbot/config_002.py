from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from .config import ConfigParseError, _require_unique, _section

__all__ = [
    "Config002",
    "load_config_002",
    "find_preregistration_002",
]

_PREREG_NAME = "PREREG_002.md"


def find_preregistration_002(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. Experiment 002 cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


@dataclass(frozen=True, slots=True)
class Config002:
    source_path: Path
    source_sha256: str
    committed_on: str
    signed_by: str
    signed_date: str

    configurations_tried: int

    sleeves: Mapping[str, tuple[str, ...]]
    universe: tuple[str, ...]
    declared_universe_size: int
    universe_history_required_from: str

    formation_days: int
    skip_days: int
    n_quantiles: int
    declared_quantile_size: int
    long_only: bool

    gross_exposure_cap: float

    rebalance: str
    has_drift_band: bool
    cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]

    sample_start: str

    support_min_sharpe: float
    support_min_sharpe_excess_over_buy_and_hold: float
    abandon_below_sharpe: float
    requires_quintile_monotonicity: bool

    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float
    expected_min_drawdown: float

    n_universe: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_universe", len(self.universe))
        object.__setattr__(self, "sleeves", MappingProxyType(dict(self.sleeves)))
        self._validate()

    def _validate(self) -> None:
        if self.n_universe == 0:
            raise ConfigParseError("universe is empty")
        flat = [t for tickers in self.sleeves.values() for t in tickers]
        if sorted(flat) != sorted(self.universe):
            raise ConfigParseError("sleeve membership does not reconcile with the universe")
        if len(set(self.universe)) != len(self.universe):
            raise ConfigParseError("duplicate ticker in universe")
        if self.declared_universe_size != self.n_universe:
            raise ConfigParseError(
                f"section 2 declares {self.declared_universe_size} instruments but lists "
                f"{self.n_universe}"
            )
        if self.formation_days <= self.skip_days:
            raise ConfigParseError(
                f"formation window {self.formation_days} must be longer than the skip "
                f"{self.skip_days}; P(t-{self.skip_days})/P(t-{self.formation_days}) would "
                "otherwise run backwards"
            )
        if self.skip_days < 0:
            raise ConfigParseError("skip cannot be negative")
        if self.n_quantiles < 2:
            raise ConfigParseError("a quantile sort needs at least two buckets")
        if not self.gross_exposure_cap > 0:
            raise ConfigParseError("gross exposure cap must be positive")
        if self.cost_bps_per_side < 0:
            raise ConfigParseError("cost cannot be negative")
        if self.configurations_tried < 1:
            raise ConfigParseError("configurations tried must be at least 1")
        if self.cost_bps_per_side not in self.cost_sensitivity_bps:
            raise ConfigParseError(
                f"headline cost {self.cost_bps_per_side} bps is absent from the sensitivity "
                f"ladder {self.cost_sensitivity_bps}"
            )
        pd.Timestamp(self.sample_start)

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000.0

    def sleeve_of(self, ticker: str) -> str:
        for sleeve, tickers in self.sleeves.items():
            if ticker in tickers:
                return sleeve
        raise KeyError(ticker)

    def describe(self) -> str:
        return (
            f"cross-sectional momentum [formation={self.formation_days}d, "
            f"skip={self.skip_days}d, top 1/{self.n_quantiles} of {self.n_universe}, "
            f"long-only, cost={self.cost_bps_per_side:g}bps/side] "
            f"sha256={self.source_sha256[:12]}"
        )


def _parse_universe(text: str) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], int]:
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
            continue
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
    declared = int(
        _require_unique(
            r"##\s+2\.\s+Universe\s+[-–—]\s+(\d+)\s+ETFs", text, "declared universe size"
        ).group(1)
    )
    return sleeves, tuple(order), declared


def _parse_text(text: str, source_path: Path) -> Config002:
    sleeves, universe, declared_universe = _parse_universe(text)
    s2, s3, s4, s5, s6, s8, s9 = (_section(text, n) for n in (2, 3, 4, 5, 6, 8, 9))

    configurations_tried = int(
        _require_unique(
            r"\*\*Configurations tried on this dataset, cumulative:\*\*\s*(\d+)",
            text,
            "cumulative configuration counter",
        ).group(1)
    )

    history_from = _require_unique(
        r"must have price history beginning on or before\s*(\d{4}-\d{2}-\d{2})",
        s2,
        "the date from which every instrument must have history",
    ).group(1)

    formula = _require_unique(
        r"momentum_i\(t\)\s*=\s*P_i\(t-(\d+)\)\s*/\s*P_i\(t-(\d+)\)\s*-\s*1",
        s3,
        "the momentum formula",
    )
    skip_days, formation_days = int(formula.group(1)), int(formula.group(2))
    prose_skip = int(
        _require_unique(
            r"skipping the most recent month\*\*\s*\((\d+)\s+trading days\)",
            s3,
            "the skip stated in prose",
        ).group(1)
    )
    if prose_skip != skip_days:
        raise ConfigParseError(
            f"section 3's formula skips {skip_days} days but its prose says {prose_skip}"
        )

    quantile_word = _require_unique(
        r"-\s*\*\*Top (quintile|quartile|decile|tercile)\*\*", s3, "the quantile cut"
    ).group(1)
    n_quantiles = {"tercile": 3, "quartile": 4, "quintile": 5, "decile": 10}[quantile_word]
    quantile_size = _require_unique(
        r"\*\*Top \w+\*\*\s*\(highest\s*~?(\d+)\s+of\s+(\d+)\)", s3, "the declared top-bucket size"
    )
    declared_quantile_size = int(quantile_size.group(1))
    if int(quantile_size.group(2)) != declared_universe:
        raise ConfigParseError(
            f"section 3 sizes the top bucket out of {quantile_size.group(2)} instruments "
            f"but section 2 declares {declared_universe}"
        )
    long_only = bool(re.search(r"^Long-only\.", s3, re.MULTILINE))
    if not long_only:
        raise ConfigParseError("section 3 no longer declares the strategy long-only")

    gross_cap = float(
        _require_unique(
            r"Gross exposure:\s*\*\*(\d+(?:\.\d+)?)\*\*", s4, "gross exposure"
        ).group(1)
    )
    _require_unique(
        r"`w_i\s*=\s*1\s*/\s*n_selected`", s4, "the equal-weight rule"
    )
    _require_unique(r"\*\*No volatility targeting\.\*\*", s4, "the no-vol-targeting clause")

    rebalance = _require_unique(
        r"Rebalance:\s*(.+?)\.?\s*$", s5, "rebalance schedule", re.MULTILINE
    ).group(1).strip()
    has_band = not re.search(r"No drift band\.", s5)
    cost_bps = float(
        _require_unique(
            r"Cost assumption:\s*\*\*(\d+(?:\.\d+)?)\s*bps per side\*\*", s5, "cost assumption"
        ).group(1)
    )
    ladder_raw = _require_unique(
        r"Sensitivity reported at\s*([\d,\s]+?)\s*bps", s5, "cost sensitivity ladder"
    ).group(1)
    ladder = tuple(float(x) for x in re.findall(r"[\d.]+", ladder_raw))

    sample_start = _require_unique(
        r"\*\*Sample window:\s*(\d{4}-\d{2}-\d{2})\s*to present", s6, "sample window start"
    ).group(1)

    support_sharpe = float(
        _require_unique(
            r"Net Sharpe \(excess of T-bill, \d+ bps\) exceeds\s*\*\*(\d+(?:\.\d+)?)\*\*",
            s8,
            "support threshold Sharpe",
        ).group(1)
    )
    support_excess = float(
        _require_unique(
            r"by at least\s*\n?\s*\*\*(\d+(?:\.\d+)?)\*\*", s8, "excess-over-buy-and-hold threshold"
        ).group(1)
    )
    abandon = float(
        _require_unique(
            r"Net Sharpe is below\s*\*\*(\d+(?:\.\d+)?)\*\*", s8, "abandonment threshold"
        ).group(1)
    )
    _require_unique(
        r"It fails to beat equal-weight buy-and-hold at all", s8, "the buy-and-hold abandon clause"
    )
    _require_unique(
        r"Quintile ordering is non-monotonic", s8, "the monotonicity abandon clause"
    )
    monotonicity = bool(
        re.search(r"\*\*Quintile monotonicity holds\.\*\*", s8)
    )
    if not monotonicity:
        raise ConfigParseError("section 8 no longer requires quintile monotonicity")
    n_quantiles_s8 = _require_unique(
        r"into (five|four|ten|three) quintiles", s8, "the number of buckets in section 8"
    ).group(1)
    if {"three": 3, "four": 4, "five": 5, "ten": 10}[n_quantiles_s8] != n_quantiles:
        raise ConfigParseError(
            f"section 3 cuts into {n_quantiles} buckets but section 8 sorts into "
            f"{n_quantiles_s8}"
        )

    exp = _require_unique(
        r"Realistic net Sharpe:\s*\*\*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\*\*",
        s9,
        "expected Sharpe range",
    )
    bug_threshold = float(
        _require_unique(r"Above\s*(\d+(?:\.\d+)?)\s*means a bug", s9, "bug threshold Sharpe").group(1)
    )
    min_dd = float(
        _require_unique(
            r"at least one\s*\n?\s*drawdown above\s*(\d+(?:\.\d+)?)%", s9, "expected drawdown"
        ).group(1)
    ) / 100.0

    committed = _require_unique(
        r"\*\*Committed on:\*\*\s*(\d{4}-\d{2}-\d{2})", text, "commit date"
    ).group(1)
    signature = _require_unique(
        r"Signed:\s*\*\*(.+?)\*\*\s+Date:\s*\*\*(.+?)\*\*", text, "signature"
    )

    return Config002(
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        committed_on=committed,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2).strip(),
        configurations_tried=configurations_tried,
        sleeves=sleeves,
        universe=universe,
        declared_universe_size=declared_universe,
        universe_history_required_from=history_from,
        formation_days=formation_days,
        skip_days=skip_days,
        n_quantiles=n_quantiles,
        declared_quantile_size=declared_quantile_size,
        long_only=long_only,
        gross_exposure_cap=gross_cap,
        rebalance=rebalance,
        has_drift_band=has_band,
        cost_bps_per_side=cost_bps,
        cost_sensitivity_bps=ladder,
        sample_start=sample_start,
        support_min_sharpe=support_sharpe,
        support_min_sharpe_excess_over_buy_and_hold=support_excess,
        abandon_below_sharpe=abandon,
        requires_quintile_monotonicity=monotonicity,
        expected_sharpe_low=float(exp.group(1)),
        expected_sharpe_high=float(exp.group(2)),
        bug_threshold_sharpe=bug_threshold,
        expected_min_drawdown=min_dd,
    )


_CACHE: dict[Path, Config002] = {}


def load_config_002(path: Path | str | None = None, *, use_cache: bool = True) -> Config002:
    resolved = Path(path).resolve() if path is not None else find_preregistration_002()
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
