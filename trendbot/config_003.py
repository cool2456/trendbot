from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigParseError, _require_unique, _section

__all__ = [
    "Config003",
    "load_config_003",
    "find_preregistration_003",
]

_PREREG_NAME = "PREREG_003.md"


def find_preregistration_003(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. Experiment 003 cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


@dataclass(frozen=True, slots=True)
class Config003:
    source_path: Path
    source_sha256: str
    committed_on: str
    signed_by: str
    signed_date: str

    configurations_tried: int

    index_name: str
    universe_history_required_from: str
    expected_universe_low: int
    expected_universe_high: int
    survivorship_flattens_gradient: bool

    formation_days: int
    skip_days: int
    n_quantiles: int
    long_only: bool

    gross_exposure_cap: float

    rebalance: str
    cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]

    sample_start: str

    support_min_sharpe: float
    support_min_sharpe_excess_over_buy_and_hold: float
    max_inversions: int
    min_spread_t_stat: float
    min_alpha_t_stat: float
    abandon_below_sharpe: float
    abandon_below_spread_t_stat: float

    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float
    turnover_floor: float
    required_crash_months: tuple[str, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
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
        if self.expected_universe_low > self.expected_universe_high:
            raise ConfigParseError("section 2's expected universe range runs backwards")
        if self.abandon_below_spread_t_stat > self.min_spread_t_stat:
            raise ConfigParseError(
                f"section 8 abandons below t={self.abandon_below_spread_t_stat} but supports "
                f"only above t={self.min_spread_t_stat}; the abandon threshold must be the "
                "lower of the two or the rule has no inconclusive band"
            )
        if self.max_inversions < 0:
            raise ConfigParseError("the tolerated inversion count cannot be negative")
        pd.Timestamp(self.sample_start)
        for month in self.required_crash_months:
            pd.Period(month, freq="M")

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000.0

    def describe(self) -> str:
        return (
            f"equity cross-sectional momentum [formation={self.formation_days}d, "
            f"skip={self.skip_days}d, top 1/{self.n_quantiles} of the {self.index_name}, "
            f"long-only, cost={self.cost_bps_per_side:g}bps/side] "
            f"sha256={self.source_sha256[:12]}"
        )


def _parse_text(text: str, source_path: Path) -> Config003:
    s2, s3, s4, s5, s6, s8, s9 = (_section(text, n) for n in (2, 3, 4, 5, 6, 8, 9))

    configurations_tried = int(
        _require_unique(
            r"\*\*Configurations tried on this dataset, cumulative:\*\*\s*(\d+)",
            text,
            "cumulative configuration counter",
        ).group(1)
    )

    universe_rule = _require_unique(
        r"\*\*Current (S&P 500) constituents with continuous daily data from\s*"
        r"(\d{4}-\d{2}-\d{2})\s*to\s*\npresent\.\*\*",
        s2,
        "the universe definition",
    )
    index_name = universe_rule.group(1)
    history_from = universe_rule.group(2)
    expected = _require_unique(
        r"Expected yield\s*(\d+)\s*[-–—]\s*(\d+)\s*names", s2, "the expected universe size range"
    )
    _require_unique(
        r"artificially \*\*narrow\*\* and the monotonicity gradient artificially \*\*flat\*\*",
        s2,
        "section 2's statement of the direction of the survivorship bias",
    )
    _require_unique(
        r"a monotonicity pass on this universe is conservative evidence",
        s2,
        "section 2's conservative-evidence clause",
    )

    formula = _require_unique(
        r"momentum_i\(t\)\s*=\s*P_i\(t-(\d+)\)\s*/\s*P_i\(t-(\d+)\)\s*-\s*1",
        s3,
        "the momentum formula",
    )
    skip_days, formation_days = int(formula.group(1)), int(formula.group(2))
    quantile_word = _require_unique(
        r"sort into \*\*(deciles|quintiles|quartiles|terciles)\*\*", s3, "the quantile cut"
    ).group(1)
    n_quantiles = {"terciles": 3, "quartiles": 4, "quintiles": 5, "deciles": 10}[quantile_word]
    declared_buckets = _require_unique(
        r"\((ten|five|four|three) buckets", s3, "the bucket count restated in words"
    ).group(1)
    if {"three": 3, "four": 4, "five": 5, "ten": 10}[declared_buckets] != n_quantiles:
        raise ConfigParseError(
            f"section 3 says {quantile_word} but restates the count as {declared_buckets}"
        )
    _require_unique(r"\*\*Top decile:\*\*\s*equal weight, long", s3, "the top-bucket position")
    long_only = bool(re.search(r"^Long-only\.", s3, re.MULTILINE))
    if not long_only:
        raise ConfigParseError("section 3 no longer declares the strategy long-only")
    _require_unique(
        r"Names lacking \d+ days of history are excluded from that date's ranking",
        s3,
        "section 3's per-date eligibility rule",
    )

    gross_cap = float(
        _require_unique(
            r"Gross exposure\s*(\d+(?:\.\d+)?)\s*when invested", s4, "gross exposure"
        ).group(1)
    )
    _require_unique(r"no volatility targeting", s4, "the no-vol-targeting clause")
    _require_unique(r"Equal weight within the top decile", s4, "the equal-weight rule")

    rebalance = _require_unique(
        r"Rebalance:\s*(.+?)\.?\s*$", s5, "rebalance schedule", re.MULTILINE
    ).group(1).strip()
    cost_bps = float(
        _require_unique(
            r"\*\*Cost assumption:\s*(\d+(?:\.\d+)?)\s*bps per side\.\*\*", s5, "cost assumption"
        ).group(1)
    )
    ladder_raw = _require_unique(
        r"Sensitivity at\s*([\d/\s]+?)\s*\.", s5, "cost sensitivity ladder"
    ).group(1)
    ladder = tuple(float(x) for x in re.findall(r"[\d.]+", ladder_raw))
    _require_unique(
        r"split and dividend adjustment must be point-in-time",
        s5,
        "section 5's point-in-time adjustment requirement",
    )

    sample_start = _require_unique(
        r"sample window\s*\n?\s*\((\d{4}-\d{2}-\d{2})\s*to present\)", s6, "sample window start"
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
    max_inversions = {"one": 1, "two": 2, "three": 3, "zero": 0}[
        _require_unique(
            r"allowing at most \*\*(zero|one|two|three)\*\* adjacent inversion",
            s8,
            "the tolerated inversion count",
        ).group(1)
    ]
    spread_t = float(
        _require_unique(
            r"with D1[-–—−]D10 spread positive at\s*\n?\s*\*\*t\s*>\s*(\d+(?:\.\d+)?)\*\*",
            s8,
            "the spread t-statistic support threshold",
        ).group(1)
    )
    alpha_t = float(
        _require_unique(
            r"\*\*Alpha to market beta is positive with t\s*>\s*(\d+(?:\.\d+)?)\.\*\*",
            s8,
            "the alpha t-statistic support threshold",
        ).group(1)
    )
    abandon_sharpe = float(
        _require_unique(
            r"Net Sharpe below \*\*(\d+(?:\.\d+)?)\*\*", s8, "the Sharpe abandonment threshold"
        ).group(1)
    )
    abandon_spread_t = float(
        _require_unique(
            r"D1[-–—−]D10 t-statistic below\s*(\d+(?:\.\d+)?)",
            s8,
            "the spread t-statistic abandonment threshold",
        ).group(1)
    )
    _require_unique(
        r"Fails to beat equal-weight buy-and-hold at all", s8, "the buy-and-hold abandon clause"
    )
    _require_unique(r"Alpha to market is negative", s8, "the negative-alpha abandon clause")
    _require_unique(
        r"More than one decile inversion", s8, "the inversion-count abandon clause"
    )

    exp = _require_unique(
        r"Realistic net Sharpe:\s*\*\*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\*\*",
        s9,
        "expected Sharpe range",
    )
    bug_threshold = float(
        _require_unique(r"Above\s*(\d+(?:\.\d+)?)\s*means a bug", s9, "bug threshold Sharpe").group(1)
    )
    turnover_floor = float(
        _require_unique(
            r"Turnover higher than 002's\s*(\d+(?:\.\d+)?)×", s9, "the turnover expectation"
        ).group(1)
    )
    crash = _require_unique(
        r"\*\*Momentum crashes are mandatory\.\*\*\s*(\w+)[-–—](\w+)\s*(\d{4})\s*and\s*(\w+)\s*(\d{4})",
        s9,
        "the mandatory crash months",
    )
    months = {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], start=1)}
    lo, hi, year = months[crash.group(1)], months[crash.group(2)], int(crash.group(3))
    if hi < lo:
        raise ConfigParseError("section 9's crash month range runs backwards")
    required = [f"{year}-{m:02d}" for m in range(lo, hi + 1)]
    required.append(f"{int(crash.group(5))}-{months[crash.group(4)]:02d}")

    committed = _require_unique(
        r"\*\*Committed on:\*\*\s*(\d{4}-\d{2}-\d{2})", text, "commit date"
    ).group(1)
    signature = _require_unique(
        r"Signed:\s*\*\*(.+?)\*\*\s+Date:\s*\*\*(.+?)\*\*", text, "signature"
    )

    return Config003(
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        committed_on=committed,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2).strip(),
        configurations_tried=configurations_tried,
        index_name=index_name,
        universe_history_required_from=history_from,
        expected_universe_low=int(expected.group(1)),
        expected_universe_high=int(expected.group(2)),
        survivorship_flattens_gradient=True,
        formation_days=formation_days,
        skip_days=skip_days,
        n_quantiles=n_quantiles,
        long_only=long_only,
        gross_exposure_cap=gross_cap,
        rebalance=rebalance,
        cost_bps_per_side=cost_bps,
        cost_sensitivity_bps=ladder,
        sample_start=sample_start,
        support_min_sharpe=support_sharpe,
        support_min_sharpe_excess_over_buy_and_hold=support_excess,
        max_inversions=max_inversions,
        min_spread_t_stat=spread_t,
        min_alpha_t_stat=alpha_t,
        abandon_below_sharpe=abandon_sharpe,
        abandon_below_spread_t_stat=abandon_spread_t,
        expected_sharpe_low=float(exp.group(1)),
        expected_sharpe_high=float(exp.group(2)),
        bug_threshold_sharpe=bug_threshold,
        turnover_floor=turnover_floor,
        required_crash_months=tuple(required),
    )


_CACHE: dict[Path, Config003] = {}


def load_config_003(path: Path | str | None = None, *, use_cache: bool = True) -> Config003:
    resolved = Path(path).resolve() if path is not None else find_preregistration_003()
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
