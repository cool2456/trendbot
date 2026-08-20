"""Parse PREREG_004.md into a frozen configuration object.

Same contract as the three parsers before it: the pre-registration is the sole source
of truth, every parameter is *pulled* out of the text, and a missing, ambiguous or
self-contradictory value is a fatal :class:`~trendbot.config.ConfigParseError` rather
than a silent default.

What is new in this document, and therefore new in this parser
--------------------------------------------------------------
Experiment 003's universe was a rule resolved against a dated index snapshot. This
one is a rule resolved **at every rebalance date** out of price and volume alone, so
what the parser extracts from section 2 is the rule's *parameters* - the name count,
the liquidity window, the price floor, the history requirement - rather than anything
resembling a list.

Section 3 is also new and is the largest degree of freedom in the experiment: a table
mapping a vendor delist reason to a return. It is parsed as a table, with the
sensitivity ladder parsed alongside it, so that the treatment cannot be quietly
changed without changing the document's content hash.

Section 6 is a rule too. The start date is *not* in the document - it is whatever
the section 2 rule first yields 500 qualifying names on - so this parser deliberately
carries no start date at all. Anything that wants one has to compute it and report it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigParseError, _require_unique, _section

__all__ = [
    "Config004",
    "DelistTreatment",
    "load_config_004",
    "find_preregistration_004",
]

_PREREG_NAME = "PREREG_004.md"

# The three buckets section 3's table maps onto. These are *our* names for the rows,
# not the vendor's; the mapping from a vendor reason string to one of these lives in
# trendbot.sharadar, which is the only module that should know a vendor's vocabulary.
BANKRUPTCY = "bankruptcy"
ACQUISITION = "acquisition"
UNKNOWN = "unknown"


def find_preregistration_004(start: Path | None = None) -> Path:
    """Locate PREREG_004.md by walking up from ``start`` (default: this file)."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. Experiment 004 cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


@dataclass(frozen=True, slots=True)
class DelistTreatment:
    """Section 3's table: what return a held position is assigned when it delists."""

    acquisition_return: float | None  # None == "final traded price", i.e. no haircut
    bankruptcy_return: float
    unknown_return: float
    sensitivity_alternatives: tuple[float, ...]  # the two values section 3 names
    sensitivity_returns: tuple[float, ...]  # those two plus the headline, sorted

    def assigned_return(self, bucket: str, *, unknown_override: float | None = None) -> float:
        """The return to assign, per section 3, for one delisting.

        ``unknown_override`` exists only for section 3's mandatory sensitivity ladder.
        It moves the *unknown* bucket and nothing else; bankruptcy stays at -100% and
        an acquisition still pays out at its final traded price, because neither of
        those is the degree of freedom section 3 says must be shown.
        """
        if bucket == ACQUISITION:
            return 0.0 if self.acquisition_return is None else self.acquisition_return
        if bucket == BANKRUPTCY:
            return self.bankruptcy_return
        if bucket == UNKNOWN:
            return self.unknown_return if unknown_override is None else unknown_override
        raise KeyError(f"unknown delist bucket {bucket!r}")

    def describe(self) -> str:
        return (
            f"acquisition -> final traded price (0%), bankruptcy -> "
            f"{self.bankruptcy_return:.0%}, unknown/OTC -> {self.unknown_return:.0%} "
            f"(sensitivity {', '.join(f'{r:.0%}' for r in self.sensitivity_returns)})"
        )


@dataclass(frozen=True, slots=True)
class Config004:
    """Every frozen parameter of the point-in-time momentum test, from PREREG_004.md."""

    # provenance
    source_path: Path
    source_sha256: str
    committed_on: str
    signed_by: str
    signed_date: str
    configurations_tried: int

    # section 2 - the universe rule
    universe_size: int
    liquidity_window_days: int
    price_floor: float
    min_history_days: int
    adrs_included: bool
    excluded_types: tuple[str, ...]

    # section 3 - delisting
    delisting: DelistTreatment

    # section 4 - the signal
    formation_days: int
    skip_days: int
    n_quantiles: int
    long_only: bool

    # section 5 - risk scaling and execution
    gross_exposure_cap: float
    rebalance: str
    cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]
    requires_point_in_time_adjustment: bool

    # section 6 - the sample window RULE. There is deliberately no start date here.
    start_rule_min_names: int

    # section 8 - the decision rule
    market_proxy_symbol: str
    support_min_sharpe: float
    support_min_sharpe_excess_over_buy_and_hold: float
    max_inversions: int
    min_spread_t_stat: float
    min_alpha_t_stat: float
    abandon_below_sharpe: float
    abandon_below_spread_t_stat: float

    # section 9 - expectations of record
    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float
    must_be_worse_than: float
    required_crash_months: tuple[str, ...]
    third_failure_invalidates: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.formation_days <= self.skip_days:
            raise ConfigParseError(
                f"formation window {self.formation_days} must be longer than the skip "
                f"{self.skip_days}; the ratio would otherwise run backwards"
            )
        if self.skip_days < 0:
            raise ConfigParseError("skip cannot be negative")
        if self.n_quantiles < 2:
            raise ConfigParseError("a quantile sort needs at least two buckets")
        if self.universe_size < self.n_quantiles:
            raise ConfigParseError(
                f"a {self.n_quantiles}-way sort needs at least {self.n_quantiles} names, "
                f"but section 2 asks for {self.universe_size}"
            )
        if self.start_rule_min_names != self.universe_size:
            raise ConfigParseError(
                f"section 6 starts the sample when {self.start_rule_min_names} names "
                f"qualify but section 2 asks for {self.universe_size}; the two must agree "
                "or the first months of the sample run a smaller universe than the rule"
            )
        if self.liquidity_window_days < 1:
            raise ConfigParseError("the liquidity window must be at least one day")
        if not self.price_floor > 0:
            raise ConfigParseError("the price floor must be positive")
        if not self.gross_exposure_cap > 0:
            raise ConfigParseError("gross exposure cap must be positive")
        if self.cost_bps_per_side < 0:
            raise ConfigParseError("cost cannot be negative")
        if self.configurations_tried < 1:
            raise ConfigParseError("configurations tried must be at least 1")
        if self.cost_bps_per_side not in self.cost_sensitivity_bps:
            raise ConfigParseError(
                f"headline cost {self.cost_bps_per_side} bps is absent from the "
                f"sensitivity ladder {self.cost_sensitivity_bps}"
            )
        if self.abandon_below_spread_t_stat > self.min_spread_t_stat:
            raise ConfigParseError(
                "section 8's abandon threshold on the spread t-statistic is above its "
                "support threshold, which leaves the rule with no inconclusive band"
            )
        if self.delisting.bankruptcy_return != -1.0:
            raise ConfigParseError(
                f"section 3 maps bankruptcy to {self.delisting.bankruptcy_return}, not -100%"
            )
        # Section 3's ladder is "the headline, plus the two alternatives it names". If the
        # headline coincides with one of the alternatives the ladder silently collapses
        # from three points to two and stops showing the size of the choice, which is
        # the one thing section 3 says it is mandatory for.
        ladder = self.delisting.sensitivity_returns
        if len(set(ladder)) != len(self.delisting.sensitivity_alternatives) + 1:
            raise ConfigParseError(
                f"section 3's sensitivity ladder collapses to {sorted(set(ladder))}: the "
                f"headline unknown-bucket treatment of {self.delisting.unknown_return:.0%} "
                "coincides with one of the alternatives it is supposed to be compared against"
            )
        if not re.fullmatch(r"[A-Z]{1,5}", self.market_proxy_symbol):
            raise ConfigParseError(
                f"section 8 names {self.market_proxy_symbol!r} as the alpha benchmark, which "
                "is not a ticker"
            )
        for month in self.required_crash_months:
            pd.Period(month, freq="M")

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000.0

    def describe(self) -> str:
        return (
            f"point-in-time cross-sectional momentum [top {self.universe_size} by "
            f"{self.liquidity_window_days}d dollar volume, >= ${self.price_floor:.2f}, "
            f"formation={self.formation_days}d, skip={self.skip_days}d, "
            f"top 1/{self.n_quantiles}, cost={self.cost_bps_per_side:g}bps/side] "
            f"sha256={self.source_sha256[:12]}"
        )


_WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def _parse_delisting(s3: str) -> DelistTreatment:
    """Section 3's table, read as a table rather than as three separate greps."""
    rows: dict[str, str] = {}
    for line in s3.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        reason, treatment = cells
        if reason.lower() == "reason" or set(reason) <= set("- :"):
            continue
        rows[reason] = treatment
    if len(rows) != 3:
        raise ConfigParseError(
            f"section 3's delisting table has {len(rows)} rows, expected 3 "
            f"(acquisition, bankruptcy, unknown): {list(rows)}"
        )

    def _find(pattern: str, what: str) -> str:
        matches = [v for k, v in rows.items() if re.search(pattern, k, re.IGNORECASE)]
        if len(matches) != 1:
            raise ConfigParseError(
                f"section 3's table does not map {what} exactly once (matched {len(matches)})"
            )
        return matches[0]

    acquisition = _find(r"acquisition|merger", "acquisition/merger")
    if not re.search(r"final traded price", acquisition, re.IGNORECASE):
        raise ConfigParseError(
            f"section 3's acquisition row no longer says 'final traded price': {acquisition!r}"
        )
    if not re.search(r"proceeds to cash", acquisition, re.IGNORECASE):
        raise ConfigParseError(
            "section 3's acquisition row no longer says the proceeds go to cash, which is "
            "what distinguishes it from simply holding a frozen price"
        )

    def _percent(text: str, what: str) -> float:
        match = re.search(r"\*\*[-–—−]?(\d+(?:\.\d+)?)%\*\*", text)
        if match is None:
            raise ConfigParseError(f"section 3's {what} row has no bolded percentage: {text!r}")
        sign = -1.0 if re.search(r"\*\*[-–—−]", text) else 1.0
        return sign * float(match.group(1)) / 100.0

    bankruptcy = _percent(_find(r"bankrupt|liquidat", "bankruptcy/liquidation"), "bankruptcy")
    unknown = _percent(_find(r"OTC|unknown|missing", "moved-to-OTC/unknown"), "unknown")

    ladder_text = _require_unique(
        r"unknown bucket set to\s*([-–—−]?\d+(?:\.\d+)?)%\s*and to\s*([-–—−]?\d+(?:\.\d+)?)%",
        s3,
        "section 3's mandatory sensitivity ladder",
    )

    def _signed(raw: str) -> float:
        return (-1.0 if raw[0] in "-–—−" else 1.0) * float(re.sub(r"^[-–—−]", "", raw)) / 100.0

    alternatives = tuple(sorted({_signed(ladder_text.group(1)), _signed(ladder_text.group(2))}))
    return DelistTreatment(
        acquisition_return=None,
        bankruptcy_return=bankruptcy,
        unknown_return=unknown,
        sensitivity_alternatives=alternatives,
        sensitivity_returns=tuple(sorted({unknown, *alternatives})),
    )


def _parse_text(text: str, source_path: Path) -> Config004:
    s2, s3, s4, s5, s6, s8, s9 = (_section(text, n) for n in (2, 3, 4, 5, 6, 8, 9))

    configurations_tried = int(
        _require_unique(
            r"\*\*Configurations tried on this dataset, cumulative:\*\*\s*(\d+)",
            text,
            "cumulative configuration counter",
        ).group(1)
    )

    # ---- section 2: the universe rule -------------------------------------------------
    rule = _require_unique(
        r"\*\*The (\d+) US common stocks with the highest median dollar volume over the "
        r"trailing\s*\n?(\d+) trading days\*\*",
        s2,
        "the universe rule",
    )
    universe_size, liquidity_window = int(rule.group(1)), int(rule.group(2))
    price_floor = float(
        _require_unique(
            r"Unadjusted close on date t is at least \*\*\$(\d+(?:\.\d+)?)\*\*",
            s2,
            "the price floor",
        ).group(1)
    )
    # "Unadjusted" is load-bearing: the floor applied to today's adjusted price would be
    # a lookahead. Asserted so that removing the word is a parse error.
    min_history = int(
        _require_unique(
            r"At least (?:\*\*)?(\d+)(?:\*\*)? trading days of price history as of date t",
            s2,
            "the history requirement",
        ).group(1)
    )
    adrs_included = bool(re.search(r"ADRs are \*\*included\*\*", s2))
    if not adrs_included:
        raise ConfigParseError("section 2 no longer states that ADRs are included")
    excluded_raw = _require_unique(
        r"Exclude ([^.]+)\.", s2, "the excluded security types"
    ).group(1)
    excluded_types = tuple(
        t.strip() for t in re.split(r",|\band\b", excluded_raw) if t.strip()
    )
    _require_unique(
        r"not yet delisted", s2, "section 2's tradeable-on-date-t requirement"
    )

    # ---- section 3: delisting ---------------------------------------------------------
    delisting = _parse_delisting(s3)

    # ---- section 4: the signal --------------------------------------------------------
    formula = _require_unique(
        r"momentum_i\(t\)\s*=\s*P_i\(t-(\d+)\)\s*/\s*P_i\(t-(\d+)\)\s*-\s*1",
        s4,
        "the momentum formula",
    )
    skip_days, formation_days = int(formula.group(1)), int(formula.group(2))
    quantile_word = _require_unique(
        r"into \*\*(deciles|quintiles|quartiles|terciles)\*\*", s4, "the quantile cut"
    ).group(1)
    n_quantiles = {"terciles": 3, "quartiles": 4, "quintiles": 5, "deciles": 10}[quantile_word]
    long_only = bool(re.search(r"Long-only\.", s4))
    if not long_only:
        raise ConfigParseError("section 4 no longer declares the strategy long-only")

    # ---- section 5: risk scaling and execution ----------------------------------------
    gross_cap = float(
        _require_unique(
            r"Gross exposure\s*(\d+(?:\.\d+)?)\s*when invested", s5, "gross exposure"
        ).group(1)
    )
    _require_unique(r"no volatility targeting", s5, "the no-vol-targeting clause")
    rebalance = (
        _require_unique(r"Rebalance:\s*(.+?)\.?\s*$", s5, "rebalance schedule", re.MULTILINE)
        .group(1)
        .strip()
    )
    cost_bps = float(
        _require_unique(
            r"\*\*Cost assumption:\s*(\d+(?:\.\d+)?)\s*bps per side\.\*\*", s5, "cost assumption"
        ).group(1)
    )
    ladder_raw = _require_unique(
        r"Sensitivity at\s*([\d/\s]+?)\s*\.", s5, "cost sensitivity ladder"
    ).group(1)
    cost_ladder = tuple(float(x) for x in re.findall(r"[\d.]+", ladder_raw))
    requires_pit = bool(
        re.search(r"split-and-dividend adjusted using point-in-time factors", s5)
    )
    if not requires_pit:
        raise ConfigParseError("section 5 no longer requires point-in-time adjustment")

    # ---- section 6: the start RULE, not a date ----------------------------------------
    start_rule = _require_unique(
        r"\*\*Start:\*\*\s*the earliest month at which the .2 rule yields at least "
        r"(\d+)\s*qualifying\s*\n?\s*names",
        s6,
        "section 6's start rule",
    )
    start_min_names = int(start_rule.group(1))
    # If this ever parses to a literal date, the window has stopped being a rule.
    if re.search(r"\*\*Start:\*\*\s*\d{4}-\d{2}-\d{2}", s6):
        raise ConfigParseError(
            "section 6 now names a start date. The window is supposed to be a rule whose "
            "answer depends on vendor coverage; a date here would be a chosen window."
        )

    # ---- section 8: the decision rule -------------------------------------------------
    support_sharpe = float(
        _require_unique(
            r"Net Sharpe \(excess of T-bill, \d+ bps\) exceeds\s*\*\*(\d+(?:\.\d+)?)\*\*",
            s8,
            "support threshold Sharpe",
        ).group(1)
    )
    support_excess = float(
        _require_unique(
            r"by at least \*\*(\d+(?:\.\d+)?)\*\*", s8, "excess-over-buy-and-hold threshold"
        ).group(1)
    )
    max_inversions = _WORD_NUMBERS[
        _require_unique(
            r"at\s*\n?\s*most \*\*(zero|one|two|three)\*\* adjacent inversion",
            s8,
            "the tolerated inversion count",
        ).group(1)
    ]
    spread_t = float(
        _require_unique(
            r"D1[-–—−]D10 spread positive at \*\*t\s*>\s*(\d+(?:\.\d+)?)\*\*",
            s8,
            "the spread t-statistic support threshold",
        ).group(1)
    )
    # The index section 8 names is CAPTURED, not assumed. Two reasons: the document is
    # the source of truth for which series "alpha" is measured against, and a literal
    # ticker in this file would be a universe member inlined outside the parser, which
    # tests/test_repo_invariants.py bans for good reason.
    alpha_clause = _require_unique(
        r"\*\*Alpha to ([A-Z]{1,5}) is positive with t\s*>\s*(\d+(?:\.\d+)?)\.\*\*",
        s8,
        "the alpha t-statistic support threshold",
    )
    market_proxy_symbol = alpha_clause.group(1)
    alpha_t = float(alpha_clause.group(2))
    abandon_sharpe = float(
        _require_unique(
            r"net Sharpe below \*\*(\d+(?:\.\d+)?)\*\*", s8, "the Sharpe abandonment threshold"
        ).group(1)
    )
    abandon_spread_t = float(
        _require_unique(
            r"D1[-–—−]D10 t below (\d+(?:\.\d+)?)", s8, "the spread-t abandonment threshold"
        ).group(1)
    )
    for clause, what in (
        (r"fails to beat buy-and-hold at all", "the buy-and-hold abandon clause"),
        (rf"alpha to {market_proxy_symbol} negative", "the negative-alpha abandon clause"),
        (r"more than one inversion", "the inversion-count abandon clause"),
    ):
        _require_unique(clause, s8, what)
    # Section 8's closing paragraph is the one that makes a size effect reportable rather
    # than ignorable. It carries no number, so its presence is asserted.
    _require_unique(
        r"evidence of a\s+size effect rather than momentum",
        s8,
        "section 8's size-effect reporting clause",
    )

    # ---- section 9: expectations of record --------------------------------------------
    exp = _require_unique(
        r"Realistic net Sharpe:\s*\*\*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\*\*",
        s9,
        "expected Sharpe range",
    )
    bug_threshold = float(
        _require_unique(r"Above (\d+(?:\.\d+)?) means a bug", s9, "bug threshold Sharpe").group(1)
    )
    worse_than = float(
        _require_unique(
            r"be \*\*worse\*\* than 003's (\d+(?:\.\d+)?) headline",
            s9,
            "the must-be-worse-than-003 expectation",
        ).group(1)
    )
    crash = _require_unique(
        r"\*\*Momentum crashes are mandatory\.\*\*\s*(\w+)[-–—](\w+)\s*(\d{4})\s*and\s*(\w+)\s*(\d{4})",
        s9,
        "the mandatory crash months",
    )
    low, high, year = _MONTHS[crash.group(1)], _MONTHS[crash.group(2)], int(crash.group(3))
    if high < low:
        raise ConfigParseError("section 9's crash month range runs backwards")
    required = [f"{year}-{m:02d}" for m in range(low, high + 1)]
    required.append(f"{int(crash.group(5))}-{_MONTHS[crash.group(4)]:02d}")
    third_failure_invalid = bool(
        re.search(r"reported as invalid rather than as a verdict", s9)
    )
    if not third_failure_invalid:
        raise ConfigParseError(
            "section 9 no longer says a third crash-check failure invalidates the result, "
            "which is the clause that decides whether a verdict may be reported at all"
        )

    # ---- section 10: the paired diagnostic is a required output ------------------------
    # In the section HEADING, not its body, so this searches the whole document.
    _require_unique(
        r"required output regardless of verdict", text, "section 10's unconditional requirement"
    )

    committed = _require_unique(
        r"\*\*Committed on:\*\*\s*(\d{4}-\d{2}-\d{2})", text, "commit date"
    ).group(1)
    signature = _require_unique(
        r"Signed:\s*\*\*(.+?)\*\*\s+Date:\s*\*\*(.+?)\*\*", text, "signature"
    )

    return Config004(
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        committed_on=committed,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2).strip(),
        configurations_tried=configurations_tried,
        universe_size=universe_size,
        liquidity_window_days=liquidity_window,
        price_floor=price_floor,
        min_history_days=min_history,
        adrs_included=adrs_included,
        excluded_types=excluded_types,
        delisting=delisting,
        formation_days=formation_days,
        skip_days=skip_days,
        n_quantiles=n_quantiles,
        long_only=long_only,
        gross_exposure_cap=gross_cap,
        rebalance=rebalance,
        cost_bps_per_side=cost_bps,
        cost_sensitivity_bps=cost_ladder,
        requires_point_in_time_adjustment=requires_pit,
        start_rule_min_names=start_min_names,
        market_proxy_symbol=market_proxy_symbol,
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
        must_be_worse_than=worse_than,
        required_crash_months=tuple(required),
        third_failure_invalidates=third_failure_invalid,
    )


_CACHE: dict[Path, Config004] = {}


def load_config_004(path: Path | str | None = None, *, use_cache: bool = True) -> Config004:
    """Parse PREREG_004.md into a frozen :class:`Config004`."""
    resolved = Path(path).resolve() if path is not None else find_preregistration_004()
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
