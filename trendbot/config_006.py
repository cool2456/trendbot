"""Parse PREREG_006.md into a frozen configuration object.

Same contract as :mod:`trendbot.config` and :mod:`trendbot.config_002` through
:mod:`trendbot.config_005`: the pre-registration is the sole source of truth, every
parameter is *pulled* out of the text, and a missing, ambiguous or self-contradictory
value is a fatal :class:`~trendbot.config.ConfigParseError` rather than a silent
default.

What is different about this document
-------------------------------------
**There is no signal in it.** Sections 2 to 7 describe a portfolio construction over
return series three earlier experiments already produced, so the parser extracts no
lookback, no formation window and no universe of instruments. What it extracts instead
is an allocation rule, a sleeve *manifest*, and — unusually — several clauses that are
prohibitions rather than parameters.

Three of those prohibitions are load-bearing and are asserted rather than assumed:

* **Section 3's "covariance adapts, nothing else does"** is the single clause that
  separates this experiment from automated data mining. If it is ever deleted from the
  document, this parser stops rather than quietly permitting a performance-based
  allocation.
* **Section 2's exclusion of 003 on validity, not performance.** The document says
  recording the reason "prevents it being reread later as a performance exclusion", so
  the reason is asserted to still be there.
* **The header's admission that the multiple-testing correction is understated.** That
  clause is the reason a passing verdict here would not be confirmatory, and
  :mod:`trendbot.engine.ms_validation` restates it next to every deflated Sharpe. A
  document that dropped it while keeping ``configurations_tried = 5`` would be claiming
  a stronger result than it had earned, so the parse fails.

Two readings, deliberately not resolved here
--------------------------------------------
Section 3 never says where the volatility target enters its weight pipeline, and
section 4 never says whether sleeve C's benchmark carries the same carry correction
section 2 applies to sleeve C itself. Both readings of both questions move a section 8
clause across zero. The parser therefore extracts the *parameters* and refuses to pick
a reading: :data:`~trendbot.engine.allocation.WEIGHT_ORDERS` and the benchmark variant
are arguments with no default, chosen and reported at the call site. See FINDINGS_006.md
section on unpinned levers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigParseError, _require_unique, _section

__all__ = [
    "Config006",
    "load_config_006",
    "find_preregistration_006",
    "SLEEVE_SOURCES",
]

_PREREG_NAME = "PREREG_006.md"

# Section 2's table names each sleeve by the experiment that produced it. The mapping
# from that label to the module that reproduces the series lives in trendbot.sleeves;
# what belongs here is only the manifest the document actually fixes, so that adding or
# removing a sleeve becomes a parse error rather than an edit to a Python literal.
SLEEVE_SOURCES: tuple[str, ...] = ("001", "002", "005")


def find_preregistration_006(start: Path | None = None) -> Path:
    """Locate PREREG_006.md by walking up from ``start`` (default: this file)."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. Experiment 006 cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


@dataclass(frozen=True, slots=True)
class Config006:
    """Every frozen parameter of multi-strategy risk allocation, from PREREG_006.md."""

    # provenance
    source_path: Path
    source_sha256: str
    committed_on: str
    signed_by: str
    signed_date: str

    # header
    configurations_tried: int
    correction_is_a_floor: bool

    # section 2 - the sleeve manifest
    sleeves: tuple[str, ...]
    excluded_experiment: str
    exclusion_is_on_validity: bool
    sleeve_c_is_carry_corrected: bool
    no_sleeve_may_be_substituted: bool

    # section 3 - allocation
    covariance_halflife_days: int
    portfolio_vol_target: float
    gross_exposure_cap: float
    min_sleeve_weight: float
    max_sleeve_weight: float
    performance_is_input_to_nothing: bool

    # section 4 - benchmark
    benchmark_is_erc_of_benchmarks: bool
    sharpe_is_excess_of_tbill: bool

    # section 5 - sample window
    window_is_the_intersection: bool

    # section 6 - execution
    overlay_cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]
    rebalance: str

    # section 8 - pre-committed decision rule
    min_excess_over_benchmark: float
    min_excess_over_best_sleeve: float
    correlation_clause_required: bool

    # section 9 - expectations of record
    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.sleeves != SLEEVE_SOURCES:
            raise ConfigParseError(
                f"section 2's sleeve set is {self.sleeves}, not {SLEEVE_SOURCES}. "
                "'No sleeve may be added, removed or substituted after this document is "
                "committed' — changing it creates experiment 007."
            )
        if len(set(self.sleeves)) != len(self.sleeves):
            raise ConfigParseError(f"section 2 names a sleeve twice: {self.sleeves}")
        if self.excluded_experiment in self.sleeves:
            raise ConfigParseError(
                f"experiment {self.excluded_experiment} is both excluded and allocated to"
            )
        if self.covariance_halflife_days < 1:
            raise ConfigParseError("covariance halflife must be at least one day")
        if not self.portfolio_vol_target > 0:
            raise ConfigParseError("volatility target must be positive")
        if not self.gross_exposure_cap > 0:
            raise ConfigParseError("gross exposure cap must be positive")
        if not 0.0 <= self.min_sleeve_weight <= self.max_sleeve_weight <= 1.0:
            raise ConfigParseError(
                f"section 3's weight bounds [{self.min_sleeve_weight}, "
                f"{self.max_sleeve_weight}] are not an ordered pair inside [0, 1]"
            )
        n = len(self.sleeves)
        if self.min_sleeve_weight * n > 1.0 or self.max_sleeve_weight * n < 1.0:
            raise ConfigParseError(
                f"section 3's bounds cannot produce {n} weights summing to one: "
                f"{n} x {self.min_sleeve_weight} > 1 or {n} x {self.max_sleeve_weight} < 1"
            )
        if self.overlay_cost_bps_per_side < 0:
            raise ConfigParseError("cost cannot be negative")
        if self.overlay_cost_bps_per_side not in self.cost_sensitivity_bps:
            raise ConfigParseError(
                f"headline cost {self.overlay_cost_bps_per_side} bps is absent from the "
                f"sensitivity ladder {self.cost_sensitivity_bps}"
            )
        if self.configurations_tried < 1:
            raise ConfigParseError("configurations tried must be at least 1")
        if self.min_excess_over_benchmark <= 0 or self.min_excess_over_best_sleeve <= 0:
            raise ConfigParseError("section 8's margins must be positive to be gates")
        if self.expected_sharpe_low > self.expected_sharpe_high:
            raise ConfigParseError("section 9's expected Sharpe range runs backwards")
        if self.bug_threshold_sharpe <= self.expected_sharpe_high:
            raise ConfigParseError(
                "section 9's bug threshold is inside its realistic range, so the range "
                "cannot be checked against it"
            )
        # The clauses that are prohibitions rather than parameters. Each one is the
        # reason a specific failure mode cannot happen; a document that dropped one
        # while keeping the numbers would be a different experiment wearing this one's
        # provenance hash.
        if not self.correction_is_a_floor:
            raise ConfigParseError(
                "the header no longer records that configs_tried = 5 is a FLOOR rather "
                "than the true multiple-testing burden. That clause is why a passing "
                "verdict here is suggestive rather than confirmatory; without it the "
                "deflated Sharpe would be reported as stronger than it is."
            )
        if not self.performance_is_input_to_nothing:
            raise ConfigParseError(
                "section 3 no longer states that realised P&L, drawdown, hit rate and "
                "rolling Sharpe are inputs to nothing. That clause is 'the single clause "
                "that separates this experiment from automated data mining'."
            )
        if not self.exclusion_is_on_validity:
            raise ConfigParseError(
                "section 2 no longer records that 003 is excluded on validity rather than "
                "performance. The document says recording the reason 'prevents it being "
                "reread later as a performance exclusion'."
            )
        if not self.sleeve_c_is_carry_corrected:
            raise ConfigParseError("section 2 no longer pins sleeve C to carry-corrected returns")
        if not self.no_sleeve_may_be_substituted:
            raise ConfigParseError("section 2 no longer forbids substituting a sleeve")
        if not self.benchmark_is_erc_of_benchmarks:
            raise ConfigParseError(
                "section 4 no longer builds the benchmark with the same ERC construction; "
                "the comparison would stop being like-for-like"
            )
        if not self.sharpe_is_excess_of_tbill:
            raise ConfigParseError("section 4 no longer computes Sharpe in excess of T-bills")
        if not self.window_is_the_intersection:
            raise ConfigParseError("section 5 no longer defines the window as an intersection")
        if not self.correlation_clause_required:
            raise ConfigParseError(
                "section 8's third clause is gone. It is the mechanism test — 'if it is "
                "false, any outperformance came from somewhere other than the stated "
                "cause' — and without it the rule tests only outcome."
            )
        if self.rebalance != "monthly":
            raise ConfigParseError(f"section 6 specifies a {self.rebalance!r} rebalance, not monthly")

    @property
    def n_sleeves(self) -> int:
        return len(self.sleeves)

    @property
    def cost_rate_per_side(self) -> float:
        return self.overlay_cost_bps_per_side / 10_000.0

    def describe(self) -> str:
        return (
            f"multi-strategy risk allocation [ERC over {self.n_sleeves} sleeves "
            f"{'/'.join(self.sleeves)}, EWMA halflife={self.covariance_halflife_days}d, "
            f"vol_target={self.portfolio_vol_target:.0%}, gross_cap={self.gross_exposure_cap:g}, "
            f"weights in [{self.min_sleeve_weight:.0%}, {self.max_sleeve_weight:.0%}], "
            f"overlay cost={self.overlay_cost_bps_per_side:g}bps/side] "
            f"sha256={self.source_sha256[:12]}"
        )


_DASH = r"[-‐‑‒–—−]"


def _phrase(text: str) -> str:
    """A prose assertion as a regex, tolerant of where the document wrapped its lines.

    Identical in purpose to :func:`trendbot.config_005._phrase`: markdown hard-wraps at
    whatever column the author used, so matching literal spaces would make these
    assertions fail on reflow rather than on meaning.
    """
    return r"\s+".join(re.escape(word) for word in text.split())


def _parse_text(text: str, source_path: Path) -> Config006:
    s2, s3, s4, s5, s6, s8, s9 = (_section(text, n) for n in (2, 3, 4, 5, 6, 8, 9))

    # ---- header ----------------------------------------------------------------------
    committed_on = _require_unique(
        _phrase("**Committed on:**") + r"\s*(\d{4}-\d{2}-\d{2})", text, "the commit date"
    ).group(1)
    configurations_tried = int(
        _require_unique(
            _phrase("**Configurations tried, cumulative:**") + r"\s*(\d+)",
            text,
            "the cumulative configuration counter",
        ).group(1)
    )
    # The header's admission that the counter understates the burden. Both halves are
    # required: the word "floor" alone could survive an edit that dropped the reasoning.
    correction_is_a_floor = bool(
        re.search(_phrase("`configs_tried = 5` is therefore a *floor*, not the true multiple-testing"), text)
        and re.search(_phrase("**A passing verdict here is suggestive, not confirmatory,**"), text)
    )
    _require_unique(
        _phrase("Frozen on commit. Changes to §2–7 create experiment 007."),
        text,
        "the freeze clause",
    )

    # ---- section 2: the sleeve manifest ----------------------------------------------
    # The table rows, read as rows rather than as prose, so that deleting one is a parse
    # failure rather than a silently shorter portfolio. The label pattern is ANY single
    # uppercase letter rather than the three that happen to be there: restricting it to
    # [ABC] would make a fourth row labelled D invisible to the parser, so a sleeve could
    # be ADDED to the document and silently ignored - the exact failure section 2's
    # "no sleeve may be added" clause exists to prevent.
    sleeves = tuple(
        match.group(2)
        for match in re.finditer(r"^\|\s*([A-Z])\s*\|\s*(\d{3})\s*\|", s2, re.MULTILINE)
    )
    if not sleeves:
        raise ConfigParseError(
            "section 2's sleeve table could not be read; refusing to guess which "
            "experiments supply the sleeves"
        )
    _require_unique(
        _phrase("**Every completed sleeve is included, including those that lost to their"),
        s2,
        "section 2's no-selection clause",
    )
    no_sleeve_may_be_substituted = bool(
        re.search(
            _phrase("**No sleeve may be added, removed or substituted after this document is"), s2
        )
    )
    sleeve_c_is_carry_corrected = bool(
        re.search(_phrase("**carry-corrected**"), s2) and re.search(_phrase("Sleeve C is included despite a negative raw Sharpe."), s2)
    )
    excluded = _require_unique(
        r"\*\*(\d{3})\s+" + _phrase("is excluded on validity, not performance.**"),
        s2,
        "section 2's excluded experiment",
    ).group(1)
    exclusion_is_on_validity = bool(
        re.search(_phrase("an unmeasurable sleeve cannot be allocated to"), s2)
        and re.search(_phrase("prevents it being reread later as a performance exclusion"), s2)
    )

    # ---- section 3: allocation --------------------------------------------------------
    halflife = int(
        _require_unique(
            _phrase("**EWMA of daily sleeve returns, halflife") + r"\s*(\d+)\s*\n?\s*"
            + _phrase("trading days**"),
            s3,
            "the covariance halflife",
        ).group(1)
    )
    _require_unique(
        _phrase("using data through the rebalance date only"),
        s3,
        "section 3's point-in-time requirement",
    )
    vol_target = (
        float(
            _require_unique(
                _phrase("ex-ante portfolio volatility targets **") + r"(\d+(?:\.\d+)?)%\s*annualised\*\*",
                s3,
                "the portfolio volatility target",
            ).group(1)
        )
        / 100.0
    )
    gross_cap = float(
        _require_unique(
            _phrase("Gross exposure capped at **") + r"(\d+(?:\.\d+)?)\*\*", s3, "the gross exposure cap"
        ).group(1)
    )
    bounds = _require_unique(
        _phrase("Minimum weight per sleeve **")
        + r"(\d+(?:\.\d+)?)%\*\*,\s*\n?\s*"
        + _phrase("maximum **")
        + r"(\d+(?:\.\d+)?)%\*\*",
        s3,
        "section 3's per-sleeve weight bounds",
    )
    _require_unique(
        _phrase("applied before renormalisation"), s3, "the order of section 3's weight bounds"
    )
    # The clause that makes this experiment not data mining.
    performance_is_input_to_nothing = bool(
        re.search(_phrase("**Covariance adapts. Nothing else does.**"), s3)
        and re.search(
            _phrase("Realised P&L, drawdown, hit rate and rolling Sharpe are inputs to nothing."), s3
        )
    )

    # ---- section 4: benchmark ---------------------------------------------------------
    benchmark_is_erc_of_benchmarks = bool(
        re.search(
            _phrase(
                "**The same equal-risk-contribution construction applied to the three sleeves' own"
            ),
            s4,
        )
        and re.search(
            _phrase("using identical covariance estimation, identical vol target, identical caps"), s4
        )
    )
    sharpe_is_excess_of_tbill = bool(
        re.search(_phrase("Sharpe computed as excess of the T-bill rate for both."), s4)
    )

    # ---- section 5: sample window -----------------------------------------------------
    window_is_the_intersection = bool(
        re.search(_phrase("**The intersection of all three sleeves' available histories.**"), s5)
        and re.search(_phrase("Determined by the data, reported, not chosen."), s5)
    )

    # ---- section 6: execution ---------------------------------------------------------
    cost = float(
        _require_unique(
            _phrase("**Additional cost for portfolio rebalancing:")
            + r"\s*(\d+(?:\.\d+)?)\s*bps\s+per\s+side",
            s6,
            "the overlay cost",
        ).group(1)
    )
    ladder_raw = _require_unique(
        _phrase("Sensitivity at") + r"\s*([\d/\s]+?)\s*bps", s6, "the cost sensitivity ladder"
    ).group(1)
    cost_sensitivity = tuple(float(part) for part in ladder_raw.split("/") if part.strip())
    if not cost_sensitivity:
        raise ConfigParseError(f"section 6's cost ladder {ladder_raw!r} parsed to nothing")
    rebalance = _require_unique(
        r"Rebalance\s+(\w+),\s*first\s+trading\s+day", s6, "the rebalance schedule"
    ).group(1)

    # ---- section 8: the pre-committed decision rule ------------------------------------
    over_benchmark = float(
        _require_unique(
            _phrase("Net Sharpe exceeds the §4 benchmark by at least **") + r"(\d+(?:\.\d+)?)\*\*",
            s8,
            "section 8's margin over the benchmark",
        ).group(1)
    )
    over_best_sleeve = float(
        _require_unique(
            _phrase("Net Sharpe exceeds **the best individual sleeve's Sharpe** by at least **")
            + r"(\d+(?:\.\d+)?)\*\*",
            s8,
            "section 8's margin over the best sleeve",
        ).group(1)
    )
    correlation_clause_required = bool(
        re.search(
            _phrase(
                "The **realised average pairwise correlation between sleeves is lower than the"
            ),
            s8,
        )
        and re.search(_phrase("This is §1's mechanism."), s8)
    )

    # ---- section 9: expectations of record ---------------------------------------------
    expected = _require_unique(
        _phrase("Realistic combined Sharpe: **")
        + r"(\d+(?:\.\d+)?)\s*"
        + _DASH
        + r"\s*(\d+(?:\.\d+)?)\*\*",
        s9,
        "section 9's realistic Sharpe range",
    )
    bug_threshold = float(
        _require_unique(
            _phrase("Above") + r"\s*(\d+(?:\.\d+)?)\s*" + _phrase("means a bug"),
            s9,
            "section 9's bug threshold",
        ).group(1)
    )

    # ---- section 10: signature ---------------------------------------------------------
    signature = _require_unique(
        _phrase("Signed: **") + r"(.+?)\*\*\s+" + _phrase("Date: **") + r"(\d{4}-\d{2}-\d{2})\*\*",
        text,
        "the signature block",
    )

    return Config006(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        committed_on=committed_on,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2),
        configurations_tried=configurations_tried,
        correction_is_a_floor=correction_is_a_floor,
        sleeves=sleeves,
        excluded_experiment=excluded,
        exclusion_is_on_validity=exclusion_is_on_validity,
        sleeve_c_is_carry_corrected=sleeve_c_is_carry_corrected,
        no_sleeve_may_be_substituted=no_sleeve_may_be_substituted,
        covariance_halflife_days=halflife,
        portfolio_vol_target=vol_target,
        gross_exposure_cap=gross_cap,
        min_sleeve_weight=float(bounds.group(1)) / 100.0,
        max_sleeve_weight=float(bounds.group(2)) / 100.0,
        performance_is_input_to_nothing=performance_is_input_to_nothing,
        benchmark_is_erc_of_benchmarks=benchmark_is_erc_of_benchmarks,
        sharpe_is_excess_of_tbill=sharpe_is_excess_of_tbill,
        window_is_the_intersection=window_is_the_intersection,
        overlay_cost_bps_per_side=cost,
        cost_sensitivity_bps=cost_sensitivity,
        rebalance=rebalance,
        min_excess_over_benchmark=over_benchmark,
        min_excess_over_best_sleeve=over_best_sleeve,
        correlation_clause_required=correlation_clause_required,
        expected_sharpe_low=float(expected.group(1)),
        expected_sharpe_high=float(expected.group(2)),
        bug_threshold_sharpe=bug_threshold,
    )


def load_config_006(path: Path | None = None) -> Config006:
    """Parse PREREG_006.md into a frozen :class:`Config006`."""
    source = path or find_preregistration_006()
    return _parse_text(source.read_text(encoding="utf-8"), source)
