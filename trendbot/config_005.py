"""Parse PREREG_005.md into a frozen configuration object.

Same contract as :mod:`trendbot.config`, :mod:`trendbot.config_002`,
:mod:`trendbot.config_003` and :mod:`trendbot.config_004`: the pre-registration is the
sole source of truth, every parameter is *pulled* out of the text, and a missing,
ambiguous or self-contradictory value is a fatal
:class:`~trendbot.config.ConfigParseError` rather than a silent default.

What is different about this document
-------------------------------------
Three things, and each of them is why this is a fifth parser rather than a fifth
branch in the fourth:

* **The universe is a rule over a published release**, not a ticker list and not an
  index snapshot. What the parser can extract is the release name, the expected yield
  range, and the *requirement that the quote convention be normalised*. Resolving it
  needs FRED, and that lives in :mod:`trendbot.fx`.
* **The section numbers moved.** 002 and 003 put the sample window in section 6; here
  section 3 is the sample window and section 6 is the freeze list. Anything that
  addressed sections by number would read the wrong paragraph, quietly.
* **Section 4 carries a required diagnostic** - the interest-rate approximation - which
  is explicitly "not a configuration". The parser asserts the clause is present so that
  deleting it from the document becomes a parse error rather than a dropped obligation.

The counter is 4, not 5
------------------------
PREREG_005.md's header argues at length that blocked experiment 004 does not count
toward ``configurations_tried`` because it never reached data and so produced nothing
that a winner could have been selected from. That argument is load-bearing for the
deflated Sharpe, so the parser reads the number the document states **and** asserts the
justification is still in the document. A future edit that bumps the number without the
reasoning, or strips the reasoning while keeping the number, fails to parse.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigParseError, _require_unique, _section

__all__ = [
    "Config005",
    "load_config_005",
    "find_preregistration_005",
    "DISLOCATION_MONTHS",
]

_PREREG_NAME = "PREREG_005.md"

# Section 9 names five FX dislocations by label - "2008 Q4", "2011 CHF" - not by month.
# Turning a label into a set of months is a reporting choice the document does not make,
# so it is made here, once, in the open. It gates nothing: section 9 explicitly imposes
# no mandatory month for this experiment. It exists so the clustering assessment is a
# set-membership test rather than a paragraph of prose.
DISLOCATION_MONTHS: dict[str, tuple[str, ...]] = {
    # The global financial crisis quarter, named by the document as a quarter.
    "2008 Q4": ("2008-10", "2008-11", "2008-12"),
    # The franc's melt-up to near parity with the euro in August 2011 and the SNB's
    # imposition of the 1.20 floor on 6 September.
    "2011 CHF": ("2011-08", "2011-09"),
    # The SNB abandoned the floor on 15 January 2015.
    "2015 CHF de-peg": ("2015-01",),
    # The document names the month.
    "2020 March": ("2020-03",),
    # Sterling's collapse after the 23 September mini-budget, and its retracement in
    # October once the measures were reversed.
    "2022 GBP": ("2022-09", "2022-10"),
}


def find_preregistration_005(start: Path | None = None) -> Path:
    """Locate PREREG_005.md by walking up from ``start`` (default: this file)."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _PREREG_NAME if parent.is_dir() else parent.parent / _PREREG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigParseError(
        f"{_PREREG_NAME} not found above {here}. Experiment 005 cannot be configured "
        "without it; refusing to fall back to hardcoded parameters."
    )


@dataclass(frozen=True, slots=True)
class Config005:
    """Every frozen parameter of cross-sectional currency momentum, from PREREG_005.md."""

    # provenance
    source_path: Path
    source_sha256: str
    committed_on: str
    signed_by: str
    signed_date: str

    # header - the cumulative trial counter
    configurations_tried: int

    # section 2 - universe, as a specification rather than a list
    release_name: str
    expected_universe_low: int
    expected_universe_high: int
    quote_normalisation_required: bool

    # section 3 - sample window
    sample_start: str

    # section 4 - returns and signal
    spot_only: bool
    interest_rate_diagnostic_required: bool
    formation_days: int
    skip_days: int
    n_quantiles: int
    long_only: bool

    # section 7 - test protocol
    equity_proxy_symbol: str

    # section 5 - execution
    gross_exposure_cap: float
    rebalance: str
    cost_bps_per_side: float
    cost_sensitivity_bps: tuple[float, ...]
    benchmark_label: str

    # section 8 - pre-committed decision rule
    support_min_sharpe: float
    support_min_sharpe_excess_over_benchmark: float
    max_inversions: int
    min_spread_t_stat: float
    min_alpha_t_stat: float
    abandon_below_sharpe: float
    abandon_below_spread_t_stat: float

    # section 9 - expectations of record
    expected_sharpe_low: float
    expected_sharpe_high: float
    bug_threshold_sharpe: float
    dislocation_labels: tuple[str, ...]
    mandatory_crash_month_required: bool

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
        if self.abandon_below_sharpe > self.support_min_sharpe:
            raise ConfigParseError("section 8's Sharpe abandon threshold exceeds its support threshold")
        if self.max_inversions < 0:
            raise ConfigParseError("the tolerated inversion count cannot be negative")
        if not re.fullmatch(r"[A-Z]{1,5}", self.equity_proxy_symbol):
            raise ConfigParseError(
                f"section 7 names {self.equity_proxy_symbol!r} as the secondary beta benchmark, "
                "which is not a ticker"
            )
        if not self.quote_normalisation_required:
            raise ConfigParseError(
                "section 2 no longer requires the quote convention to be normalised; that "
                "requirement is the reason this experiment can produce a correct answer"
            )
        if not self.spot_only:
            raise ConfigParseError("section 4 no longer defines the return as spot-only")
        if not self.interest_rate_diagnostic_required:
            raise ConfigParseError("section 4 no longer requires the interest-rate diagnostic")
        unknown = [label for label in self.dislocation_labels if label not in DISLOCATION_MONTHS]
        if unknown:
            raise ConfigParseError(
                f"section 9 names FX dislocations this module cannot date: {unknown}. "
                "Add them to DISLOCATION_MONTHS rather than dropping them."
            )
        pd.Timestamp(self.sample_start)  # raises if section 3's date is unparseable

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000.0

    @property
    def dislocation_months(self) -> dict[str, tuple[str, ...]]:
        """Section 9's labels expanded to the months the clustering test looks in."""
        return {label: DISLOCATION_MONTHS[label] for label in self.dislocation_labels}

    def describe(self) -> str:
        return (
            f"currency cross-sectional momentum [formation={self.formation_days}d, "
            f"skip={self.skip_days}d, top 1/{self.n_quantiles} of the {self.release_name} "
            f"daily USD rates, long-only, spot-only, "
            f"cost={self.cost_bps_per_side:g}bps/side] sha256={self.source_sha256[:12]}"
        )


# Q1-Q5 is written with a Unicode minus sign in this document; earlier ones used a
# hyphen or an en dash. Accepting all of them means a typographic change cannot silently
# turn a required clause into a missing one.
_DASH = r"[-‐‑‒–—−]"


def _phrase(text: str) -> str:
    """A prose assertion as a regex, tolerant of where the document wrapped its lines.

    Markdown hard-wraps at whatever column the author used, so a sentence quoted from
    the document may contain a newline anywhere a space appears. Matching literal
    spaces would make these assertions fail on reflow rather than on meaning, which is
    the opposite of what they are for. Regex metacharacters in the phrase are escaped;
    only the spaces become flexible.
    """
    return r"\s+".join(re.escape(word) for word in text.split())


def _parse_text(text: str, source_path: Path) -> Config005:
    s2, s3, s4, s5, s7, s8, s9 = (_section(text, n) for n in (2, 3, 4, 5, 7, 8, 9))

    configurations_tried = int(
        _require_unique(
            _phrase("**Configurations tried, cumulative:**") + r"\s*(\d+)", text, "the cumulative configuration counter"
        ).group(1)
    )
    # The header's argument for why blocked experiment 004 does not advance the counter.
    # It is the justification for the single most consequential number in the deflated
    # Sharpe, so its presence is asserted rather than assumed.
    _require_unique(
        _phrase("**Experiment 004 does not count, and here is why.**"), text, "the header's exclusion of experiment 004"
    )
    _require_unique(
        _phrase("PSR corrects for configurations *tried*"), text, "the header's definition of the counter"
    )

    # ---- section 2: the universe is a rule over a published release -------------------
    release_rule = _require_unique(
        _phrase("**Every daily USD exchange rate series published in the Federal Reserve")
        + r"\s+(H\.\d+)\s+"
        + _phrase("release and available via FRED that has continuous daily data over the full sample window.**"),
        s2,
        "the universe definition",
    )
    expected = _require_unique(
        _phrase("Expected yield: roughly") + r"\s*(\d+)\s*" + _DASH + r"\s*(\d+)\s*pairs",
        s2,
        "the expected universe size range",
    )
    _require_unique(
        _phrase("Series discontinued mid-sample are **excluded entirely**, not truncated"),
        s2,
        "section 2's exclusion rule for discontinued series",
    )
    _require_unique(_phrase("No substitutions, no additions"), s2, "section 2's no-substitution clause")
    # The clause the whole of trendbot.fx exists to satisfy.
    _require_unique(
        _phrase("**Quote convention must be normalised.**"), s2, "section 2's quote-normalisation requirement"
    )
    _require_unique(
        _phrase("converted to a common convention (foreign currency value expressed in USD) before ranking"),
        s2,
        "section 2's statement of which convention is common",
    )

    # ---- section 3: sample window -----------------------------------------------------
    sample_start = _require_unique(
        r"\*\*(\d{4}-\d{2}-\d{2})\s+" + _phrase("to present.** Fixed now."), s3, "the sample window start"
    ).group(1)

    # ---- section 4: returns and signal ------------------------------------------------
    _require_unique(
        _phrase("**Return definition:** log change in the USD value of the foreign currency"),
        s4,
        "the return definition",
    )
    spot_only = bool(re.search(_phrase("**Spot only.** No interest-rate component."), s4))
    if not spot_only:
        raise ConfigParseError("section 4 no longer declares the return spot-only")
    diagnostic = bool(
        re.search(
            _phrase("**Required diagnostic (not a configuration):**")
            + r".*?"
            + _phrase("short-term interest rate differentials from FRED"),
            s4,
            re.S,
        )
    )

    formula = _require_unique(
        r"momentum_i\(t\)\s*=\s*P_i\(t-(\d+)\)\s*/\s*P_i\(t-(\d+)\)\s*-\s*1", s4, "the momentum formula"
    )
    skip_days, formation_days = int(formula.group(1)), int(formula.group(2))
    quantile_word = _require_unique(
        r"Sort\s+into\s+\*\*(deciles|quintiles|quartiles|terciles)\*\*", s4, "the quantile cut"
    ).group(1)
    n_quantiles = {"terciles": 3, "quartiles": 4, "quintiles": 5, "deciles": 10}[quantile_word]
    _require_unique(_phrase("Top quintile: equal weight, long"), s4, "the top-bucket position")
    _require_unique(_phrase("All others: zero"), s4, "the zero-weight rule for the other buckets")
    long_only = bool(re.search(r"^Long-only\b", s4, re.MULTILINE))
    if not long_only:
        raise ConfigParseError("section 4 no longer declares the strategy long-only")

    # ---- section 5: execution ---------------------------------------------------------
    gross_cap = float(
        _require_unique(r"Gross\s+exposure\s*(\d+(?:\.\d+)?)\s+when\s+invested", s5, "gross exposure").group(1)
    )
    _require_unique(_phrase("No leverage, no volatility targeting"), s5, "the no-leverage clause")
    _require_unique(
        _phrase("Signal on close of bar t → position taken at bar t+1"), s5, "the execution lag"
    )
    rebalance = (
        _require_unique(r"Rebalance:\s*(.+?)\.\s*$", s5, "the rebalance schedule", re.MULTILINE).group(1).strip()
    )
    cost_bps = float(
        _require_unique(
            r"\*\*Cost\s+assumption:\s*(\d+(?:\.\d+)?)\s+bps\s+per\s+side\.\*\*", s5, "the cost assumption"
        ).group(1)
    )
    ladder = tuple(
        float(x)
        for x in re.findall(
            r"[\d.]+",
            _require_unique(r"Sensitivity\s+at\s*([\d/]+)\s*\.", s5, "the cost sensitivity ladder").group(1),
        )
    )
    benchmark_label = (
        _require_unique(
            r"Benchmark:\s*\*\*(.+?)\*\*", s5, "the benchmark definition", re.S
        ).group(1).replace("\n", " ").strip()
    )
    _require_unique(
        _phrase("Sharpe computed as excess of the T-bill rate for both strategy and benchmark"),
        s5,
        "section 5's excess-return convention",
    )

    # ---- section 7: the secondary beta benchmark --------------------------------------
    # Section 7.6 names it, so it is parsed rather than written here. The repository
    # invariant that forbids inlining a ticker outside a config parser is what makes
    # that mandatory rather than merely tidy.
    equity_proxy = _require_unique(
        _phrase("against the dollar factor (headline) and against") + r"\s+([A-Z]{1,5})\b",
        s7,
        "the secondary beta benchmark",
    ).group(1)
    _require_unique(
        _phrase("to check whether currency momentum is a disguised equity beta"),
        s7,
        "section 7's statement of what the secondary regression is for",
    )

    # ---- section 8: the decision rule -------------------------------------------------
    support_sharpe = float(
        _require_unique(
            _phrase("Net Sharpe (excess of T-bill,") + r"\s*\d+\s*" + _phrase("bps) exceeds") + r"\s*\*\*(\d+(?:\.\d+)?)\*\*",
            s8,
            "the support threshold Sharpe",
        ).group(1)
    )
    support_excess = float(
        _require_unique(r"by at least \*\*(\d+(?:\.\d+)?)\*\*", s8, "the excess-over-benchmark threshold").group(1)
    )
    max_inversions = {"zero": 0, "one": 1, "two": 2, "three": 3}[
        _require_unique(
            r"at\s+most\s+\*\*(zero|one|two|three)\*\*\s+adjacent\s+inversion",
            s8,
            "the tolerated inversion count",
        ).group(1)
    ]
    spread_t = float(
        _require_unique(
            r"Q1" + _DASH + r"Q5\s+spread\s+positive\s+at\s+\*\*t\s*>\s*(\d+(?:\.\d+)?)\*\*",
            s8,
            "the spread t-statistic support threshold",
        ).group(1)
    )
    alpha_t = float(
        _require_unique(
            _phrase("**Alpha to the dollar factor is positive with t >") + r"\s*(\d+(?:\.\d+)?)\.\*\*",
            s8,
            "the alpha t-statistic support threshold",
        ).group(1)
    )
    abandon = _require_unique(
        _phrase("**Abandon** if any of:") + r"\s*(.+?)\n\nAnything else", s8, "the abandon clauses", re.S
    ).group(1)
    abandon_sharpe = float(
        _require_unique(
            _phrase("net Sharpe below") + r"\s*\*\*(\d+(?:\.\d+)?)\*\*",
            abandon,
            "the Sharpe abandonment threshold",
        ).group(1)
    )
    abandon_spread_t = float(
        _require_unique(
            r"Q1" + _DASH + r"Q5\s+t\s+below\s+(\d+(?:\.\d+)?)",
            abandon,
            "the spread t-statistic abandonment threshold",
        ).group(1)
    )
    _require_unique(_phrase("fails to beat the benchmark at all"), abandon, "the benchmark abandon clause")
    _require_unique(_phrase("more than one inversion"), abandon, "the inversion-count abandon clause")
    _require_unique(_phrase("alpha to the dollar factor negative"), abandon, "the negative-alpha abandon clause")

    # ---- section 9: expectations of record --------------------------------------------
    exp = _require_unique(
        _phrase("Realistic net Sharpe: **") + r"(\d+(?:\.\d+)?)\s*" + _DASH + r"\s*(\d+(?:\.\d+)?)\*\*",
        s9,
        "the expected Sharpe range",
    )
    bug_threshold = float(
        _require_unique(r"Above\s+(\d+(?:\.\d+)?)\s+means\s+a\s+bug", s9, "the bug-threshold Sharpe").group(1)
    )
    # Section 9 explicitly declines to name a mandatory crash month for this experiment,
    # which is a difference from 002 and 003 that the protocol has to know about.
    no_mandatory = bool(
        re.search(
            _phrase("currency momentum has no single canonical crash date, so")
            + r"\s+.{0,4}9\s+"
            + _phrase("imposes no specific month requirement here"),
            s9,
            re.S,
        )
    )
    if not no_mandatory:
        raise ConfigParseError(
            "section 9 no longer states that no specific crash month is required; the "
            "protocol would need a mandatory-month gate it does not have"
        )
    _require_unique(
        _phrase("Absence of any clustering is a warning sign about the implementation"),
        s9,
        "section 9's clustering warning",
    )
    labels = tuple(
        part.strip()
        for part in _require_unique(
            r"\(2008 Q4,([^)]*)\)", s9, "the list of known FX dislocations"
        ).group(0)[1:-1].split(",")
    )

    committed = _require_unique(r"\*\*Committed on:\*\*\s*(\d{4}-\d{2}-\d{2})", text, "the commit date").group(1)
    signature = _require_unique(r"Signed:\s*\*\*(.+?)\*\*\s+Date:\s*\*\*(.+?)\*\*", text, "the signature")

    return Config005(
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        committed_on=committed,
        signed_by=signature.group(1).strip(),
        signed_date=signature.group(2).strip(),
        configurations_tried=configurations_tried,
        release_name=release_rule.group(1),
        expected_universe_low=int(expected.group(1)),
        expected_universe_high=int(expected.group(2)),
        quote_normalisation_required=True,
        sample_start=sample_start,
        spot_only=spot_only,
        interest_rate_diagnostic_required=diagnostic,
        formation_days=formation_days,
        skip_days=skip_days,
        n_quantiles=n_quantiles,
        long_only=long_only,
        gross_exposure_cap=gross_cap,
        rebalance=rebalance,
        cost_bps_per_side=cost_bps,
        cost_sensitivity_bps=ladder,
        benchmark_label=benchmark_label,
        equity_proxy_symbol=equity_proxy,
        support_min_sharpe=support_sharpe,
        support_min_sharpe_excess_over_benchmark=support_excess,
        max_inversions=max_inversions,
        min_spread_t_stat=spread_t,
        min_alpha_t_stat=alpha_t,
        abandon_below_sharpe=abandon_sharpe,
        abandon_below_spread_t_stat=abandon_spread_t,
        expected_sharpe_low=float(exp.group(1)),
        expected_sharpe_high=float(exp.group(2)),
        bug_threshold_sharpe=bug_threshold,
        dislocation_labels=labels,
        mandatory_crash_month_required=False,
    )


_CACHE: dict[Path, Config005] = {}


def load_config_005(path: Path | str | None = None, *, use_cache: bool = True) -> Config005:
    """Parse PREREG_005.md into a frozen :class:`Config005`."""
    resolved = Path(path).resolve() if path is not None else find_preregistration_005()
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
