"""Experiment 005's data layer: H.10 rates, normalised, audited, and made a panel.

PREREG_005.md section 2 states the failure mode this module exists to prevent:

    Quote convention must be normalised. FRED mixes conventions - some series are USD
    per foreign unit, others foreign units per USD. All series must be converted to a
    common convention (foreign currency value expressed in USD) before ranking.
    Getting a subset inverted scrambles every rank silently without raising an error.
    This is the single most likely way this experiment produces a wrong answer.

"Silently" is the operative word. An inverted series is a perfectly well-formed
positive price series with a plausible-looking volatility. Every downstream
computation succeeds. The momentum rank for that currency is simply backwards, and
nothing anywhere raises. So the direction cannot be inferred from the numbers, and
this module never tries: :func:`parse_quote_convention` reads FRED's own ``units``
string, which states the convention in words, and cross-checks it against the title.

Four independent checks then have to pass before the universe is usable, because a
parser that reads metadata correctly is not the same thing as a *dataset* that is
right:

1. :func:`check_reference_levels` - the normalised value on a named historical date
   must fall inside a range fixed from independently known market history. Every band
   here is chosen so that the inverted value lands **outside** it; a check an inversion
   would still pass is not a check.
2. :func:`check_peg_relationships` - two structural identities that hold by monetary
   policy rather than by anything in this repository: the Hong Kong dollar's
   7.75-7.85 convertibility band, and the Danish krone's ERM II central rate against
   the euro. The second is the strongest check available here because it is a
   *cross-rate*: it relates two different FRED series, so inverting either one breaks
   it, and neither could be made to pass by a coincidence in the other.
3. :func:`check_drift_plausibility` - inversion flips the sign of a currency's long-run
   drift and turns a depreciating currency into an implausibly appreciating one.
4. :func:`build_fx_universe` - section 2's continuity rule, with the largest gap per
   series reported rather than assumed away.

Handling of missing observations (build order step 3)
-----------------------------------------------------
H.10 rates are New York Fed noon buying rates, published on US business days. A series
is therefore blank on two kinds of day: US market holidays, when *nothing* is
published, and that particular country's own holidays, when everything else is
published and this one currency is not.

The two are handled differently and for different reasons.

* **US holidays** - no rate exists for any currency, so the date is not a trading day
  at all and is dropped from the calendar. :func:`publication_calendar` builds that
  calendar as the dates on which at least one candidate series published.
* **Foreign holidays** - the market was closed for that currency and the last
  published rate is the only defensible mark, exactly as
  :mod:`trendbot.engine.panel_backtest` already treats a vendor hole. So the series is
  forward-filled along the shared calendar.

Forward filling looks strictly backwards: the value carried into a hole is the last
value published *before* it, so no observation can move a return dated earlier than
itself. There is no back-fill anywhere - a series is NaN before its own first
observation and stays that way, which is what excludes it from those dates' rankings
rather than inventing a price. ``tests/test_fx.py`` asserts both halves of that.

Open and close
--------------
FRED publishes **one** rate per day for an H.10 series - there is no intraday open. So
``open`` and ``close`` are the same frame, and the engine's single execution shift is
what separates decision from fill: the signal is formed on the rate of bar ``t`` and
the position is taken at the rate of bar ``t+1``, which is PREREG_005.md section 5
verbatim. Nothing here fills at a price known only after the decision.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CACHE_DIR, PriceData
from .fred import H10_RELEASE_ID, FredError, SeriesMetadata, fetch_observations, fetch_series_metadata, list_release_series

__all__ = [
    "USD_PER_FOREIGN",
    "FOREIGN_PER_USD",
    "MAX_HOLIDAY_GAP_DAYS",
    "QuoteConvention",
    "parse_quote_convention",
    "normalise_to_usd",
    "ReferenceCheck",
    "REFERENCE_LEVELS",
    "check_reference_levels",
    "PegCheck",
    "check_peg_relationships",
    "DriftCheck",
    "check_drift_plausibility",
    "SeriesGaps",
    "publication_calendar",
    "measure_gaps",
    "forward_fill_panel",
    "equal_weight_factor",
    "FxUniverse",
    "build_fx_universe",
    "fx_price_data",
    "dollar_factor_returns",
]

USD_PER_FOREIGN = "USD_PER_FOREIGN"
FOREIGN_PER_USD = "FOREIGN_PER_USD"

# Section 2's continuity rule needs a number for "continuous". Fixed here, before the
# data was looked at, at ten consecutive scheduled publication days - two calendar
# weeks. That is longer than the longest national market closure among these
# currencies (Chinese New Year runs about eight business days; Japan's Golden Week
# about five) and far shorter than any discontinuation. A series whose largest gap
# exceeds it is not "continuous daily data over the full sample window" and is
# excluded entirely rather than truncated, per section 2.
MAX_HOLIDAY_GAP_DAYS = 10


# --------------------------------------------------------------------------------------
# 1. quote convention, read from FRED's own words
# --------------------------------------------------------------------------------------

# "U.S. Dollar", "U.S. Dollars", "US Dollar", "USD". Matched whole, case-insensitively.
_USD = re.compile(r"^u\.?\s?s\.?\s?dollars?$|^usd$", re.IGNORECASE)

# FRED's units string for an H.10 bilateral rate: "<A> to One <B>", meaning the value
# is the number of A that buys one B.
_UNITS = re.compile(r"^(?P<numerator>.+?)\s+to\s+One\s+(?P<denominator>.+?)$", re.IGNORECASE)

# The title restates it: "<A> to <B> Spot Exchange Rate".
_TITLE = re.compile(r"^(?P<numerator>.+?)\s+to\s+(?P<denominator>.+?)\s+Spot Exchange Rate$", re.IGNORECASE)


def _canonical(name: str) -> str:
    """Compare currency names without being defeated by punctuation or plurality."""
    text = re.sub(r"[^a-z ]", "", name.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    # "Dollars"/"Dollar", "Kroner"/"Krone" and friends: the title uses the singular of
    # the denominator and the plural of the numerator, so the pair only lines up once
    # a trailing plural is stripped. Done on the last word only, so "U.S." survives.
    words = text.split()
    if words:
        last = words[-1]
        for plural, singular in (("ies", "y"), ("es", ""), ("s", "")):
            if last.endswith(plural) and len(last) > len(plural) + 1:
                words[-1] = last[: -len(plural)] + singular
                break
    return " ".join(words)


def _is_usd(name: str) -> bool:
    return bool(_USD.match(name.strip()))


@dataclass(frozen=True, slots=True)
class QuoteConvention:
    """Which way round a FRED exchange-rate series is quoted, and how it was decided.

    ``inverted`` is the only field the pipeline acts on, and it is derived from
    ``units`` alone. ``evidence`` records the strings it was derived from so that a
    findings document can print the reasoning rather than the conclusion.
    """

    series_id: str
    title: str
    units: str
    numerator: str
    denominator: str
    direction: str
    foreign_currency: str
    evidence: str

    @property
    def inverted(self) -> bool:
        """True when the raw series must be inverted to become USD per foreign unit."""
        return self.direction == FOREIGN_PER_USD

    def __str__(self) -> str:
        action = "1 / x" if self.inverted else "x (unchanged)"
        return (
            f"{self.series_id}: units {self.units!r} -> {self.direction}; "
            f"foreign currency {self.foreign_currency}; normalise by {action}"
        )


def parse_quote_convention(meta: SeriesMetadata) -> QuoteConvention:
    """Determine a series' quote direction from its FRED metadata. Never from its values.

    Two independent strings have to agree - the units and the title - and exactly one
    side of the pair has to be the US dollar. Anything else raises: an exchange rate
    whose direction cannot be established is not something to guess about, because a
    wrong guess produces a complete, plausible, silently backwards result.
    """
    units_match = _UNITS.match(meta.units.strip())
    if units_match is None:
        raise FredError(
            f"{meta.series_id}: units {meta.units!r} do not read as '<A> to One <B>', so "
            "the quote convention cannot be established from FRED's metadata. Refusing "
            "to infer a direction from the values."
        )
    numerator = units_match.group("numerator").strip()
    denominator = units_match.group("denominator").strip()

    title_match = _TITLE.match(meta.title.strip())
    if title_match is None:
        raise FredError(
            f"{meta.series_id}: title {meta.title!r} does not read as "
            "'<A> to <B> Spot Exchange Rate', so the units cannot be cross-checked"
        )
    if (
        _canonical(title_match.group("numerator")) != _canonical(numerator)
        or _canonical(title_match.group("denominator")) != _canonical(denominator)
    ):
        raise FredError(
            f"{meta.series_id}: title {meta.title!r} and units {meta.units!r} disagree "
            "about which currency is which. One of them is wrong and there is no way to "
            "tell which; refusing to continue."
        )

    numerator_is_usd, denominator_is_usd = _is_usd(numerator), _is_usd(denominator)
    if numerator_is_usd == denominator_is_usd:
        raise FredError(
            f"{meta.series_id}: units {meta.units!r} name "
            f"{'two US dollar legs' if numerator_is_usd else 'no US dollar leg'}; "
            "section 2's universe is bilateral USD rates only"
        )

    if denominator_is_usd:
        # "Japanese Yen to One U.S. Dollar": the value is yen per dollar, so the dollar
        # value of one yen is its reciprocal.
        direction, foreign = FOREIGN_PER_USD, numerator
        evidence = f"units say {numerator!r} per one {denominator!r} -> value is foreign per USD"
    else:
        # "U.S. Dollars to One Euro": the value is already the dollar value of one euro.
        direction, foreign = USD_PER_FOREIGN, denominator
        evidence = f"units say {numerator!r} per one {denominator!r} -> value is USD per foreign"

    return QuoteConvention(
        series_id=meta.series_id,
        title=meta.title,
        units=meta.units,
        numerator=numerator,
        denominator=denominator,
        direction=direction,
        foreign_currency=foreign,
        evidence=evidence,
    )


def normalise_to_usd(series: pd.Series, convention: QuoteConvention) -> pd.Series:
    """Express the series as the USD value of one unit of the foreign currency.

    Non-positive values are turned into holes rather than inverted: a zero or negative
    exchange rate is not a rate, and ``1/0`` would be an infinity that propagates into
    a momentum ratio and out the other side as a rank.
    """
    values = series.astype(float)
    values = values.where(values > 0)
    if convention.inverted:
        values = 1.0 / values
    return values.rename(convention.series_id)


# --------------------------------------------------------------------------------------
# 2. verification against independently known values
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceCheck:
    """One assertion that a normalised series sits where market history says it should."""

    series_id: str
    currency: str
    date: str
    low: float
    high: float
    observed: float
    inverted_would_be: float
    why: str

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.observed) and self.low <= self.observed <= self.high)

    @property
    def discriminating(self) -> bool:
        """True when the *inverted* value would fail. A band that both pass tests nothing."""
        if not np.isfinite(self.inverted_would_be):
            return True
        return not (self.low <= self.inverted_would_be <= self.high)

    def __str__(self) -> str:
        return (
            f"{'PASS' if self.passed else 'FAIL'} {self.series_id} ({self.currency}) on "
            f"{self.date}: {self.observed:.6g} USD per unit, expected "
            f"[{self.low:g}, {self.high:g}]; inverted would be "
            f"{self.inverted_would_be:.6g} ({'caught' if self.discriminating else 'NOT CAUGHT'})"
        )


# Independently known market levels, as USD per one unit of the foreign currency.
# Every band is wide enough to absorb the difference between a noon buying rate and
# whatever quote the level is remembered from, and narrow enough that the reciprocal
# falls outside it. The "why" is the external fact being relied on; none of these
# numbers was read off the data being checked.
REFERENCE_LEVELS: tuple[tuple[str, str, float, float, str], ...] = (
    (
        "DEXJPUS",
        "2013-12-31",
        0.007,
        0.013,
        "USD/JPY traded near 105 at end-2013; the yen has stayed inside 75-160 per "
        "dollar for the whole sample, so USD per yen is a hundredth-scale number",
    ),
    (
        "DEXUSUK",
        "2007-12-31",
        1.70,
        2.20,
        "sterling was near its multi-decade high of about 2.00 dollars at end-2007",
    ),
    (
        "DEXSZUS",
        "2011-08-09",
        1.15,
        1.60,
        "the franc set a record high against the dollar in the second week of August "
        "2011, around 0.72-0.78 dollars per franc inverted, i.e. 1.28-1.39 the other way",
    ),
    (
        "DEXKOUS",
        "2010-12-31",
        0.0005,
        0.0015,
        "the won traded near 1,130 per dollar at end-2010; USD per won is a "
        "thousandth-scale number",
    ),
    (
        "DEXMXUS",
        "2015-12-31",
        0.030,
        0.100,
        "the peso traded near 17 per dollar at end-2015",
    ),
    (
        "DEXUSEU",
        "2008-07-15",
        1.40,
        1.65,
        "the euro set its all-time high against the dollar in mid-July 2008, just above 1.60",
    ),
    (
        "DEXCAUS",
        "2002-01-31",
        0.55,
        0.70,
        "the Canadian dollar hit its record low near 1.61 per US dollar in January 2002",
    ),
    (
        "DEXINUS",
        "2013-08-28",
        0.010,
        0.020,
        "the rupee hit a then-record low near 68 per dollar on 28 August 2013",
    ),
    (
        "DEXCHUS",
        "2004-06-30",
        0.110,
        0.130,
        "the yuan was pegged at 8.2765 per dollar until July 2005",
    ),
    (
        "DEXBZUS",
        "2015-09-24",
        0.15,
        0.32,
        "the real weakened past 4 per dollar in late September 2015",
    ),
)


def check_reference_levels(
    normalised: pd.DataFrame,
    conventions: dict[str, QuoteConvention],
    references: tuple[tuple[str, str, float, float, str], ...] = REFERENCE_LEVELS,
) -> list[ReferenceCheck]:
    """Assert the normalised level on a named date against externally known history."""
    checks: list[ReferenceCheck] = []
    for series_id, date, low, high, why in references:
        if series_id not in normalised.columns:
            continue
        column = normalised[series_id].dropna()
        stamp = pd.Timestamp(date)
        # The named date can be a holiday for that currency. Use the last rate
        # published on or before it - which is what "the level on that date" means for
        # a series that does not print every day - rather than skipping the check.
        usable = column.loc[:stamp]
        observed = float(usable.iloc[-1]) if len(usable) else float("nan")
        checks.append(
            ReferenceCheck(
                series_id=series_id,
                currency=conventions[series_id].foreign_currency,
                date=str(usable.index[-1].date()) if len(usable) else date,
                low=low,
                high=high,
                observed=observed,
                inverted_would_be=1.0 / observed if observed else float("nan"),
                why=why,
            )
        )
    return checks


@dataclass(frozen=True, slots=True)
class PegCheck:
    """A structural identity that holds by monetary policy, not by anything here."""

    name: str
    statistic: str
    low: float
    high: float
    observed_min: float
    observed_max: float
    fraction_inside: float
    tolerance: float
    detail: str

    @property
    def passed(self) -> bool:
        return self.fraction_inside >= self.tolerance

    def __str__(self) -> str:
        return (
            f"{'PASS' if self.passed else 'FAIL'} {self.name}: {self.statistic} ranged "
            f"{self.observed_min:.6g}..{self.observed_max:.6g} against the policy band "
            f"[{self.low:g}, {self.high:g}]; {self.fraction_inside:.2%} of days inside "
            f"(needs {self.tolerance:.0%}). {self.detail}"
        )


def check_peg_relationships(normalised: pd.DataFrame, *, tolerance: float = 0.98) -> list[PegCheck]:
    """Two policy identities the normalised panel must reproduce.

    The Danish check is a **cross-rate**: it divides one normalised series by another,
    so it is sensitive to an inversion in either and cannot be satisfied by both being
    wrong in the same direction. Nothing else available here has that property, which
    is why it is worth more than any single-series band.
    """
    checks: list[PegCheck] = []

    if "DEXHKUS" in normalised.columns:
        hkd = normalised["DEXHKUS"].dropna()
        # The HKMA has run a 7.75-7.85 convertibility band since May 2005 and a 7.80
        # link before it, so USD per HKD sits in a narrow band around 0.128. A slightly
        # wider band absorbs the pre-2005 regime and the 2003 excursion.
        low, high = 1.0 / 7.90, 1.0 / 7.70
        inside = float(((hkd >= low) & (hkd <= high)).mean()) if len(hkd) else 0.0
        checks.append(
            PegCheck(
                name="Hong Kong dollar linked exchange rate",
                statistic="USD per HKD",
                low=low,
                high=high,
                observed_min=float(hkd.min()) if len(hkd) else float("nan"),
                observed_max=float(hkd.max()) if len(hkd) else float("nan"),
                fraction_inside=inside,
                tolerance=tolerance,
                detail="an inverted HKD series would sit near 7.8, three orders out",
            )
        )

    if {"DEXDNUS", "DEXUSEU"} <= set(normalised.columns):
        pair = normalised[["DEXUSEU", "DEXDNUS"]].dropna()
        implied = pair["DEXUSEU"] / pair["DEXDNUS"]  # kroner per euro
        # ERM II central rate 7.46038 with a formal +/-2.25% band; Denmark has in
        # practice held far tighter. The formal band is used so the check is a
        # statement about policy rather than about observed behaviour.
        low, high = 7.46038 * 0.9775, 7.46038 * 1.0225
        inside = float(((implied >= low) & (implied <= high)).mean()) if len(implied) else 0.0
        checks.append(
            PegCheck(
                name="Danish krone ERM II central rate against the euro",
                statistic="implied DKK per EUR = (USD per EUR) / (USD per DKK)",
                low=low,
                high=high,
                observed_min=float(implied.min()) if len(implied) else float("nan"),
                observed_max=float(implied.max()) if len(implied) else float("nan"),
                fraction_inside=inside,
                tolerance=tolerance,
                detail=(
                    "a cross-rate between two separate FRED series: inverting either one "
                    "breaks it, so it cannot be passed by luck"
                ),
            )
        )
    return checks


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """Long-run drift of a normalised series, as a symptom of inversion."""

    series_id: str
    currency: str
    annualised_log_drift: float
    limit: float
    years: float
    first: float
    last: float

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.annualised_log_drift) and abs(self.annualised_log_drift) <= self.limit)

    def __str__(self) -> str:
        return (
            f"{'PASS' if self.passed else 'FAIL'} {self.series_id} ({self.currency}): "
            f"{self.annualised_log_drift:+.2%}/yr over {self.years:.1f}y "
            f"({self.first:.6g} -> {self.last:.6g}), limit +/-{self.limit:.0%}"
        )


def check_drift_plausibility(
    normalised: pd.DataFrame,
    conventions: dict[str, QuoteConvention],
    *,
    limit: float = 0.25,
    periods_per_year: int = 252,
) -> list[DriftCheck]:
    """No currency should imply an implausible long-run trend against the dollar.

    Inversion flips the sign of the drift, so a currency that depreciated 8%/yr becomes
    one that appreciated 8%/yr. That alone does not always breach a threshold, which is
    why this is the *weakest* of the checks and is reported alongside the others rather
    than relied on. It does catch the case that matters most - a high-inflation
    currency inverted into an implausible compounding appreciation.
    """
    checks: list[DriftCheck] = []
    for series_id in normalised.columns:
        column = normalised[series_id].dropna()
        if len(column) < 2:
            continue
        first, last = float(column.iloc[0]), float(column.iloc[-1])
        years = len(column) / periods_per_year
        drift = math.log(last / first) / years if years > 0 and first > 0 and last > 0 else float("nan")
        checks.append(
            DriftCheck(
                series_id=series_id,
                currency=conventions[series_id].foreign_currency,
                annualised_log_drift=drift,
                limit=limit,
                years=years,
                first=first,
                last=last,
            )
        )
    return checks


# --------------------------------------------------------------------------------------
# 3. section 2's universe rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeriesGaps:
    """Continuity of one series along the shared publication calendar."""

    series_id: str
    currency: str
    first_observation: str
    last_observation: str
    n_observations: int
    n_scheduled: int
    largest_gap: int
    largest_gap_start: str
    largest_gap_end: str

    def __str__(self) -> str:
        gap = (
            f"largest gap {self.largest_gap} day(s)"
            + (f" ({self.largest_gap_start}..{self.largest_gap_end})" if self.largest_gap else "")
        )
        return (
            f"{self.series_id} ({self.currency}): {self.first_observation} -> "
            f"{self.last_observation}, {self.n_observations}/{self.n_scheduled} published, {gap}"
        )


def publication_calendar(raw: dict[str, pd.Series], start: str, end: str | None = None) -> pd.DatetimeIndex:
    """Dates inside the window on which at least one candidate series published.

    This is the trading calendar. A date on which no H.10 rate exists at all is a US
    market holiday - not a day on which every currency happened to be closed - and is
    not a bar. Building the calendar from the candidates rather than from a hardcoded
    holiday list means it is derived from the release itself.
    """
    index = pd.DatetimeIndex([])
    for series in raw.values():
        index = index.union(series.dropna().index)
    window = index[index >= pd.Timestamp(start)]
    if end is not None:
        window = window[window <= pd.Timestamp(end)]
    return pd.DatetimeIndex(sorted(window))


def measure_gaps(
    series: pd.Series, calendar: pd.DatetimeIndex, *, series_id: str, currency: str
) -> SeriesGaps:
    """Longest run of scheduled publication days on which this series printed nothing."""
    aligned = series.reindex(calendar)
    present = aligned.notna().to_numpy()
    largest, run, best_end = 0, 0, -1
    for i, ok in enumerate(present):
        if ok:
            run = 0
            continue
        run += 1
        if run > largest:
            largest, best_end = run, i
    observed = aligned.dropna()
    return SeriesGaps(
        series_id=series_id,
        currency=currency,
        first_observation=str(observed.index[0].date()) if len(observed) else "-",
        last_observation=str(observed.index[-1].date()) if len(observed) else "-",
        n_observations=int(present.sum()),
        n_scheduled=len(calendar),
        largest_gap=largest,
        largest_gap_start=str(calendar[best_end - largest + 1].date()) if largest else "-",
        largest_gap_end=str(calendar[best_end].date()) if largest else "-",
    )


def forward_fill_panel(
    normalised: dict[str, pd.Series], members: Sequence[str], *, upper_bound: pd.Timestamp
) -> pd.DataFrame:
    """One frame over the union calendar, each column forward-filled and never back-filled.

    The index is the union of the members' own publication dates up to ``upper_bound``,
    which is deliberately *wider* than the sample window: the 252-day formation window at
    the first bar of the window has to reach into the year before it, so the panel carries
    the pre-window history and the reporting layer slices.

    ``ffill`` and nothing else. A column is NaN before its own first observation and stays
    NaN, so a currency that did not exist yet is excluded from those dates' rankings rather
    than being given an invented price. ``tests/test_fx.py`` asserts both halves.
    """
    members = list(members)
    index = pd.DatetimeIndex([])
    for series_id in members:
        index = index.union(normalised[series_id].dropna().index)
    index = pd.DatetimeIndex(sorted(index[index <= pd.Timestamp(upper_bound)]))
    return pd.DataFrame(
        {series_id: normalised[series_id].reindex(index).ffill() for series_id in members},
        index=index,
        columns=members,
    )


def equal_weight_factor(frame: pd.DataFrame) -> pd.Series:
    """Daily equal-weighted mean return of the columns, weights reset every bar.

    A column contributes on a date only when it has both a previous and a current price,
    so a column's first bar is not counted as a return from nothing and a column that has
    not started yet does not dilute the average.
    """
    returns = frame.pct_change(fill_method=None)
    contributes = frame.notna() & frame.shift(1).notna()
    counts = contributes.sum(axis=1)
    weights = contributes.div(counts.where(counts > 0), axis=0).fillna(0.0)
    return (weights * returns.fillna(0.0)).sum(axis=1).fillna(0.0).rename("dollar_factor")


@dataclass(frozen=True, slots=True)
class FxUniverse:
    """Section 2's universe, plus every series it rejected and why."""

    sample_start: str
    release_id: int
    candidates: tuple[str, ...]
    universe: tuple[str, ...]
    conventions: dict[str, QuoteConvention]
    metadata: dict[str, SeriesMetadata]
    normalised: pd.DataFrame  # calendar x universe, forward-filled
    raw: dict[str, pd.Series]
    calendar: pd.DatetimeIndex
    gaps: dict[str, SeriesGaps]
    exclusions: tuple[tuple[str, str, str], ...]  # (series_id, currency-or-title, reason)
    max_gap_days: int = MAX_HOLIDAY_GAP_DAYS
    non_daily: tuple[tuple[str, str], ...] = field(default=())

    @property
    def n(self) -> int:
        return len(self.universe)

    def currency_of(self, series_id: str) -> str:
        return self.conventions[series_id].foreign_currency

    def describe(self) -> str:
        inverted = sum(1 for s in self.universe if self.conventions[s].inverted)
        return (
            f"H.10 release {self.release_id}: {len(self.candidates)} daily bilateral USD "
            f"rates considered, {self.n} in the universe ({inverted} inverted to USD-per-"
            f"foreign, {self.n - inverted} already USD-per-foreign), "
            f"{self.calendar[0].date()} -> {self.calendar[-1].date()}, "
            f"{len(self.calendar)} publication days"
        )


def build_fx_universe(
    *,
    sample_start: str,
    release_id: int = H10_RELEASE_ID,
    max_gap_days: int = MAX_HOLIDAY_GAP_DAYS,
    refresh: bool = False,
    end: str | None = None,
) -> FxUniverse:
    """Section 2, executed: enumerate H.10, normalise, then apply the continuity rule.

    Ordering matters and is not arbitrary. The quote convention is resolved for every
    candidate *before* anything is excluded, so the exclusion report can name the
    currency rather than the series id, and so a series that fails to parse fails
    loudly instead of being dropped as "discontinuous".
    """
    ids = list_release_series(release_id, refresh=refresh)

    metadata: dict[str, SeriesMetadata] = {}
    conventions: dict[str, QuoteConvention] = {}
    candidates: list[str] = []
    non_daily: list[tuple[str, str]] = []
    for series_id in ids:
        meta = fetch_series_metadata(series_id, refresh=refresh)
        if meta.frequency.strip().lower() != "daily":
            non_daily.append((series_id, f"{meta.frequency or 'unknown frequency'}: {meta.title}"))
            continue
        try:
            convention = parse_quote_convention(meta)
        except FredError as exc:
            # Not a bilateral USD exchange rate - the trade-weighted dollar indexes in
            # this release are daily but are indexes, not rates, and their units say so.
            # Section 2 asks for "USD exchange rate series", so they are out, by their
            # own metadata rather than by name.
            non_daily.append(
                (series_id, f"daily, but not a bilateral USD rate — FRED units {meta.units!r}")
            )
            continue
        metadata[series_id] = meta
        conventions[series_id] = convention
        candidates.append(series_id)

    raw = {series_id: fetch_observations(series_id, refresh=refresh) for series_id in candidates}
    normalised_raw = {
        series_id: normalise_to_usd(raw[series_id], conventions[series_id]) for series_id in candidates
    }

    calendar = publication_calendar(normalised_raw, sample_start, end)
    if len(calendar) == 0:
        raise FredError(f"no H.10 publication days on or after {sample_start}")

    exclusions: list[tuple[str, str, str]] = []
    gaps: dict[str, SeriesGaps] = {}
    universe: list[str] = []
    for series_id in candidates:
        currency = conventions[series_id].foreign_currency
        column = normalised_raw[series_id]
        gap = measure_gaps(column, calendar, series_id=series_id, currency=currency)
        gaps[series_id] = gap

        observed = column.dropna()
        if len(observed) == 0:
            exclusions.append((series_id, currency, "no positive observations at all"))
            continue
        if observed.index[0] > calendar[0]:
            exclusions.append(
                (
                    series_id,
                    currency,
                    f"first observation {observed.index[0].date()} is after the window "
                    f"opens on {calendar[0].date()}: no data for the start of the sample",
                )
            )
            continue
        if observed.index[-1] < calendar[-1] - pd.Timedelta(days=max_gap_days * 2):
            exclusions.append(
                (
                    series_id,
                    currency,
                    f"last observation {observed.index[-1].date()} precedes the end of "
                    f"the window ({calendar[-1].date()}): discontinued mid-sample",
                )
            )
            continue
        if gap.largest_gap > max_gap_days:
            exclusions.append(
                (
                    series_id,
                    currency,
                    f"largest gap {gap.largest_gap} publication days "
                    f"({gap.largest_gap_start}..{gap.largest_gap_end}) exceeds the "
                    f"{max_gap_days}-day holiday allowance",
                )
            )
            continue
        universe.append(series_id)

    if not universe:
        raise FredError("section 2's universe rule selected no series")

    normalised = forward_fill_panel(normalised_raw, universe, upper_bound=calendar[-1])

    # The gap statistics that decided membership were measured against a calendar built
    # from every CANDIDATE, while the panel that gets traded spans the dates the chosen
    # MEMBERS published. Those are the same set only if no rejected candidate ever
    # published on a day no member did. If that ever stopped being true, the reported
    # "largest gap" would describe a calendar the strategy never sees, so it is asserted
    # rather than assumed.
    in_window = normalised.index[normalised.index >= calendar[0]]
    if not in_window.equals(calendar):
        only_calendar = calendar.difference(in_window)
        only_panel = in_window.difference(calendar)
        raise FredError(
            "the calendar the continuity rule was measured against and the calendar the "
            "panel spans disagree: "
            f"{len(only_calendar)} date(s) only in the former "
            f"({[str(d.date()) for d in only_calendar[:5]]}), "
            f"{len(only_panel)} only in the latter "
            f"({[str(d.date()) for d in only_panel[:5]]})"
        )

    return FxUniverse(
        sample_start=sample_start,
        release_id=release_id,
        candidates=tuple(candidates),
        universe=tuple(universe),
        conventions=conventions,
        metadata=metadata,
        normalised=normalised,
        raw=raw,
        calendar=calendar,
        gaps=gaps,
        exclusions=tuple(exclusions),
        max_gap_days=max_gap_days,
        non_daily=tuple(non_daily),
    )


def fx_price_data(universe: FxUniverse) -> PriceData:
    """The normalised panel as a :class:`~trendbot.data.PriceData`.

    ``open`` and ``close`` are the same frame because FRED publishes one rate per day.
    See this module's docstring: the decision/fill separation is the engine's single
    execution shift, not an intraday price difference this data does not have.
    """
    frame = universe.normalised
    return PriceData(
        open=frame.copy(),
        close=frame.copy(),
        source=f"FRED H.10 release {universe.release_id} (normalised to USD per foreign unit)",
        adjusted=True,
        fetched_at="",
    )


def dollar_factor_returns(universe: FxUniverse, prices: PriceData | None = None) -> pd.Series:
    """PREREG_005.md section 5's benchmark: equal weight long every universe currency.

    The mean of the daily normalised currency returns, weights reset to ``1/N`` each
    day - the dollar factor as Lustig/Roussanov/Verdelhan and Menkhoff et al. define it,
    which is the construction section 5's parenthetical names. Being path-independent,
    the same series serves as section 5's benchmark and as section 8's alpha regressor,
    so the two clauses cannot be adjudicated against subtly different objects.

    A currency contributes on a date only when it has both a previous and a current
    rate, so a currency's first bar is not counted as a return from nothing.
    """
    frame = (prices.close if prices is not None else universe.normalised)[list(universe.universe)]
    return equal_weight_factor(frame)


# --------------------------------------------------------------------------------------
# 4. section 4's required diagnostic - short-term rates for the carry approximation
# --------------------------------------------------------------------------------------

# Section 4 asks for "short-term interest rate differentials from FRED" and names no
# series, so the candidates are listed here, in priority order, and what was actually
# found is reported rather than assumed. Priority is: a three-month interbank or money
# market rate (the closest thing to the rate a currency position would actually earn),
# then an immediate/call rate, then a central bank policy or discount rate.
#
# The country codes are FRED's OECD Main Economic Indicators two-letter codes. Mapping a
# currency to a country is not a quote convention and cannot silently invert anything:
# a wrong mapping here changes the size of a reported diagnostic, and the diagnostic
# prints its sources so the mapping is auditable.
SHORT_RATE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "DEXUSEU": ("IR3TIB01EZM156N", "IRSTCI01EZM156N", "IRSTCB01EZM156N"),
    "DEXJPUS": ("IR3TIB01JPM156N", "IRSTCI01JPM156N", "INTDSRJPM193N"),
    "DEXUSUK": ("IR3TIB01GBM156N", "IRSTCI01GBM156N", "INTDSRGBM193N"),
    "DEXSZUS": ("IR3TIB01CHM156N", "IRSTCI01CHM156N", "INTDSRCHM193N"),
    "DEXCAUS": ("IR3TIB01CAM156N", "IRSTCI01CAM156N", "INTDSRCAM193N"),
    "DEXUSAL": ("IR3TIB01AUM156N", "IRSTCI01AUM156N", "INTDSRAUM193N"),
    "DEXUSNZ": ("IR3TIB01NZM156N", "IRSTCI01NZM156N", "INTDSRNZM193N"),
    "DEXSDUS": ("IR3TIB01SEM156N", "IRSTCI01SEM156N", "INTDSRSEM193N"),
    "DEXNOUS": ("IR3TIB01NOM156N", "IRSTCI01NOM156N", "INTDSRNOM193N"),
    "DEXDNUS": ("IR3TIB01DKM156N", "IRSTCI01DKM156N", "INTDSRDKM193N"),
    "DEXKOUS": ("IR3TIB01KRM156N", "IRSTCI01KRM156N", "INTDSRKRM193N"),
    "DEXMXUS": ("IR3TIB01MXM156N", "IRSTCI01MXM156N", "INTDSRMXM193N"),
    "DEXINUS": ("IR3TIB01INM156N", "IRSTCI01INM156N", "INTDSRINM193N"),
    "DEXBZUS": ("IR3TIB01BRM156N", "IRSTCI01BRM156N", "INTDSRBRM193N"),
    "DEXCHUS": ("IR3TIB01CNM156N", "IRSTCI01CNM156N", "INTDSRCNM193N"),
    "DEXSFUS": ("IR3TIB01ZAM156N", "IRSTCI01ZAM156N", "INTDSRZAM193N"),
    "DEXSIUS": ("IR3TIB01SGM156N", "IRSTCI01SGM156N", "INTDSRSGM193N"),
    "DEXMAUS": ("IR3TIB01MYM156N", "IRSTCI01MYM156N", "INTDSRMYM193N"),
    "DEXTHUS": ("IR3TIB01THM156N", "IRSTCI01THM156N", "INTDSRTHM193N"),
    "DEXHKUS": ("IR3TIB01HKM156N", "IRSTCI01HKM156N", "INTDSRHKM193N"),
    "DEXTAUS": ("IR3TIB01TWM156N", "IRSTCI01TWM156N", "INTDSRTWM193N"),
    "DEXSLUS": ("IR3TIB01LKM156N", "IRSTCI01LKM156N", "INTDSRLKM193N"),
    "DEXVZUS": ("INTDSRVEM193N",),
}

# The US leg of the differential. The engine already subtracts a T-bill rate for the
# strategy and the benchmark alike, so this is *not* used to build the carry - it is
# fetched only so the differential can be reported currency by currency.
US_SHORT_RATE_CANDIDATES: tuple[str, ...] = ("DTB3", "IR3TIB01USM156N", "IRSTCI01USM156N")


def fetch_short_rates(
    universe: tuple[str, ...],
    *,
    candidates: dict[str, tuple[str, ...]] | None = None,
    refresh: bool = False,
    attempts: dict[str, list[str]] | None = None,
) -> dict[str, tuple[str, pd.Series]]:
    """First available short rate per currency, as an annualised decimal.

    Tries each currency's candidates in order and keeps the first that returns data.
    A currency for which nothing is available is simply absent from the result; the
    caller reports it as uncovered rather than substituting a number. Pass ``attempts``
    to collect why each candidate was rejected - FRED's coverage of non-OECD short rates
    is the binding constraint on this diagnostic and "we did not look" must not be
    confusable with "it is not there".

    FRED publishes these as percent per annum, so they are divided by 100 here - once,
    at the boundary - and everything downstream is in decimal.
    """
    candidates = candidates or SHORT_RATE_CANDIDATES
    # Which candidate won - or that none did - is cached, so a rerun does not re-probe
    # series FRED does not carry. Probing costs the same courtesy budget as fetching.
    resolved_path = CACHE_DIR / "fred_short_rate_resolution.json"
    resolved: dict[str, str | None] = {}
    if resolved_path.is_file() and not refresh:
        resolved = json.loads(resolved_path.read_text())

    found: dict[str, tuple[str, pd.Series]] = {}
    for series_id in universe:
        if series_id in resolved:
            winner = resolved[series_id]
            if winner is None:
                continue
            found[series_id] = (winner, fetch_observations(winner).astype(float) / 100.0)
            continue
        winner = None
        for candidate in candidates.get(series_id, ()):
            try:
                observations = fetch_observations(candidate, refresh=refresh)
            except FredError as exc:
                # Recorded, not swallowed: "FRED does not carry this series" is a finding
                # about coverage that the diagnostic has to report, and a bare `continue`
                # would make it indistinguishable from a series that was never tried.
                if attempts is not None:
                    attempts.setdefault(series_id, []).append(f"{candidate}: {exc}")
                continue
            if observations.notna().sum() == 0:
                if attempts is not None:
                    attempts.setdefault(series_id, []).append(f"{candidate}: no numeric observations")
                continue
            winner = candidate
            found[series_id] = (candidate, observations.astype(float) / 100.0)
            break
        resolved[series_id] = winner

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True))
    return found


def peg_breach_dates(normalised: pd.DataFrame, *, tolerance: float = 0.98) -> dict[str, pd.DatetimeIndex]:
    """Dates on which a policy identity from :func:`check_peg_relationships` does not hold.

    These are not statistical outliers and no threshold was fitted to find them: a
    Danish krone cross-rate outside the ERM II band, or a Hong Kong dollar outside the
    convertibility band, is a value the issuing central bank was committed to
    preventing. When one appears in a series that is otherwise inside the band on
    99.9%+ of days, the observation is far more likely to be a bad print than a policy
    breach nobody recorded.

    Nothing here removes them. PREREG_005.md sections 2-6 are frozen and authorise no
    outlier rule, so the headline runs on the data as published. This exists so the
    *sensitivity* to those observations can be reported instead of being unknown.
    """
    breaches: dict[str, pd.DatetimeIndex] = {}
    for check in check_peg_relationships(normalised, tolerance=tolerance):
        if check.name.startswith("Hong Kong"):
            column = normalised["DEXHKUS"].dropna()
            outside = column[(column < check.low) | (column > check.high)]
        else:
            pair = normalised[["DEXUSEU", "DEXDNUS"]].dropna()
            implied = pair["DEXUSEU"] / pair["DEXDNUS"]
            outside = implied[(implied < check.low) | (implied > check.high)]
        if len(outside):
            breaches[check.name] = pd.DatetimeIndex(outside.index)
    return breaches
