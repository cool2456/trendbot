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

MAX_HOLIDAY_GAP_DAYS = 10


_USD = re.compile(r"^u\.?\s?s\.?\s?dollars?$|^usd$", re.IGNORECASE)

_UNITS = re.compile(r"^(?P<numerator>.+?)\s+to\s+One\s+(?P<denominator>.+?)$", re.IGNORECASE)

_TITLE = re.compile(r"^(?P<numerator>.+?)\s+to\s+(?P<denominator>.+?)\s+Spot Exchange Rate$", re.IGNORECASE)


def _canonical(name: str) -> str:
    text = re.sub(r"[^a-z ]", "", name.lower()).strip()
    text = re.sub(r"\s+", " ", text)
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
        return self.direction == FOREIGN_PER_USD

    def __str__(self) -> str:
        action = "1 / x" if self.inverted else "x (unchanged)"
        return (
            f"{self.series_id}: units {self.units!r} -> {self.direction}; "
            f"foreign currency {self.foreign_currency}; normalise by {action}"
        )


def parse_quote_convention(meta: SeriesMetadata) -> QuoteConvention:
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
        direction, foreign = FOREIGN_PER_USD, numerator
        evidence = f"units say {numerator!r} per one {denominator!r} -> value is foreign per USD"
    else:
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
    values = series.astype(float)
    values = values.where(values > 0)
    if convention.inverted:
        values = 1.0 / values
    return values.rename(convention.series_id)


@dataclass(frozen=True, slots=True)
class ReferenceCheck:
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
    checks: list[ReferenceCheck] = []
    for series_id, date, low, high, why in references:
        if series_id not in normalised.columns:
            continue
        column = normalised[series_id].dropna()
        stamp = pd.Timestamp(date)
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
    checks: list[PegCheck] = []

    if "DEXHKUS" in normalised.columns:
        hkd = normalised["DEXHKUS"].dropna()
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
        implied = pair["DEXUSEU"] / pair["DEXDNUS"]
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


@dataclass(frozen=True, slots=True)
class SeriesGaps:
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
    returns = frame.pct_change(fill_method=None)
    contributes = frame.notna() & frame.shift(1).notna()
    counts = contributes.sum(axis=1)
    weights = contributes.div(counts.where(counts > 0), axis=0).fillna(0.0)
    return (weights * returns.fillna(0.0)).sum(axis=1).fillna(0.0).rename("dollar_factor")


@dataclass(frozen=True, slots=True)
class FxUniverse:
    sample_start: str
    release_id: int
    candidates: tuple[str, ...]
    universe: tuple[str, ...]
    conventions: dict[str, QuoteConvention]
    metadata: dict[str, SeriesMetadata]
    normalised: pd.DataFrame
    raw: dict[str, pd.Series]
    calendar: pd.DatetimeIndex
    gaps: dict[str, SeriesGaps]
    exclusions: tuple[tuple[str, str, str], ...]
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
    frame = universe.normalised
    return PriceData(
        open=frame.copy(),
        close=frame.copy(),
        source=f"FRED H.10 release {universe.release_id} (normalised to USD per foreign unit)",
        adjusted=True,
        fetched_at="",
    )


def dollar_factor_returns(universe: FxUniverse, prices: PriceData | None = None) -> pd.Series:
    frame = (prices.close if prices is not None else universe.normalised)[list(universe.universe)]
    return equal_weight_factor(frame)


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

US_SHORT_RATE_CANDIDATES: tuple[str, ...] = ("DTB3", "IR3TIB01USM156N", "IRSTCI01USM156N")


def fetch_short_rates(
    universe: tuple[str, ...],
    *,
    candidates: dict[str, tuple[str, ...]] | None = None,
    refresh: bool = False,
    attempts: dict[str, list[str]] | None = None,
) -> dict[str, tuple[str, pd.Series]]:
    candidates = candidates or SHORT_RATE_CANDIDATES
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
