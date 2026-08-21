"""Federal Reserve Economic Data access, for experiment 005's H.10 universe.

PREREG_005.md section 2 defines the universe as a *rule* over a published release —
"every daily USD exchange rate series published in the Federal Reserve H.10 release
and available via FRED" — so this module's job is to enumerate that release and pull
each series' observations **and its metadata**, never a ticker list written from
memory. The metadata is not decoration: section 2 says the quote convention has to be
normalised, and the only non-circular source for a series' direction is what FRED
itself says the series' units are.

Two access paths, one interface
-------------------------------
``api``
    ``api.stlouisfed.org``, used when a ``FRED_KEY`` is present in the environment or
    ``.env``. Structured JSON, and the documented way to do this.

``web``
    ``fred.stlouisfed.org``, used when there is no key. ``fredgraph.csv`` serves the
    full observation history and the series page carries the same ``title``/``units``
    strings the API returns, so the *content* is identical - it is the same database -
    and only the transport differs. Which path produced a given artefact is recorded on
    it, so a findings document can state how the data was obtained rather than imply.

There is deliberately no third path and no bundled fallback list. If FRED cannot be
reached, the experiment stops with a :class:`FredError`, exactly as experiment 004
stopped rather than substituting data it could not fetch.

Everything is cached under ``data/cache`` as parquet plus a JSON sidecar, so a rerun
is reproducible without a network round trip and without re-hammering an endpoint the
Fed operates as a courtesy.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path

import pandas as pd
import requests

from .data import CACHE_DIR, DataError, read_dotenv

__all__ = [
    "FredError",
    "SeriesMetadata",
    "H10_RELEASE_ID",
    "FRED_API_ROOT",
    "FRED_WEB_ROOT",
    "fred_api_key",
    "access_path",
    "fetch_series_metadata",
    "fetch_observations",
    "list_release_series",
]

FRED_API_ROOT = "https://api.stlouisfed.org/fred"
FRED_WEB_ROOT = "https://fred.stlouisfed.org"

# The H.10 "Foreign Exchange Rates" release. Section 2 names the release, not the
# series; this is the release's FRED identifier and is the only hardcoded pointer in
# the universe path.
H10_RELEASE_ID = 17

_USER_AGENT = "trendbot/1.0 (research; experiment 005; contact via repository)"
_TIMEOUT = 60
_RETRIES = 4
_BACKOFF = 2.0
# Minimum seconds between requests. FRED is a courtesy service and its edge applies an
# IP-level block to bursts: enumerating candidate series concurrently during this build
# earned a 403 across both fred.stlouisfed.org and api.stlouisfed.org for several
# minutes, which is the same class of failure that blocked experiment 004. Pacing is
# cheap - a full refresh of this experiment is a few dozen requests - and the cache
# means it is paid once.
_MIN_INTERVAL = 1.5
_last_request_at = 0.0


class FredError(DataError):
    """Raised when FRED data or metadata cannot be obtained or fails a sanity check."""


# --------------------------------------------------------------------------------------
# credentials and transport
# --------------------------------------------------------------------------------------


def fred_api_key() -> str | None:
    """``FRED_KEY`` from the environment or ``.env``; ``None`` when absent.

    Absence is not an error. It selects the ``web`` access path, which reaches the
    same database. It *is* reported, so that a findings document says which path ran.
    """
    env = {**read_dotenv(), **os.environ}
    for name in ("FRED_KEY", "FRED_API_KEY"):
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def access_path() -> str:
    """``"api"`` when a key is configured, otherwise ``"web"``."""
    return "api" if fred_api_key() else "web"


def _get(url: str, params: dict | None = None) -> requests.Response:
    """One GET with bounded retries on the failures that are worth retrying.

    Experiment 004 was blocked by a vendor throttle, so this backs off on 429 and 5xx
    rather than hammering, and it gives up loudly instead of returning something empty
    that would look like "the series has no data".
    """
    global _last_request_at
    last: Exception | None = None
    for attempt in range(_RETRIES):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            response = requests.get(
                url, params=params, timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
            )
        except requests.RequestException as exc:  # pragma: no cover - network path
            last = exc
        else:
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429, 500, 502, 503, 504):
                # 403 here is the edge's burst block, not an authorisation failure: the
                # same URL succeeds after a pause. Retrying with backoff is correct;
                # treating it as fatal would abandon a run that only needed to wait.
                last = FredError(f"{url} returned HTTP {response.status_code}")
            else:
                raise FredError(
                    f"{url} returned HTTP {response.status_code}: {response.text[:300]}"
                )
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF * (2**attempt))
    raise FredError(f"could not fetch {url} after {_RETRIES} attempts: {last}")


# --------------------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
    """What FRED says a series *is*, as opposed to what its numbers are.

    ``units`` is the load-bearing field. For an H.10 exchange rate it reads
    ``"<A> to One <B>"``, which states the quote convention explicitly and is the only
    thing in the dataset that does. :mod:`trendbot.fx` parses it; nothing anywhere
    infers a direction from the values.
    """

    series_id: str
    title: str
    units: str
    frequency: str
    observation_start: str
    observation_end: str
    obtained_via: str

    def as_dict(self) -> dict:
        return asdict(self)


def _metadata_cache(series_id: str) -> Path:
    return CACHE_DIR / f"fred_{series_id}.meta.json"


def _observations_cache(series_id: str) -> Path:
    return CACHE_DIR / f"fred_{series_id}.parquet"


def _metadata_from_api(series_id: str, key: str) -> SeriesMetadata:
    payload = _get(
        f"{FRED_API_ROOT}/series",
        {"series_id": series_id, "api_key": key, "file_type": "json"},
    ).json()
    entries = payload.get("seriess") or []
    if not entries:
        raise FredError(f"FRED returned no metadata for {series_id}")
    entry = entries[0]
    missing = [f for f in ("title", "units", "frequency") if not entry.get(f)]
    if missing:
        raise FredError(f"FRED metadata for {series_id} is missing {missing}")
    return SeriesMetadata(
        series_id=series_id,
        title=str(entry["title"]).strip(),
        units=str(entry["units"]).strip(),
        frequency=str(entry["frequency"]).strip(),
        observation_start=str(entry.get("observation_start", "")).strip(),
        observation_end=str(entry.get("observation_end", "")).strip(),
        obtained_via="api",
    )


# The series page renders the same strings the API serves. Each pattern is anchored on
# a label rather than on layout, and every one of them is *required*: a page redesign
# must break this loudly, because a silently missing ``units`` would take the quote
# convention with it.
_OG_TITLE = re.compile(r'property="og:title"\s+content="([^"]*)"')
_UNITS = re.compile(r"Units:\s*</span>(.*?)(?:<span[^>]*>\s*Frequency|</div>)", re.S)
_FREQUENCY = re.compile(r"Frequency:\s*</span>(.*?)</div>", re.S)
_RANGE = re.compile(r"from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})")


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _metadata_from_web(series_id: str) -> SeriesMetadata:
    html = _get(f"{FRED_WEB_ROOT}/series/{series_id}").text

    title_match = _OG_TITLE.search(html)
    if title_match is None:
        raise FredError(f"could not read a title for {series_id} from its FRED series page")
    title = unescape(title_match.group(1)).strip()

    units_match = _UNITS.search(html)
    if units_match is None:
        raise FredError(
            f"could not read the units for {series_id} from its FRED series page. The "
            "units string is what states the quote convention; refusing to continue "
            "without it rather than guessing a direction."
        )
    # "Japanese Yen to One U.S. Dollar , Not Seasonally Adjusted" -> the part before
    # the seasonal-adjustment clause, which is a separate field FRED renders inline.
    units = _text(units_match.group(1)).split(",")[0].strip()
    if not units:
        raise FredError(f"the units string for {series_id} parsed to empty")

    frequency_match = _FREQUENCY.search(html)
    frequency = _text(frequency_match.group(1)) if frequency_match else ""

    description = re.search(r'<meta name="description" content="([^"]*)"', html)
    span = _RANGE.search(unescape(description.group(1))) if description else None
    start, end = (span.group(1), span.group(2)) if span else ("", "")

    return SeriesMetadata(
        series_id=series_id,
        title=title,
        units=units,
        frequency=frequency,
        observation_start=start,
        observation_end=end,
        obtained_via="web",
    )


def fetch_series_metadata(
    series_id: str, *, cache: bool = True, refresh: bool = False
) -> SeriesMetadata:
    """FRED's own description of ``series_id``: title, units, frequency, coverage."""
    path = _metadata_cache(series_id)
    if cache and not refresh and path.is_file():
        return SeriesMetadata(**json.loads(path.read_text()))

    key = fred_api_key()
    meta = _metadata_from_api(series_id, key) if key else _metadata_from_web(series_id)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta.as_dict(), indent=2))
    return meta


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------


def _observations_from_api(series_id: str, key: str) -> pd.Series:
    payload = _get(
        f"{FRED_API_ROOT}/series/observations",
        {"series_id": series_id, "api_key": key, "file_type": "json"},
    ).json()
    rows = payload.get("observations") or []
    if not rows:
        raise FredError(f"FRED returned no observations for {series_id}")
    frame = pd.DataFrame(rows)[["date", "value"]]
    # FRED's own missing-value marker. Coercing it to NaN is the point: a day a rate
    # was not published is a hole, never a zero and never a stale carry - what to do
    # about the hole is decided downstream, in one place, in trendbot.fx.
    values = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    return pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
        name=series_id,
    )


def _observations_from_web(series_id: str) -> pd.Series:
    text = _get(f"{FRED_WEB_ROOT}/graph/fredgraph.csv", {"id": series_id}).text
    if not text.lstrip().lower().startswith("observation_date"):
        raise FredError(
            f"fredgraph.csv for {series_id} did not return a CSV header; got "
            f"{text[:200]!r}"
        )
    frame = pd.read_csv(pd.io.common.StringIO(text))
    if len(frame.columns) != 2:
        raise FredError(f"fredgraph.csv for {series_id} returned {len(frame.columns)} columns")
    dates, values = frame.columns
    if values != series_id:
        raise FredError(
            f"fredgraph.csv returned a column named {values!r} when asked for {series_id!r}"
        )
    series = pd.Series(
        pd.to_numeric(frame[values], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame[dates])),
        name=series_id,
    )
    if series.notna().sum() == 0:
        raise FredError(f"fredgraph.csv for {series_id} contained no numeric observations")
    return series


def fetch_observations(
    series_id: str, *, cache: bool = True, refresh: bool = False
) -> pd.Series:
    """The full published history of ``series_id``, NaN where FRED publishes no value.

    The index is FRED's own observation calendar for the series, which for a daily
    H.10 rate is every weekday including the ones on which no rate was published. That
    is deliberate: the holes are data - they are what section 3 of the build order asks
    to be handled explicitly - and dropping them here would hide them.
    """
    path = _observations_cache(series_id)
    if cache and not refresh and path.is_file():
        return pd.read_parquet(path)[series_id]

    key = fred_api_key()
    series = _observations_from_api(series_id, key) if key else _observations_from_web(series_id)
    series = series.sort_index()
    if series.index.has_duplicates:
        raise FredError(f"{series_id} came back with duplicate observation dates")

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_parquet(path)
    return series


# --------------------------------------------------------------------------------------
# release enumeration - section 2's "published in the H.10 release"
# --------------------------------------------------------------------------------------


def _release_series_from_api(release_id: int, key: str) -> tuple[str, ...]:
    found: list[str] = []
    offset, limit = 0, 1000
    while True:
        payload = _get(
            f"{FRED_API_ROOT}/release/series",
            {
                "release_id": release_id,
                "api_key": key,
                "file_type": "json",
                "limit": limit,
                "offset": offset,
            },
        ).json()
        batch = payload.get("seriess") or []
        found.extend(str(entry["id"]) for entry in batch)
        offset += len(batch)
        if len(batch) < limit or offset >= int(payload.get("count", offset)):
            break
    if not found:
        raise FredError(f"FRED listed no series for release {release_id}")
    return tuple(sorted(set(found)))


_RELEASE_LINK = re.compile(r'href="/series/([A-Z0-9]+)"')


def _release_series_from_web(release_id: int) -> tuple[str, ...]:
    html = _get(f"{FRED_WEB_ROOT}/release", {"rid": release_id}).text
    found = sorted(set(_RELEASE_LINK.findall(html)))
    if not found:
        raise FredError(
            f"the FRED release page for rid={release_id} listed no series; the page "
            "layout may have changed. Refusing to fall back to a hardcoded list."
        )
    # The release page renders its series list in one block with no pager. If FRED ever
    # paginates it, this catches the truncation instead of silently shrinking the
    # universe - which is precisely the failure section 2's "no substitutions" clause
    # is written against.
    if re.search(r"\bpageID=(?!1\b)\d+", html):
        raise FredError(
            f"the FRED release page for rid={release_id} appears paginated; the "
            "enumeration would be truncated. Refusing to build a partial universe."
        )
    return tuple(found)


def list_release_series(
    release_id: int = H10_RELEASE_ID, *, cache: bool = True, refresh: bool = False
) -> tuple[str, ...]:
    """Every FRED series identifier published under ``release_id``.

    Section 2's universe is a rule over this list, not a list. Filtering it down to the
    daily bilateral USD rates happens in :mod:`trendbot.fx`, against each series'
    metadata, so the filter is auditable and the discarded series are reportable.
    """
    path = CACHE_DIR / f"fred_release_{release_id}.json"
    if cache and not refresh and path.is_file():
        return tuple(json.loads(path.read_text())["series"])

    key = fred_api_key()
    found = (
        _release_series_from_api(release_id, key) if key else _release_series_from_web(release_id)
    )
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"release_id": release_id, "obtained_via": access_path(), "series": list(found)}, indent=2)
        )
    return found
