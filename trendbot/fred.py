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

H10_RELEASE_ID = 17

_USER_AGENT = "trendbot/1.0 (research; experiment 005; contact via repository)"
_TIMEOUT = 60
_RETRIES = 4
_BACKOFF = 2.0
_MIN_INTERVAL = 1.5
_last_request_at = 0.0


class FredError(DataError):
    pass


def fred_api_key() -> str | None:
    env = {**read_dotenv(), **os.environ}
    for name in ("FRED_KEY", "FRED_API_KEY"):
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def access_path() -> str:
    return "api" if fred_api_key() else "web"


def _get(url: str, params: dict | None = None) -> requests.Response:
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
        except requests.RequestException as exc:  # pragma: no cover
            last = exc
        else:
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429, 500, 502, 503, 504):
                last = FredError(f"{url} returned HTTP {response.status_code}")
            else:
                raise FredError(
                    f"{url} returned HTTP {response.status_code}: {response.text[:300]}"
                )
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF * (2**attempt))
    raise FredError(f"could not fetch {url} after {_RETRIES} attempts: {last}")


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
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
    path = _metadata_cache(series_id)
    if cache and not refresh and path.is_file():
        return SeriesMetadata(**json.loads(path.read_text()))

    key = fred_api_key()
    meta = _metadata_from_api(series_id, key) if key else _metadata_from_web(series_id)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta.as_dict(), indent=2))
    return meta


def _observations_from_api(series_id: str, key: str) -> pd.Series:
    payload = _get(
        f"{FRED_API_ROOT}/series/observations",
        {"series_id": series_id, "api_key": key, "file_type": "json"},
    ).json()
    rows = payload.get("observations") or []
    if not rows:
        raise FredError(f"FRED returned no observations for {series_id}")
    frame = pd.DataFrame(rows)[["date", "value"]]
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
    if re.search(r"\bpageID=(?!1\b)\d+", html):
        raise FredError(
            f"the FRED release page for rid={release_id} appears paginated; the "
            "enumeration would be truncated. Refusing to build a partial universe."
        )
    return tuple(found)


def list_release_series(
    release_id: int = H10_RELEASE_ID, *, cache: bool = True, refresh: bool = False
) -> tuple[str, ...]:
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
