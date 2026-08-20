"""Sharadar (Nasdaq Data Link) client: security master, prices, corporate actions.

This is the only module in the package that knows a vendor's vocabulary. Everything
above it speaks in **permanent security IDs**, never ticker strings.

Why that distinction is the whole point of experiment 004
---------------------------------------------------------
A ticker is a lease, not a name. ``FB`` became ``META``; ``GM`` in 2008 is a different
company from ``GM`` in 2011; symbols are recycled within months of a delisting. Any
structure keyed on the ticker string silently merges those into one price series, and
a merged series produces a twelve-month momentum number for a company that did not
exist. Sharadar's ``permaticker`` is stable across ticker changes and is never reused,
so it is used as the key everywhere and the ticker is carried only as a label.

Three tables are used:

``SHARADAR/TICKERS``
    The security master. Gives ``permaticker``, ``category`` (which is how section 2's
    "common stock, exclude ETFs/ETNs/CEFs/units/warrants/preferred" is applied),
    ``isdelisted``, and the first and last dates on which the vendor has a price.

``SHARADAR/SEP``
    Daily equity prices. Carries three different closes and the distinction matters:
    ``closeunadj`` is what section 2's $5.00 floor is applied to, ``closeadj`` is what
    the signal and the returns are computed on, and ``close`` is split-adjusted only.

``SHARADAR/ACTIONS``
    Corporate actions, including the delisting events section 3's table maps onto and
    the ticker changes that make the permaticker necessary.

Nothing here is called by the test suite. Every function that reaches the network is
kept behind :class:`SharadarClient` so the pipeline above can be driven from fixtures.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config_004 import ACQUISITION, BANKRUPTCY, UNKNOWN
from .data import REPO_ROOT, DataError, read_dotenv

__all__ = [
    "SharadarClient",
    "SharadarError",
    "CACHE_DIR",
    "classify_delist_reason",
    "COMMON_STOCK_CATEGORIES",
    "SEP_COLUMNS",
]

CACHE_DIR = REPO_ROOT / "data" / "sharadar"
BASE_URL = "https://data.nasdaq.com/api/v3/datatables"

# Columns this module relies on. Asserted on every fetch: a vendor that renames a
# column must break loudly rather than silently deliver a frame missing the one
# distinction section 2 turns on.
SEP_COLUMNS = ("ticker", "date", "open", "close", "closeadj", "closeunadj", "volume")
TICKERS_COLUMNS = (
    "permaticker",
    "ticker",
    "name",
    "category",
    "isdelisted",
    "firstpricedate",
    "lastpricedate",
)

# Section 2: "Security type is common stock. Exclude ETFs, ETNs, CEFs, closed-end debt,
# units, warrants, preferred shares. ADRs are **included**."
#
# Sharadar's `category` is a free-text taxonomy, so the inclusion list is written out
# rather than inferred by excluding strings. Everything not on this list is out, which
# is the safe direction: a category this project has not seen before is excluded rather
# than silently admitted.
COMMON_STOCK_CATEGORIES = frozenset(
    {
        "Domestic Common Stock",
        "Domestic Common Stock Primary Class",
        "Domestic Common Stock Secondary Class",
        "Canadian Common Stock",
        "Canadian Common Stock Primary Class",
        "Canadian Common Stock Secondary Class",
        "ADR Common Stock",
        "ADR Common Stock Primary Class",
        "ADR Common Stock Secondary Class",
    }
)


class SharadarError(DataError):
    """Raised when the vendor cannot be reached or returns something unexpected."""


# --------------------------------------------------------------------------------------
# section 3 - mapping a vendor reason onto one of section 3's three buckets
# --------------------------------------------------------------------------------------

# Sharadar records delistings in ACTIONS with an `action` string. These are grouped onto
# the three rows of PREREG_004.md section 3's table. The mapping is deliberately
# conservative: anything not recognised falls into `unknown`, which section 3 haircuts
# at -30%, rather than into `acquisition`, which pays out in full. Guessing in the
# generous direction is how a delisting treatment quietly manufactures a return.
_ACQUISITION_ACTIONS = frozenset({"acquisitionby", "merger", "merging", "acquired"})
_BANKRUPTCY_ACTIONS = frozenset(
    {"bankruptcyliquidation", "bankruptcy", "liquidation", "chapter11", "chapter7"}
)


def classify_delist_reason(action: str | None) -> str:
    """Map a vendor delist action string onto one of section 3's three buckets.

    Returns one of ``acquisition``, ``bankruptcy`` or ``unknown``. A missing, empty or
    unrecognised reason is ``unknown`` by construction - section 3 explicitly groups
    "reason unknown/missing" with "moved to OTC" and haircuts both.
    """
    if action is None:
        return UNKNOWN
    key = str(action).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if not key or key in {"nan", "none", "null"}:
        return UNKNOWN
    if key in _ACQUISITION_ACTIONS:
        return ACQUISITION
    if key in _BANKRUPTCY_ACTIONS:
        return BANKRUPTCY
    return UNKNOWN


# --------------------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SharadarClient:
    """Thin, cached client over the three Sharadar tables this experiment needs."""

    api_key: str
    cache_dir: Path = CACHE_DIR
    timeout: int = 120
    max_retries: int = 5

    @classmethod
    def from_env(cls, **kwargs) -> "SharadarClient":
        """Read the key from the environment or .env.

        Both spellings are accepted because the repository's .env carries the
        misspelled one; refusing to read it would be pedantry, but silently inventing
        a key would not be, so a missing key is fatal.
        """
        env = {**read_dotenv(), **os.environ}
        for name in ("SHARADAR_KEY", "SHARDAR_KEY", "NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY"):
            value = env.get(name)
            if value:
                return cls(api_key=value.strip().strip("'\""), **kwargs)
        raise SharadarError(
            "no Sharadar API key found. Set SHARADAR_KEY (or SHARDAR_KEY) in .env or the "
            "environment; this module will not fall back to an unauthenticated request."
        )

    # ---- low-level -------------------------------------------------------------------

    def _get(self, table: str, params: dict) -> "requests.Response":  # noqa: F821
        import requests  # noqa: PLC0415

        last: Exception | None = None
        for attempt in range(self.max_retries):
            response = requests.get(
                f"{BASE_URL}/{table}.json",
                params={**params, "api_key": self.api_key},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response
            if response.status_code in (429, 503):
                # The vendor rate-limits aggressively and answers 429 for both "too fast"
                # and "temporarily disabled". Backing off is worth trying; failing after
                # a bounded number of attempts is better than an unbounded loop.
                last = SharadarError(
                    f"{table}: HTTP {response.status_code} {response.text[:200]}"
                )
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            raise SharadarError(f"{table}: HTTP {response.status_code} {response.text[:300]}")
        raise SharadarError(f"{table}: gave up after {self.max_retries} attempts ({last})")

    def _paged(self, table: str, params: dict, *, expect: tuple[str, ...]) -> pd.DataFrame:
        """Fetch a datatable, following the cursor until the vendor stops paging."""
        frames: list[pd.DataFrame] = []
        cursor: str | None = None
        columns: list[str] | None = None
        while True:
            page = dict(params)
            if cursor:
                page["qopts.cursor_id"] = cursor
            payload = self._get(table, page).json()
            datatable = payload.get("datatable")
            if datatable is None:
                raise SharadarError(f"{table}: no datatable in response {json.dumps(payload)[:300]}")
            columns = [c["name"] for c in datatable["columns"]]
            frames.append(pd.DataFrame(datatable["data"], columns=columns))
            cursor = (payload.get("meta") or {}).get("next_cursor_id")
            if not cursor:
                break
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(expect))
        missing = [c for c in expect if c not in frame.columns]
        if missing:
            raise SharadarError(
                f"{table} is missing columns {missing}; got {list(frame.columns)}. The vendor "
                "schema has changed and this experiment depends on those columns meaning "
                "what they meant when it was written."
            )
        return frame

    # ---- tables ----------------------------------------------------------------------

    def security_master(self, *, refresh: bool = False) -> pd.DataFrame:
        """SHARADAR/TICKERS for the equity price table, one row per security.

        Returned indexed by ``permaticker``, with dates parsed. ``ticker`` is present
        but is a label; nothing downstream may key on it.
        """
        path = self.cache_dir / "tickers.parquet"
        if path.is_file() and not refresh:
            return pd.read_parquet(path)

        frame = self._paged("SHARADAR/TICKERS", {"table": "SEP"}, expect=TICKERS_COLUMNS)
        frame = frame[list(TICKERS_COLUMNS)].copy()
        frame["permaticker"] = frame["permaticker"].astype("int64")
        for column in ("firstpricedate", "lastpricedate"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame["isdelisted"] = frame["isdelisted"].astype(str).str.upper().eq("Y")
        frame = frame.drop_duplicates(subset=["permaticker"], keep="first")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    def actions(self, *, refresh: bool = False) -> pd.DataFrame:
        """SHARADAR/ACTIONS - delistings, ticker changes, splits."""
        path = self.cache_dir / "actions.parquet"
        if path.is_file() and not refresh:
            return pd.read_parquet(path)

        frame = self._paged("SHARADAR/ACTIONS", {}, expect=("date", "action", "ticker"))
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "permaticker" in frame.columns:
            frame["permaticker"] = pd.to_numeric(frame["permaticker"], errors="coerce")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    def prices(
        self,
        tickers: tuple[str, ...] | list[str] | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        refresh: bool = False,
        label: str = "sep",
    ) -> pd.DataFrame:
        """SHARADAR/SEP daily prices.

        ``tickers=None`` asks the vendor for every security over the date range, which
        is what the point-in-time universe needs and what the free tier will refuse.
        """
        import hashlib  # noqa: PLC0415

        key = hashlib.sha256(
            f"{sorted(tickers) if tickers else 'ALL'}|{start}|{end}".encode()
        ).hexdigest()[:16]
        path = self.cache_dir / f"{label}_{key}.parquet"
        if path.is_file() and not refresh:
            return pd.read_parquet(path)

        params: dict = {}
        if tickers:
            params["ticker"] = ",".join(tickers)
        if start:
            params["date.gte"] = start
        if end:
            params["date.lte"] = end
        frame = self._paged("SHARADAR/SEP", params, expect=SEP_COLUMNS)
        frame = frame[list(SEP_COLUMNS)].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "close", "closeadj", "closeunadj", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    def bulk_prices(self, *, refresh: bool = False, label: str = "sep_bulk") -> pd.DataFrame:
        """Whole-table SEP export, which is the only sane way to get every security.

        The datatable cursor API pages a few thousand rows at a time; SEP is tens of
        millions of rows. Nasdaq Data Link's export endpoint hands back the entire table
        as one zipped CSV instead, which turns hours of paging into one download.
        """
        import requests  # noqa: PLC0415

        path = self.cache_dir / f"{label}.parquet"
        if path.is_file() and not refresh:
            return pd.read_parquet(path)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(
            f"{BASE_URL}/SHARADAR/SEP.json",
            params={"qopts.export": "true", "api_key": self.api_key},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise SharadarError(f"SEP export request: HTTP {response.status_code} {response.text[:300]}")
        payload = response.json()
        link = (payload.get("datatable_bulk_download") or {}).get("file", {})
        status, url = link.get("status"), link.get("link")

        waited = 0
        while status not in ("fresh", "regenerating") or not url:
            if waited > 900:
                raise SharadarError(f"SEP export never became available (status {status})")
            time.sleep(15)
            waited += 15
            payload = requests.get(
                f"{BASE_URL}/SHARADAR/SEP.json",
                params={"qopts.export": "true", "api_key": self.api_key},
                timeout=self.timeout,
            ).json()
            link = (payload.get("datatable_bulk_download") or {}).get("file", {})
            status, url = link.get("status"), link.get("link")

        blob = requests.get(url, timeout=1800)
        if blob.status_code != 200:
            raise SharadarError(f"SEP export download: HTTP {blob.status_code}")
        with zipfile.ZipFile(io.BytesIO(blob.content)) as archive:
            name = archive.namelist()[0]
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, usecols=list(SEP_COLUMNS))
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame.to_parquet(path)
        return frame

    def describe(self) -> dict:
        return {
            "vendor": "Sharadar via Nasdaq Data Link",
            "tables": ["SHARADAR/TICKERS", "SHARADAR/SEP", "SHARADAR/ACTIONS"],
            "key_prefix": f"{self.api_key[:2]}…",
            "cache_dir": str(self.cache_dir),
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
