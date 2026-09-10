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
    pass


_ACQUISITION_ACTIONS = frozenset({"acquisitionby", "merger", "merging", "acquired"})
_BANKRUPTCY_ACTIONS = frozenset(
    {"bankruptcyliquidation", "bankruptcy", "liquidation", "chapter11", "chapter7"}
)


def classify_delist_reason(action: str | None) -> str:
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


@dataclass(frozen=True, slots=True)
class SharadarClient:
    api_key: str
    cache_dir: Path = CACHE_DIR
    timeout: int = 120
    max_retries: int = 5

    @classmethod
    def from_env(cls, **kwargs) -> "SharadarClient":
        env = {**read_dotenv(), **os.environ}
        for name in ("SHARADAR_KEY", "SHARDAR_KEY", "NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY"):
            value = env.get(name)
            if value:
                return cls(api_key=value.strip().strip("'\""), **kwargs)
        raise SharadarError(
            "no Sharadar API key found. Set SHARADAR_KEY (or SHARDAR_KEY) in .env or the "
            "environment; this module will not fall back to an unauthenticated request."
        )

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
                last = SharadarError(
                    f"{table}: HTTP {response.status_code} {response.text[:200]}"
                )
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            raise SharadarError(f"{table}: HTTP {response.status_code} {response.text[:300]}")
        raise SharadarError(f"{table}: gave up after {self.max_retries} attempts ({last})")

    def _paged(self, table: str, params: dict, *, expect: tuple[str, ...]) -> pd.DataFrame:
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

    def security_master(self, *, refresh: bool = False) -> pd.DataFrame:
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
