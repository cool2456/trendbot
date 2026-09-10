from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "PriceData",
    "load_prices",
    "last_trade_prices",
    "load_risk_free_rate",
    "DataError",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"

ALPACA_DATA_URL = "https://data.alpaca.markets"


class DataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PriceData:
    open: pd.DataFrame
    close: pd.DataFrame
    source: str
    adjusted: bool
    fetched_at: str

    def __post_init__(self) -> None:
        if list(self.open.columns) != list(self.close.columns):
            raise DataError("open and close frames disagree on columns")
        if not self.open.index.equals(self.close.index):
            raise DataError("open and close frames disagree on the trading calendar")

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.close.columns)

    def first_valid(self) -> pd.Series:
        return self.close.apply(lambda s: s.first_valid_index())

    def slice(self, start: str | None = None, end: str | None = None) -> "PriceData":
        return PriceData(
            open=self.open.loc[start:end],
            close=self.close.loc[start:end],
            source=self.source,
            adjusted=self.adjusted,
            fetched_at=self.fetched_at,
        )

    def describe(self) -> str:
        return (
            f"{len(self.close)} bars x {len(self.tickers)} instruments from {self.source} "
            f"({'adjusted' if self.adjusted else 'RAW'}), "
            f"{self.close.index[0].date()} -> {self.close.index[-1].date()}"
        )


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path or (REPO_ROOT / ".env")
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def alpaca_credentials() -> tuple[str, str]:
    env = {**read_dotenv(), **os.environ}
    key, secret = env.get("ALPACA_KEY"), env.get("ALPACA_SECRET")
    if not key or not secret:
        raise DataError(
            "ALPACA_KEY / ALPACA_SECRET not found in the environment or .env. "
            "These must be paper-trading credentials."
        )
    if not key.startswith("PK"):
        raise DataError(
            f"ALPACA_KEY {key[:4]}... does not look like a paper key (paper keys start "
            "with 'PK'). This repository refuses to run against a live account."
        )
    return key, secret


def _fetch_yahoo(tickers: tuple[str, ...], start: str, end: str | None) -> PriceData:
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise DataError(
            "yfinance is not installed. Install the research extra: "
            "pip install -e '.[research]'"
        ) from exc

    raw = yfinance.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        actions=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned nothing for {tickers}")

    close = raw["Close"].reindex(columns=list(tickers))
    open_ = raw["Open"].reindex(columns=list(tickers))
    missing = [t for t in tickers if close[t].notna().sum() == 0]
    if missing:
        raise DataError(f"yfinance returned no usable closes for {missing}")

    return PriceData(
        open=open_.astype(float),
        close=close.astype(float),
        source="yahoo",
        adjusted=True,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


def _fetch_alpaca(tickers: tuple[str, ...], start: str, end: str | None) -> PriceData:
    import requests  # noqa: PLC0415

    key, secret = alpaca_credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        rows: list[dict] = []
        page: str | None = None
        while True:
            params = {
                "timeframe": "1Day",
                "start": start,
                "adjustment": "all",
                "feed": "sip",
                "limit": 10_000,
            }
            if end:
                params["end"] = end
            if page:
                params["page_token"] = page
            resp = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars",
                headers=headers,
                params=params,
                timeout=60,
            )
            if resp.status_code != 200:
                raise DataError(f"alpaca bars for {ticker}: HTTP {resp.status_code} {resp.text[:200]}")
            payload = resp.json()
            rows.extend(payload.get("bars") or [])
            page = payload.get("next_page_token")
            if not page:
                break
        if not rows:
            raise DataError(f"alpaca returned no bars for {ticker}")
        frame = pd.DataFrame(rows)
        frame["t"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        frames[ticker] = frame.set_index("t")[["o", "c"]]

    index = frames[tickers[0]].index
    for frame in frames.values():
        index = index.union(frame.index)
    index = index.sort_values()

    return PriceData(
        open=pd.DataFrame({t: frames[t]["o"].reindex(index) for t in tickers}).astype(float),
        close=pd.DataFrame({t: frames[t]["c"].reindex(index) for t in tickers}).astype(float),
        source="alpaca",
        adjusted=True,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


_VENDORS = {"yahoo": _fetch_yahoo, "alpaca": _fetch_alpaca}


def _sanity_check(data: PriceData, *, max_abs_daily_move: float | None = 0.5) -> None:
    for name, frame in (("open", data.open), ("close", data.close)):
        if frame.index.has_duplicates:
            raise DataError(f"{name} has duplicate dates")
        if not frame.index.is_monotonic_increasing:
            raise DataError(f"{name} index is not sorted")
        finite = frame.to_numpy()
        if (finite[~pd.isna(finite)] <= 0).any():
            raise DataError(f"{name} contains a non-positive price")
    if max_abs_daily_move is None:
        return
    moves = data.close.pct_change(fill_method=None).abs()
    stacked = moves.stack(future_stack=True).dropna()
    bad = stacked[stacked > max_abs_daily_move]
    if len(bad):
        raise DataError(f"implausible one-day moves, likely a bad split adjustment:\n{bad.head(10)}")


def load_prices(
    tickers: tuple[str, ...] | list[str],
    *,
    source: str = "yahoo",
    start: str = "1990-01-01",
    end: str | None = None,
    cache: bool = True,
    refresh: bool = False,
    max_abs_daily_move: float | None = 0.5,
) -> PriceData:
    tickers = tuple(tickers)
    if source not in _VENDORS:
        raise DataError(f"unknown source {source!r}; expected one of {sorted(_VENDORS)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joined = "-".join(tickers)
    label = joined if len(joined) <= 180 else f"{len(tickers)}names-{hashlib.sha256(joined.encode()).hexdigest()[:16]}"
    stem = CACHE_DIR / f"{source}_{label}_{start}_{end or 'latest'}"
    meta_path, open_path, close_path = (
        stem.with_suffix(".meta.json"),
        stem.with_suffix(".open.parquet"),
        stem.with_suffix(".close.parquet"),
    )

    if cache and not refresh and meta_path.is_file() and open_path.is_file() and close_path.is_file():
        meta = json.loads(meta_path.read_text())
        cached = PriceData(
            open=pd.read_parquet(open_path),
            close=pd.read_parquet(close_path),
            source=meta["source"],
            adjusted=meta["adjusted"],
            fetched_at=meta["fetched_at"],
        )
        _sanity_check(cached, max_abs_daily_move=max_abs_daily_move)
        return cached

    data = _VENDORS[source](tickers, start, end)
    _sanity_check(data, max_abs_daily_move=max_abs_daily_move)

    if cache:
        data.open.to_parquet(open_path)
        data.close.to_parquet(close_path)
        meta_path.write_text(
            json.dumps(
                {
                    "source": data.source,
                    "adjusted": data.adjusted,
                    "fetched_at": data.fetched_at,
                    "tickers": list(tickers),
                    "rows": len(data.close),
                },
                indent=2,
            )
        )
    return data


def last_trade_prices(tickers: tuple[str, ...] | list[str], *, lookback_days: int = 10) -> pd.Series:
    import requests  # noqa: PLC0415

    key, secret = alpaca_credentials()
    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)
    start = end - dt.timedelta(days=lookback_days + 10)

    prices: dict[str, float] = {}
    asof: dict[str, str] = {}
    for ticker in tickers:
        resp = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={
                "timeframe": "1Day",
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "adjustment": "raw",
                "feed": "sip",
                "limit": 100,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise DataError(f"alpaca bars for {ticker}: HTTP {resp.status_code} {resp.text[:200]}")
        bars = resp.json().get("bars") or []
        if not bars:
            raise DataError(f"no recent daily bar for {ticker}")
        prices[ticker] = float(bars[-1]["c"])
        asof[ticker] = bars[-1]["t"][:10]

    series = pd.Series(prices, name="price")
    series.attrs["asof"] = asof
    return series


def load_risk_free_rate(
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    cache: bool = True,
    refresh: bool = False,
) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"riskfree_IRX_{start}_{end or 'latest'}.parquet"
    if cache and not refresh and path.is_file():
        return pd.read_parquet(path)["rate"]

    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise DataError(
            "yfinance is not installed. Install the research extra: pip install -e '.[research]'"
        ) from exc

    raw = yfinance.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise DataError("could not fetch ^IRX (13-week T-bill rate)")
    series = raw["Close"].squeeze().dropna().astype(float) / 100.0
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series.name = "rate"
    if not (-0.01 <= series.min() and series.max() <= 0.25):
        raise DataError(f"implausible T-bill rates: {series.min():.4f} to {series.max():.4f}")
    if cache:
        series.to_frame().to_parquet(path)
    return series


def daily_risk_free(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = rate.reindex(index.union(rate.index)).ffill().reindex(index)
    if aligned.isna().any():
        aligned = aligned.bfill()
    if aligned.isna().any():
        raise DataError("risk-free series does not cover the requested calendar")
    return aligned / 252.0
