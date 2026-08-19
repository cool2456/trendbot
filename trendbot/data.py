"""Price history for the research path.

Nothing in :mod:`tests` imports this module: the test suite is entirely offline and
runs on fixtures. This is the only place in the package that touches a price vendor.

Two sources, used for two different jobs:

``yahoo``
    Split- and dividend-adjusted OHLC via ``yfinance``, going back to each ETF's
    inception (SPY 1993, the last of the twelve 2007). Used for the backtest,
    because a momentum signal on TLT/IEF/VNQ computed on price-only series would be
    measuring something other than the return an investor earns.

``alpaca``
    The broker's own adjusted bars. Correct for the live path and for feasibility
    (an affordability calculation must use the price you would actually pay), but
    the market-data plan floors history at 2016-01-04 for every symbol, which is
    too short for the section 7 protocol.

Both are cached to parquet under ``data/cache`` so a backtest is reproducible
without a network round trip.
"""

from __future__ import annotations

import datetime as dt
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
    """Raised when price data cannot be obtained or fails a sanity check."""


@dataclass(frozen=True, slots=True)
class PriceData:
    """Adjusted OHLC for a fixed universe, on a shared trading-day calendar."""

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


# --------------------------------------------------------------------------------------
# environment / credentials
# --------------------------------------------------------------------------------------


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE pairs from .env. Absent file yields an empty mapping."""
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
    """Alpaca paper key/secret from the environment, falling back to .env."""
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


# --------------------------------------------------------------------------------------
# vendors
# --------------------------------------------------------------------------------------


def _fetch_yahoo(tickers: tuple[str, ...], start: str, end: str | None) -> PriceData:
    try:
        import yfinance  # noqa: PLC0415  (optional research-only dependency)
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise DataError(
            "yfinance is not installed. Install the research extra: "
            "pip install -e '.[research]'"
        ) from exc

    raw = yfinance.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,  # split- AND dividend-adjusted OHLC
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
                "adjustment": "all",  # splits and dividends
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


# --------------------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------------------


def _sanity_check(data: PriceData) -> None:
    """Refuse obviously corrupt price data rather than backtesting on it."""
    for name, frame in (("open", data.open), ("close", data.close)):
        if frame.index.has_duplicates:
            raise DataError(f"{name} has duplicate dates")
        if not frame.index.is_monotonic_increasing:
            raise DataError(f"{name} index is not sorted")
        finite = frame.to_numpy()
        if (finite[~pd.isna(finite)] <= 0).any():
            raise DataError(f"{name} contains a non-positive price")
    # A single-day move beyond +/-50% in a broad ETF is a bad adjustment, not a market.
    moves = data.close.pct_change(fill_method=None).abs()
    stacked = moves.stack(future_stack=True).dropna()
    bad = stacked[stacked > 0.5]
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
) -> PriceData:
    """Load adjusted daily open/close for ``tickers``, cached to parquet."""
    tickers = tuple(tickers)
    if source not in _VENDORS:
        raise DataError(f"unknown source {source!r}; expected one of {sorted(_VENDORS)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = CACHE_DIR / f"{source}_{'-'.join(tickers)}_{start}_{end or 'latest'}"
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
        # A cache read is the path every backtest in this repository actually takes,
        # so it gets the same sanity check as a fresh fetch. A parquet file that was
        # written before a check existed, or edited since, must not be trusted merely
        # because it is on disk.
        _sanity_check(cached)
        return cached

    data = _VENDORS[source](tickers, start, end)
    _sanity_check(data)

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
    """Latest *unadjusted* daily close per ticker, from the broker's own feed.

    Feasibility is an affordability question, so it must use the price actually
    quoted in the market (``adjustment=raw``), not a back-adjusted research series.

    The free Alpaca market-data plan refuses SIP data inside a 15-minute recency
    window, so this asks for the last completed session rather than a live quote.
    That is the right price for a monthly rebalance anyway: the decision is made on
    a close.
    """
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
    """Daily 13-week US Treasury bill rate, annualised, as a decimal fraction.

    PREREGISTRATION.md section 4 mandates a cash account with gross exposure capped
    at 1.0, so the book carries a large and time-varying idle cash balance - about
    30% on average. Neither document says whether that cash earns anything, and it
    plainly would. This series is used for two things, symmetrically:

    * uninvested cash accrues at this rate inside the engine, and
    * Sharpe is computed in excess of this same rate, for the strategy **and** for
      the section 8 equal-weight buy-and-hold benchmark.

    Doing only the first would credit the strategy with cash income while charging
    it no opportunity cost, which inflates the Sharpe of a partly-invested book by
    0.2-0.4 against a fully-invested benchmark.

    Source is ``^IRX``, which is a *discount* rate rather than a bond-equivalent
    yield; at 13 weeks and 4% the two differ by a basis point or two, which is far
    below the resolution of anything decided here. Values are forward-filled across
    market holidays and converted to a daily accrual by simple division by 252,
    matching the annualisation convention used everywhere else in the package.
    """
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
    """Align an annualised rate series to a trading calendar as a per-day accrual."""
    aligned = rate.reindex(index.union(rate.index)).ffill().reindex(index)
    if aligned.isna().any():
        aligned = aligned.bfill()
    if aligned.isna().any():
        raise DataError("risk-free series does not cover the requested calendar")
    return aligned / 252.0
