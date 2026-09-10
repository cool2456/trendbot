from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REPO_ROOT, PriceData

__all__ = [
    "UNIVERSE_DIR",
    "MarketProxy",
    "load_market_proxy",
    "ConstituentSnapshot",
    "load_constituent_snapshot",
    "EquityUniverse",
    "resolve_universe",
    "CorporateActionAudit",
    "corporate_action_audit",
    "SplitReconciliation",
    "split_reconciliation",
    "adjustment_convention",
    "ratio_invariance_report",
]

UNIVERSE_DIR = REPO_ROOT / "data" / "universe"


@dataclass(frozen=True, slots=True)
class MarketProxy:
    symbol: str
    role: str
    rationale: str
    alternative: str
    adjudicates_section_8: bool
    chosen_at: str

    def describe(self) -> str:
        return f"{self.symbol} — {self.role}"


def load_market_proxy(path: Path | str | None = None) -> MarketProxy:
    path = Path(path) if path is not None else UNIVERSE_DIR / "market_proxy.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Section 7.5's regression needs a declared market proxy "
            "and this repository refuses to pick one implicitly."
        )
    payload = json.loads(path.read_text())
    return MarketProxy(
        symbol=payload["symbol"],
        role=payload["role"],
        rationale=payload["rationale"],
        alternative=payload["alternative_reported_alongside"],
        adjudicates_section_8=bool(payload["adjudicates_section_8"]),
        chosen_at=payload["chosen_at"],
    )


@dataclass(frozen=True, slots=True)
class ConstituentSnapshot:
    table: pd.DataFrame
    source_url: str
    retrieved_at: str
    sha256: str
    path: Path

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.table["yahoo_symbol"])

    @property
    def n(self) -> int:
        return len(self.table)

    def index_symbol(self, vendor_symbol: str) -> str:
        row = self.table.loc[self.table["yahoo_symbol"] == vendor_symbol]
        if row.empty:
            raise KeyError(vendor_symbol)
        return str(row.iloc[0]["symbol"])

    def sector_of(self, vendor_symbol: str) -> str:
        row = self.table.loc[self.table["yahoo_symbol"] == vendor_symbol]
        if row.empty:
            raise KeyError(vendor_symbol)
        return str(row.iloc[0]["gics_sector"])

    def describe(self) -> str:
        return (
            f"{self.n} constituents, retrieved {self.retrieved_at} from {self.source_url}, "
            f"sha256={self.sha256[:12]}"
        )


def load_constituent_snapshot(path: Path | str | None = None) -> ConstituentSnapshot:
    if path is None:
        candidates = sorted(UNIVERSE_DIR.glob("sp500_constituents_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"no constituent snapshot in {UNIVERSE_DIR}. Experiment 003's universe is "
                "'current S&P 500 constituents', which must be pinned to a dated file "
                "before any backtest; refusing to resolve it live."
            )
        path = candidates[-1]
    path = Path(path)
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.is_file():
        meta_path = path.parent / (path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text())

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(
            f"{path.name} does not match the digest recorded in {meta_path.name}: "
            f"{digest[:12]} vs {meta['sha256'][:12]}. The universe has been edited since "
            "it was pinned."
        )
    table = pd.read_csv(path)
    for column in ("symbol", "yahoo_symbol", "gics_sector"):
        if column not in table.columns:
            raise ValueError(f"{path.name} has no {column!r} column")
    if table["yahoo_symbol"].duplicated().any():
        raise ValueError(f"{path.name} lists a duplicate symbol")
    return ConstituentSnapshot(
        table=table,
        source_url=meta["source_url"],
        retrieved_at=meta["retrieved_at"],
        sha256=digest,
        path=path,
    )


@dataclass(frozen=True, slots=True)
class EquityUniverse:
    universe: tuple[str, ...]
    table: pd.DataFrame
    excluded: pd.DataFrame
    required_from: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_window_bars: int
    snapshot: ConstituentSnapshot

    @property
    def n(self) -> int:
        return len(self.universe)

    @property
    def with_holes(self) -> pd.Series:
        holes = self.table["missing_bars_in_window"]
        return holes[holes > 0]

    def __str__(self) -> str:
        return (
            f"section 2 resolves to {self.n} of {self.snapshot.n} current constituents "
            f"with continuous data from {self.required_from.date()}; "
            f"{len(self.excluded)} excluded for insufficient history"
        )


def resolve_universe(
    prices: PriceData,
    snapshot: ConstituentSnapshot,
    *,
    required_from: str,
    window_end: pd.Timestamp | None = None,
) -> EquityUniverse:
    required = pd.Timestamp(required_from)
    close = prices.close
    end = pd.Timestamp(window_end) if window_end is not None else close.index[-1]
    window = close.loc[required:end]
    n_bars = len(window)
    if n_bars == 0:
        raise ValueError(f"no bars between {required.date()} and {end.date()}")

    rows = []
    for symbol in snapshot.symbols:
        if symbol not in close.columns:
            rows.append(
                {
                    "ticker": symbol,
                    "sector": snapshot.sector_of(symbol),
                    "first_bar": pd.NaT,
                    "last_bar": pd.NaT,
                    "eligible": False,
                    "reason": "no data returned by the vendor",
                    "bars_in_window": 0,
                    "missing_bars_in_window": n_bars,
                }
            )
            continue
        series = close[symbol]
        first, last = series.first_valid_index(), series.last_valid_index()
        present = int(window[symbol].notna().sum())
        starts_early = first is not None and first <= required
        survives = last is not None and last >= end
        if not starts_early:
            reason = f"first bar {first.date() if first is not None else 'none'} is after {required.date()}"
        elif not survives:
            reason = f"last bar {last.date() if last is not None else 'none'} is before {end.date()}"
        else:
            reason = ""
        rows.append(
            {
                "ticker": symbol,
                "sector": snapshot.sector_of(symbol),
                "first_bar": first,
                "last_bar": last,
                "eligible": bool(starts_early and survives),
                "reason": reason,
                "bars_in_window": present,
                "missing_bars_in_window": n_bars - present,
            }
        )

    frame = pd.DataFrame(rows).set_index("ticker")
    frame["eligible"] = frame["eligible"].astype(bool)
    eligible = frame[frame["eligible"]]
    return EquityUniverse(
        universe=tuple(eligible.index),
        table=eligible.drop(columns=["reason"]),
        excluded=frame[~frame["eligible"]][["sector", "first_bar", "last_bar", "reason"]],
        required_from=required,
        window_start=window.index[0],
        window_end=window.index[-1],
        n_window_bars=n_bars,
        snapshot=snapshot,
    )


@dataclass(frozen=True, slots=True)
class CorporateActionAudit:
    events: pd.DataFrame
    threshold: float
    n_observations: int
    splits_checked: int
    tickers_checked: int
    fetch_errors: tuple[str, ...]

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def unexplained(self) -> pd.DataFrame:
        return self.events[self.events["classification"] == "market move"]

    @property
    def split_suspected(self) -> pd.DataFrame:
        return self.events[self.events["classification"] == "UNADJUSTED SPLIT"]

    @property
    def passes(self) -> bool:
        return len(self.split_suspected) == 0 and not self.fetch_errors

    def by_year(self) -> pd.Series:
        if self.events.empty:
            return pd.Series(dtype=int)
        return self.events.groupby(self.events["date"].dt.year).size()

    def __str__(self) -> str:
        return (
            f"corporate-action audit: {self.n_events} daily moves beyond "
            f"±{self.threshold:.0%} across {self.tickers_checked} tickers and "
            f"{self.n_observations:,} observations; {len(self.split_suspected)} coincide "
            f"with a known split ({'FAIL' if self.split_suspected.size else 'none - PASS'}), "
            f"{len(self.unexplained)} are market moves"
        )


def _split_cache_path(tickers, start: str) -> Path:
    key = hashlib.sha256(("|".join(sorted(tickers)) + start).encode()).hexdigest()[:16]
    return REPO_ROOT / "data" / "cache" / f"corporate_actions_{len(tickers)}names_{key}.json"


def _split_dates(
    tickers, start: str, *, cache: bool = True, refresh: bool = False
) -> tuple[dict[str, pd.DatetimeIndex], list[str]]:
    tickers = list(tickers)
    path = _split_cache_path(tickers, start)
    if cache and not refresh and path.is_file():
        payload = json.loads(path.read_text())
        return (
            {k: pd.DatetimeIndex(pd.to_datetime(v)) for k, v in payload["splits"].items()},
            list(payload["errors"]),
        )

    import yfinance  # noqa: PLC0415

    out: dict[str, pd.DatetimeIndex] = {}
    errors: list[str] = []
    for symbol in tickers:
        try:
            splits = yfinance.Ticker(symbol).splits
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        if splits is None or len(splits) == 0:
            out[symbol] = pd.DatetimeIndex([])
            continue
        index = pd.DatetimeIndex(splits.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        index = index.normalize()
        out[symbol] = index[index >= pd.Timestamp(start)]

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "start": start,
                    "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "splits": {k: [str(d.date()) for d in v] for k, v in out.items()},
                    "errors": errors,
                },
                indent=2,
            )
        )
    return out, errors


def corporate_action_audit(
    prices: PriceData,
    universe,
    *,
    threshold: float = 0.35,
    start: str | None = None,
    split_dates: dict[str, pd.DatetimeIndex] | None = None,
    tolerance_days: int = 3,
) -> CorporateActionAudit:
    universe = list(universe)
    close = prices.close[universe]
    if start is not None:
        close = close.loc[start:]
    returns = close.pct_change(fill_method=None)

    stacked = returns.stack(future_stack=True).dropna()
    n_observations = int(len(stacked))
    extreme = stacked[stacked.abs() > threshold]

    if split_dates is None:
        affected = sorted({symbol for _, symbol in extreme.index})
        split_dates, errors = _split_dates(affected, start or str(close.index[0].date()))
    else:
        errors = []

    tol = pd.Timedelta(days=tolerance_days)
    rows = []
    for (date, symbol), move in extreme.items():
        dates = split_dates.get(symbol, pd.DatetimeIndex([]))
        near = dates[(dates >= date - tol) & (dates <= date + tol)]
        if len(near):
            classification = "UNADJUSTED SPLIT"
            detail = f"split recorded {', '.join(str(d.date()) for d in near)}"
        else:
            classification = "market move"
            detail = "no split within ±%dd" % tolerance_days
        rows.append(
            {
                "date": date,
                "ticker": symbol,
                "return": float(move),
                "classification": classification,
                "detail": detail,
            }
        )

    events = pd.DataFrame(rows, columns=["date", "ticker", "return", "classification", "detail"])
    if not events.empty:
        events = events.sort_values("return").reset_index(drop=True)
    return CorporateActionAudit(
        events=events,
        threshold=threshold,
        n_observations=n_observations,
        splits_checked=int(sum(len(v) for v in split_dates.values())),
        tickers_checked=len(universe),
        fetch_errors=tuple(errors),
    )


@dataclass(frozen=True, slots=True)
class SplitReconciliation:
    table: pd.DataFrame
    threshold: float
    n_splits: int
    tickers_with_splits: int

    @property
    def suspicious(self) -> pd.DataFrame:
        return self.table[self.table["suspicious"]]

    @property
    def passes(self) -> bool:
        return self.suspicious.empty

    def __str__(self) -> str:
        return (
            f"split reconciliation: {self.n_splits} recorded actions across "
            f"{self.tickers_with_splits} tickers; {len(self.suspicious)} show an adjusted "
            f"move beyond ±{self.threshold:.0%} on the action date "
            f"({'PASS' if self.passes else 'FAIL'})"
        )


def split_reconciliation(
    prices: PriceData,
    universe,
    *,
    split_dates: dict[str, pd.DatetimeIndex] | None = None,
    start: str | None = None,
    threshold: float = 0.20,
) -> SplitReconciliation:
    universe = list(universe)
    close = prices.close[universe]
    if start is not None:
        close = close.loc[start:]
    returns = close.pct_change(fill_method=None)

    if split_dates is None:
        split_dates, _ = _split_dates(universe, start or str(close.index[0].date()))

    rows = []
    for symbol, dates in split_dates.items():
        for date in pd.DatetimeIndex(dates).normalize():
            if date < close.index[0] or date > close.index[-1]:
                continue
            position = close.index.searchsorted(date)
            if position >= len(close.index):
                continue
            bar = close.index[position]
            value = returns.loc[bar, symbol]
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "action_date": date,
                    "bar": bar,
                    "ticker": symbol,
                    "adjusted_return": float(value),
                    "suspicious": bool(abs(value) > threshold),
                }
            )

    table = pd.DataFrame(
        rows, columns=["action_date", "bar", "ticker", "adjusted_return", "suspicious"]
    )
    if not table.empty:
        table["suspicious"] = table["suspicious"].astype(bool)
        table = table.reindex(table["adjusted_return"].abs().sort_values(ascending=False).index)
        table = table.reset_index(drop=True)
    return SplitReconciliation(
        table=table,
        threshold=threshold,
        n_splits=len(table),
        tickers_with_splits=int(sum(1 for v in split_dates.values() if len(v))),
    )


def adjustment_convention(prices: PriceData) -> dict:
    return {
        "source": prices.source,
        "vendor_call": "yfinance.download(..., auto_adjust=True)",
        "convention": "retroactive back-adjustment for splits AND cash dividends",
        "point_in_time": False,
        "as_of": prices.fetched_at,
        "what_that_means": (
            "every price before a corporate action is multiplied by a factor derived "
            "from actions known at download time, so the series is the one an observer "
            "would reconstruct today, not the one a trader saw then"
        ),
    }


@dataclass(frozen=True, slots=True)
class RatioInvarianceReport:
    max_abs_momentum_difference: float
    n_compared: int
    split_factor: float
    split_offset_days: int
    detail: str

    @property
    def invariant(self) -> bool:
        return self.max_abs_momentum_difference < 1e-12

    def __str__(self) -> str:
        return (
            f"ratio invariance: applying a {self.split_factor:g}:1 back-adjustment for a "
            f"split {self.split_offset_days} bars in the FUTURE moves the momentum "
            f"signal by at most {self.max_abs_momentum_difference:.3e} across "
            f"{self.n_compared:,} values -> "
            f"{'INVARIANT' if self.invariant else 'NOT INVARIANT'}"
        )


def ratio_invariance_report(
    prices: pd.DataFrame,
    formation_days: int,
    skip_days: int,
    *,
    split_factor: float = 2.0,
    split_offset_days: int = 5,
) -> RatioInvarianceReport:
    from .xsmom import cross_sectional_momentum

    if split_offset_days < 1:
        raise ValueError("the injected split must be in the future")
    base = cross_sectional_momentum(prices, formation_days, skip_days)

    event_position = len(prices) - split_offset_days
    if event_position <= formation_days:
        raise ValueError("price history is too short to place a split after a full lookback")
    event_date = prices.index[event_position]

    adjusted = prices.copy()
    adjusted.iloc[:event_position] = adjusted.iloc[:event_position] / split_factor
    after = cross_sectional_momentum(adjusted, formation_days, skip_days)

    compare_to = prices.index[event_position - 1]
    a = base.loc[:compare_to].to_numpy()
    b = after.loc[:compare_to].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    difference = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0

    return RatioInvarianceReport(
        max_abs_momentum_difference=difference,
        n_compared=int(both.sum()),
        split_factor=split_factor,
        split_offset_days=split_offset_days,
        detail=(
            f"artificial {split_factor:g}:1 split placed at {event_date.date()}; every "
            f"momentum value dated on or before {compare_to.date()} compared"
        ),
    )


def write_snapshot_metadata(path: Path, source_url: str, n: int) -> dict:
    meta = {
        "source_url": source_url,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_constituents": int(n),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    Path(path).parent.joinpath(Path(path).stem + ".meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta
