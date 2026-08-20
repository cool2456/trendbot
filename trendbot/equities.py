"""Universe resolution and the corporate-action audit for an equity universe.

Experiments 001 and 002 ran on large, liquid ETFs, where the only data question worth
asking was "did the vendor publish a bar". A universe of individual companies raises
two that ETFs could not.

**Corporate actions.** An unadjusted 2-for-1 split reads as a -50% return. On a
*momentum* ranking that is not a small error: it drops the name straight into the
bottom decile and holds it there for a year. Spin-offs and large special dividends do
the same thing. :func:`corporate_action_audit` finds every extreme daily move in the
panel and classifies it against the vendor's own split and dividend records, so the
question "is this a market move or a broken adjustment" is answered per event rather
than assumed either way.

**Point-in-time adjustment.** A series adjusted with *today's* factors encodes future
corporate actions into past prices. That is a genuine lookahead and no ``.shift()``
catches it, because nothing about it is a shift.
:func:`adjustment_convention` states plainly what the vendor does, and
:func:`ratio_invariance_report` quantifies what it costs *this* signal, rather than
either ignoring it or waving at it.

The quantification matters and is not obvious, so it is spelled out here and
demonstrated numerically in ``tests/test_equities.py``:

    Back-adjustment multiplies every price strictly before an event by a constant
    factor ``f``. The signal is a **ratio** of two prices, ``P(t-21) / P(t-252)``.

    * An event *after* bar ``t`` scales both endpoints by the same ``f``, and the
      ratio is unchanged. Future splits and future dividends therefore cannot leak
      into a past signal value through this route.
    * An event *between* ``t-252`` and ``t-21`` scales only the older endpoint, which
      is precisely the correction that makes the ratio a true return.

    So for a ratio signal the back-adjusted series and a hypothetical point-in-time
    series produce **identical** momentum values. What back-adjustment does change is
    the price *level*, which this strategy never uses: it sizes by equal weight, not by
    share count.

    What that argument does **not** cover, and what is therefore reported as live
    exposure rather than dismissed: an event the vendor has *wrong or missing*, which
    the audit is for; and delisting/index-membership survivorship, which is
    PREREG_003.md section 2's problem and is not a data-adjustment question.
"""

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


# --------------------------------------------------------------------------------------
# the market proxy for the section 7.5 regression
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketProxy:
    """Which series section 7.5's regression calls "the market", and why.

    PREREG_003.md section 7.5 requires a regression on "market excess returns" and names
    no index, so the choice is a reporting decision the document leaves open. Section 8
    then turns the resulting alpha into both a support clause and an abandon clause,
    which makes it a decision worth recording rather than burying at a call site.

    It lives in a tracked JSON file next to the constituent snapshot for the same reason
    the snapshot does: it is *disclosed data about the experiment*, not logic, and
    keeping it out of the source is what lets ``tests/test_repo_invariants.py`` keep
    banning inlined tickers everywhere without an exception carved out for this one.
    """

    symbol: str
    role: str
    rationale: str
    alternative: str
    adjudicates_section_8: bool
    chosen_at: str

    def describe(self) -> str:
        return f"{self.symbol} — {self.role}"


def load_market_proxy(path: Path | str | None = None) -> MarketProxy:
    """Load the declared market proxy. No default symbol: absence is an error."""
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


# --------------------------------------------------------------------------------------
# the constituent snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstituentSnapshot:
    """A dated, hashed list of index members.

    PREREG_003.md section 2 defines the universe as a *rule* - "current S&P 500
    constituents" - which is a moving target. Pinning it to a file with a retrieval
    date and a digest is what makes the experiment reproducible: re-running it next
    month against a live index would silently be a different experiment.
    """

    table: pd.DataFrame
    source_url: str
    retrieved_at: str
    sha256: str
    path: Path

    @property
    def symbols(self) -> tuple[str, ...]:
        """Vendor-format symbols, in the order the snapshot lists them."""
        return tuple(self.table["yahoo_symbol"])

    @property
    def n(self) -> int:
        return len(self.table)

    def index_symbol(self, vendor_symbol: str) -> str:
        """The index vendor's spelling of a symbol, e.g. ``BRK-B`` -> ``BRK.B``."""
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
    """Load the pinned constituent list, verifying its recorded digest.

    A snapshot whose bytes no longer match the digest recorded beside it is refused:
    the universe is part of the experiment's definition, and an edited universe is a
    different experiment whatever the file is called.
    """
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


# --------------------------------------------------------------------------------------
# section 2 - universe resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityUniverse:
    """The concrete universe section 2's rule resolves to, and what it excluded."""

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
    """Turn section 2's rule into a ticker tuple, and record what it threw away.

    "Continuous daily data from ``required_from``" is read as **continuously listed**:
    the name's first bar is on or before the required date and its last bar reaches the
    end of the window. It is not read as "the vendor published a bar on every single
    session", which is a statement about the feed rather than about the company - the
    same reading experiment 002 used and disclosed. Per-ticker missing-bar counts are
    reported so the stricter reading can be applied to a number.
    """
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


# --------------------------------------------------------------------------------------
# section 5 - the corporate action audit
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorporateActionAudit:
    """Every extreme daily move in the panel, classified against vendor action records."""

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
        """Moves with no split within a day of them - i.e. market moves, not artefacts."""
        return self.events[self.events["classification"] == "market move"]

    @property
    def split_suspected(self) -> pd.DataFrame:
        """Moves coinciding with a split, which would mean the adjustment failed."""
        return self.events[self.events["classification"] == "UNADJUSTED SPLIT"]

    @property
    def passes(self) -> bool:
        """True when no extreme move coincides with a split the vendor knows about.

        A move that coincides with a known split is an adjustment failure and is fatal.
        A move with no split behind it is a market move; individual equities have those
        and no threshold can make them go away.
        """
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
    """Split dates per ticker, straight from the vendor's own action records.

    Cached, because fetching per-ticker action records for a 400-name universe is a
    few hundred round trips and the audit is run by every protocol command.
    """
    tickers = list(tickers)
    path = _split_cache_path(tickers, start)
    if cache and not refresh and path.is_file():
        payload = json.loads(path.read_text())
        return (
            {k: pd.DatetimeIndex(pd.to_datetime(v)) for k, v in payload["splits"].items()},
            list(payload["errors"]),
        )

    import yfinance  # noqa: PLC0415  (research-only dependency)

    out: dict[str, pd.DatetimeIndex] = {}
    errors: list[str] = []
    for symbol in tickers:
        try:
            splits = yfinance.Ticker(symbol).splits
        except Exception as exc:  # noqa: BLE001 - the vendor, not our logic
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        if splits is None or len(splits) == 0:
            out[symbol] = pd.DatetimeIndex([])
            continue
        index = pd.DatetimeIndex(splits.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        # The vendor stamps actions at the 09:30 open; the price index is midnight-
        # normalised. Without this normalise() every action sorts *after* its own bar,
        # and any check that snaps "to the first bar on or after the action" silently
        # inspects the following session - which makes the whole reconciliation pass
        # vacuously. This bit us once; ``tests/test_equities.py`` pins it.
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
    """Find every daily move beyond ``threshold`` and reconcile it against splits.

    PREREG_003.md section 5 requires this before any result is trusted, and the build
    order gates everything downstream on it. The classification is deliberately
    conservative in the direction that matters: a move is called ``UNADJUSTED SPLIT``
    whenever a split is recorded within ``tolerance_days`` of it *and* the move's
    direction matches what an unapplied split of that size would look like. Everything
    else is a market move, which single stocks genuinely have.

    ``split_dates`` may be injected so the audit can be tested offline; when omitted it
    is fetched from the vendor's action records.
    """
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


# --------------------------------------------------------------------------------------
# the stronger audit: every recorded action, not every large move
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitReconciliation:
    """Adjusted return on **every** recorded split date, whatever its size.

    The ±35% scan the build order specifies is a *return* filter, and it therefore
    only finds a broken adjustment big enough to clear the threshold. A 5-for-4 split
    that failed to adjust is a −20% return: badly wrong, invisible to a 35% scan, and
    quite enough to move a name several deciles.

    This pass inverts the question. Instead of asking "which large moves have an action
    behind them", it asks "what did the series do on every date an action is recorded",
    which is the complete test. On a correctly adjusted series a split date is an
    ordinary trading day and the return is an ordinary market return.
    """

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
    """Check the adjusted return on every recorded corporate action date."""
    universe = list(universe)
    close = prices.close[universe]
    if start is not None:
        close = close.loc[start:]
    returns = close.pct_change(fill_method=None)

    if split_dates is None:
        split_dates, _ = _split_dates(universe, start or str(close.index[0].date()))

    rows = []
    for symbol, dates in split_dates.items():
        # Normalised defensively as well as at the source: an action stamped 09:30
        # against a midnight-normalised price index sorts after its own bar, and the
        # snap below would then inspect the following session instead of the ex-date.
        for date in pd.DatetimeIndex(dates).normalize():
            if date < close.index[0] or date > close.index[-1]:
                continue
            # Snap to the first trading bar on or after the ex-date: the ex-date is
            # normally a session, but a holiday in the feed must not skip the check.
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


# --------------------------------------------------------------------------------------
# section 5 - the point-in-time question
# --------------------------------------------------------------------------------------


def adjustment_convention(prices: PriceData) -> dict:
    """State what the vendor's adjustment actually is. No inference, no reassurance."""
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
    """How much a non-point-in-time adjustment can move THIS signal. See module docs."""

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
    """Quantify the exposure the module docstring argues is zero, rather than asserting it.

    Takes a real price panel, injects a *future* split into it by back-adjusting every
    price before an artificial event date, and measures how far the momentum signal
    moves at every date **before** that event. The claim under test is that it does not
    move at all, because a constant factor applied to both endpoints of a ratio cancels.

    Reporting a measured number instead of an argument is the point: if the signal ever
    stops being a pure ratio, this stops returning zero.
    """
    from .xsmom import cross_sectional_momentum  # local import: keeps xsmom dependency-free

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

    # Only dates strictly before the event are covered by the invariance claim; on and
    # after it the two series legitimately differ, because one of them has had a split.
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
    """Record a snapshot's provenance beside it. Used by the fetch script."""
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
