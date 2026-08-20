"""Point-in-time universe construction and delisting treatment.

This is the module experiment 004 exists for. Experiments 001-003 all took a fixed
list of tickers that survive to today and ran a strategy over it. Section 2 of
PREREG_004.md instead specifies a **rule evaluated at every rebalance date over a
population that includes securities which no longer exist**, and section 3 specifies
what happens to a position when one of them stops existing.

Three failure modes this module is built around
------------------------------------------------

**Everything is keyed on the permanent security ID, never the ticker.** Tickers are
recycled and reassigned. The panel's columns are permatickers rendered as strings;
:meth:`SecurityMaster.label` is the only place a ticker is used, and only for display.

**A delisting is not a missing price.** Forward-filling a delisted security holds its
last price forever and assigns it a 0% return; dropping it silently removes a loss from
the record; treating the gap as zero return does the same. All three flatter the
strategy, because delistings are not symmetric - a stock that stops trading has usually
stopped for a bad reason. Section 3's table is implemented in :func:`build_panels`, and
:func:`audit_position_exits` asserts that no position ever leaves the book without a
return having been assigned to it.

**The universe rule must not see the future.** The trailing-60-day median dollar volume
at date ``t`` is computed on data through ``t`` and no further; the $5.00 floor is
applied to the *unadjusted* close at ``t``, not to a back-adjusted price that encodes
later splits. Both are enforced by construction here and asserted in the tests.

How the delisting return reaches the engine
--------------------------------------------
The engine marks positions from a price panel and knows nothing about corporate events,
and it is not going to be rewritten to. So the delisting return is expressed *in the
panel*: on the delist date a security's price is set to ``final_price * (1 + assigned)``
and from then on it accrues at the risk-free rate, which is section 5's treatment of
cash. A position in a delisted name therefore earns exactly the assigned return on the
delist date and the cash rate afterwards, which is section 3's "proceeds to cash for
remainder of month" applied to all three buckets rather than only to acquisitions.

The bankruptcy case wants a price of exactly zero, which the engine cannot represent
(it divides by prices). A floor of ``BANKRUPT_PRICE_FLOOR`` is used instead, chosen so
the realised return is -100% to twelve decimal places, and
:func:`verify_delisting_returns` asserts the realised return against the assigned one
rather than trusting the construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config_004 import UNKNOWN, Config004
from .data import PriceData

__all__ = [
    "SecurityMaster",
    "Delisting",
    "PointInTimeUniverse",
    "build_security_master",
    "build_pit_universe",
    "build_panels",
    "resolve_permatickers",
    "assemble_wide_panels",
    "verify_delisting_returns",
    "audit_position_exits",
    "BANKRUPT_PRICE_FLOOR",
]

# Small enough that the realised return on a bankruptcy is -1.0 to ~1e-12 for any
# plausible share price, large enough to stay far from denormal arithmetic.
BANKRUPT_PRICE_FLOOR = 1e-8


# --------------------------------------------------------------------------------------
# the security master
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Delisting:
    """One security's exit from the market, with section 3's bucket already assigned."""

    permaticker: int
    date: pd.Timestamp
    bucket: str  # acquisition | bankruptcy | unknown
    raw_reason: str


@dataclass(frozen=True, slots=True)
class SecurityMaster:
    """Metadata for every security, keyed by permanent ID.

    ``table`` is indexed by ``permaticker`` and carries ``ticker`` (the *latest* one, a
    label only), ``category``, ``isdelisted``, ``firstpricedate`` and ``lastpricedate``.
    ``delistings`` maps a permaticker to its :class:`Delisting`.
    """

    table: pd.DataFrame
    delistings: Mapping[int, Delisting]
    ticker_history: Mapping[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def permatickers(self) -> tuple[int, ...]:
        return tuple(self.table.index)

    def label(self, permaticker: int) -> str:
        """A human-readable label. Display only - nothing may key on this."""
        if permaticker not in self.table.index:
            return str(permaticker)
        row = self.table.loc[permaticker]
        return f"{row['ticker']}"

    def is_common_stock(self, permaticker: int) -> bool:
        from .sharadar import COMMON_STOCK_CATEGORIES  # noqa: PLC0415

        if permaticker not in self.table.index:
            return False
        return str(self.table.loc[permaticker, "category"]) in COMMON_STOCK_CATEGORIES

    def delist_date(self, permaticker: int) -> pd.Timestamp | None:
        delisting = self.delistings.get(int(permaticker))
        return None if delisting is None else delisting.date

    def tradeable_on(self, permaticker: int, date: pd.Timestamp) -> bool:
        """Section 2's "actually tradeable on date t (not yet delisted)"."""
        delist = self.delist_date(permaticker)
        return delist is None or date < delist

    def survives_to(self, date: pd.Timestamp) -> set[int]:
        """The permatickers still listed as of ``date`` - the (B) arm of section 10."""
        return {
            int(p)
            for p in self.table.index
            if self.delist_date(int(p)) is None or self.delist_date(int(p)) > date
        }

    def reused_tickers(self) -> pd.DataFrame:
        """Ticker strings carried by more than one security. Reported, never merged."""
        counts = self.table.groupby("ticker").size()
        shared = counts[counts > 1]
        if shared.empty:
            return pd.DataFrame(columns=["ticker", "permatickers", "n"])
        rows = []
        for ticker in shared.index:
            members = sorted(int(p) for p in self.table.index[self.table["ticker"] == ticker])
            rows.append({"ticker": ticker, "permatickers": members, "n": len(members)})
        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def build_security_master(
    tickers_table: pd.DataFrame,
    actions_table: pd.DataFrame | None = None,
) -> SecurityMaster:
    """Assemble the security master and the delisting map from vendor tables.

    ``actions_table`` supplies the *reason* for each delisting. Where a security is
    flagged delisted in the master but has no matching action row, the reason is
    ``unknown``, which section 3 haircuts - the conservative direction. Its date is then
    the vendor's last price date, because a security with no price after date *d* was
    not tradeable after *d* whatever the actions table says.
    """
    from .sharadar import classify_delist_reason  # noqa: PLC0415

    table = tickers_table.copy()
    if "permaticker" in table.columns:
        table["permaticker"] = table["permaticker"].astype("int64")
        table = table.drop_duplicates(subset=["permaticker"], keep="first").set_index("permaticker")
    table = table.sort_index()

    ticker_history: dict[int, tuple[str, ...]] = {
        int(p): (str(table.loc[p, "ticker"]),) for p in table.index
    }

    reasons: dict[int, tuple[pd.Timestamp, str]] = {}
    if actions_table is not None and len(actions_table):
        actions = actions_table.copy()
        actions["date"] = pd.to_datetime(actions["date"], errors="coerce")
        if "permaticker" in actions.columns:
            actions = actions[actions["permaticker"].notna()].copy()
            actions["permaticker"] = actions["permaticker"].astype("int64")
        else:
            # No permaticker column: fall back to the latest-ticker map. This is the one
            # place a ticker string is used to join, and it is why the fallback is
            # reported rather than silent - a reused ticker could attach the wrong
            # reason to the wrong security.
            lookup = {str(table.loc[p, "ticker"]): int(p) for p in table.index}
            actions["permaticker"] = actions["ticker"].map(lookup)
            actions = actions[actions["permaticker"].notna()].copy()
            actions["permaticker"] = actions["permaticker"].astype("int64")

        delist_like = actions[
            actions["action"].astype(str).str.contains("delist|bankrupt|liquidat|acquisitionby|merger", case=False, na=False)
        ]
        for permaticker, group in delist_like.groupby("permaticker"):
            group = group.sort_values("date")
            row = group.iloc[-1]
            reasons[int(permaticker)] = (row["date"], str(row["action"]))

        # Ticker changes, recorded so the master can report them even though nothing
        # keys on a ticker.
        changes = actions[
            actions["action"].astype(str).str.contains("tickerchange", case=False, na=False)
        ]
        for permaticker, group in changes.groupby("permaticker"):
            names = tuple(dict.fromkeys(str(t) for t in group["ticker"]))
            base = ticker_history.get(int(permaticker), ())
            ticker_history[int(permaticker)] = tuple(dict.fromkeys(names + base))

    delistings: dict[int, Delisting] = {}
    for permaticker in table.index:
        permaticker = int(permaticker)
        row = table.loc[permaticker]
        if not bool(row.get("isdelisted", False)):
            continue
        recorded = reasons.get(permaticker)
        last_price = pd.to_datetime(row.get("lastpricedate"), errors="coerce")
        if recorded is not None and pd.notna(recorded[0]):
            date, raw = recorded
            bucket = classify_delist_reason(raw)
        else:
            date, raw, bucket = last_price, "", UNKNOWN
        if pd.isna(date):
            date = last_price
        if pd.isna(date):
            continue
        # A security stops being tradeable when its prices stop, whatever the actions
        # table says. Taking the earlier of the two keeps the universe rule honest.
        if pd.notna(last_price):
            date = min(pd.Timestamp(date), pd.Timestamp(last_price))
        delistings[permaticker] = Delisting(
            permaticker=permaticker,
            date=pd.Timestamp(date),
            bucket=bucket,
            raw_reason=raw,
        )

    return SecurityMaster(
        table=table, delistings=delistings, ticker_history=ticker_history
    )


# --------------------------------------------------------------------------------------
# turning the vendor's long format into wide panels, keyed by permanent ID
# --------------------------------------------------------------------------------------


def resolve_permatickers(sep: pd.DataFrame, master: SecurityMaster) -> pd.DataFrame:
    """Attach a permanent security ID to every price row. **This is the reuse defence.**

    The vendor's price table is keyed by ``(ticker, date)``, and a ticker string is not
    a security: ``ZZZ`` can be one company until 2011 and a different one from 2015.
    Joining on the ticker alone would concatenate the two into a single price series
    with a gap in the middle, and a twelve-month momentum computed across that gap is a
    number about no company at all.

    The join is therefore on ``(ticker, date)`` against each candidate security's own
    ``[firstpricedate, lastpricedate]`` window. A row whose date falls in no candidate's
    window, or in more than one, is **dropped and counted** rather than assigned to a
    guess - the returned frame carries the count on ``.attrs`` so the caller reports it.

    The join is vectorised rather than looped: a real price table is tens of millions of
    rows and a per-row Python loop over it is an overnight job.
    """
    table = master.table
    windows = pd.DataFrame(
        {
            "ticker": table["ticker"].astype(str).to_numpy(),
            "permaticker": [int(p) for p in table.index],
            "first": pd.to_datetime(table.get("firstpricedate"), errors="coerce")
            .fillna(pd.Timestamp.min)
            .to_numpy(),
            "last": pd.to_datetime(table.get("lastpricedate"), errors="coerce")
            .fillna(pd.Timestamp.max)
            .to_numpy(),
        }
    )
    reused = int((windows.groupby("ticker").size() > 1).sum())

    frame = sep.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_row"] = np.arange(len(frame), dtype="int64")

    merged = frame[["_row", "ticker", "date"]].merge(windows, on="ticker", how="left")
    # A ticker carried by one security matches unconditionally; a ticker carried by
    # several is resolved by which security's price window contains the date. Both are
    # the same expression, because a sole candidate's window still has to contain it.
    inside = merged["permaticker"].notna() & (merged["date"] >= merged["first"]) & (
        merged["date"] <= merged["last"]
    )
    hits = merged[inside]

    per_row = hits.groupby("_row")["permaticker"].agg(["first", "size"])
    unique = per_row[per_row["size"] == 1]
    ambiguous = int((per_row["size"] > 1).sum())

    frame = frame.set_index("_row")
    frame["permaticker"] = unique["first"].astype("int64")
    out = frame[frame["permaticker"].notna()].copy()
    out["permaticker"] = out["permaticker"].astype("int64")
    out = out.reset_index(drop=True)

    out.attrs["unmatched_rows"] = int(len(frame) - len(per_row))
    out.attrs["ambiguous_rows"] = ambiguous
    out.attrs["reused_tickers"] = reused
    return out


def assemble_wide_panels(
    sep: pd.DataFrame, *, calendar: pd.DatetimeIndex | None = None
) -> dict[str, pd.DataFrame]:
    """Pivot resolved price rows into the wide panels the rest of the pipeline uses.

    Columns are permatickers rendered as strings, in ascending numeric order, so that
    every panel shares one column order and a positional operation cannot silently
    misalign two of them.

    The dividend-adjusted **open** is derived rather than fetched: the vendor supplies
    a split-adjusted open and both a split-adjusted and a split-and-dividend-adjusted
    close, so the day's dividend adjustment factor is ``closeadj / close`` and applying
    it to the open puts the two on the same basis. Section 5 requires the strategy to
    trade a split-and-dividend-adjusted series; trading a split-only open against a
    split-and-dividend close would leak the dividend into the execution price.
    """
    required = {"permaticker", "date", "open", "close", "closeadj", "closeunadj", "volume"}
    missing = required - set(sep.columns)
    if missing:
        raise ValueError(f"price rows are missing columns {sorted(missing)}")

    frame = sep.copy()
    frame["key"] = frame["permaticker"].astype("int64").astype(str)
    if frame.duplicated(subset=["key", "date"]).any():
        n = int(frame.duplicated(subset=["key", "date"]).sum())
        raise ValueError(
            f"{n} duplicate (security, date) rows; a pivot would silently pick one of "
            "each and the panel would stop being reproducible"
        )

    def pivot(column: str) -> pd.DataFrame:
        wide = frame.pivot(index="date", columns="key", values=column)
        wide.index = pd.DatetimeIndex(wide.index)
        return wide.sort_index()

    closeadj = pivot("closeadj")
    close = pivot("close")
    open_ = pivot("open")
    closeunadj = pivot("closeunadj")
    volume = pivot("volume")

    order = sorted(closeadj.columns, key=int)
    closeadj, close, open_, closeunadj, volume = (
        f.reindex(columns=order) for f in (closeadj, close, open_, closeunadj, volume)
    )
    if calendar is not None:
        index = pd.DatetimeIndex(calendar)
        closeadj, close, open_, closeunadj, volume = (
            f.reindex(index=index) for f in (closeadj, close, open_, closeunadj, volume)
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        dividend_factor = closeadj / close.where(close > 0)
    openadj = open_ * dividend_factor
    # Dollar volume is a *traded* quantity, so it uses the unadjusted close and the
    # as-reported share volume. Using adjusted prices here would restate a 2009 stock's
    # liquidity with today's split factors and change which names the rule picks.
    dollar_volume = closeunadj * volume

    return {
        "closeadj": closeadj,
        "openadj": openadj,
        "closeunadj": closeunadj,
        "dollar_volume": dollar_volume,
        "volume": volume,
    }


# --------------------------------------------------------------------------------------
# section 2 - the point-in-time universe rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointInTimeUniverse:
    """Section 2's rule, evaluated at every rebalance date."""

    membership: pd.DataFrame  # rebalance date x permaticker, boolean
    counts: pd.Series  # eligible names per rebalance date
    start_date: pd.Timestamp | None  # section 6's realised start
    rebalances: tuple[pd.Timestamp, ...]
    cfg_universe_size: int

    @property
    def qualifying_dates(self) -> pd.DatetimeIndex:
        """Dates at which the rule yields at least the required name count."""
        return pd.DatetimeIndex(self.counts.index[self.counts >= self.cfg_universe_size])

    def members(self, date: pd.Timestamp) -> tuple[int, ...]:
        row = self.membership.loc[date]
        return tuple(int(c) for c in self.membership.columns[row.to_numpy()])

    def entries_and_exits(self) -> pd.DataFrame:
        """How many names join and leave the universe at each rebalance."""
        member = self.membership.to_numpy()
        rows = []
        for i, date in enumerate(self.membership.index):
            if i == 0:
                rows.append({"date": date, "n": int(member[i].sum()), "entered": int(member[i].sum()), "left": 0})
                continue
            entered = int((member[i] & ~member[i - 1]).sum())
            left = int((~member[i] & member[i - 1]).sum())
            rows.append({"date": date, "n": int(member[i].sum()), "entered": entered, "left": left})
        return pd.DataFrame(rows).set_index("date")

    def all_members(self) -> tuple[int, ...]:
        """Every security that is in the universe on at least one date."""
        ever = self.membership.any(axis=0)
        return tuple(int(c) for c in self.membership.columns[ever.to_numpy()])


def build_pit_universe(
    cfg: Config004,
    *,
    closeunadj: pd.DataFrame,
    closeadj: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    master: SecurityMaster,
    rebalances: Sequence[pd.Timestamp],
    eligible_ids: Sequence[int] | None = None,
    restrict_to: Sequence[int] | None = None,
) -> PointInTimeUniverse:
    """Evaluate section 2's rule at each rebalance date.

    All three frames are indexed by trading date and columned by permaticker (as int).
    Every filter reads data at or before the rebalance date and nothing after it.

    ``restrict_to`` is section 10's (B) arm: the identical rule confined to a subset of
    securities. It is applied *before* the top-N ranking, which is what makes (B) a
    survivors-only universe rather than a survivors-only slice of arm (A)'s universe.
    """
    columns = list(closeunadj.columns)
    if list(closeadj.columns) != columns or list(dollar_volume.columns) != columns:
        raise ValueError("the three panels must share one column order of permatickers")

    ids = np.array([int(c) for c in columns])
    allowed = None
    if eligible_ids is not None:
        allowed = np.isin(ids, np.array(sorted({int(i) for i in eligible_ids}), dtype=ids.dtype))
    if restrict_to is not None:
        subset = np.isin(ids, np.array(sorted({int(i) for i in restrict_to}), dtype=ids.dtype))
        allowed = subset if allowed is None else (allowed & subset)
    if allowed is None:
        allowed = np.ones(len(ids), dtype=bool)

    # Trailing median dollar volume. min_periods equal to the window means a security
    # without a full window of history is not ranked at all, rather than being ranked
    # on a handful of days.
    median_dv = dollar_volume.rolling(cfg.liquidity_window_days, min_periods=cfg.liquidity_window_days).median()
    # History: count of observed closes up to and including each date.
    history = closeadj.notna().cumsum()

    delist_positions = np.full(len(ids), np.datetime64("NaT", "ns"))
    for position, permaticker in enumerate(ids):
        date = master.delist_date(int(permaticker))
        if date is not None:
            delist_positions[position] = np.datetime64(pd.Timestamp(date), "ns")

    rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for date in rebalances:
        if date not in closeunadj.index:
            continue
        floor_ok = closeunadj.loc[date].to_numpy(dtype=float) >= cfg.price_floor
        history_ok = history.loc[date].to_numpy() >= cfg.min_history_days
        tradeable = np.isnat(delist_positions) | (np.datetime64(pd.Timestamp(date), "ns") < delist_positions)
        liquidity = median_dv.loc[date].to_numpy(dtype=float)
        liquid_ok = np.isfinite(liquidity)

        candidate = allowed & floor_ok & history_ok & tradeable & liquid_ok
        selected = np.zeros(len(ids), dtype=bool)
        if candidate.any():
            scores = np.where(candidate, liquidity, -np.inf)
            n_take = min(cfg.universe_size, int(candidate.sum()))
            # argsort on the negated score gives descending order; ties break by column
            # position, which is permaticker order and therefore deterministic.
            order = np.argsort(-scores, kind="stable")[:n_take]
            selected[order] = True
        rows.append(selected)
        dates.append(pd.Timestamp(date))

    membership = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="rebalance"), columns=columns)
    counts = membership.sum(axis=1).rename("n_eligible")
    qualifying = counts.index[counts >= cfg.universe_size]
    start = pd.Timestamp(qualifying[0]) if len(qualifying) else None
    return PointInTimeUniverse(
        membership=membership,
        counts=counts,
        start_date=start,
        rebalances=tuple(pd.Timestamp(d) for d in dates),
        cfg_universe_size=cfg.universe_size,
    )


# --------------------------------------------------------------------------------------
# section 3 - the delisting treatment, expressed in the price panel
# --------------------------------------------------------------------------------------


def build_panels(
    cfg: Config004,
    *,
    closeadj: pd.DataFrame,
    openadj: pd.DataFrame,
    master: SecurityMaster,
    rf_daily: pd.Series | None = None,
    unknown_override: float | None = None,
) -> tuple[PriceData, pd.DataFrame]:
    """Build the close/open panels with section 3's delisting returns baked in.

    Returns the :class:`~trendbot.data.PriceData` the engine consumes, plus a frame
    recording, per delisted security, the assigned return and the date it was applied,
    so the treatment can be audited instead of trusted.

    On the delist date the price becomes ``last_price * (1 + assigned_return)``; from
    the next bar it accrues at the risk-free rate, which is section 5's cash treatment.
    Before the delist date nothing is touched.
    """
    close = closeadj.copy()
    open_ = openadj.copy()
    index = close.index
    if not index.equals(open_.index) or list(close.columns) != list(open_.columns):
        raise ValueError("close and open panels must share an index and column order")

    if rf_daily is None:
        accrual = pd.Series(0.0, index=index)
    else:
        accrual = rf_daily.reindex(index).fillna(0.0)
    growth = (1.0 + accrual.to_numpy(dtype=float)).cumprod()

    close_v = close.to_numpy(dtype=float).copy()
    open_v = open_.to_numpy(dtype=float).copy()
    positions = {d: i for i, d in enumerate(index)}

    records: list[dict] = []
    for permaticker, delisting in master.delistings.items():
        column = str(permaticker) if str(permaticker) in close.columns else permaticker
        if column not in close.columns:
            continue
        j = close.columns.get_loc(column)
        series = close_v[:, j]
        # The final traded bar is the last observed price **on or before the delist
        # date**, not simply the last observed price. Those differ when the vendor's
        # price table runs past the recorded delisting, and taking the panel's word for
        # it would let a security keep trading after it stopped existing. Bars after the
        # event are overwritten below, which neutralises any such stray prices rather
        # than leaving them where a signal could read them.
        on_or_before = np.flatnonzero(
            np.isfinite(series) & (index.to_numpy() <= np.datetime64(delisting.date, "ns"))
        )
        observed = on_or_before if on_or_before.size else np.flatnonzero(np.isfinite(series))
        if observed.size == 0:
            continue
        last_i = int(observed[-1])
        last_price = float(series[last_i])
        if not np.isfinite(last_price) or last_price <= 0:
            continue

        assigned = cfg.delisting.assigned_return(
            delisting.bucket, unknown_override=unknown_override
        )
        terminal = max(last_price * (1.0 + assigned), BANKRUPT_PRICE_FLOOR)

        # The delisting return lands on the first bar AFTER the final traded bar. That
        # bar is the one on which the position stops existing, so it is the bar that
        # must carry the loss; putting it on the final traded bar would overwrite a
        # real observed price with a synthetic one.
        event_i = last_i + 1
        if event_i >= len(index):
            # The security's last price is the last bar of the sample. There is no bar
            # on which to realise the delisting, and no position can survive past the
            # end of the sample either, so nothing is assigned.
            continue

        close_v[event_i:, j] = terminal * (growth[event_i:] / growth[event_i])
        open_v[event_i:, j] = close_v[event_i:, j]
        records.append(
            {
                "permaticker": permaticker,
                "ticker": master.label(permaticker),
                "bucket": delisting.bucket,
                "raw_reason": delisting.raw_reason,
                "delist_date": delisting.date,
                "last_traded_bar": index[last_i],
                "event_bar": index[event_i],
                "last_price": last_price,
                "assigned_return": assigned,
                "terminal_price": terminal,
            }
        )

    panels = PriceData(
        open=pd.DataFrame(open_v, index=index, columns=close.columns),
        close=pd.DataFrame(close_v, index=index, columns=close.columns),
        source="sharadar-pit",
        adjusted=True,
        fetched_at="",
    )
    return panels, pd.DataFrame(records)


def verify_delisting_returns(
    panels: PriceData, applied: pd.DataFrame, *, tolerance: float = 1e-9
) -> pd.DataFrame:
    """Assert the realised return on each delist bar equals the assigned one.

    Constructing the panel correctly and *checking* the panel are different claims.
    This measures the return the engine will actually see and compares it against
    section 3's table, which is the only way to know the treatment was applied rather
    than merely intended.
    """
    if applied.empty:
        return applied.assign(realised_return=pd.Series(dtype=float), agrees=pd.Series(dtype=bool))
    close = panels.close
    realised = []
    for _, row in applied.iterrows():
        column = row["permaticker"] if row["permaticker"] in close.columns else str(row["permaticker"])
        series = close[column]
        before = float(series.loc[row["last_traded_bar"]])
        after = float(series.loc[row["event_bar"]])
        realised.append(after / before - 1.0 if before > 0 else np.nan)
    out = applied.copy()
    out["realised_return"] = realised
    out["agrees"] = (out["realised_return"] - out["assigned_return"]).abs() <= tolerance
    return out


def audit_position_exits(
    weights: pd.DataFrame,
    panels: PriceData,
    master: SecurityMaster,
) -> pd.DataFrame:
    """Every position that left the book, and how its exit was priced.

    Section 3's requirement, made checkable: a held position must never disappear
    without a return. An exit is either a *trade* (the name was sold at a market price
    on a rebalance) or a *delisting* (the name stopped trading and section 3 assigned
    it a return). Anything else - a position whose price simply became NaN while it was
    held - is the silent drop this audit exists to catch.
    """
    held = weights.abs() > 1e-12
    close = panels.close
    rows: list[dict] = []
    held_v = held.to_numpy()
    for j, column in enumerate(weights.columns):
        column_held = held_v[:, j]
        if not column_held.any():
            continue
        # positions where the name was held on the previous bar and not on this one
        exits = np.flatnonzero((~column_held[1:]) & column_held[:-1]) + 1
        for i in exits:
            date = weights.index[i]
            price_now = close.iloc[i][column]
            price_before = close.iloc[i - 1][column]
            permaticker = int(column)
            delist = master.delist_date(permaticker)
            rows.append(
                {
                    "permaticker": permaticker,
                    "ticker": master.label(permaticker),
                    "exit_date": date,
                    "priced": bool(np.isfinite(price_now) and np.isfinite(price_before)),
                    "delisted": delist is not None and pd.Timestamp(delist) <= date,
                }
            )
    return pd.DataFrame(rows)
