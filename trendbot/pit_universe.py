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

BANKRUPT_PRICE_FLOOR = 1e-8


@dataclass(frozen=True, slots=True)
class Delisting:
    permaticker: int
    date: pd.Timestamp
    bucket: str
    raw_reason: str


@dataclass(frozen=True, slots=True)
class SecurityMaster:
    table: pd.DataFrame
    delistings: Mapping[int, Delisting]
    ticker_history: Mapping[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def permatickers(self) -> tuple[int, ...]:
        return tuple(self.table.index)

    def label(self, permaticker: int) -> str:
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
        delist = self.delist_date(permaticker)
        return delist is None or date < delist

    def survives_to(self, date: pd.Timestamp) -> set[int]:
        return {
            int(p)
            for p in self.table.index
            if self.delist_date(int(p)) is None or self.delist_date(int(p)) > date
        }

    def reused_tickers(self) -> pd.DataFrame:
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


def resolve_permatickers(sep: pd.DataFrame, master: SecurityMaster) -> pd.DataFrame:
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
    dollar_volume = closeunadj * volume

    return {
        "closeadj": closeadj,
        "openadj": openadj,
        "closeunadj": closeunadj,
        "dollar_volume": dollar_volume,
        "volume": volume,
    }


@dataclass(frozen=True, slots=True)
class PointInTimeUniverse:
    membership: pd.DataFrame
    counts: pd.Series
    start_date: pd.Timestamp | None
    rebalances: tuple[pd.Timestamp, ...]
    cfg_universe_size: int

    @property
    def qualifying_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.counts.index[self.counts >= self.cfg_universe_size])

    def members(self, date: pd.Timestamp) -> tuple[int, ...]:
        row = self.membership.loc[date]
        return tuple(int(c) for c in self.membership.columns[row.to_numpy()])

    def entries_and_exits(self) -> pd.DataFrame:
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

    median_dv = dollar_volume.rolling(cfg.liquidity_window_days, min_periods=cfg.liquidity_window_days).median()
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


def build_panels(
    cfg: Config004,
    *,
    closeadj: pd.DataFrame,
    openadj: pd.DataFrame,
    master: SecurityMaster,
    rf_daily: pd.Series | None = None,
    unknown_override: float | None = None,
) -> tuple[PriceData, pd.DataFrame]:
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

        event_i = last_i + 1
        if event_i >= len(index):
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
    held = weights.abs() > 1e-12
    close = panels.close
    rows: list[dict] = []
    held_v = held.to_numpy()
    for j, column in enumerate(weights.columns):
        column_held = held_v[:, j]
        if not column_held.any():
            continue
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
