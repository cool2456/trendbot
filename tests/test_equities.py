from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trendbot.data import PriceData
from trendbot.equities import (
    adjustment_convention,
    corporate_action_audit,
    load_constituent_snapshot,
    ratio_invariance_report,
    resolve_universe,
    split_reconciliation,
)
from trendbot.xsmom import cross_sectional_momentum


def _snapshot(tmp_path, symbols):
    frame = pd.DataFrame(
        {
            "symbol": list(symbols),
            "security": [f"{s} Inc" for s in symbols],
            "gics_sector": ["Industrials"] * len(symbols),
            "date_added": ["1990-01-01"] * len(symbols),
            "cik": range(len(symbols)),
            "yahoo_symbol": list(symbols),
        }
    )
    path = tmp_path / "sp500_constituents_2020-01-01.csv"
    frame.to_csv(path, index=False)
    (tmp_path / "sp500_constituents_2020-01-01.meta.json").write_text(
        json.dumps(
            {
                "source_url": "test",
                "retrieved_at": "2020-01-01T00:00:00+00:00",
                "n_constituents": len(symbols),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    return path


def _panel(symbols, periods=600, start="2018-01-01") -> PriceData:
    index = pd.bdate_range(start, periods=periods)
    frame = pd.DataFrame(100.0, index=index, columns=list(symbols))
    return PriceData(
        open=frame.copy(), close=frame.copy(), source="test", adjusted=True, fetched_at=""
    )


def test_a_snapshot_whose_bytes_changed_is_refused(tmp_path):
    path = _snapshot(tmp_path, ["AAA", "BBB"])
    assert load_constituent_snapshot(path).n == 2

    path.write_text(path.read_text() + "CCC,CCC Inc,Industrials,1990-01-01,99,CCC\n")
    with pytest.raises(ValueError, match="does not match the digest"):
        load_constituent_snapshot(path)


def test_a_snapshot_with_a_duplicate_symbol_is_refused(tmp_path):
    path = _snapshot(tmp_path, ["AAA", "AAA"])
    meta = tmp_path / "sp500_constituents_2020-01-01.meta.json"
    payload = json.loads(meta.read_text())
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate symbol"):
        load_constituent_snapshot(path)


def test_a_name_starting_after_the_required_date_is_excluded_and_named(tmp_path):
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols)
    close = prices.close.copy()
    required = close.index[100]
    close.loc[: close.index[150], "BBB"] = np.nan
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )

    universe = resolve_universe(
        prices, load_constituent_snapshot(_snapshot(tmp_path, symbols)),
        required_from=str(required.date()),
    )
    assert universe.universe == ("AAA", "CCC")
    assert list(universe.excluded.index) == ["BBB"]
    assert "is after" in universe.excluded.loc["BBB", "reason"]
    assert universe.n == 2


def test_a_name_that_stops_trading_before_the_window_ends_is_excluded(tmp_path):
    symbols = ["AAA", "BBB"]
    prices = _panel(symbols)
    close = prices.close.copy()
    close.loc[close.index[400] :, "BBB"] = np.nan
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )
    universe = resolve_universe(
        prices, load_constituent_snapshot(_snapshot(tmp_path, symbols)),
        required_from=str(prices.close.index[10].date()),
    )
    assert universe.universe == ("AAA",)
    assert "is before" in universe.excluded.loc["BBB", "reason"]


def test_a_vendor_hole_does_not_cost_a_name_its_place(tmp_path):
    symbols = ["AAA", "BBB"]
    prices = _panel(symbols)
    close = prices.close.copy()
    close.iloc[300, close.columns.get_loc("BBB")] = np.nan
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )
    universe = resolve_universe(
        prices, load_constituent_snapshot(_snapshot(tmp_path, symbols)),
        required_from=str(prices.close.index[10].date()),
    )
    assert set(universe.universe) == {"AAA", "BBB"}
    assert universe.with_holes.to_dict() == {"BBB": 1}


def test_an_unadjusted_split_is_classified_as_one_not_as_a_market_move():
    symbols = ["AAA", "BBB"]
    prices = _panel(symbols)
    close = prices.close.copy()
    event = close.index[300]
    close.loc[event:, "AAA"] = 50.0
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )

    audit = corporate_action_audit(
        prices, symbols, threshold=0.35, split_dates={"AAA": pd.DatetimeIndex([event]), "BBB": pd.DatetimeIndex([])}
    )
    assert audit.n_events == 1
    assert audit.events.iloc[0]["ticker"] == "AAA"
    assert audit.events.iloc[0]["return"] == pytest.approx(-0.5)
    assert audit.events.iloc[0]["classification"] == "UNADJUSTED SPLIT"
    assert not audit.passes, "a move coinciding with a known split must fail the gate"


def test_a_large_move_with_no_action_behind_it_is_a_market_move():
    symbols = ["AAA"]
    prices = _panel(symbols)
    close = prices.close.copy()
    close.loc[close.index[300] :, "AAA"] = 50.0
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )
    audit = corporate_action_audit(
        prices, symbols, threshold=0.35, split_dates={"AAA": pd.DatetimeIndex([])}
    )
    assert audit.n_events == 1
    assert audit.events.iloc[0]["classification"] == "market move"
    assert audit.passes, "a market move must not fail the gate"
    assert len(audit.unexplained) == 1


def test_the_threshold_is_what_it_says():
    symbols = ["AAA"]
    prices = _panel(symbols)
    close = prices.close.copy()
    close.loc[close.index[300] :, "AAA"] = 70.0
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )
    assert corporate_action_audit(
        prices, symbols, threshold=0.35, split_dates={"AAA": pd.DatetimeIndex([])}
    ).n_events == 0
    assert corporate_action_audit(
        prices, symbols, threshold=0.25, split_dates={"AAA": pd.DatetimeIndex([])}
    ).n_events == 1


def test_split_reconciliation_finds_a_break_too_small_for_the_extreme_move_scan():
    symbols = ["AAA"]
    prices = _panel(symbols)
    close = prices.close.copy()
    event = close.index[300]
    close.loc[event:, "AAA"] = 80.0
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )
    dates = {"AAA": pd.DatetimeIndex([event])}

    assert corporate_action_audit(prices, symbols, threshold=0.35, split_dates=dates).n_events == 0

    reconciliation = split_reconciliation(prices, symbols, split_dates=dates, threshold=0.15)
    assert reconciliation.n_splits == 1
    assert len(reconciliation.suspicious) == 1
    assert reconciliation.suspicious.iloc[0]["adjusted_return"] == pytest.approx(-0.2)
    assert not reconciliation.passes


def test_an_action_stamped_at_the_open_is_checked_against_its_own_bar():
    symbols = ["AAA"]
    prices = _panel(symbols)
    close = prices.close.copy()
    event = close.index[300]
    close.loc[event:, "AAA"] = 50.0
    prices = PriceData(
        open=prices.open, close=close, source="test", adjusted=True, fetched_at=""
    )

    stamped = {"AAA": pd.DatetimeIndex([event + pd.Timedelta(hours=9, minutes=30)])}
    reconciliation = split_reconciliation(prices, symbols, split_dates=stamped, threshold=0.15)
    assert reconciliation.n_splits == 1
    assert reconciliation.table.iloc[0]["bar"] == event, (
        "the 09:30 action was checked against the wrong bar; the whole reconciliation "
        "would pass vacuously"
    )
    assert reconciliation.table.iloc[0]["adjusted_return"] == pytest.approx(-0.5)
    assert not reconciliation.passes


def test_a_clean_series_passes_reconciliation_on_every_action_date():
    symbols = ["AAA"]
    index = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(600, 1)), axis=0)),
        index=index,
        columns=symbols,
    )
    prices = PriceData(open=close.copy(), close=close, source="test", adjusted=True, fetched_at="")
    dates = {"AAA": pd.DatetimeIndex([index[100], index[300], index[500]])}
    reconciliation = split_reconciliation(prices, symbols, split_dates=dates, threshold=0.20)
    assert reconciliation.n_splits == 3
    assert reconciliation.passes


def test_the_vendor_is_reported_as_not_point_in_time():
    prices = _panel(["AAA"])
    prices = PriceData(
        open=prices.open, close=prices.close, source="yahoo", adjusted=True, fetched_at="x"
    )
    convention = adjustment_convention(prices)
    assert convention["point_in_time"] is False
    assert "back-adjust" in convention["convention"]


def test_a_future_split_cannot_move_a_past_momentum_value():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2015-01-01", periods=900)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(900, 12)), axis=0)),
        index=index,
        columns=[f"N{i}" for i in range(12)],
    )
    report = ratio_invariance_report(close, 252, 21, split_factor=2.0, split_offset_days=5)
    assert report.invariant, report
    assert report.max_abs_momentum_difference < 1e-12
    assert report.n_compared > 5000


def test_a_split_inside_the_formation_window_does_change_the_signal():
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2015-01-01", periods=900)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(900, 4)), axis=0)),
        index=index,
        columns=[f"N{i}" for i in range(4)],
    )
    base = cross_sectional_momentum(close, 252, 21)

    adjusted = close.copy()
    adjusted.iloc[: len(close) - 100] = adjusted.iloc[: len(close) - 100] / 2.0
    after = cross_sectional_momentum(adjusted, 252, 21)

    a, b = base.iloc[-1].to_numpy(), after.iloc[-1].to_numpy()
    assert np.nanmax(np.abs(a - b)) > 0.5, (
        "an event inside the formation window must change the ratio; if it does not, "
        "the invariance test above is vacuous"
    )
