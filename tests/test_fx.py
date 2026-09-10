from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.fred import FredError, SeriesMetadata
from trendbot.fx import (
    FOREIGN_PER_USD,
    MAX_HOLIDAY_GAP_DAYS,
    REFERENCE_LEVELS,
    USD_PER_FOREIGN,
    check_drift_plausibility,
    check_peg_relationships,
    check_reference_levels,
    equal_weight_factor,
    forward_fill_panel,
    measure_gaps,
    normalise_to_usd,
    parse_quote_convention,
    peg_breach_dates,
    publication_calendar,
)


def meta(series_id: str, title: str, units: str, frequency: str = "Daily") -> SeriesMetadata:
    return SeriesMetadata(
        series_id=series_id,
        title=title,
        units=units,
        frequency=frequency,
        observation_start="1999-01-04",
        observation_end="2026-08-14",
        obtained_via="fixture",
    )


YEN = meta("DEXJPUS", "Japanese Yen to U.S. Dollar Spot Exchange Rate", "Japanese Yen to One U.S. Dollar")
EUR = meta("DEXUSEU", "U.S. Dollars to Euro Spot Exchange Rate", "U.S. Dollars to One Euro")
GBP = meta(
    "DEXUSUK",
    "U.S. Dollars to U.K. Pound Sterling Spot Exchange Rate",
    "U.S. Dollars to One U.K. Pound Sterling",
)


def test_foreign_per_usd_is_recognised_and_inverted():
    convention = parse_quote_convention(YEN)
    assert convention.direction == FOREIGN_PER_USD
    assert convention.inverted is True
    assert convention.foreign_currency == "Japanese Yen"


def test_usd_per_foreign_is_recognised_and_left_alone():
    for metadata, currency in ((EUR, "Euro"), (GBP, "U.K. Pound Sterling")):
        convention = parse_quote_convention(metadata)
        assert convention.direction == USD_PER_FOREIGN
        assert convention.inverted is False
        assert convention.foreign_currency == currency


def test_direction_does_not_depend_on_the_values():
    assert parse_quote_convention(YEN).inverted is True
    series = pd.Series([1.2, 1.3], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert normalise_to_usd(series, parse_quote_convention(YEN)).tolist() == [1 / 1.2, 1 / 1.3]


def test_title_and_units_must_agree():
    liar = meta("DEXJPUS", "U.S. Dollars to Japanese Yen Spot Exchange Rate", "Japanese Yen to One U.S. Dollar")
    with pytest.raises(FredError, match="disagree"):
        parse_quote_convention(liar)


def test_units_that_are_not_a_bilateral_rate_are_refused():
    index = meta("DTWEXBGS", "Nominal Broad U.S. Dollar Index", "Index Jan 2006=100")
    with pytest.raises(FredError, match="do not read as"):
        parse_quote_convention(index)


def test_units_naming_no_dollar_leg_are_refused():
    cross = meta("XXX", "Japanese Yen to Euro Spot Exchange Rate", "Japanese Yen to One Euro")
    with pytest.raises(FredError, match="no US dollar leg"):
        parse_quote_convention(cross)


def test_units_naming_two_dollar_legs_are_refused():
    silly = meta("XXX", "U.S. Dollars to U.S. Dollar Spot Exchange Rate", "U.S. Dollars to One U.S. Dollar")
    with pytest.raises(FredError, match="two US dollar legs"):
        parse_quote_convention(silly)


def test_title_that_is_not_a_spot_rate_is_refused():
    odd = meta("XXX", "Japanese Yen per Dollar, monthly average", "Japanese Yen to One U.S. Dollar")
    with pytest.raises(FredError, match="cannot be cross-checked"):
        parse_quote_convention(odd)


def test_non_positive_values_become_holes_not_infinities():
    convention = parse_quote_convention(YEN)
    series = pd.Series([100.0, 0.0, -5.0, 110.0], index=pd.bdate_range("2020-01-01", periods=4))
    out = normalise_to_usd(series, convention)
    assert np.isfinite(out.iloc[0]) and np.isfinite(out.iloc[3])
    assert out.iloc[1:3].isna().all()


def _correct_panel() -> tuple[pd.DataFrame, dict]:
    index = pd.bdate_range("1999-01-04", "2026-08-14")
    levels = {
        "DEXJPUS": 0.0095,
        "DEXUSUK": 1.98,
        "DEXSZUS": 1.37,
        "DEXKOUS": 0.00088,
        "DEXMXUS": 0.058,
        "DEXUSEU": 1.55,
        "DEXCAUS": 0.63,
        "DEXINUS": 0.0145,
        "DEXCHUS": 0.1208,
        "DEXBZUS": 0.24,
        "DEXHKUS": 1 / 7.80,
        "DEXDNUS": 1.55 / 7.46038,
    }
    frame = pd.DataFrame({k: np.full(len(index), v) for k, v in levels.items()}, index=index)
    conventions = {
        k: parse_quote_convention(meta(k, f"X to U.S. Dollar Spot Exchange Rate", "X to One U.S. Dollar"))
        for k in levels
    }
    return frame, conventions


def test_reference_levels_pass_on_a_correct_panel():
    frame, conventions = _correct_panel()
    checks = check_reference_levels(frame, conventions)
    assert checks, "the reference table matched nothing — the check would be vacuous"
    assert all(c.passed for c in checks)


def test_every_reference_band_would_catch_an_inversion():
    frame, conventions = _correct_panel()
    for check in check_reference_levels(frame, conventions):
        assert check.discriminating, f"{check.series_id}'s band would not catch an inversion"


def test_reference_levels_fail_on_an_inverted_panel():
    frame, conventions = _correct_panel()
    checks = check_reference_levels(1.0 / frame, conventions)
    assert not any(c.passed for c in checks), "an inverted panel passed the reference check"


def test_peg_checks_fail_when_either_leg_is_inverted():
    frame, _ = _correct_panel()
    assert all(c.passed for c in check_peg_relationships(frame))

    for leg in ("DEXUSEU", "DEXDNUS"):
        broken = frame.copy()
        broken[leg] = 1.0 / broken[leg]
        results = {c.name: c for c in check_peg_relationships(broken)}
        krone = next(c for name, c in results.items() if name.startswith("Danish"))
        assert not krone.passed, f"inverting {leg} did not break the ERM II cross-rate check"

    inverted_hkd = frame.copy()
    inverted_hkd["DEXHKUS"] = 1.0 / inverted_hkd["DEXHKUS"]
    hk = next(c for c in check_peg_relationships(inverted_hkd) if c.name.startswith("Hong Kong"))
    assert not hk.passed


def test_peg_breach_dates_finds_the_days_outside_the_band():
    frame, _ = _correct_panel()
    assert peg_breach_dates(frame) == {}
    tampered = frame.copy()
    day = tampered.index[100]
    tampered.loc[day, "DEXDNUS"] = tampered.loc[day, "DEXDNUS"] * 1.10
    breaches = peg_breach_dates(tampered)
    assert any(day in dates for dates in breaches.values())


def test_drift_check_fails_on_an_implausibly_compounding_series():
    index = pd.bdate_range("1999-01-04", periods=2520)
    exploding = pd.DataFrame({"DEXBZUS": np.exp(np.linspace(0, 10, len(index)))}, index=index)
    conventions = {
        "DEXBZUS": parse_quote_convention(
            meta("DEXBZUS", "Brazilian Reals to U.S. Dollar Spot Exchange Rate", "Brazilian Reals to One U.S. Dollar")
        )
    }
    assert not check_drift_plausibility(exploding, conventions)[0].passed


def _raw(index, values):
    return pd.Series(values, index=index, dtype=float)


def test_publication_calendar_is_the_union_of_days_anything_published():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    a = _raw(index, [1.0, np.nan, 3.0, 4.0])
    b = _raw(index, [np.nan, 2.0, np.nan, 4.0])
    calendar = publication_calendar({"A": a, "B": b}, "2020-01-01")
    assert list(calendar) == list(index)

    c = _raw(index, [1.0, np.nan, 3.0, 4.0])
    d = _raw(index, [1.0, np.nan, 3.0, 4.0])
    calendar = publication_calendar({"C": c, "D": d}, "2020-01-01")
    assert pd.Timestamp("2020-01-02") not in calendar


def test_publication_calendar_respects_the_window_start():
    index = pd.bdate_range("1998-01-01", periods=400)
    calendar = publication_calendar({"A": _raw(index, np.ones(len(index)))}, "1999-01-01")
    assert calendar[0] >= pd.Timestamp("1999-01-01")


def test_measure_gaps_reports_the_longest_run_of_missing_days():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    values = np.ones(10)
    values[[3, 4, 5]] = np.nan
    values[8] = np.nan
    gaps = measure_gaps(_raw(calendar, values), calendar, series_id="X", currency="Xs")
    assert gaps.largest_gap == 3
    assert gaps.largest_gap_start == str(calendar[3].date())
    assert gaps.largest_gap_end == str(calendar[5].date())
    assert gaps.n_observations == 6
    assert gaps.n_scheduled == 10


def test_the_holiday_allowance_is_longer_than_any_national_market_closure():
    assert MAX_HOLIDAY_GAP_DAYS > 8


def test_forward_fill_carries_values_forward_only():
    index = pd.bdate_range("2020-01-01", periods=6)
    a = _raw(index, [1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    b = _raw(index, np.ones(6))
    filled = forward_fill_panel({"A": a, "B": b}, ["A", "B"], upper_bound=index[-1])["A"]
    assert filled.tolist() == [1.0, 1.0, 1.0, 4.0, 4.0, 6.0]


def test_forward_fill_never_reaches_backwards():
    index = pd.bdate_range("2020-01-01", periods=6)
    a = _raw(index, [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0])
    b = _raw(index, np.ones(6))
    filled = forward_fill_panel({"A": a, "B": b}, ["A", "B"], upper_bound=index[-1])["A"]
    assert filled.iloc[:2].isna().all(), "a series was back-filled before its first observation"


def test_a_later_observation_cannot_change_an_earlier_filled_value():
    index = pd.bdate_range("2020-01-01", periods=8)
    a = _raw(index, [1.0, np.nan, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0])
    b = _raw(index, np.ones(8))
    panel = {"A": a, "B": b}
    before = forward_fill_panel(panel, ["A", "B"], upper_bound=index[-1])["A"]
    tampered = a.copy()
    tampered.iloc[5:] = tampered.iloc[5:] * 100.0
    after = forward_fill_panel({"A": tampered, "B": b}, ["A", "B"], upper_bound=index[-1])["A"]
    assert before.iloc[:5].equals(after.iloc[:5])


def test_a_date_only_one_series_published_is_still_a_bar():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    only = _raw(index, [1.0, 2.0, 3.0])
    other = _raw(index, [10.0, np.nan, 30.0])
    panel = forward_fill_panel({"A": only, "B": other}, ["A", "B"], upper_bound=index[-1])
    assert len(panel) == 3
    assert panel.loc["2020-01-02", "B"] == 10.0


def test_forward_fill_panel_index_is_the_union_of_member_observations():
    a = _raw(pd.to_datetime(["2020-01-01", "2020-01-03"]), [1.0, 2.0])
    b = _raw(pd.to_datetime(["2020-01-02", "2020-01-03"]), [3.0, 4.0])
    panel = forward_fill_panel({"A": a, "B": b}, ["A", "B"], upper_bound=pd.Timestamp("2020-01-03"))
    assert list(panel.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert np.isnan(panel.loc["2020-01-01", "B"])
    assert panel.loc["2020-01-02", "A"] == 1.0


def test_forward_fill_panel_stops_at_the_upper_bound():
    index = pd.bdate_range("2020-01-01", periods=10)
    panel = forward_fill_panel({"A": _raw(index, np.ones(10))}, ["A"], upper_bound=index[4])
    assert panel.index[-1] == index[4]


def test_equal_weight_factor_is_the_mean_of_the_column_returns():
    index = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 90.0]}, index=index)
    factor = equal_weight_factor(frame)
    assert factor.iloc[0] == 0.0
    assert factor.iloc[1] == pytest.approx(0.05)
    assert factor.iloc[2] == pytest.approx(-0.05)


def test_equal_weight_factor_ignores_a_column_that_has_not_started():
    index = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [np.nan, np.nan, 100.0]}, index=index)
    factor = equal_weight_factor(frame)
    assert factor.iloc[1] == pytest.approx(0.10)
    assert factor.iloc[2] == pytest.approx(0.10)


def test_equal_weight_factor_is_never_nan():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.DataFrame({"A": [np.nan, np.nan, 100.0, 101.0]}, index=index)
    assert not equal_weight_factor(frame).isna().any()


def test_reference_table_covers_both_quote_directions():
    ids = {series_id for series_id, *_ in REFERENCE_LEVELS}
    assert "DEXUSEU" in ids or "DEXUSUK" in ids
    assert "DEXJPUS" in ids
