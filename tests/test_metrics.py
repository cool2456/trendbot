"""Closed-form checks on trendbot.engine.metrics.

Every fixture here is hand-built so the expected value can be written down by hand
rather than read off the implementation. Where a convention is disputable (Sortino's
denominator, ddof, turnover annualisation) the test pins the documented convention
*and* shows that the rival convention would give a different number, so the test
fails if the convention silently flips.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from trendbot.engine import metrics as m


# --------------------------------------------------------------------------------------
# hand-built fixtures
# --------------------------------------------------------------------------------------

# equity: 1.0 -> 1.25 -> 1.00 -> 0.75 -> 1.125.  Every value is exact in binary
# floating point, so the drawdown assertions below can be exact.
DRAWDOWN_PATH = pd.Series([0.0, 0.25, -0.20, -0.25, 0.5])

# mean and sd computed by hand below; deliberately mixed-sign with one zero day.
MIXED = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015, -0.02, 0.03])


def _daily_index(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# --------------------------------------------------------------------------------------
# CAGR and the degenerate zero-variance series
# --------------------------------------------------------------------------------------


def test_cagr_of_a_constant_daily_return_matches_the_closed_form():
    # 252 observations is exactly one year at the module's annualisation factor,
    # so CAGR is just the one-year compounded growth.
    r = pd.Series([0.001] * 252)
    assert m.cagr(r) == pytest.approx(1.001**252 - 1.0, rel=1e-12)
    assert m.cagr(r) == pytest.approx(0.28643404, abs=1e-8)


def test_cagr_annualises_a_multi_year_sample_rather_than_reporting_total_growth():
    # Two years of the same daily return: total growth squares, CAGR does not move.
    one_year = pd.Series([0.001] * 252)
    two_years = pd.Series([0.001] * 504)
    assert m.cagr(two_years) == pytest.approx(m.cagr(one_year), rel=1e-12)
    total = float((1.0 + two_years).prod() - 1.0)
    assert total == pytest.approx((1.001**252) ** 2 - 1.0, rel=1e-12)


def test_cagr_of_a_wiped_out_account_is_minus_one_not_a_complex_root():
    # -100% on one day makes total growth exactly zero; a fractional power of a
    # non-positive number is what the -1.0 short circuit exists to avoid.
    r = pd.Series([0.01] * 100 + [-1.0] + [0.0] * 151)
    assert m.cagr(r) == -1.0


def test_cagr_of_an_empty_series_is_nan_not_an_exception():
    assert math.isnan(m.cagr(pd.Series([], dtype=float)))


def test_sharpe_of_a_zero_variance_series_is_nan():
    # pd.Series([0.001]*252).std(ddof=1) is 2.2e-19 rather than 0.0, because 0.001 is
    # not exactly representable. An exact `sd == 0` guard misses that and returns a
    # Sharpe of 7e16; the guard is therefore scale-relative.
    assert math.isnan(m.sharpe(pd.Series([0.001] * 252)))
    assert math.isnan(m.sharpe(pd.Series([1e-9] * 252)))


def test_sharpe_returns_nan_when_the_standard_deviation_is_exactly_zero():
    # The exactly-zero case, which the naive guard also caught.
    assert math.isnan(m.sharpe(pd.Series([0.0] * 252)))
    assert math.isnan(m.sharpe(pd.Series([0.25] * 252)))


def test_sharpe_of_a_single_observation_is_nan():
    assert math.isnan(m.sharpe(pd.Series([0.01])))


def test_sortino_of_a_series_with_no_losing_day_is_nan_not_infinity():
    # Documented explicitly in the source: "no losing day: ratio is undefined, not
    # infinite". An infinity here would propagate into a summary table as a winner.
    r = pd.Series([0.001] * 252)
    assert math.isnan(m.sortino(r))
    assert math.isnan(m.sortino(pd.Series([0.01, 0.02, 0.0, 0.03])))


# --------------------------------------------------------------------------------------
# Sharpe: hand computation and the ddof convention
# --------------------------------------------------------------------------------------


def test_sharpe_equals_mean_over_sd_times_sqrt_252_with_ddof_one():
    v = MIXED.to_numpy()
    expected = v.mean() / v.std(ddof=1) * math.sqrt(252)
    assert m.sharpe(MIXED) == pytest.approx(expected, rel=1e-12)
    # observed on this machine: 4.7556
    assert m.sharpe(MIXED) == pytest.approx(4.755563543, abs=1e-8)


def test_sharpe_uses_the_sample_standard_deviation_not_the_population_one():
    v = MIXED.to_numpy()
    with_ddof1 = v.mean() / v.std(ddof=1) * math.sqrt(252)
    with_ddof0 = v.mean() / v.std(ddof=0) * math.sqrt(252)
    # 4.7556 vs 5.0839 on this 8-observation sample: the two conventions are far
    # enough apart here that the test cannot pass under both.
    assert abs(with_ddof1 - with_ddof0) > 0.3
    assert m.sharpe(MIXED) == pytest.approx(with_ddof1, rel=1e-12)
    assert m.sharpe(MIXED) != pytest.approx(with_ddof0, rel=1e-6)


def test_sharpe_annualisation_scales_with_the_square_root_of_the_period_count():
    monthly = pd.Series([0.02, -0.01, 0.03, 0.0, -0.005, 0.015] * 4)
    daily_ann = m.sharpe(monthly, periods_per_year=252)
    monthly_ann = m.sharpe(monthly, periods_per_year=12)
    assert daily_ann / monthly_ann == pytest.approx(math.sqrt(252 / 12), rel=1e-12)


def test_sharpe_sign_follows_the_mean():
    assert m.sharpe(MIXED) > 0
    assert m.sharpe(-MIXED) == pytest.approx(-m.sharpe(MIXED), rel=1e-12)


# --------------------------------------------------------------------------------------
# Sortino: the full-sample denominator convention
# --------------------------------------------------------------------------------------


def test_sortino_divides_the_squared_downside_by_the_total_observation_count():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    # By hand: mean = 0.005; sum of squared negatives = 1e-4 + 4e-4 = 5e-4.
    # Full-sample convention divides by 4, the rival convention divides by 2.
    dd_full = math.sqrt(5e-4 / 4)
    dd_negatives_only = math.sqrt(5e-4 / 2)
    expected_full = 0.005 / dd_full * math.sqrt(252)
    expected_rival = 0.005 / dd_negatives_only * math.sqrt(252)
    assert expected_full == pytest.approx(7.0992957, abs=1e-6)
    assert expected_rival == pytest.approx(5.0199602, abs=1e-6)
    assert m.sortino(r) == pytest.approx(expected_full, rel=1e-12)
    assert m.sortino(r) != pytest.approx(expected_rival, rel=1e-3)


def test_sortino_exceeds_sharpe_when_the_downside_is_the_quiet_side():
    # Large upside days, small downside days: downside deviation is well below the
    # full standard deviation, so Sortino must be the larger number.
    r = pd.Series([0.05, -0.001, 0.06, -0.002, 0.04, -0.001] * 8)
    assert m.sortino(r) > m.sharpe(r) > 0


def test_sortino_treats_zero_return_days_as_neither_upside_nor_downside():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    padded = pd.Series([0.02, -0.01, 0.03, -0.02, 0.0, 0.0])
    # Adding flat days leaves the squared-downside sum alone but grows the
    # denominator's count, so the ratio must fall - that is the whole content of
    # the full-sample convention.
    assert m.sortino(padded) < m.sortino(r)
    dd_padded = math.sqrt(5e-4 / 6)
    assert m.sortino(padded) == pytest.approx(
        (padded.mean() / dd_padded) * math.sqrt(252), rel=1e-12
    )


# --------------------------------------------------------------------------------------
# drawdown family, all on one hand-built equity path
# --------------------------------------------------------------------------------------


def test_equity_curve_compounds_geometrically_from_the_initial_value():
    eq = m.equity_curve(DRAWDOWN_PATH)
    assert eq.tolist() == [1.0, 1.25, 1.0, 0.75, 1.125]
    # a leading zero return means the curve genuinely starts at `initial`
    assert eq.iloc[0] == 1.0
    scaled = m.equity_curve(DRAWDOWN_PATH, initial=10_000.0)
    assert scaled.iloc[0] == 10_000.0
    assert scaled.tolist() == [x * 10_000.0 for x in eq.tolist()]
    # and it compounds rather than sums: the sum of returns is +0.30, the
    # compounded result is +0.125.
    assert float(DRAWDOWN_PATH.sum()) == pytest.approx(0.30, abs=1e-12)
    assert eq.iloc[-1] / 1.0 - 1.0 == pytest.approx(0.125, abs=1e-12)


def test_equity_curve_is_the_running_product_of_one_plus_the_return():
    eq = m.equity_curve(MIXED, initial=250.0)
    expected = 250.0 * np.cumprod(1.0 + MIXED.to_numpy())
    assert np.allclose(eq.to_numpy(), expected, rtol=0, atol=1e-12)
    assert list(eq.index) == list(MIXED.index)


def test_max_drawdown_finds_the_peak_to_trough_decline_not_the_worst_single_day():
    # peak 1.25 at position 1, trough 0.75 at position 3 -> -40%.
    assert m.max_drawdown(DRAWDOWN_PATH) == pytest.approx(-0.40, abs=1e-12)
    # the worst single day is only -25%, so this cannot be a per-bar minimum
    assert DRAWDOWN_PATH.min() == -0.25


def test_drawdown_series_is_zero_at_new_highs_and_negative_below_them():
    dd = m.drawdown_series(DRAWDOWN_PATH)
    assert np.allclose(dd.to_numpy(), [0.0, 0.0, -0.20, -0.40, -0.10], rtol=0, atol=1e-12)
    assert (dd <= 1e-15).all()
    assert list(dd.index) == list(DRAWDOWN_PATH.index)


def test_time_underwater_is_the_fraction_of_bars_below_a_prior_peak():
    # positions 2, 3, 4 are underwater; positions 0 and 1 are at a new high.
    assert m.time_underwater(DRAWDOWN_PATH) == pytest.approx(3 / 5, abs=1e-12)


def test_longest_drawdown_days_counts_the_longest_consecutive_underwater_run():
    assert m.longest_drawdown_days(DRAWDOWN_PATH) == 3
    # two separate short runs must not be added together
    two_runs = pd.Series([0.5, -0.2, 0.5, -0.2, 0.5, 0.5])
    dd = m.drawdown_series(two_runs)
    assert (dd < 0).sum() == 2
    assert m.longest_drawdown_days(two_runs) == 1
    # a monotonically rising path is never underwater
    assert m.longest_drawdown_days(pd.Series([0.01] * 10)) == 0
    assert m.time_underwater(pd.Series([0.01] * 10)) == 0.0


def test_calmar_is_cagr_over_the_absolute_max_drawdown():
    r = pd.Series([0.001] * 200 + [-0.02] * 10 + [0.001] * 42)
    expected = m.cagr(r) / abs(m.max_drawdown(r))
    assert m.calmar(r) == pytest.approx(expected, rel=1e-12)
    # no drawdown at all -> undefined, not a division by zero
    assert math.isnan(m.calmar(pd.Series([0.001] * 50)))


# --------------------------------------------------------------------------------------
# turnover and exposure
# --------------------------------------------------------------------------------------


def test_annual_weight_churn_of_a_single_trade_to_full_investment_is_gross_over_years():
    # Flat on day 0, then [0.5, 0.3, 0.2] forever: one-way turnover 1.0 in total,
    # spread over 504 bars = 2 years, so 0.5 per year.
    w = pd.DataFrame(0.0, index=range(504), columns=["A", "B", "C"])
    w.iloc[1:] = [0.5, 0.3, 0.2]
    assert m.annual_weight_churn(w) == pytest.approx(0.5, rel=1e-12)
    assert m.average_gross_exposure(w) == pytest.approx(503 / 504, rel=1e-12)


def test_annual_weight_churn_counts_establishing_the_initial_book_as_real_turnover():
    # Same portfolio, but invested from the very first row. Buying the book still
    # costs a full round of turnover, so the answer must be identical.
    w = pd.DataFrame([[0.5, 0.3, 0.2]] * 504, columns=["A", "B", "C"])
    assert m.annual_weight_churn(w) == pytest.approx(0.5, rel=1e-12)


def test_annual_weight_churn_scales_linearly_with_the_number_of_round_trips():
    # In and out of a 100% gross book every other bar over one year: 252 bars, the
    # first row establishes 1.0 and each subsequent flip costs 1.0.
    rows = [[1.0], [0.0]] * 126
    w = pd.DataFrame(rows, columns=["A"])
    assert len(w) == 252
    assert m.annual_weight_churn(w) == pytest.approx(252.0, rel=1e-12)


def test_annual_weight_churn_is_nan_for_an_empty_book():
    assert math.isnan(m.annual_weight_churn(pd.DataFrame()))


def test_exposure_measures_separate_gross_from_net():
    w = pd.DataFrame({"A": [0.6, 0.6], "B": [-0.4, -0.4]})
    assert m.average_gross_exposure(w) == pytest.approx(1.0, rel=1e-12)
    assert m.average_net_exposure(w) == pytest.approx(0.2, rel=1e-12)
    assert m.invested_fraction(w) == 1.0
    flat = pd.DataFrame({"A": [0.0, 0.5], "B": [0.0, 0.5]})
    assert m.invested_fraction(flat) == pytest.approx(0.5, abs=1e-12)


# --------------------------------------------------------------------------------------
# calendar-year compounding
# --------------------------------------------------------------------------------------


def test_calendar_year_returns_compound_within_each_year():
    idx = pd.to_datetime(["2020-06-01", "2020-07-01", "2021-03-01", "2021-04-01"])
    r = pd.Series([0.1, 0.1, -0.5, 0.5], index=idx)
    cy = m.calendar_year_returns(r)
    assert list(cy.index) == [2020, 2021]
    assert cy.loc[2020] == pytest.approx(1.1 * 1.1 - 1.0, rel=1e-12)  # +21%, not +20%
    assert cy.loc[2021] == pytest.approx(0.5 * 1.5 - 1.0, rel=1e-12)  # -25%, not 0%
    assert cy.loc[2020] != pytest.approx(0.20, rel=1e-6)


def test_calendar_year_returns_chain_to_the_full_sample_total_return():
    idx = pd.to_datetime(
        ["2019-02-01", "2019-08-01", "2020-05-01", "2020-11-01", "2021-01-04"]
    )
    r = pd.Series([0.05, -0.03, 0.10, 0.02, -0.07], index=idx)
    cy = m.calendar_year_returns(r)
    assert float((1.0 + cy).prod()) == pytest.approx(float((1.0 + r).prod()), rel=1e-12)


def test_calendar_year_returns_refuses_an_index_that_has_no_calendar():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        m.calendar_year_returns(pd.Series([0.01, 0.02]))


# --------------------------------------------------------------------------------------
# NaN discipline
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        m.cagr,
        m.sharpe,
        m.sortino,
        m.max_drawdown,
        m.drawdown_series,
        m.time_underwater,
        m.longest_drawdown_days,
        m.equity_curve,
        m.calmar,
        m.hit_rate,
        m.summarise,
    ],
)
def test_a_nan_in_the_return_series_raises_rather_than_being_skipped(fn):
    idx = _daily_index(5)
    r = pd.Series([0.01, 0.02, np.nan, -0.01, 0.005], index=idx)
    with pytest.raises(ValueError, match="NaN"):
        fn(r)


def test_a_nan_is_not_silently_treated_as_a_flat_day():
    idx = _daily_index(5)
    clean = pd.Series([0.01, 0.02, 0.0, -0.01, 0.005], index=idx)
    dirty = clean.copy()
    dirty.iloc[2] = np.nan
    # the clean version works, so the raise is about the NaN and nothing else
    assert math.isfinite(m.sharpe(clean))
    with pytest.raises(ValueError):
        m.sharpe(dirty)


def test_a_nan_in_a_numpy_array_input_also_raises():
    with pytest.raises(ValueError, match="NaN"):
        m.sharpe(np.array([0.01, np.nan, 0.02]))


def test_empty_input_yields_nan_rather_than_an_exception():
    empty = pd.Series([], dtype=float)
    assert math.isnan(m.max_drawdown(empty))
    assert math.isnan(m.time_underwater(empty))
    assert m.longest_drawdown_days(empty) == 0
    assert len(m.equity_curve(empty)) == 0


# --------------------------------------------------------------------------------------
# summarise()
# --------------------------------------------------------------------------------------


def _normal_sample() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(11)
    idx = _daily_index(1260)
    r = pd.Series(rng.normal(0.0004, 0.009, len(idx)), index=idx)
    w = pd.DataFrame(
        rng.uniform(0.0, 0.25, size=(len(idx), 4)), index=idx, columns=list("ABCD")
    )
    return r, w


def test_summarise_returns_a_frozen_dataclass():
    r, w = _normal_sample()
    stats = m.summarise(r, w)
    assert isinstance(stats, m.PerformanceStats)
    assert dataclasses.is_dataclass(stats)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.sharpe = 99.0  # type: ignore[misc]


def test_summarise_fields_are_all_finite_for_a_normal_input():
    r, w = _normal_sample()
    stats = m.summarise(r, w)
    for field in dataclasses.fields(stats):
        value = getattr(stats, field.name)
        assert isinstance(value, (int, float)), field.name
        assert math.isfinite(value), f"{field.name} = {value}"


def test_summarise_agrees_with_the_individual_metric_functions():
    r, w = _normal_sample()
    stats = m.summarise(r, w)
    assert stats.n_periods == len(r)
    assert stats.years == pytest.approx(len(r) / 252, rel=1e-12)
    assert stats.sharpe == pytest.approx(m.sharpe(r), rel=1e-12)
    assert stats.sortino == pytest.approx(m.sortino(r), rel=1e-12)
    assert stats.cagr == pytest.approx(m.cagr(r), rel=1e-12)
    assert stats.max_drawdown == pytest.approx(m.max_drawdown(r), rel=1e-12)
    assert stats.ann_weight_churn == pytest.approx(m.annual_weight_churn(w), rel=1e-12)
    assert stats.avg_gross_exposure == pytest.approx(m.average_gross_exposure(w), rel=1e-12)
    assert stats.total_return == pytest.approx(float((1.0 + r).prod() - 1.0), rel=1e-12)
    assert stats.n_years_observed == len(m.calendar_year_returns(r))
    assert stats.longest_drawdown_days == m.longest_drawdown_days(r)


def test_summarise_without_weights_reports_zero_turnover_rather_than_nan():
    r, _ = _normal_sample()
    stats = m.summarise(r)
    assert stats.ann_turnover == 0.0
    assert stats.avg_gross_exposure == 0.0
    assert stats.invested_fraction == 0.0
    assert math.isfinite(stats.sharpe)


def test_summarise_counts_losing_calendar_years():
    idx = pd.to_datetime(["2018-03-01", "2019-03-01", "2020-03-01", "2021-03-01"])
    r = pd.Series([0.10, -0.05, 0.20, -0.30], index=idx)
    stats = m.summarise(r)
    assert stats.n_years_observed == 4
    assert stats.losing_years == 2


def test_summarise_str_is_a_single_readable_line():
    r, w = _normal_sample()
    text = str(m.summarise(r, w))
    assert "\n" not in text
    assert "Sharpe" in text and "maxDD" in text


def test_hit_rate_ignores_flat_days():
    r = pd.Series([0.01, -0.01, 0.0, 0.0, 0.02, -0.02, 0.03])
    # five active days, three of them up
    assert m.hit_rate(r) == pytest.approx(3 / 5, abs=1e-12)
    assert math.isnan(m.hit_rate(pd.Series([0.0, 0.0, 0.0])))


# --------------------------------------------------------------------------------------
# turnover measures TRADING; churn measures drift. Conflating them overstates costs.
# --------------------------------------------------------------------------------------


def test_annual_turnover_counts_only_executed_trades():
    trades = pd.DataFrame(
        [[0.5, 0.5], [-0.25, 0.25]],
        index=pd.to_datetime(["2020-01-02", "2020-07-01"]),
        columns=["A", "B"],
    )
    # 1.0 + 0.5 = 1.5 of one-way turnover over one year of daily observations
    assert m.annual_turnover(trades, n_periods=252) == pytest.approx(1.5)
    assert m.annual_turnover(trades, n_periods=504) == pytest.approx(0.75)


def test_annual_turnover_is_nan_without_trades():
    assert math.isnan(m.annual_turnover(None, 252))
    assert math.isnan(m.annual_turnover(pd.DataFrame(), 252))
    assert math.isnan(m.annual_turnover(pd.DataFrame([[1.0]]), 0))


def test_a_book_that_never_trades_has_zero_turnover_but_non_zero_churn():
    """The distinction the two functions exist for.

    Buy two assets once and never touch them again. Turnover is zero because nothing
    was traded; churn is large because the weights drift every day. Charging a cost
    per side against churn would invent a cost that was never paid.
    """
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", periods=252)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (252, 2)), axis=0)),
        index=index,
        columns=["A", "B"],
    )
    shares = np.array([0.5, 0.5]) / prices.iloc[0]
    value = prices * shares
    weights = value.div(value.sum(axis=1), axis=0)

    no_trades = pd.DataFrame(columns=["A", "B"], dtype=float)
    assert math.isnan(m.annual_turnover(no_trades, len(weights)))
    assert m.annual_weight_churn(weights) > 1.0  # observed ~2.4
