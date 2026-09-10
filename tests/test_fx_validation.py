from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_005 import load_config_005
from trendbot.engine.fx_validation import (
    carry_adjusted_prices,
    evaluate_decision_rule_005,
    short_rate_panel,
    worst_month_clustering,
)


@pytest.fixture(scope="module")
def cfg005():
    return load_config_005()


def verdict(cfg, **overrides) -> str:
    baseline = dict(
        strategy_sharpe=0.90,
        benchmark_sharpe=0.20,
        n_inversions=0,
        spread_mean=0.004,
        spread_t_stat=3.0,
        alpha_annualised=0.05,
        alpha_t_stat=3.0,
    )
    return evaluate_decision_rule_005(cfg, **{**baseline, **overrides}).verdict


def test_the_baseline_is_supported(cfg005):
    assert verdict(cfg005) == "SUPPORTED"


def test_sharpe_below_the_support_threshold_is_not_supported(cfg005):
    assert verdict(cfg005, strategy_sharpe=0.30, benchmark_sharpe=0.05) != "SUPPORTED"


def test_the_support_threshold_is_strict(cfg005):
    assert verdict(cfg005, strategy_sharpe=0.40, benchmark_sharpe=0.10) != "SUPPORTED"
    assert verdict(cfg005, strategy_sharpe=0.4001, benchmark_sharpe=0.10) == "SUPPORTED"


def test_the_benchmark_margin_is_inclusive(cfg005):
    assert verdict(cfg005, strategy_sharpe=0.90, benchmark_sharpe=0.75) == "SUPPORTED"
    assert verdict(cfg005, strategy_sharpe=0.90, benchmark_sharpe=0.7501) != "SUPPORTED"


def test_more_than_one_inversion_abandons(cfg005):
    assert verdict(cfg005, n_inversions=1) == "SUPPORTED"
    assert verdict(cfg005, n_inversions=2) == "ABANDON"


def test_spread_t_below_one_abandons(cfg005):
    assert verdict(cfg005, spread_t_stat=0.9) == "ABANDON"


def test_spread_t_between_one_and_two_is_inconclusive(cfg005):
    assert verdict(cfg005, spread_t_stat=1.5) == "INCONCLUSIVE - do not trade"


def test_negative_alpha_to_the_dollar_factor_abandons(cfg005):
    assert verdict(cfg005, alpha_annualised=-0.01, alpha_t_stat=-0.4) == "ABANDON"


def test_failing_to_beat_the_benchmark_at_all_abandons(cfg005):
    assert verdict(cfg005, strategy_sharpe=0.20, benchmark_sharpe=0.30) == "ABANDON"


def test_sharpe_below_the_abandon_threshold_abandons(cfg005):
    assert verdict(cfg005, strategy_sharpe=0.10, benchmark_sharpe=-0.50) == "ABANDON"


def test_a_negative_sharpe_that_still_beats_the_benchmark_abandons(cfg005):
    decision = evaluate_decision_rule_005(
        cfg005,
        strategy_sharpe=-0.27,
        benchmark_sharpe=-0.43,
        n_inversions=1,
        spread_mean=0.0008,
        spread_t_stat=0.62,
        alpha_annualised=0.004,
        alpha_t_stat=0.50,
    )
    assert decision.verdict == "ABANDON"
    fired = [name for name, ok, _ in decision.abandon_clauses if ok]
    assert "net Sharpe below 0.15" in fired
    assert any("t-statistic below" in name for name in fired)
    assert decision.support_clauses[1][1] is True


def test_every_support_and_abandon_clause_is_represented(cfg005):
    decision = evaluate_decision_rule_005(
        cfg005,
        strategy_sharpe=0.5,
        benchmark_sharpe=0.1,
        n_inversions=0,
        spread_mean=0.001,
        spread_t_stat=2.5,
        alpha_annualised=0.02,
        alpha_t_stat=2.5,
    )
    assert len(decision.support_clauses) == 4, "section 8 states four support clauses"
    assert len(decision.abandon_clauses) == 5, "section 8 states five abandon clauses"


def test_verdict_string_lists_every_clause(cfg005):
    text = str(
        evaluate_decision_rule_005(
            cfg005,
            strategy_sharpe=0.5,
            benchmark_sharpe=0.1,
            n_inversions=0,
            spread_mean=0.001,
            spread_t_stat=2.5,
            alpha_annualised=0.02,
            alpha_t_stat=2.5,
        )
    )
    assert text.count("[YES]") + text.count("[NO ]") == 9


def _worst(months):
    return pd.DataFrame(
        {"return": [-0.09, -0.06, -0.05]},
        index=pd.PeriodIndex(months, freq="M"),
    )


def test_clustering_matches_a_month_inside_a_named_window():
    windows = {"2011 CHF": ("2011-08", "2011-09"), "2020 March": ("2020-03",)}
    clustering = worst_month_clustering(_worst(["2011-09", "2004-02", "2007-11"]), windows)
    assert clustering.n_matched == 1
    assert clustering.labels_hit == ("2011 CHF",)
    assert clustering.any_clustering is True


def test_no_clustering_is_reported_as_the_section_9_warning():
    windows = {"2020 March": ("2020-03",)}
    clustering = worst_month_clustering(_worst(["2004-02", "2005-06", "2013-01"]), windows)
    assert clustering.n_matched == 0
    assert clustering.any_clustering is False
    assert "warning sign" in str(clustering)


def test_a_month_one_off_a_window_does_not_count_as_a_match():
    windows = {"2008 Q4": ("2008-10", "2008-11", "2008-12")}
    clustering = worst_month_clustering(_worst(["2008-09", "2008-08", "2007-08"]), windows)
    assert clustering.n_matched == 0


def test_zero_rates_leave_prices_untouched():
    index = pd.bdate_range("2020-01-01", periods=10)
    spot = pd.DataFrame({"A": np.linspace(1.0, 1.1, 10)}, index=index)
    rates = pd.DataFrame({"A": np.zeros(10)}, index=index)
    pd.testing.assert_frame_equal(carry_adjusted_prices(spot, rates), spot)


def test_a_missing_rate_accrues_nothing():
    index = pd.bdate_range("2020-01-01", periods=10)
    spot = pd.DataFrame({"A": np.ones(10)}, index=index)
    rates = pd.DataFrame({"A": np.full(10, np.nan)}, index=index)
    assert carry_adjusted_prices(spot, rates)["A"].tolist() == [1.0] * 10


def test_a_constant_rate_compounds_at_that_rate():
    index = pd.bdate_range("2020-01-01", periods=253)
    spot = pd.DataFrame({"A": np.ones(len(index))}, index=index)
    rates = pd.DataFrame({"A": np.full(len(index), 0.05)}, index=index)
    out = carry_adjusted_prices(spot, rates)["A"]
    assert out.iloc[-1] == pytest.approx(np.exp(0.05), rel=1e-6)


def test_the_accrual_uses_the_previous_bar_s_rate():
    index = pd.bdate_range("2020-01-01", periods=4)
    spot = pd.DataFrame({"A": np.ones(4)}, index=index)
    rates = pd.DataFrame({"A": [0.0, 0.0, 100.0, 100.0]}, index=index)
    out = carry_adjusted_prices(spot, rates)["A"]
    assert out.iloc[0] == out.iloc[1] == out.iloc[2] == 1.0
    assert out.iloc[3] > 1.0


def test_a_later_rate_cannot_change_an_earlier_carried_price():
    index = pd.bdate_range("2020-01-01", periods=20)
    spot = pd.DataFrame({"A": np.ones(20)}, index=index)
    base = pd.DataFrame({"A": np.full(20, 0.02)}, index=index)
    before = carry_adjusted_prices(spot, base)["A"]
    tampered = base.copy()
    tampered.iloc[12:] = 0.99
    after = carry_adjusted_prices(spot, tampered)["A"]
    pd.testing.assert_series_equal(before.iloc[:13], after.iloc[:13])


def test_mismatched_panels_are_refused():
    index = pd.bdate_range("2020-01-01", periods=5)
    spot = pd.DataFrame({"A": np.ones(5), "B": np.ones(5)}, index=index)
    rates = pd.DataFrame({"B": np.zeros(5), "A": np.zeros(5)}, index=index)
    with pytest.raises(ValueError, match="column order"):
        carry_adjusted_prices(spot, rates)


def test_an_uncovered_currency_is_reported_not_filled():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    monthly = pd.Series([0.03], index=pd.to_datetime(["2019-12-01"]))
    coverage = short_rate_panel({"A": ("SRC_A", monthly)}, ("A", "B"), calendar)
    assert coverage.covered == ("A",)
    assert coverage.uncovered == ("B",)
    assert coverage.rates["B"].isna().all(), "an uncovered currency was given a rate"
    assert coverage.rates["A"].notna().all(), "a monthly rate was not carried onto the calendar"
    assert coverage.sources["A"] == "SRC_A"


def test_a_rate_is_carried_forward_but_never_backward():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    late = pd.Series([0.03], index=pd.to_datetime(["2020-01-08"]))
    rates = short_rate_panel({"A": ("SRC", late)}, ("A",), calendar).rates["A"]
    assert rates.iloc[:5].isna().all()
    assert rates.iloc[-1] == 0.03
