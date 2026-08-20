"""Experiment 003's decile sort, market regression and decision rule.

The decile cut and the market regression are both pass/fail inputs to PREREG_003.md
section 8, so both get fixtures whose answer is known independently of the code under
test: a regression with an alpha and beta planted in it, and a bucket sort whose sizes
are arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_003 import load_config_003
from trendbot.engine.eq_validation import evaluate_decision_rule_003, market_regression
from trendbot.engine.metrics import TRADING_DAYS_PER_YEAR
from trendbot.engine.panel_backtest import run_panel_backtest
from trendbot.engine.validation import synthetic_prices
from trendbot.engine.xs_validation import bucket_study
from trendbot.strategies import EquityCrossSectionalMomentum
from trendbot.xsmom import quantile_labels, quantile_sizes


@pytest.fixture(scope="session")
def cfg003():
    return load_config_003()


# --------------------------------------------------------------------------------------
# the decile cut
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (409, (41, 41, 41, 41, 41, 41, 41, 41, 41, 40)),  # the real universe
        (405, (41, 40, 41, 40, 41, 40, 41, 40, 41, 40)),
        (400, (40,) * 10),
        (10, (1,) * 10),
        (9, (0,) * 10),  # fewer names than buckets: no sort exists
    ],
)
def test_even_deciles_differ_by_at_most_one_name(n, expected):
    sizes = quantile_sizes(n, 10, "even")
    assert sizes == expected
    if n >= 10:
        assert sum(sizes) == n
        assert max(sizes) - min(sizes) <= 1, "that is what 'even' means"
        assert sizes[0] >= sizes[-1], "the top bucket never gives up a name to the bottom"


def test_the_two_conventions_disagree_exactly_where_it_matters():
    """At 409 names the choice moves 9 names into or out of the bottom decile."""
    even = quantile_sizes(409, 10, "even")
    floor = quantile_sizes(409, 10, "floor")
    assert even[0] == 41 and even[-1] == 40
    assert floor[0] == 40 and floor[-1] == 49
    assert floor[-1] / floor[0] == pytest.approx(1.225, abs=0.001)


@pytest.mark.parametrize("method", ["floor", "even"])
@pytest.mark.parametrize("n", [10, 41, 100, 405, 409])
def test_the_reported_sizes_are_the_sizes_actually_dealt(method, n):
    """``quantile_sizes`` and ``quantile_labels`` must not be able to disagree.

    They did once: the even-split sizes function put the short bucket first while the
    labeller put it last, so a report would have printed one partition while the engine
    traded another.
    """
    momentum = pd.DataFrame(
        [np.arange(n, dtype=float)],
        index=pd.to_datetime(["2020-01-01"]),
        columns=[f"N{i}" for i in range(n)],
    )
    labels = quantile_labels(momentum, 10, method).iloc[0]
    histogram = tuple(int((labels == bucket).sum()) for bucket in range(10))
    assert histogram == quantile_sizes(n, 10, method)


def test_the_traded_set_is_exactly_d1(cfg003):
    prices = synthetic_prices([f"N{i:03d}" for i in range(60)], seed=3, n_days=700)
    universe = tuple(prices.close.columns)
    strategy = EquityCrossSectionalMomentum(cfg003, universe, "even")
    weights = strategy(price_panel_of(prices, universe))
    labels = quantile_labels(strategy.momentum(price_panel_of(prices, universe)), 10, "even")
    assert ((labels == 0.0) == (weights > 0)).all().all()


def price_panel_of(prices, universe):
    from trendbot.engine.panel import price_panel

    return price_panel(prices, universe)


def test_d1_is_exactly_what_the_engine_earns_open_to_open(cfg003):
    """The consistency check between section 3's position and section 8's gate.

    Section 3's traded portfolio *is* D1, so at zero cost the engine's equity measured
    open-of-rebalance to open-of-next-rebalance must equal D1's forward return series
    exactly. If they ever diverge the monotonicity gate has stopped describing the
    strategy that was run.
    """
    prices = synthetic_prices([f"N{i:03d}" for i in range(60)], seed=5, n_days=1200)
    universe = tuple(prices.close.columns)
    result = run_panel_backtest(
        prices,
        universe,
        EquityCrossSectionalMomentum(cfg003, universe, "even"),
        cost_bps=0.0,
        gross_cap=cfg003.gross_exposure_cap,
    )
    study = bucket_study(
        prices,
        universe,
        formation_days=cfg003.formation_days,
        skip_days=cfg003.skip_days,
        n_quantiles=cfg003.n_quantiles,
        method="even",
        label_prefix="D",
    )
    equity_open = result.diagnostics["equity_open"]
    open_to_open = pd.Series(
        equity_open.to_numpy()[1:] / equity_open.to_numpy()[:-1] - 1.0,
        index=equity_open.index[:-1],
    )
    shared = study.returns.index.intersection(open_to_open.index)
    assert len(shared) > 20
    delta = (study.returns.loc[shared, "D1"] - open_to_open.loc[shared]).abs().max()
    assert delta < 1e-12, f"D1 and the traded book disagree by {delta:.3e}"


def test_the_study_labels_and_counts_ten_buckets(cfg003):
    prices = synthetic_prices([f"N{i:03d}" for i in range(60)], seed=6, n_days=900)
    universe = tuple(prices.close.columns)
    study = bucket_study(
        prices,
        universe,
        formation_days=cfg003.formation_days,
        skip_days=cfg003.skip_days,
        n_quantiles=10,
        method="even",
        label_prefix="D",
    )
    assert study.labels == [f"D{i}" for i in range(1, 11)]
    assert study.spread_name == "D1-D10"
    assert list(study.returns.columns) == study.labels
    assert (study.membership.sum(axis=1) == 60).all()


# --------------------------------------------------------------------------------------
# the market regression
# --------------------------------------------------------------------------------------


def test_a_planted_alpha_and_beta_are_recovered():
    rng = np.random.default_rng(0)
    n = 4000
    index = pd.bdate_range("2008-01-02", periods=n)
    market = pd.Series(rng.normal(0.0003, 0.011, n), index=index)
    alpha_per_day = 0.06 / TRADING_DAYS_PER_YEAR
    # Residual noise kept small on purpose: with 0.4%/day of idiosyncratic vol the
    # standard error on an annualised alpha is ~1.6%, and a test asserting recovery to
    # a tighter tolerance than its own standard error is testing the seed, not the code.
    strategy = alpha_per_day + 1.3 * market + pd.Series(rng.normal(0, 0.001, n), index=index)

    reg = market_regression(strategy, market, market_label="planted")
    assert reg.beta == pytest.approx(1.3, abs=0.005)
    assert reg.alpha_annualised == pytest.approx(0.06, abs=0.015)
    assert reg.alpha_t_stat > 2.0
    assert reg.alpha_positive
    assert reg.clears(2.0)
    assert reg.n_observations == n
    assert reg.r_squared > 0.98


def test_a_pure_beta_tilt_shows_no_alpha():
    """The 002 failure mode: all beta, nothing else. Alpha must not clear."""
    rng = np.random.default_rng(1)
    n = 3000
    index = pd.bdate_range("2008-01-02", periods=n)
    market = pd.Series(rng.normal(0.0004, 0.011, n), index=index)
    strategy = 1.1 * market  # exactly a levered market, no skill at all

    reg = market_regression(strategy, market, market_label="planted")
    assert reg.beta == pytest.approx(1.1, abs=1e-9)
    assert abs(reg.alpha_annualised) < 1e-6
    # An exact fit leaves residuals at 1e-19; the t-statistic on that is one rounding
    # error over another and is reported as undefined rather than as a large number.
    assert not np.isfinite(reg.alpha_t_stat)
    assert not reg.clears(2.0)


def test_a_negative_alpha_is_reported_as_negative():
    rng = np.random.default_rng(2)
    n = 3000
    index = pd.bdate_range("2008-01-02", periods=n)
    market = pd.Series(rng.normal(0.0004, 0.011, n), index=index)
    strategy = -0.05 / TRADING_DAYS_PER_YEAR + market + pd.Series(rng.normal(0, 0.003, n), index=index)
    reg = market_regression(strategy, market, market_label="planted")
    assert reg.alpha_annualised < 0
    assert not reg.alpha_positive
    assert not reg.clears(2.0)


def test_series_are_aligned_on_their_shared_dates_not_zero_filled():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=100)
    strategy = pd.Series(rng.normal(0, 0.01, 100), index=index)
    market = pd.Series(rng.normal(0, 0.01, 80), index=index[20:])
    reg = market_regression(strategy, market, market_label="planted")
    assert reg.n_observations == 80

    with pytest.raises(ValueError, match="at least 3"):
        market_regression(strategy.iloc[:2], market.iloc[:2], market_label="x")


def test_a_market_series_with_no_dispersion_is_refused():
    """Beta is unidentified, so alpha is not a number and must not be reported as one."""
    index = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError, match="no dispersion"):
        market_regression(
            pd.Series(np.linspace(0, 0.01, 100), index=index),
            pd.Series(0.001, index=index),
            market_label="degenerate",
        )


# --------------------------------------------------------------------------------------
# section 8, applied
# --------------------------------------------------------------------------------------


def _decide(cfg, **overrides):
    kwargs = dict(
        strategy_sharpe=0.70,
        benchmark_sharpe=0.50,
        n_inversions=0,
        spread_mean=0.004,
        spread_t_stat=3.0,
        alpha_annualised=0.05,
        alpha_t_stat=3.0,
    )
    kwargs.update(overrides)
    return evaluate_decision_rule_003(cfg, **kwargs)


def test_all_four_support_clauses_are_required(cfg003):
    assert _decide(cfg003).verdict == "SUPPORTED"
    assert _decide(cfg003, strategy_sharpe=0.40).verdict != "SUPPORTED"  # not strictly above
    assert _decide(cfg003, benchmark_sharpe=0.60).verdict == "INCONCLUSIVE - do not trade"
    assert _decide(cfg003, spread_t_stat=1.5).verdict == "INCONCLUSIVE - do not trade"
    assert _decide(cfg003, alpha_t_stat=1.5).verdict == "INCONCLUSIVE - do not trade"


def test_one_inversion_is_tolerated_and_two_is_not(cfg003):
    """The change from 002's rule, pre-committed: ten buckets are noisier than five."""
    assert _decide(cfg003, n_inversions=1).verdict == "SUPPORTED"
    assert _decide(cfg003, n_inversions=2).verdict == "ABANDON"


def test_a_spread_t_below_one_abandons_however_good_the_sharpe(cfg003):
    assert _decide(cfg003, strategy_sharpe=3.0, spread_t_stat=0.9).verdict == "ABANDON"
    assert _decide(cfg003, strategy_sharpe=3.0, spread_t_stat=1.5).verdict != "ABANDON"


def test_negative_alpha_abandons_whatever_the_sharpe(cfg003):
    """Section 8: "A result that is purely a beta tilt does not count, whatever its Sharpe"."""
    assert _decide(cfg003, strategy_sharpe=3.0, alpha_annualised=-0.01).verdict == "ABANDON"
    assert _decide(cfg003, alpha_annualised=0.0).verdict == "INCONCLUSIVE - do not trade"


def test_failing_to_beat_the_benchmark_at_all_abandons(cfg003):
    assert _decide(cfg003, benchmark_sharpe=0.75).verdict == "ABANDON"
    assert _decide(cfg003, benchmark_sharpe=0.70).verdict == "ABANDON"  # "exceeds" is strict


def test_a_sharpe_below_the_floor_abandons(cfg003):
    assert _decide(cfg003, strategy_sharpe=0.14, benchmark_sharpe=0.05).verdict == "ABANDON"


def test_the_comparators_follow_the_documents_own_words(cfg003):
    exact = _decide(cfg003, strategy_sharpe=0.65, benchmark_sharpe=0.50)
    assert exact.verdict == "SUPPORTED", "'at least 0.15' is inclusive"
    assert _decide(cfg003, spread_t_stat=2.0).verdict != "SUPPORTED", "'t > 2.0' is strict"
    assert _decide(cfg003, alpha_t_stat=2.0).verdict != "SUPPORTED"


def test_the_observed_003_result_abandons_on_three_clauses(cfg003):
    """The numbers this experiment actually produced, pinned as a regression."""
    decision = evaluate_decision_rule_003(
        cfg003,
        strategy_sharpe=0.698752,
        benchmark_sharpe=0.716387,
        n_inversions=4,
        spread_mean=-0.003987,
        spread_t_stat=-0.846,
        alpha_annualised=0.0494,
        alpha_t_stat=1.742,
    )
    assert decision.verdict == "ABANDON"
    triggered = {name for name, ok, _ in decision.abandon_clauses if ok}
    assert triggered == {
        "fails to beat equal-weight buy-and-hold at all",
        "more than 1 decile inversion",
        "D1-D10 t-statistic below 1.0",
    }
