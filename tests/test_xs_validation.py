from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_002 import load_config_002
from trendbot.data import PriceData
from trendbot.engine.panel_backtest import run_panel_backtest
from trendbot.engine.validation import synthetic_prices
from trendbot.engine.xs_validation import (
    Decision002,
    common_factor_prices,
    factor_attribution,
    evaluate_decision_rule_002,
    panel_noise_test,
    quintile_study,
    universe_verification,
    worst_months,
)
from trendbot.strategies import CrossSectionalMomentum
from trendbot.xsmom import cross_sectional_momentum


@pytest.fixture(scope="session")
def cfg002():
    return load_config_002()


def test_common_factor_prices_hold_total_volatility_flat_across_instruments(cfg002):
    prices = common_factor_prices(cfg002.universe, seed=0, n_days=3000, annual_vol=0.16)
    vols = prices.close.pct_change().std() * np.sqrt(252)
    assert vols.min() > 0.14 and vols.max() < 0.18
    assert (vols.max() - vols.min()) < 0.03


def test_common_factor_prices_are_actually_correlated(cfg002):
    corr = common_factor_prices(cfg002.universe, seed=1, n_days=3000).close.pct_change().corr()
    off_diagonal = corr.to_numpy()[np.triu_indices(cfg002.n_universe, 1)]
    assert 0.35 < off_diagonal.mean() < 0.65, off_diagonal.mean()

    independent = synthetic_prices(cfg002.universe, seed=1, n_days=3000).close.pct_change().corr()
    flat = independent.to_numpy()[np.triu_indices(cfg002.n_universe, 1)]
    assert abs(flat.mean()) < 0.05, "the independent generator is not independent"


def test_an_infeasible_factor_share_is_refused_not_clipped(cfg002):
    with pytest.raises(ValueError, match="infeasible"):
        common_factor_prices(
            cfg002.universe, seed=0, n_days=100, common_variance_share=0.95,
            beta_low=0.5, beta_high=1.5,
        )
    with pytest.raises(ValueError, match="common_variance_share"):
        common_factor_prices(cfg002.universe, seed=0, n_days=100, common_variance_share=1.0)


@pytest.mark.slow
@pytest.mark.parametrize("correlated", [False, True])
def test_the_rule_earns_nothing_on_either_null(cfg002, correlated):
    result = panel_noise_test(
        cfg002,
        label="test",
        seeds=tuple(range(8)),
        n_days=2500,
        correlated=correlated,
    )
    assert abs(result.mean_gross_sharpe) < 0.2, result
    assert abs(result.mean_buy_and_hold_sharpe) < 0.2, result
    assert result.passes()
    assert result.mean_sharpe < result.mean_gross_sharpe, "costs did not cost anything"


@pytest.mark.slow
def test_on_correlated_noise_the_rule_is_a_factor_tilt_that_earns_nothing(cfg002):
    attribution = factor_attribution(
        cfg002, seeds=tuple(range(8)), n_days=2500, common_variance_share=0.5
    )
    assert attribution.mean_beta > 0.7, "the rule is supposed to load on the factor here"
    assert abs(attribution.mean_alpha_annualised) < 0.01, attribution
    assert attribution.sharpe_correlation > 0.6, attribution


def test_factor_attribution_finds_no_tilt_where_there_is_no_factor(cfg002):
    attribution = factor_attribution(
        cfg002, seeds=(0, 1, 2), n_days=1500, common_variance_share=0.0
    )
    assert attribution.mean_beta > 0.7, (
        "with equal weights and no factor the basket is still a diversified average of "
        "the same names, so some beta survives"
    )
    assert abs(attribution.mean_alpha_annualised) < 0.02, attribution


def _hand_panel() -> PriceData:
    index = pd.bdate_range("2020-01-01", periods=130)
    frame = pd.DataFrame(100.0, index=index, columns=[f"N{i}" for i in range(10)])
    return PriceData(
        open=frame.copy(), close=frame.copy(), source="hand", adjusted=True, fetched_at=""
    )


def test_quintile_study_reads_the_forward_return_off_consecutive_rebalances(cfg002):
    from dataclasses import replace

    index = pd.bdate_range("2020-01-01", periods=90)
    names = [f"N{i}" for i in range(10)]
    close = pd.DataFrame(100.0, index=index, columns=names)

    months = index.to_period("M")
    first_of_month = index[np.r_[True, months[1:] != months[:-1]]]
    reb_a, reb_b = first_of_month[1], first_of_month[2]

    winners, losers = names[:5], names[5:]
    close.loc[: index[index.get_loc(reb_a) - 1], winners] = 200.0
    close.loc[: index[index.get_loc(reb_a) - 1], losers] = 50.0
    close.iloc[0] = 100.0
    close.loc[reb_a:, winners] = 300.0
    close.loc[reb_a:, losers] = 40.0
    close.loc[reb_b:, winners] = 600.0
    close.loc[reb_b:, losers] = 20.0

    prices = PriceData(
        open=close.copy(), close=close.copy(), source="hand", adjusted=True, fetched_at=""
    )
    cfg = replace(
        cfg002,
        sleeves={"hand": tuple(names)},
        universe=tuple(names),
        declared_universe_size=10,
        n_quantiles=2,
        declared_quantile_size=5,
        formation_days=index.get_loc(reb_a) - 1,
        skip_days=0,
    )
    study = quintile_study(prices, cfg)
    row = study.returns.loc[reb_a]
    assert row["Q1"] == pytest.approx(1.0), "winners doubled; Q1's forward return must be +100%"
    assert row["Q2"] == pytest.approx(-0.5), "losers halved; Q2's forward return must be -50%"
    assert study.membership.loc[reb_a].tolist() == [5, 5]
    assert study.is_monotonic
    assert study.spread_positive


def test_q1_is_exactly_what_the_engine_earns_open_to_open(cfg002):
    prices = synthetic_prices(cfg002.universe, seed=11, n_days=1500)
    result = run_panel_backtest(
        prices,
        cfg002.universe,
        CrossSectionalMomentum(cfg002),
        cost_bps=0.0,
        gross_cap=cfg002.gross_exposure_cap,
    )
    study = quintile_study(prices, cfg002)
    equity_open = result.diagnostics["equity_open"]
    open_to_open = pd.Series(
        equity_open.to_numpy()[1:] / equity_open.to_numpy()[:-1] - 1.0, index=equity_open.index[:-1]
    )
    common = study.returns.index.intersection(open_to_open.index)
    assert len(common) > 30
    delta = (study.returns.loc[common, "Q1"] - open_to_open.loc[common]).abs().max()
    assert delta < 1e-12, f"Q1 and the traded book disagree by {delta:.3e}"


def test_the_quintile_study_sorts_on_information_available_before_the_trade(cfg002):
    prices = synthetic_prices(cfg002.universe, seed=12, n_days=900)
    study = quintile_study(prices, cfg002)
    close = prices.close[list(cfg002.universe)]
    momentum = cross_sectional_momentum(close, cfg002.formation_days, cfg002.skip_days)
    index = close.index

    reb = study.returns.index[5]
    i = index.get_loc(reb)
    decision_row = momentum.iloc[i - 1]
    top = decision_row.rank(ascending=False, method="first") <= 8
    mark_open = prices.open[list(cfg002.universe)].ffill()
    nxt = study.returns.index[6]
    expected = float(
        (mark_open.loc[nxt][top].to_numpy() / mark_open.loc[reb][top].to_numpy() - 1.0).mean()
    )
    assert study.returns.loc[reb, "Q1"] == pytest.approx(expected, abs=1e-12)


def test_monotonicity_and_inversion_counting(cfg002):
    from trendbot.engine.xs_validation import QuintileStudy

    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=24, freq="MS"), name="rebalance")
    cols = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    membership = pd.DataFrame(8, index=idx, columns=cols)

    rng = np.random.default_rng(0)
    means = [0.05, 0.04, 0.03, 0.02, 0.01]
    clean = pd.DataFrame(
        np.array(means)[None, :] + rng.normal(0, 1e-4, size=(24, 5)), index=idx, columns=cols
    )
    study = QuintileStudy(returns=clean, membership=membership, n_quantiles=5, convention="x")
    assert study.is_monotonic
    assert study.n_inversions == 0
    assert study.spread_positive
    assert study.spread_t_stat > 10

    flipped = clean.copy()
    flipped[["Q4", "Q5"]] = flipped[["Q5", "Q4"]].to_numpy()
    study = QuintileStudy(returns=flipped, membership=membership, n_quantiles=5, convention="x")
    assert not study.is_monotonic
    assert study.n_inversions == 1
    assert study.spread_positive, "a single inversion low down still leaves Q1 above Q5"


def _decide(cfg002, sharpe, benchmark, monotonic, spread_positive=True) -> Decision002:
    return evaluate_decision_rule_002(
        cfg002,
        strategy_sharpe=sharpe,
        benchmark_sharpe=benchmark,
        monotonic=monotonic,
        spread_positive=spread_positive,
    )


def test_supported_requires_all_three_clauses(cfg002):
    assert _decide(cfg002, 0.70, 0.50, True).verdict == "SUPPORTED"
    assert _decide(cfg002, 0.40, 0.10, True).verdict != "SUPPORTED"
    assert _decide(cfg002, 0.70, 0.60, True).verdict == "INCONCLUSIVE - do not trade"
    assert _decide(cfg002, 0.70, 0.50, False).verdict == "ABANDON"


def test_non_monotonic_is_sufficient_to_abandon_however_good_the_sharpe(cfg002):
    assert _decide(cfg002, 2.00, 0.10, False).verdict == "ABANDON"
    assert _decide(cfg002, 2.00, 0.10, True).verdict == "SUPPORTED"


def test_a_positive_spread_does_not_rescue_a_broken_ordering(cfg002):
    decision = _decide(cfg002, 0.70, 0.50, False, spread_positive=True)
    assert decision.verdict == "ABANDON"
    assert decision.beats_benchmark_at_all, "the fixture is meant to clear the other clauses"
    assert decision.exceeds_min_sharpe and decision.beats_benchmark_by_margin
    triggered = {name: ok for name, ok, _ in decision.abandon_clauses}
    assert triggered["quintile ordering is non-monotonic"]
    assert not triggered["fails to beat equal-weight buy-and-hold at all"]


def test_failing_to_beat_the_benchmark_at_all_abandons(cfg002):
    assert _decide(cfg002, 0.46, 0.49, True).verdict == "ABANDON"
    assert _decide(cfg002, 0.49, 0.49, True).verdict == "ABANDON"


def test_a_sharpe_below_the_floor_abandons(cfg002):
    assert _decide(cfg002, 0.14, 0.05, True).verdict == "ABANDON"
    assert _decide(cfg002, 0.15, 0.05, True).verdict == "INCONCLUSIVE - do not trade"


def test_the_comparators_follow_the_documents_own_words(cfg002):
    exactly_the_margin = _decide(cfg002, 0.65, 0.50, True)
    assert exactly_the_margin.beats_benchmark_by_margin, "'at least 0.15' is inclusive"
    assert _decide(cfg002, 0.4000001, 0.1, True).exceeds_min_sharpe
    assert not _decide(cfg002, 0.40, 0.1, True).exceeds_min_sharpe, "'exceeds 0.40' is strict"


def test_universe_verification_flags_a_ticker_that_starts_too_late(cfg002):
    prices = synthetic_prices(cfg002.universe, seed=0, n_days=3000, start="2000-01-03")
    late = cfg002.universe[3]
    close = prices.close.copy()
    close.loc[: pd.Timestamp("2010-01-01"), late] = np.nan
    prices = PriceData(
        open=prices.open, close=close, source="x", adjusted=True, fetched_at=""
    )
    verification = universe_verification(prices, cfg002)
    assert verification.dropped == (late,)
    assert verification.n_kept == cfg002.n_universe - 1
    assert late not in verification.kept


def test_worst_months_compounds_within_the_month():
    index = pd.bdate_range("2020-01-01", periods=60)
    returns = pd.Series(0.0, index=index)
    jan = index[index.month == 1]
    returns.loc[jan[0]] = -0.10
    returns.loc[jan[1]] = -0.10
    worst = worst_months(returns, n=2)
    assert str(worst.index[0]) == "2020-01"
    assert worst.iloc[0, 0] == pytest.approx(0.9 * 0.9 - 1.0)
