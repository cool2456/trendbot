from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from trendbot.engine import validation as v

TRADING_DAYS = 252


def _returns(n: int, *, mean: float = 0.0003, sd: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(rng.normal(mean, sd, n), index=idx)


def test_synthetic_prices_is_deterministic_for_a_given_seed():
    a = v.synthetic_prices(("AAA", "BBB"), seed=3, n_days=500)
    b = v.synthetic_prices(("AAA", "BBB"), seed=3, n_days=500)
    pd.testing.assert_frame_equal(a.close, b.close)
    pd.testing.assert_frame_equal(a.open, b.open)
    assert a.source == b.source == "synthetic(seed=3)"


def test_synthetic_prices_differs_across_seeds():
    a = v.synthetic_prices(("AAA", "BBB"), seed=3, n_days=500)
    b = v.synthetic_prices(("AAA", "BBB"), seed=4, n_days=500)
    assert not a.close.equals(b.close)
    assert (a.close.iloc[1:] != b.close.iloc[1:]).all().all()


def test_synthetic_prices_has_a_real_overnight_gap_between_close_and_next_open():
    p = v.synthetic_prices(("AAA", "BBB", "CCC"), seed=0, n_days=1000)
    prev_close = p.close.shift(1).iloc[1:]
    opens = p.open.iloc[1:]
    assert (opens != prev_close).all().all()
    log_gap = np.log(opens / prev_close)
    observed = float(log_gap.to_numpy().std(ddof=1))
    assert 0.0030 < observed < 0.0055
    assert float(log_gap.abs().to_numpy().mean()) > 1e-3


def test_synthetic_prices_gap_can_be_switched_off_which_proves_it_is_the_gap_term():
    p = v.synthetic_prices(("AAA", "BBB"), seed=0, n_days=200, overnight_gap_fraction=0.0)
    pd.testing.assert_frame_equal(
        p.open.iloc[1:], p.close.shift(1).iloc[1:], check_names=False
    )


def test_synthetic_prices_are_strictly_positive_and_share_one_calendar():
    p = v.synthetic_prices(("AAA", "BBB", "CCC"), seed=1, n_days=3000)
    assert p.close.notna().all().all()
    assert p.open.notna().all().all()
    assert (p.close.to_numpy() > 0).all()
    assert (p.open.to_numpy() > 0).all()
    assert p.open.index.equals(p.close.index)
    assert list(p.open.columns) == list(p.close.columns) == ["AAA", "BBB", "CCC"]
    assert p.open.index.is_monotonic_increasing
    assert not p.open.index.has_duplicates


def test_synthetic_prices_realises_the_requested_volatility_and_zero_drift():
    p = v.synthetic_prices(tuple(f"T{i}" for i in range(12)), seed=0, n_days=6000, annual_vol=0.16)
    daily = p.close.pct_change().dropna()
    ann_vol = float((daily.std(ddof=1) * math.sqrt(TRADING_DAYS)).mean())
    assert 0.150 < ann_vol < 0.172
    log_drift = float(np.log(p.close.iloc[-1] / p.close.iloc[0]).mean()) / (6000 / TRADING_DAYS)
    assert abs(log_drift) < 0.06


def test_synthetic_prices_with_positive_drift_actually_drifts_up():
    flat = v.synthetic_prices(("AAA",), seed=5, n_days=4000, annual_drift=0.0)
    up = v.synthetic_prices(("AAA",), seed=5, n_days=4000, annual_drift=0.10)
    assert float(up.close.iloc[-1].iloc[0]) > float(flat.close.iloc[-1].iloc[0])


def test_synthetic_prices_refuses_a_degenerate_length():
    with pytest.raises(ValueError, match="at least 2 days"):
        v.synthetic_prices(("AAA",), seed=0, n_days=1)


def test_synth_fixture_covers_the_pre_registered_universe(synth, cfg):
    assert synth.tickers == tuple(cfg.universe)
    assert len(synth.close) == 2000


def test_psr_is_exactly_one_half_when_observed_equals_the_benchmark():
    for sr in (-0.05, 0.0, 0.02, 0.08):
        assert v.probabilistic_sharpe_ratio(sr, 1000, 0.0, 3.0, sr) == pytest.approx(0.5, abs=1e-12)
    assert v.probabilistic_sharpe_ratio(0.05, 500, -0.6, 6.0, 0.05) == pytest.approx(0.5, abs=1e-12)


def test_psr_increases_monotonically_in_the_observed_sharpe():
    grid = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    values = [v.probabilistic_sharpe_ratio(sr, 1000, -0.2, 4.0, 0.02) for sr in grid]
    assert all(b > a for a, b in zip(values, values[1:])), values


def test_psr_decreases_monotonically_in_the_benchmark_sharpe():
    grid = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.08]
    values = [v.probabilistic_sharpe_ratio(0.05, 1000, -0.2, 4.0, b) for b in grid]
    assert all(b < a for a, b in zip(values, values[1:])), values


def test_psr_lies_strictly_inside_the_unit_interval():
    for sr in (-0.10, -0.01, 0.0, 0.01, 0.10):
        for n in (10, 250, 1000):
            for benchmark in (0.0, 0.03):
                p = v.probabilistic_sharpe_ratio(sr, n, -0.3, 5.0, benchmark)
                assert 0.0 < p < 1.0, (sr, n, benchmark, p)


def test_psr_increases_with_sample_size_for_a_positive_sharpe():
    values = [v.probabilistic_sharpe_ratio(0.04, n, 0.0, 3.0, 0.0) for n in (50, 250, 1000, 4000)]
    assert all(b > a for a, b in zip(values, values[1:])), values


def test_psr_is_near_one_for_a_strong_sharpe_over_many_observations():
    from scipy import stats

    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0008, 0.008, 3000))
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    assert sr * math.sqrt(TRADING_DAYS) > 0.9
    psr = v.probabilistic_sharpe_ratio(sr, len(r), skew, kurt, 0.0)
    assert psr > 0.99


def test_psr_requires_at_least_two_observations():
    with pytest.raises(ValueError, match="at least 2 observations"):
        v.probabilistic_sharpe_ratio(0.05, 1, 0.0, 3.0)


def test_psr_refuses_higher_moments_that_make_the_estimator_undefined():
    with pytest.raises(ValueError, match="non-positive"):
        v.probabilistic_sharpe_ratio(1.0, 500, 5.0, 1.0, 0.0)


def test_expected_max_sharpe_is_exactly_zero_for_a_single_configuration():
    for variance in (0.0, 1e-6, 0.01, 4.0):
        result = v.expected_max_sharpe(1, variance)
        assert result == 0.0
        assert math.isfinite(result)


def test_expected_max_sharpe_increases_with_the_number_of_configurations():
    values = [v.expected_max_sharpe(n, 0.01) for n in (2, 3, 5, 10, 50, 100, 1000)]
    assert all(x > 0 for x in values)
    assert all(b > a for a, b in zip(values, values[1:])), values
    assert values[0] == pytest.approx(0.051976, abs=1e-5)
    assert values[-2] == pytest.approx(0.253060, abs=1e-5)


def test_expected_max_sharpe_increases_with_the_trial_variance():
    values = [v.expected_max_sharpe(100, s) for s in (0.0, 1e-4, 1e-3, 1e-2, 1e-1)]
    assert values[0] == 0.0
    assert all(b > a for a, b in zip(values, values[1:])), values
    assert v.expected_max_sharpe(100, 0.04) == pytest.approx(
        2.0 * v.expected_max_sharpe(100, 0.01), rel=1e-12
    )


def test_expected_max_sharpe_rejects_fewer_than_one_configuration():
    for n in (0, -1, -100):
        with pytest.raises(ValueError, match="at least 1"):
            v.expected_max_sharpe(n, 0.01)


def test_expected_max_sharpe_rejects_a_negative_variance():
    with pytest.raises(ValueError, match="variance cannot be negative"):
        v.expected_max_sharpe(10, -0.01)


def test_deflated_sharpe_requires_the_configuration_count_as_a_positional_argument():
    r = _returns(1000)
    with pytest.raises(TypeError):
        v.deflated_sharpe_ratio(r)  # type: ignore[call-arg]

    parameter = inspect.signature(v.deflated_sharpe_ratio).parameters["n_configurations"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_deflated_sharpe_rejects_a_non_integer_configuration_count():
    r = _returns(1000)
    for bad in (1.0, "3", None, True):
        with pytest.raises(TypeError, match="must be an int"):
            v.deflated_sharpe_ratio(r, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        v.deflated_sharpe_ratio(r, 0)


def test_deflated_sharpe_requires_trial_sharpes_once_more_than_one_config_was_tried():
    r = _returns(1000)
    with pytest.raises(ValueError, match="requires trial_sharpes"):
        v.deflated_sharpe_ratio(r, 25)
    with pytest.raises(ValueError, match="at least 2 trial Sharpes"):
        v.deflated_sharpe_ratio(r, 25, trial_sharpes=[0.01])


def test_deflated_sharpe_rejects_a_degenerate_return_series():
    idx = pd.bdate_range("2000-01-03", periods=100)
    with pytest.raises(ValueError, match="zero-variance"):
        v.deflated_sharpe_ratio(pd.Series(0.0, index=idx), 1)
    with pytest.raises(ValueError, match="at least 4 observations"):
        v.deflated_sharpe_ratio(pd.Series([0.01, -0.02, 0.03]), 1)


def test_deflated_sharpe_at_one_configuration_is_the_psr_against_zero():
    from scipy import stats

    r = _returns(1500, mean=0.0005, seed=2)
    result = v.deflated_sharpe_ratio(r, 1)

    assert result.n_configurations == 1
    assert result.benchmark_sharpe == 0.0
    assert result.deflated_sharpe == result.psr_vs_zero

    sr = float(r.mean() / r.std(ddof=1))
    expected = v.probabilistic_sharpe_ratio(
        sr,
        len(r),
        float(stats.skew(r, bias=False)),
        float(stats.kurtosis(r, fisher=False, bias=False)),
        0.0,
    )
    assert result.deflated_sharpe == pytest.approx(expected, rel=1e-12)
    assert result.per_period_sharpe == pytest.approx(sr, rel=1e-12)
    assert result.annualised_sharpe == pytest.approx(sr * math.sqrt(TRADING_DAYS), rel=1e-12)
    assert "N=1" in result.note


def test_deflated_sharpe_is_strictly_below_the_naive_psr_once_more_than_one_was_tried():
    r = _returns(1500, mean=0.0005, seed=2)
    rng = np.random.default_rng(99)
    trials = rng.normal(0.0, 0.03, 40)
    for n in (2, 5, 20, 40, 500):
        result = v.deflated_sharpe_ratio(r, n, trial_sharpes=trials)
        assert result.benchmark_sharpe > 0.0
        assert result.deflated_sharpe < result.psr_vs_zero, n
        assert 0.0 < result.deflated_sharpe < 1.0


def test_deflating_harder_never_makes_a_result_look_better():
    r = _returns(1500, mean=0.0005, seed=2)
    rng = np.random.default_rng(99)
    trials = rng.normal(0.0, 0.03, 40)
    values = [
        v.deflated_sharpe_ratio(r, n, trial_sharpes=trials).deflated_sharpe
        for n in (1, 2, 5, 20, 100, 1000)
    ]
    assert all(b < a for a, b in zip(values, values[1:])), values


def test_deflated_sharpe_significance_flag_follows_the_stated_level():
    r = _returns(1500, mean=0.0005, seed=2)
    result = v.deflated_sharpe_ratio(r, 1)
    assert result.is_significant == (result.deflated_sharpe > result.significance_level)
    assert result.significance_level == 0.95
    lenient = v.deflated_sharpe_ratio(r, 1, significance_level=0.0)
    assert lenient.is_significant
    strict = v.deflated_sharpe_ratio(r, 1, significance_level=0.999999)
    assert not strict.is_significant


def test_deflated_sharpe_ignores_nan_observations_in_the_count():
    r = _returns(500, seed=4)
    padded = pd.concat([r, pd.Series([np.nan] * 10, index=pd.bdate_range("2010-01-04", periods=10))])
    assert v.deflated_sharpe_ratio(padded, 1).n_observations == 500


def test_in_sample_out_of_sample_splits_fifty_fifty_and_reports_both_halves():
    from trendbot.engine.metrics import sharpe

    r = _returns(2000, seed=1)
    split = v.in_sample_out_of_sample(r)
    assert split.first_half_n == split.second_half_n == 1000
    assert split.split_date == r.index[1000]
    assert split.first_half_sharpe == pytest.approx(sharpe(r.iloc[:1000]), rel=1e-12)
    assert split.second_half_sharpe == pytest.approx(sharpe(r.iloc[1000:]), rel=1e-12)
    assert split.asymmetry == pytest.approx(
        split.first_half_sharpe - split.second_half_sharpe, rel=1e-12
    )


def test_in_sample_out_of_sample_gives_the_extra_observation_to_the_second_half():
    r = _returns(1001, seed=1)
    split = v.in_sample_out_of_sample(r)
    assert (split.first_half_n, split.second_half_n) == (500, 501)
    assert split.first_half_n + split.second_half_n == len(r)


def test_in_sample_out_of_sample_raises_on_a_series_too_small_to_split():
    for n in (0, 1, 2, 3):
        with pytest.raises(ValueError, match="at least 4 observations"):
            v.in_sample_out_of_sample(_returns(max(n, 1)).iloc[:n])


def test_in_sample_out_of_sample_drops_nans_before_splitting():
    r = _returns(100, seed=1)
    dirty = r.copy()
    dirty.iloc[10] = np.nan
    split = v.in_sample_out_of_sample(dirty)
    assert split.first_half_n + split.second_half_n == 99


def test_walk_forward_produces_the_expected_number_of_windows():
    r = _returns(2016, seed=1)
    wf = v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    assert len(wf.windows) == 5
    assert (wf.train_years, wf.test_years, wf.step_years) == (3.0, 1.0, 1.0)

    half_step = v.walk_forward(r, train_years=2.0, test_years=1.0, step_years=0.5)
    assert len(half_step.windows) == 11


def test_walk_forward_windows_do_not_overlap_their_own_test_period():
    r = _returns(2016, seed=1)
    wf = v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    for _, row in wf.windows.iterrows():
        assert row["train_start"] < row["train_end"] < row["test_start"] < row["test_end"]
        train = r.loc[row["train_start"] : row["train_end"]]
        test = r.loc[row["test_start"] : row["test_end"]]
        assert len(train) == 756
        assert len(test) == 252
        assert train.index.intersection(test.index).empty


def test_walk_forward_test_period_starts_on_the_bar_immediately_after_training_ends():
    r = _returns(2016, seed=1)
    wf = v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    for _, row in wf.windows.iterrows():
        i = r.index.get_loc(row["train_end"])
        j = r.index.get_loc(row["test_start"])
        assert j == i + 1


def test_walk_forward_windows_advance_by_exactly_one_step():
    r = _returns(2016, seed=1)
    wf = v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    starts = [r.index.get_loc(d) for d in wf.windows["train_start"]]
    assert starts == [0, 252, 504, 756, 1008]


def test_walk_forward_aggregates_agree_with_the_window_table():
    from trendbot.engine.metrics import sharpe

    r = _returns(2016, seed=1)
    wf = v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    assert wf.mean_train_sharpe == pytest.approx(wf.windows["train_sharpe"].mean(), rel=1e-12)
    assert wf.mean_test_sharpe == pytest.approx(wf.windows["test_sharpe"].mean(), rel=1e-12)
    assert wf.gap == pytest.approx(wf.mean_train_sharpe - wf.mean_test_sharpe, rel=1e-12)
    assert 0.0 <= wf.fraction_positive <= 1.0
    first = wf.windows.iloc[0]
    assert first["train_sharpe"] == pytest.approx(sharpe(r.iloc[:756]), rel=1e-12)
    assert first["test_sharpe"] == pytest.approx(sharpe(r.iloc[756:1008]), rel=1e-12)
    assert first["test_return"] == pytest.approx(float((1 + r.iloc[756:1008]).prod() - 1), rel=1e-12)


def test_walk_forward_raises_when_the_series_cannot_hold_one_window():
    r = _returns(1000, seed=1)
    with pytest.raises(ValueError, match="need at least 1008 observations"):
        v.walk_forward(r, train_years=3.0, test_years=1.0, step_years=1.0)
    assert len(v.walk_forward(_returns(1008, seed=1)).windows) == 1
    with pytest.raises(ValueError):
        v.walk_forward(_returns(1007, seed=1))


def test_walk_forward_rejects_a_sub_period_window_length():
    r = _returns(2016, seed=1)
    for kwargs in ({"step_years": 0.0}, {"train_years": 0.0}, {"test_years": 0.001}):
        with pytest.raises(ValueError, match="at least one period"):
            v.walk_forward(r, **kwargs)


def test_the_engine_earns_nothing_on_driftless_synthetic_data(cfg):
    result = v.noise_test(cfg, seeds=(0, 1), n_days=1500)
    assert abs(result.mean_sharpe) < 0.35, result
    assert abs(result.mean_buy_and_hold_sharpe) < 0.35, result
    assert result.seeds == (0, 1)
    assert result.n_days == 1500
    assert result.annual_drift == 0.0
    assert len(result.strategy_sharpes) == 2
    assert result.mean_sharpe == pytest.approx(float(np.mean(result.strategy_sharpes)), rel=1e-12)
    assert result.passes(tolerance=0.35)


def test_noise_test_result_reports_a_readable_summary(cfg):
    result = v.noise_test(cfg, seeds=(0,), n_days=800)
    text = str(result)
    assert "noise test" in text and "800" in text
    assert math.isnan(result.std_sharpe)


@pytest.mark.slow
def test_step_5_gate_mining_noise_manufactures_a_sharpe_that_deflation_rejects(cfg):
    study = v.noise_sweep_study(cfg)

    assert len(study.sweeps) == 6
    assert all(s.n_configurations == 100 for s in study.sweeps)

    assert study.n_mined_above_one >= 1, study
    assert study.max_best_sharpe > 1.0, study

    assert study.any_significant is False, study.table.to_string()

    table = study.table
    fooled = table[(table["psr_vs_zero"] > 0.95) & (table["deflated_sharpe"] <= 0.95)]
    assert len(fooled) >= 1, table.to_string()
    assert float(fooled["psr_vs_zero"].max()) > 0.95
    assert float((table["psr_vs_zero"] - table["deflated_sharpe"]).max()) > 0.2

    assert (table["benchmark_sharpe_ann"] > 0).all()
    assert all(np.std(s.per_period_sharpes, ddof=1) > 0 for s in study.sweeps)


@pytest.mark.slow
def test_a_single_noise_sweep_mines_a_positive_sharpe_out_of_nothing(cfg):
    sweep = v.noise_sweep(cfg, seed=0, n_days=750)
    assert sweep.n_configurations == 100 == len(sweep.grid)
    assert set(sweep.best_config) == {"lookback", "halflife", "variant"}
    assert sweep.best_sharpe > 0.5
    assert sweep.best_sharpe == pytest.approx(float(sweep.grid["sharpe"].max()), rel=1e-12)
    deflated = sweep.deflated()
    assert deflated.n_configurations == 100
    assert deflated.deflated_sharpe < deflated.psr_vs_zero
    assert not deflated.is_significant
    assert "SIGNIFICANT" not in str(sweep).replace("not significant", "")
