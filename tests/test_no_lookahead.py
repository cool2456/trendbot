from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trendbot
from trendbot.data import PriceData
from trendbot.engine import backtest as backtest_module
from trendbot.engine.backtest import lag_for_execution, rebalance_dates, run_backtest
from trendbot.engine.metrics import sharpe
from trendbot.engine.validation import synthetic_prices
from trendbot.signal import trend_signal

PACKAGE_DIR = Path(trendbot.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

NEGATIVE_SHIFT = re.compile(r"shift\(\s*(?:periods\s*=\s*)?-")


def _package_sources() -> list[Path]:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    if SCRIPTS_DIR.is_dir():
        files += sorted(SCRIPTS_DIR.rglob("*.py"))
    assert files, "found no python sources to scan; the scan would be vacuous"
    return files


def _replace_prices_from(prices: PriceData, j: int, *, mode: str) -> PriceData:
    assert 1 <= j < len(prices.close)
    open_, close = prices.open.copy(), prices.close.copy()
    n_tail, n_cols = len(close) - j, close.shape[1]

    if mode == "scale":
        rng = np.random.default_rng(12345)
        factor = rng.uniform(3.0, 9.0, size=n_cols)
        open_.iloc[j:, :] = open_.iloc[j:, :].to_numpy() * factor
        close.iloc[j:, :] = close.iloc[j:, :].to_numpy() * factor
    elif mode == "noise":
        rng = np.random.default_rng(999)
        anchor = close.iloc[j - 1].to_numpy()
        tail = anchor * np.exp(np.cumsum(rng.normal(0.0, 0.04, size=(n_tail, n_cols)), axis=0))
        close.iloc[j:, :] = tail
        open_.iloc[j:, :] = tail * np.exp(rng.normal(0.0, 0.02, size=(n_tail, n_cols)))
    else:  # pragma: no cover
        raise ValueError(f"unknown mode {mode!r}")

    return PriceData(open=open_, close=close, source=f"perturbed@{j}:{mode}", adjusted=True, fetched_at="")


def _assert_identical_before(base, alt, cut: pd.Timestamp) -> None:
    for name in ("returns", "gross_returns", "equity", "costs"):
        a, b = getattr(base, name), getattr(alt, name)
        pd.testing.assert_series_equal(a[a.index < cut], b[b.index < cut], check_exact=True, obj=name)
    for name in ("weights", "trades"):
        a, b = getattr(base, name), getattr(alt, name)
        pd.testing.assert_frame_equal(a[a.index < cut], b[b.index < cut], check_exact=True, obj=name)
    a, b = base.targets, alt.targets
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut], check_exact=True, obj="targets")


def _cut_positions(index: pd.DatetimeIndex) -> dict[str, int]:
    rebals = rebalance_dates(index)
    on_rebalance = index.get_loc(rebals[len(rebals) // 2])
    mid_month = on_rebalance + 9
    assert index[mid_month] not in set(rebals), "the 'mid-month' cut landed on a rebalance"
    return {"rebalance_boundary": on_rebalance, "mid_month": mid_month}


@pytest.mark.parametrize("cut_kind", ["rebalance_boundary", "mid_month"])
@pytest.mark.parametrize("mode", ["scale", "noise"])
def test_no_future_bar_can_change_a_past_decision(synth, cfg, cut_kind, mode):
    j = _cut_positions(synth.close.index)[cut_kind]
    cut = synth.close.index[j]

    base = run_backtest(synth, cfg)
    alt = run_backtest(_replace_prices_from(synth, j, mode=mode), cfg)

    _assert_identical_before(base, alt, cut)

    assert not base.returns[base.returns.index >= cut].equals(alt.returns[alt.returns.index >= cut])
    assert len(base.returns[base.returns.index < cut]) > 1000
    assert len(base.targets[base.targets.index < cut]) > 40


def test_the_harness_detects_the_engines_own_lag_switch(synth, cfg):
    j = _cut_positions(synth.close.index)["rebalance_boundary"]
    cut = synth.close.index[j]
    base = run_backtest(synth, cfg, _unsafe_disable_execution_lag=True)
    alt = run_backtest(
        _replace_prices_from(synth, j, mode="scale"), cfg, _unsafe_disable_execution_lag=True
    )
    with pytest.raises(AssertionError):
        _assert_identical_before(base, alt, cut)


def test_saturated_caps_make_the_targets_independent_of_the_vol_estimate(synth, cfg):
    result = run_backtest(synth, cfg)
    diagnostics = result.diagnostics
    saturated = diagnostics[(diagnostics.n_capped == diagnostics.n_active) & (diagnostics.n_active > 1)]
    assert len(saturated) > 0, "the fixture must contain a saturated rebalance or this proves nothing"
    for date, row in saturated.iterrows():
        weights = result.targets.loc[date]
        live = weights[weights != 0]
        assert len(live) == int(row.n_active)
        assert live.round(12).nunique() == 1, f"{date}: saturated book should be equal-weight"
        expected = min(cfg.per_instrument_cap, 1.0 / row.n_active)
        assert live.iloc[0] == pytest.approx(expected)


def test_the_causality_harness_detects_a_five_bar_peek(synth, cfg):
    index = synth.close.index
    rebals = rebalance_dates(index)
    reb = rebals[len(rebals) // 2]
    j = index.get_loc(reb) + 3
    cut = index[j]
    alt_prices = _replace_prices_from(synth, j, mode="scale")

    lookback = cfg.lookback_days
    base = run_backtest(synth, cfg, signal=trend_signal(synth.close.shift(-5), lookback, long_only=cfg.long_only))
    alt = run_backtest(alt_prices, cfg, signal=trend_signal(alt_prices.close.shift(-5), lookback, long_only=cfg.long_only))

    before_base = base.targets[base.targets.index < cut]
    before_alt = alt.targets[alt.targets.index < cut]
    assert not before_base.equals(before_alt), "harness failed to notice a 5-bar peek"
    assert float((before_base - before_alt).abs().to_numpy().max()) > 0.01


def _same_bar_return_signal(prices: PriceData, cfg) -> pd.DataFrame:
    return np.sign(prices.close[list(cfg.universe)].pct_change(fill_method=None)).fillna(0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_execution_lag_is_present_and_load_bearing(cfg, seed):
    prices = synthetic_prices(cfg.universe, seed=seed, n_days=2000)
    sig = _same_bar_return_signal(prices, cfg)

    lagged = run_backtest(prices, cfg, signal=sig)
    unlagged = run_backtest(prices, cfg, signal=sig, _unsafe_disable_execution_lag=True)

    sr_lagged, sr_unlagged = sharpe(lagged.returns), sharpe(unlagged.returns)
    assert abs(sr_lagged) < 0.5, f"lagged Sharpe {sr_lagged:+.3f} - the shift is not suppressing the edge"
    assert sr_unlagged > 0.6
    assert sr_unlagged - sr_lagged > 0.7

    rebals = rebalance_dates(prices.close.index)
    lagged_reb, unlagged_reb = lagged.returns.loc[rebals], unlagged.returns.loc[rebals]
    t_lagged = lagged_reb.mean() / lagged_reb.std(ddof=1) * np.sqrt(len(lagged_reb))
    t_unlagged = unlagged_reb.mean() / unlagged_reb.std(ddof=1) * np.sqrt(len(unlagged_reb))

    assert t_unlagged > 10.0, "removing the shift did not create a fortune, so the shift does nothing"
    assert abs(t_lagged) < 4.0, f"lagged rebalance-day t-stat {t_lagged:+.2f} is too large to be costs"
    assert lagged_reb.mean() < 0.0


PEEK_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


@pytest.fixture(scope="module")
def nl_peek_sharpes():
    from trendbot.config import load_config

    cfg = load_config()
    lagged, unlagged = [], []
    for seed in PEEK_SEEDS:
        prices = synthetic_prices(cfg.universe, seed=seed, n_days=2000)
        peek = trend_signal(prices.close.shift(-1), cfg.lookback_days, long_only=cfg.long_only)
        lagged.append(sharpe(run_backtest(prices, cfg, signal=peek).returns))
        unlagged.append(
            sharpe(run_backtest(prices, cfg, signal=peek, _unsafe_disable_execution_lag=True).returns)
        )
    return np.array(lagged), np.array(unlagged)


def test_signal_built_from_close_shift_minus_one_earns_no_abnormal_return(nl_peek_sharpes):
    lagged, _ = nl_peek_sharpes
    mean_sharpe = float(lagged.mean())
    assert abs(mean_sharpe) < 0.3, f"mean peek Sharpe {mean_sharpe:+.3f} - noise should pay nothing"


def test_the_literal_peek_test_cannot_discriminate(nl_peek_sharpes):
    _, unlagged = nl_peek_sharpes
    mean_unlagged = float(unlagged.mean())
    assert abs(mean_unlagged) < 0.3, (
        "the unlagged peek run now earns something, which would make the literal "
        f"test discriminating after all (mean Sharpe {mean_unlagged:+.3f})"
    )


def test_lag_for_execution_is_the_only_shift():
    offenders = []
    for path in _package_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if NEGATIVE_SHIFT.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "negative shift(s) found in production source:\n" + "\n".join(offenders)


def test_lag_for_execution_shifts_by_exactly_one_bar():
    index = pd.date_range("2020-01-06", periods=5, freq="B")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=index)

    lagged = lag_for_execution(frame)

    assert list(lagged.index) == list(index)
    assert list(lagged.columns) == ["A", "B"]
    assert lagged.iloc[0].isna().all(), "row 0 must be unknowable: there is no bar -1"
    assert lagged["A"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0]
    assert lagged["B"].tolist()[1:] == [10.0, 20.0, 30.0, 40.0]
    assert not np.allclose(lagged.iloc[1:].to_numpy(), frame.iloc[1:].to_numpy())
    assert not np.allclose(lagged.iloc[2:].to_numpy(), frame.iloc[:-2].to_numpy())

    series = frame["A"]
    lagged_series = lag_for_execution(series)
    assert pd.isna(lagged_series.iloc[0])
    assert lagged_series.tolist()[1:] == [1.0, 2.0, 3.0, 4.0]


def test_the_signal_and_the_vol_estimate_are_themselves_backward_looking(synth, cfg):
    from trendbot.sizing import annualised_vol

    j = _cut_positions(synth.close.index)["mid_month"]
    alt = _replace_prices_from(synth, j, mode="scale")

    for frame_of in (
        lambda p: trend_signal(p.close, cfg.lookback_days, long_only=cfg.long_only),
        lambda p: annualised_vol(p.close.pct_change(fill_method=None), cfg.ewma_halflife_days),
    ):
        a, b = frame_of(synth), frame_of(alt)
        pd.testing.assert_frame_equal(a.iloc[:j], b.iloc[:j], check_exact=True)
        assert not a.iloc[j:].equals(b.iloc[j:])


def test_rebalance_uses_the_previous_bars_signal(synth, cfg):
    index = synth.close.index
    result = run_backtest(synth, cfg)
    diagnostics = result.diagnostics

    assert not diagnostics.empty
    assert list(diagnostics.index) == list(rebalance_dates(index))
    for date, decision_date in diagnostics["decision_date"].items():
        assert decision_date == index[index.get_loc(date) - 1]
        assert decision_date < date
        assert (decision_date.year, decision_date.month) != (date.year, date.month)

    reb = diagnostics.index[len(diagnostics) // 2]
    previous_bar = index[index.get_loc(reb) - 1]
    blank = pd.DataFrame(0.0, index=index, columns=list(cfg.universe))

    on_previous_bar = blank.copy()
    on_previous_bar.loc[previous_bar] = 1.0
    on_the_rebalance = blank.copy()
    on_the_rebalance.loc[reb] = 1.0

    from_previous = run_backtest(synth, cfg, signal=on_previous_bar).targets.loc[reb]
    from_same_bar = run_backtest(synth, cfg, signal=on_the_rebalance).targets.loc[reb]

    assert float(from_previous.abs().sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(from_same_bar.abs().sum()) == 0.0


def test_no_production_call_site_disables_the_lag():
    flag = "_unsafe_disable_execution_lag"
    users = {p for p in _package_sources() if flag in p.read_text(encoding="utf-8")}
    definition = Path(backtest_module.__file__).resolve()

    assert users == {definition}, (
        "the test-only lookahead escape hatch is referenced outside its definition: "
        f"{sorted(str(p.relative_to(REPO_ROOT)) for p in users - {definition})}"
    )

    text = definition.read_text(encoding="utf-8")
    assert not re.search(rf"{flag}\s*=\s*True", text), "backtest.py enables the escape hatch itself"
    assert f"{flag}: bool = False" in text, "the escape hatch is no longer off by default"


def test_rebalance_dates_are_first_trading_day_of_each_month():
    index = pd.DatetimeIndex(
        [
            "2020-01-02", "2020-01-03", "2020-01-31",
            "2020-02-03", "2020-02-04", "2020-02-28",
            "2020-03-02", "2020-03-03",
            "2020-04-01",
        ]
    )

    dates = rebalance_dates(index)

    assert list(dates) == [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02"), pd.Timestamp("2020-04-01")]
    for date in dates:
        month = index[(index.year == date.year) & (index.month == date.month)]
        assert date == month[0]


def test_the_first_rebalance_of_the_sample_is_dropped(synth):
    index = synth.close.index
    dates = rebalance_dates(index)

    first_of_month = index[np.r_[True, index.to_period("M")[1:] != index.to_period("M")[:-1]]]
    assert index[0] not in set(dates), "the very first bar cannot be a rebalance"
    assert first_of_month[0] not in set(dates)
    assert list(dates) == list(first_of_month[1:])
    assert len(dates) == len(first_of_month) - 1

    for date in dates:
        assert index.get_loc(date) >= 1


def test_rebalance_dates_refuses_an_index_it_cannot_reason_about():
    with pytest.raises(TypeError):
        rebalance_dates(pd.RangeIndex(10))


def test_a_missing_close_marks_at_the_last_known_price(cfg):
    import numpy as np
    import pandas as pd

    from trendbot.data import PriceData
    from trendbot.engine.backtest import run_backtest

    n = 700
    index = pd.bdate_range("2020-01-01", periods=n)
    tickers = list(cfg.universe)
    path = 100.0 * (1.0004 ** np.arange(n))
    close = pd.DataFrame({t: path for t in tickers}, index=index)
    opens = close.shift(1).bfill()
    intact = PriceData(open=opens, close=close, source="t", adjusted=True, fetched_at="")

    holed_close = close.copy()
    hole_at = 600
    holed_close.iloc[hole_at, 0] = np.nan
    holed = PriceData(open=opens, close=holed_close, source="t", adjusted=True, fetched_at="")

    base = run_backtest(intact, cfg)
    alt = run_backtest(holed, cfg)

    pd.testing.assert_series_equal(
        base.returns.iloc[:hole_at], alt.returns.iloc[:hole_at], check_exact=False, atol=1e-15
    )
    one_day = 0.0004
    assert abs(alt.returns.iloc[hole_at] - base.returns.iloc[hole_at]) < one_day
    assert abs(alt.returns.iloc[hole_at + 1] - base.returns.iloc[hole_at + 1]) < one_day
    tail = slice(hole_at + 2, None)
    pd.testing.assert_series_equal(
        base.returns.iloc[tail], alt.returns.iloc[tail], check_exact=False, atol=1e-12
    )
    assert base.equity.iloc[-1] == pytest.approx(alt.equity.iloc[-1], rel=1e-9)
