"""HARD INVARIANT 1 - no lookahead, structurally enforced.

BUILD_PROMPT.md::

    Signal from bar t's close fills at bar t+1's open, via a .shift() inside the
    engine. Write a test proving a strategy fed close.shift(-1) earns no abnormal
    return.

What each test in this file actually proves, stated honestly, because the three
claims are not the same claim:

* ``test_no_future_bar_can_change_a_past_decision`` proves **causality**: replacing
  every price from row ``j`` onwards cannot move a single return, held weight or
  target weight dated before ``j``. This is the decisive structural test. It is not
  a statistical argument and it cannot pass by luck.
  ``test_the_causality_harness_detects_a_five_bar_peek`` is its negative control:
  the same harness, pointed at a signal that genuinely reads five bars ahead,
  fails. Without that, a bug in the harness would look like a clean engine.

* ``test_execution_lag_is_present_and_load_bearing`` proves the lag is **exactly one
  bar and is what suppresses the edge**, by running the same pathological signal
  with the shift on and with it switched off.

* ``test_signal_built_from_close_shift_minus_one_earns_no_abnormal_return`` is the
  brief's literal test, and **on its own it proves nothing at all**. Measured on
  this engine, seeds 0-7 x 2000 driftless days:

      trend_signal(close.shift(-1)) with the execution lag ON  -> mean Sharpe +0.029
      trend_signal(close.shift(-1)) with the execution lag OFF -> mean Sharpe +0.122

  Both are inside the +/-0.3 band the brief asks for, so the literal test passes
  whether or not the engine lags anything. That is not a defect in the engine; it
  is a defect in the test as a piece of evidence. A 252-day momentum sign barely
  changes when you shove the price series one bar sideways, and driftless noise has
  no edge to steal in the first place. ``test_the_literal_peek_test_cannot_
  discriminate`` asserts that non-discrimination explicitly so the weakness is
  recorded in the suite rather than assumed away.

Every number quoted in a comment below was observed on this machine with the seeds
written into the test; the assertion bounds are set well outside them.
"""

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

# shift(-1), shift( -1 ), shift(periods=-1) - any way of asking pandas to look ahead
NEGATIVE_SHIFT = re.compile(r"shift\(\s*(?:periods\s*=\s*)?-")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _package_sources() -> list[Path]:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    if SCRIPTS_DIR.is_dir():
        files += sorted(SCRIPTS_DIR.rglob("*.py"))
    assert files, "found no python sources to scan; the scan would be vacuous"
    return files


def _replace_prices_from(prices: PriceData, j: int, *, mode: str) -> PriceData:
    """Return a panel identical to ``prices`` up to row ``j-1`` and different after.

    ``mode="scale"`` multiplies every price from row ``j`` on by a large per-ticker
    factor; ``mode="noise"`` throws the tail away and replaces it with an unrelated
    random walk of four times the volatility. Both are deterministic.
    """
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
    else:  # pragma: no cover - guards a typo in a parametrisation
        raise ValueError(f"unknown mode {mode!r}")

    return PriceData(open=open_, close=close, source=f"perturbed@{j}:{mode}", adjusted=True, fetched_at="")


def _assert_identical_before(base, alt, cut: pd.Timestamp) -> None:
    """Everything the perturbation must not have touched, checked bit-for-bit.

    Two different cut-offs, and the difference is the whole point:

    * Realised quantities - returns, equity, costs, held weights - are compared
      *strictly before* ``cut``. The bar at ``cut`` has a perturbed price, so its
      return legitimately changes even in a perfectly causal engine.
    * **The target weights are compared up to and including ``cut``.** The targets for
      a rebalance on ``cut`` are computed from the signal at ``cut - 1``, which the
      perturbation never touched, so a causal engine must produce exactly the same
      targets. This is the row that catches a one-bar leak, and comparing only
      ``< cut`` misses it: a one-bar lookahead moves precisely the decision made for
      ``cut`` and nothing earlier. ``test_the_harness_detects_the_engines_own_lag_switch``
      below proves this cut-off is doing that work.
    """
    for name in ("returns", "gross_returns", "equity", "costs"):
        a, b = getattr(base, name), getattr(alt, name)
        pd.testing.assert_series_equal(a[a.index < cut], b[b.index < cut], check_exact=True, obj=name)
    for name in ("weights", "trades"):
        a, b = getattr(base, name), getattr(alt, name)
        pd.testing.assert_frame_equal(a[a.index < cut], b[b.index < cut], check_exact=True, obj=name)
    # Only the TARGET is a pure decision. The executed trade is target minus current,
    # and the current book is marked to the perturbed open of ``cut``, so it moves for
    # an entirely causal reason.
    a, b = base.targets, alt.targets
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut], check_exact=True, obj="targets")


def _cut_positions(index: pd.DatetimeIndex) -> dict[str, int]:
    """A rebalance boundary and a mid-month bar, both deep inside the sample."""
    rebals = rebalance_dates(index)
    on_rebalance = index.get_loc(rebals[len(rebals) // 2])
    mid_month = on_rebalance + 9
    assert index[mid_month] not in set(rebals), "the 'mid-month' cut landed on a rebalance"
    return {"rebalance_boundary": on_rebalance, "mid_month": mid_month}


# --------------------------------------------------------------------------------------
# (a) the decisive structural proof
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("cut_kind", ["rebalance_boundary", "mid_month"])
@pytest.mark.parametrize("mode", ["scale", "noise"])
def test_no_future_bar_can_change_a_past_decision(synth, cfg, cut_kind, mode):
    """Rewriting the future leaves the past bit-identical.

    This is the structural statement of no-lookahead. A causal engine cannot fail
    it; an engine with a leak of any size, anywhere - signal, vol estimate,
    covariance, rebalance calendar - fails it immediately, because the leak makes a
    pre-``j`` number a function of a post-``j`` price.
    """
    j = _cut_positions(synth.close.index)[cut_kind]
    cut = synth.close.index[j]

    base = run_backtest(synth, cfg)
    alt = run_backtest(_replace_prices_from(synth, j, mode=mode), cfg)

    _assert_identical_before(base, alt, cut)

    # ...and the perturbation must actually have done something, or the test above
    # is comparing two identical runs and proving nothing.
    assert not base.returns[base.returns.index >= cut].equals(alt.returns[alt.returns.index >= cut])
    assert len(base.returns[base.returns.index < cut]) > 1000
    assert len(base.targets[base.targets.index < cut]) > 40


def test_the_harness_detects_the_engines_own_lag_switch(synth, cfg):
    """The negative control that matters most.

    Point the identical harness at the engine with its execution lag removed - the
    smallest possible lookahead, exactly one bar - and it must fail. Without this the
    perturbation test could be silently blind to a one-bar leak and nobody would
    know, because a blind harness and a clean engine look identical from outside.

    ``scale`` rather than ``noise``, deliberately. The signal is a *sign*, so a
    perturbation only reaches the target weights if it flips one; a few percent of
    extra noise usually does not. And the vol estimate cannot rescue it whenever the
    per-instrument cap saturates - if every active instrument clips at 0.25, the
    normalised target is exactly ``1/n_active`` no matter what sigma says (see
    ``test_saturated_caps_make_the_targets_independent_of_the_vol_estimate``). A large
    multiplicative shift reliably flips signs and so reliably exposes the leak.
    """
    j = _cut_positions(synth.close.index)["rebalance_boundary"]
    cut = synth.close.index[j]
    base = run_backtest(synth, cfg, _unsafe_disable_execution_lag=True)
    alt = run_backtest(
        _replace_prices_from(synth, j, mode="scale"), cfg, _unsafe_disable_execution_lag=True
    )
    with pytest.raises(AssertionError):
        _assert_identical_before(base, alt, cut)


def test_saturated_caps_make_the_targets_independent_of_the_vol_estimate(synth, cfg):
    """When every active instrument clips at 0.25, vol targeting stops doing anything.

    Section 4 calls volatility "the only adaptive component". It is switched off
    entirely whenever ``k`` is large enough to push every live instrument into the
    per-instrument cap: clipping them all to 0.25 and then scaling to gross 1.0 gives
    ``1/n_active`` each, whatever their volatilities were. Recorded here because it is
    a real property of the pre-registered rule, not a bug, and because it is why the
    negative control above cannot use a small perturbation.
    """
    result = run_backtest(synth, cfg)
    diagnostics = result.diagnostics
    saturated = diagnostics[(diagnostics.n_capped == diagnostics.n_active) & (diagnostics.n_active > 1)]
    assert len(saturated) > 0, "the fixture must contain a saturated rebalance or this proves nothing"
    for date, row in saturated.iterrows():
        weights = result.targets.loc[date]
        live = weights[weights != 0]
        assert len(live) == int(row.n_active)
        assert live.round(12).nunique() == 1, f"{date}: saturated book should be equal-weight"
        # Every instrument clips to 0.25; the gross cap then only bites once there are
        # more than four of them, at which point the common weight becomes 1/n_active.
        expected = min(cfg.per_instrument_cap, 1.0 / row.n_active)
        assert live.iloc[0] == pytest.approx(expected)


def test_the_causality_harness_detects_a_five_bar_peek(synth, cfg):
    """Negative control: the harness of (a) fails on a signal that really peeks.

    ``close.shift(-5)`` puts the future five bars inside the signal, so the target
    weights at a rebalance four bars *before* the cut become a function of prices
    at and after the cut. If this test ever passes, the test above has stopped
    being able to detect anything.
    """
    index = synth.close.index
    rebals = rebalance_dates(index)
    reb = rebals[len(rebals) // 2]
    j = index.get_loc(reb) + 3  # the cut sits inside the 5-bar peek window of ``reb``
    cut = index[j]
    alt_prices = _replace_prices_from(synth, j, mode="scale")

    lookback = cfg.lookback_days
    base = run_backtest(synth, cfg, signal=trend_signal(synth.close.shift(-5), lookback, long_only=cfg.long_only))
    alt = run_backtest(alt_prices, cfg, signal=trend_signal(alt_prices.close.shift(-5), lookback, long_only=cfg.long_only))

    before_base = base.targets[base.targets.index < cut]
    before_alt = alt.targets[alt.targets.index < cut]
    assert not before_base.equals(before_alt), "harness failed to notice a 5-bar peek"
    # observed max target-weight divergence on a pre-cut rebalance: 0.125
    assert float((before_base - before_alt).abs().to_numpy().max()) > 0.01


# --------------------------------------------------------------------------------------
# (b) the discriminating test: the lag exists and is load-bearing
# --------------------------------------------------------------------------------------


def _same_bar_return_signal(prices: PriceData, cfg) -> pd.DataFrame:
    """sign of bar t's own close-to-close return - known only at the close of bar t."""
    return np.sign(prices.close[list(cfg.universe)].pct_change(fill_method=None)).fillna(0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_execution_lag_is_present_and_load_bearing(cfg, seed):
    """The same clairvoyant-by-one-bar signal is worthless lagged and rich unlagged.

    The signal is a pure function of bar ``t``'s own return. Lagged, the position
    carried into bar ``t+1`` is driven by bar ``t``'s return, which is independent
    of everything that follows, so there is nothing to earn. Unlagged, the engine
    buys at the open of bar ``t`` already knowing the sign of bar ``t``'s close.

    Observed, seeds 0-3 x 2000 days:

        full-sample Sharpe   lagged  -0.043 -0.166 +0.129 -0.217
                             unlagged +1.531 +1.801 +1.477 +0.838

    The full-sample Sharpe understates the leak badly, because the engine only
    rebalances monthly: one bar of foresight per month is diluted by twenty bars of
    noise. BUILD_PROMPT's suggested "> 3" is therefore unreachable through that
    statistic on this engine, and asserting it would be asserting something false.
    The undiluted statistic is the rebalance-day return itself, where the peek is
    actually cashed:

        mean rebalance-day return   lagged  -0.07%  (t = -1.4 to -2.3)
                                    unlagged +0.73%  (t = +25.5 to +30.8)

    The lagged rebalance-day mean is reliably *negative* because that is the day the
    5 bps per side is paid - a cost, not a signal.
    """
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
    assert lagged_reb.mean() < 0.0  # rebalance day is when the transaction cost lands


# --------------------------------------------------------------------------------------
# (c) the brief's literal test, plus an honest account of what it is worth
# --------------------------------------------------------------------------------------

PEEK_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


@pytest.fixture(scope="module")
def nl_peek_sharpes():
    """Sharpe of ``trend_signal(close.shift(-1))`` with and without the execution lag."""
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
    """The literal invariant-1 test: peeking one bar buys nothing on driftless noise.

    Observed mean over seeds 0-7 x 2000 days: **+0.029**. Individual seeds range
    -0.74 to +0.42, which is ordinary sampling noise for a 2000-day Sharpe
    (se ~ 0.36) and is exactly why the assertion is on the mean.

    Read the module docstring before treating this as evidence of anything: it is a
    statement that driftless noise contains no edge, not a statement about the
    engine's shift.
    """
    lagged, _ = nl_peek_sharpes
    mean_sharpe = float(lagged.mean())
    assert abs(mean_sharpe) < 0.3, f"mean peek Sharpe {mean_sharpe:+.3f} - noise should pay nothing"


def test_the_literal_peek_test_cannot_discriminate(nl_peek_sharpes):
    """The test above passes with the execution lag switched off, too.

    Observed mean with the lag removed: **+0.122**, comfortably inside the same
    +/-0.3 band. Recorded as an assertion so that the suite states, in executable
    form, that ``test_signal_built_from_close_shift_minus_one_earns_no_abnormal_
    return`` is not a proof of hard invariant 1. The proofs are
    ``test_no_future_bar_can_change_a_past_decision`` and
    ``test_execution_lag_is_present_and_load_bearing``.
    """
    _, unlagged = nl_peek_sharpes
    mean_unlagged = float(unlagged.mean())
    assert abs(mean_unlagged) < 0.3, (
        "the unlagged peek run now earns something, which would make the literal "
        f"test discriminating after all (mean Sharpe {mean_unlagged:+.3f})"
    )


# --------------------------------------------------------------------------------------
# (d) the shift is unique, and it is +1
# --------------------------------------------------------------------------------------


def test_lag_for_execution_is_the_only_shift():
    """No module under ``trendbot/`` or ``scripts/`` contains a negative shift."""
    offenders = []
    for path in _package_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if NEGATIVE_SHIFT.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "negative shift(s) found in production source:\n" + "\n".join(offenders)


def test_lag_for_execution_shifts_by_exactly_one_bar():
    """Behavioural check on the single shift, on a hand-built frame.

    Asserted against the values themselves rather than against ``frame.shift(1)``,
    so that the test does not restate the implementation.
    """
    index = pd.date_range("2020-01-06", periods=5, freq="B")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=index)

    lagged = lag_for_execution(frame)

    assert list(lagged.index) == list(index)
    assert list(lagged.columns) == ["A", "B"]
    assert lagged.iloc[0].isna().all(), "row 0 must be unknowable: there is no bar -1"
    assert lagged["A"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0]
    assert lagged["B"].tolist()[1:] == [10.0, 20.0, 30.0, 40.0]
    # exactly one bar: not zero (contemporaneous) and not two (stale)
    assert not np.allclose(lagged.iloc[1:].to_numpy(), frame.iloc[1:].to_numpy())
    assert not np.allclose(lagged.iloc[2:].to_numpy(), frame.iloc[:-2].to_numpy())

    series = frame["A"]
    lagged_series = lag_for_execution(series)
    assert pd.isna(lagged_series.iloc[0])
    assert lagged_series.tolist()[1:] == [1.0, 2.0, 3.0, 4.0]


def test_the_signal_and_the_vol_estimate_are_themselves_backward_looking(synth, cfg):
    """The two inputs the engine lags must already be causal before it lags them.

    A backward-looking shift in the engine cannot repair a signal that peeks, so
    both inputs are checked directly: rewriting prices from row ``j`` on must leave
    every signal and sigma row before ``j`` untouched.
    """
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


# --------------------------------------------------------------------------------------
# (e) the rebalance consumes the previous bar's signal
# --------------------------------------------------------------------------------------


def test_rebalance_uses_the_previous_bars_signal(synth, cfg):
    """``decision_date`` is the row immediately before the rebalance, and is obeyed.

    Two independent statements. First the diagnostics: every rebalance records a
    decision date that is exactly the preceding index entry, and - because a
    rebalance is the first trading day of a month - it always falls in the previous
    month. Then a behavioural probe: a signal frame that is zero everywhere except
    one row produces a position only when that row is the decision date, and
    nothing at all when it is the rebalance date itself.
    """
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

    # the engine acted on the previous close's signal: fully invested, gross 1.0
    assert float(from_previous.abs().sum()) == pytest.approx(1.0, abs=1e-9)
    # and ignored a signal dated on the rebalance bar itself
    assert float(from_same_bar.abs().sum()) == 0.0


# --------------------------------------------------------------------------------------
# (f) nothing in production turns the lag off
# --------------------------------------------------------------------------------------


def test_no_production_call_site_disables_the_lag():
    """``_unsafe_disable_execution_lag`` exists only where it is defined."""
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


# --------------------------------------------------------------------------------------
# (g) the rebalance calendar
# --------------------------------------------------------------------------------------


def test_rebalance_dates_are_first_trading_day_of_each_month():
    """First *trading* day, not first calendar day, and one per month.

    1 Feb 2020 and 1 Mar 2020 were weekends and 1 Jan is a holiday, so a calendar
    reading of section 5 would pick dates that are not in the index at all.
    """
    index = pd.DatetimeIndex(
        [
            "2020-01-02", "2020-01-03", "2020-01-31",  # Jan (1 Jan is a holiday)
            "2020-02-03", "2020-02-04", "2020-02-28",  # Feb (1 Feb is a Saturday)
            "2020-03-02", "2020-03-03",                # Mar (1 Mar is a Sunday)
            "2020-04-01",
        ]
    )

    dates = rebalance_dates(index)

    assert list(dates) == [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02"), pd.Timestamp("2020-04-01")]
    for date in dates:
        month = index[(index.year == date.year) & (index.month == date.month)]
        assert date == month[0]


def test_the_first_rebalance_of_the_sample_is_dropped(synth):
    """There is no previous bar to take a signal from, so the first month is skipped."""
    index = synth.close.index
    dates = rebalance_dates(index)

    first_of_month = index[np.r_[True, index.to_period("M")[1:] != index.to_period("M")[:-1]]]
    assert index[0] not in set(dates), "the very first bar cannot be a rebalance"
    assert first_of_month[0] not in set(dates)
    assert list(dates) == list(first_of_month[1:])
    assert len(dates) == len(first_of_month) - 1

    # every surviving rebalance has a strictly earlier bar to have decided on
    for date in dates:
        assert index.get_loc(date) >= 1


def test_rebalance_dates_refuses_an_index_it_cannot_reason_about():
    """A non-DatetimeIndex has no months; guessing one would invent a schedule."""
    with pytest.raises(TypeError):
        rebalance_dates(pd.RangeIndex(10))


def test_a_missing_close_marks_at_the_last_known_price(cfg):
    """A vendor hole must not re-mark a position to its start-of-month value.

    The engine values positions off a forward-filled price series, so a missing print
    leaves the position flat for that day and catching up the next - a one-day mark
    lag with no lasting effect. Before this was fixed the position was instead
    re-marked to whatever it was worth at the *open of the current month*, which
    produced a spurious multi-basis-point round trip and, when the hole landed on the
    bar before a rebalance, a permanent shift in the equity curve because the position
    was then traded at that stale price.
    """
    import numpy as np
    import pandas as pd

    from trendbot.data import PriceData
    from trendbot.engine.backtest import run_backtest

    n = 700
    index = pd.bdate_range("2020-01-01", periods=n)
    tickers = list(cfg.universe)
    path = 100.0 * (1.0004 ** np.arange(n))  # every instrument compounds identically
    close = pd.DataFrame({t: path for t in tickers}, index=index)
    opens = close.shift(1).bfill()
    intact = PriceData(open=opens, close=close, source="t", adjusted=True, fetched_at="")

    holed_close = close.copy()
    hole_at = 600
    holed_close.iloc[hole_at, 0] = np.nan
    holed = PriceData(open=opens, close=holed_close, source="t", adjusted=True, fetched_at="")

    base = run_backtest(intact, cfg)
    alt = run_backtest(holed, cfg)

    # Nothing before the hole moves at all.
    pd.testing.assert_series_equal(
        base.returns.iloc[:hole_at], alt.returns.iloc[:hole_at], check_exact=False, atol=1e-15
    )
    # The hole costs one day of mark on one of twelve positions and gives it back the
    # next day. Both deviations are bounded by a single instrument's daily move.
    one_day = 0.0004
    assert abs(alt.returns.iloc[hole_at] - base.returns.iloc[hole_at]) < one_day
    assert abs(alt.returns.iloc[hole_at + 1] - base.returns.iloc[hole_at + 1]) < one_day
    # ... and it leaves no permanent trace: the equity curves reconverge and stay
    # together. This is the part the old marking bug failed.
    tail = slice(hole_at + 2, None)
    pd.testing.assert_series_equal(
        base.returns.iloc[tail], alt.returns.iloc[tail], check_exact=False, atol=1e-12
    )
    assert base.equity.iloc[-1] == pytest.approx(alt.equity.iloc[-1], rel=1e-9)
