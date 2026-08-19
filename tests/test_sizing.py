"""Tests for trendbot.sizing - section 4 risk scaling, section 5 drift band, and the
whole-share rounding that is build step 3's acceptance gate.

Everything here is offline and seeded. Expected numbers in the arithmetic tests are
recomputed by hand from PREREGISTRATION.md section 4 rather than read back from the
implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trendbot.sizing import (
    ShareAllocation,
    apply_drift_band,
    annualised_vol,
    ewma_covariance,
    target_weights,
    whole_share_allocation,
)

# Real closes as of 2026-08-18, used by the build-step-3 gate test.
GATE_PRICES = {
    "SPY": 767.45,
    "EFA": 107.27,
    "EEM": 65.34,
    "TLT": 81.66,
    "IEF": 92.93,
    "GLD": 398.55,
    "SLV": 57.44,
    "DBC": 30.48,
    "UUP": 28.14,
    "FXE": 106.84,
    "FXY": 57.48,
    "VNQ": 97.62,
}

# Plausible annualised vols for the same twelve instruments.
GATE_SIGMAS = {
    "SPY": 0.14,
    "EFA": 0.15,
    "EEM": 0.19,
    "TLT": 0.14,
    "IEF": 0.06,
    "GLD": 0.13,
    "SLV": 0.28,
    "DBC": 0.12,
    "UUP": 0.07,
    "FXE": 0.07,
    "FXY": 0.09,
    "VNQ": 0.17,
}

DATE = pd.Timestamp("2026-08-18")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _series(cfg, mapping, default=0.0) -> pd.Series:
    """A universe-ordered Series with ``mapping`` overlaid on ``default``."""
    s = pd.Series(float(default), index=list(cfg.universe))
    for k, v in mapping.items():
        s[k] = v
    return s


def _equicorrelated_cov(sigma: pd.Series, rho: float) -> pd.DataFrame:
    s = sigma.to_numpy(dtype=float)
    corr = np.full((len(s), len(s)), rho)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.outer(s, s) * corr, index=sigma.index, columns=sigma.index)


def _allocation_table(alloc: ShareAllocation) -> str:
    frame = pd.DataFrame(
        {
            "price": alloc.prices,
            "target_w": alloc.target_weights,
            "target_$": alloc.target_dollars,
            "shares": alloc.shares,
            "realised_$": alloc.realised_dollars,
            "realised_w": alloc.realised_weights,
            "drift_w": alloc.drift,
            "holdable": alloc.holdable,
        }
    )
    return (
        f"\nequity=${alloc.equity:,.0f}  holdable={alloc.n_holdable}/{alloc.n_wanted}  "
        f"L2 tracking error={alloc.tracking_error:.6f}  "
        f"risk budget deployed={alloc.risk_budget_deployed:.4f}  "
        f"cash left=${alloc.cash_left:,.2f}\n" + frame.round(6).to_string()
    )


# --------------------------------------------------------------------------------------
# section 4 arithmetic, hand-computed
# --------------------------------------------------------------------------------------


def test_target_weights_reproduces_hand_computed_section_4_chain_equal_sigmas(cfg):
    """Five active instruments, all sigma=0.10, worked through by hand.

    x_i      = 1 * (0.10 / 0.10)                    = 1.0
    w_raw_i  = 1.0 / 12                             = 0.0833333...
    exante   = sqrt(5) * (0.0833333 * 0.10)         = 0.0186338998...
    k        = 0.10 / exante = 12 / sqrt(5)         = 5.366563146
    w_i      = 0.0833333 * k = 1/sqrt(5)            = 0.4472135955
    clip     -> 0.25 each                           (all five breach the 0.25 cap)
    gross    = 5 * 0.25 = 1.25 > 1.0                -> rescale by 1/1.25
    final    = 0.20 each
    """
    active = list(cfg.universe)[:5]
    trend = _series(cfg, {t: 1.0 for t in active})
    sigma = _series(cfg, {}, default=0.10)

    tw = target_weights(DATE, trend, sigma, cfg, covariance="diagonal")

    assert tw.raw[active].tolist() == pytest.approx([1.0 / 12] * 5, abs=1e-15)
    assert tw.k == pytest.approx(12.0 / math.sqrt(5.0), rel=1e-12)
    assert tw.k == pytest.approx(5.366563145999495, rel=1e-12)
    assert tw.exante_vol_raw == pytest.approx(math.sqrt(5) * (1 / 12) * 0.10, rel=1e-12)
    assert set(tw.capped_instruments) == set(active)
    assert tw.gross_cap_binding is True
    assert tw.weights[active].tolist() == pytest.approx([0.20] * 5, abs=1e-15)
    assert tw.weights.drop(active).abs().sum() == 0.0
    assert tw.gross == pytest.approx(1.0, abs=1e-12)
    assert tw.n_active == 5


def test_target_weights_reproduces_hand_computed_section_4_chain_mixed_sigmas(cfg):
    """Five active instruments with different sigmas; the 0.25 cap binds on three.

    sigma:   SPY 0.08  EFA 0.16  EEM 0.20  TLT 0.10  FXY 0.25
    x_i    = 0.10/sigma_i:      1.25    0.625    0.50     1.00     0.40
    w_raw  = x_i/12:        0.1041667 0.0520833 0.0416667 0.0833333 0.0333333
    Each w_raw_i * sigma_i = 0.10/12 = 0.00833333 by construction, so
    exante = sqrt(5) * 0.00833333 = 0.0186338998 and k = 12/sqrt(5) = 5.366563146.
    after k:                0.5590170 0.2795085 0.2236068 0.4472136 0.1788854
    clip at 0.25:           0.25      0.25      0.2236068 0.25      0.1788854
    gross  = 1.1524922 > 1.0 -> divide by 1.1524922
    final:                  0.2169212 0.2169212 0.1940202 0.2169212 0.1552162
    """
    sig = {"SPY": 0.08, "EFA": 0.16, "EEM": 0.20, "TLT": 0.10, "FXY": 0.25}
    active = list(sig)
    trend = _series(cfg, {t: 1.0 for t in active})
    sigma = _series(cfg, sig, default=0.15)  # the inactive sigmas must not matter

    tw = target_weights(DATE, trend, sigma, cfg, covariance="diagonal")

    x = {t: 0.10 / s for t, s in sig.items()}
    w_raw = {t: v / 12 for t, v in x.items()}
    assert tw.raw[active].tolist() == pytest.approx([w_raw[t] for t in active], rel=1e-12)

    k = 12.0 / math.sqrt(5.0)
    assert tw.k == pytest.approx(k, rel=1e-12)

    scaled = {t: min(w_raw[t] * k, 0.25) for t in active}
    gross = sum(scaled.values())
    assert gross == pytest.approx(1.1524922359499622, rel=1e-12)
    expected = {t: scaled[t] / gross for t in active}

    assert set(tw.capped_instruments) == {"SPY", "EFA", "TLT"}
    assert tw.gross_cap_binding is True
    assert tw.weights[active].tolist() == pytest.approx([expected[t] for t in active], rel=1e-12)
    assert tw.weights[active].tolist() == pytest.approx(
        [0.2169212010299862, 0.2169212010299862, 0.1940202205055786,
         0.2169212010299862, 0.1552161764044629],
        rel=1e-12,
    )
    assert tw.gross == pytest.approx(1.0, abs=1e-12)


def test_k_scales_exante_vol_to_exactly_the_portfolio_target(cfg):
    """The comment on section 4's k line: k scales ex-ante vol to 10% annualised."""
    trend = _series(cfg, {t: 1.0 for t in list(cfg.universe)[:7]})
    sigma = _series(cfg, {}, default=0.22)
    cov = _equicorrelated_cov(sigma, 0.25)

    for method in ("diagonal", "full"):
        tw = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance=method)
        assert tw.exante_vol_scaled == pytest.approx(cfg.portfolio_vol_target, abs=1e-14)


def test_divisor_is_the_fixed_universe_size_not_the_active_count(cfg):
    """w_i = x_i / 12 even with two active names (section 2 fixes the universe).

    k cancels any constant divisor, so the observable consequences are the raw
    pre-scaling weights and k itself: with N=12 and two active names at sigma=0.10,
    w_raw = 1/12 and k = 12/sqrt(2) = 8.485. Had the divisor been the active count 2,
    w_raw would be 0.5 and k would be 2/sqrt(2) = 1.414.
    """
    active = ["SPY", "GLD"]
    trend = _series(cfg, {t: 1.0 for t in active})
    sigma = _series(cfg, {}, default=0.10)

    tw = target_weights(DATE, trend, sigma, cfg, covariance="diagonal")

    assert cfg.n_universe == 12
    assert tw.raw[active].tolist() == pytest.approx([1.0 / 12, 1.0 / 12], abs=1e-15)
    assert tw.k == pytest.approx(12.0 / math.sqrt(2.0), rel=1e-12)
    assert tw.k != pytest.approx(2.0 / math.sqrt(2.0), rel=1e-3)
    # the inactive ten stay at zero rather than absorbing the freed budget
    assert tw.raw.drop(active).abs().sum() == 0.0
    assert tw.weights.drop(active).abs().sum() == 0.0


def test_inactive_instruments_do_not_get_the_freed_risk_budget(cfg):
    """Dropping an instrument from the signal must not enlarge anyone else's raw weight."""
    sigma = _series(cfg, {}, default=0.12)
    all_on = target_weights(DATE, _series(cfg, {}, default=1.0), sigma, cfg, covariance="diagonal")
    ten_on = target_weights(
        DATE,
        _series(cfg, {t: 1.0 for t in list(cfg.universe)[:10]}),
        sigma,
        cfg,
        covariance="diagonal",
    )
    survivors = list(cfg.universe)[:10]
    assert ten_on.raw[survivors].tolist() == pytest.approx(all_on.raw[survivors].tolist(), abs=1e-15)


# --------------------------------------------------------------------------------------
# section 4 post-conditions
# --------------------------------------------------------------------------------------


def test_caps_hold_over_many_random_draws(cfg):
    """|w_i| <= 0.25 and sum|w_i| <= 1.0 must hold for every input, always.

    600 seeded draws (300 per covariance method) mixing long-only and long-short
    signals, lognormal sigmas, and deliberately poisoned sigmas (NaN and 0).
    Observed worst case: max|w_i| = 0.25 exactly, max gross = 1.0 + 4e-16.
    """
    rng = np.random.default_rng(20260819)
    universe = list(cfg.universe)
    worst_abs = 0.0
    worst_gross = 0.0

    for i in range(300):
        low = 0 if i % 2 else -1  # alternate long-only and long-short signals
        trend = pd.Series(rng.integers(low, 2, size=12).astype(float), index=universe)
        sigma = pd.Series(np.exp(rng.normal(math.log(0.15), 0.8, size=12)), index=universe)
        sigma[rng.random(12) < 0.15] = np.nan
        sigma[rng.random(12) < 0.05] = 0.0

        loadings = rng.normal(size=(12, 3))
        corr = loadings @ loadings.T + np.diag(rng.random(12) * 2 + 0.5)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        cov = pd.DataFrame(
            np.outer(sigma.to_numpy(), sigma.to_numpy()) * corr, index=universe, columns=universe
        )

        for method in ("diagonal", "full"):
            tw = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance=method)
            w = tw.weights
            assert not w.isna().any(), f"NaN weight on draw {i} ({method})"
            assert np.isfinite(w.to_numpy()).all(), f"non-finite weight on draw {i} ({method})"
            assert w.abs().max() <= cfg.per_instrument_cap + 1e-12, f"draw {i} ({method})"
            assert w.abs().sum() <= cfg.gross_exposure_cap + 1e-9, f"draw {i} ({method})"
            worst_abs = max(worst_abs, float(w.abs().max()))
            worst_gross = max(worst_gross, float(w.abs().sum()))

    assert worst_abs == pytest.approx(0.25, abs=1e-12)  # the cap is reached, not just respected
    assert worst_gross == pytest.approx(1.0, abs=1e-9)


def test_caps_hold_on_a_realistic_synthetic_panel(cfg, synth):
    """Same post-conditions driven by the real vol/covariance estimators, offline."""
    returns = synth.close.pct_change(fill_method=None)
    sigma = annualised_vol(returns, cfg.ewma_halflife_days)
    cov = ewma_covariance(returns, cfg.ewma_halflife_days)
    momentum = synth.close / synth.close.shift(cfg.lookback_days) - 1.0

    dates = synth.close.index[cfg.lookback_days + 50 :: 120]
    assert len(dates) >= 10
    for date in dates:
        raw = momentum.loc[date]
        trend = (raw > 0).astype(float).where(raw.notna(), 0.0)
        tw = target_weights(date, trend, sigma.loc[date], cfg, cov=cov.loc[date], covariance="full")
        assert not tw.weights.isna().any()
        assert tw.weights.abs().max() <= cfg.per_instrument_cap + 1e-12
        assert tw.gross <= cfg.gross_exposure_cap + 1e-9
        assert (tw.weights >= -1e-15).all(), "long-only variant must never go short"


# --------------------------------------------------------------------------------------
# section 4 degenerate inputs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad_sigma", [np.nan, 0.0])
def test_unusable_sigma_gives_a_zero_weight_not_nan_or_inf(cfg, bad_sigma):
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {"SLV": bad_sigma}, default=0.20)
    cov = _equicorrelated_cov(sigma.fillna(0.0), 0.2)

    for method in ("diagonal", "full"):
        tw = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance=method)
        assert tw.weights["SLV"] == 0.0
        assert tw.raw["SLV"] == 0.0
        assert not tw.weights.isna().any()
        assert np.isfinite(tw.weights.to_numpy()).all()
        assert tw.weights.drop("SLV").gt(0).all(), "the other eleven must still be sized"


def test_a_nan_row_in_the_covariance_matrix_does_not_poison_the_other_weights(cfg):
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {"FXY": np.nan}, default=0.18)
    cov = _equicorrelated_cov(_series(cfg, {}, default=0.18), 0.3)
    cov.loc["FXY", :] = np.nan
    cov.loc[:, "FXY"] = np.nan

    tw = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance="full")
    assert tw.weights["FXY"] == 0.0
    assert not tw.weights.isna().any()
    # eleven survivors, equal sigma, rho=0.3: exante = (0.10/12) * sqrt(11 + 110*0.3),
    # k = 12/sqrt(44), so each weight is (0.10/0.18)/sqrt(44) = 0.0837531
    expected = (0.10 / 0.18) / math.sqrt(44.0)
    assert tw.weights.drop("FXY").tolist() == pytest.approx([expected] * 11, rel=1e-12)
    assert tw.gross == pytest.approx(11 * expected, rel=1e-12)
    assert tw.gross < cfg.gross_exposure_cap


def test_all_zero_trend_gives_all_zero_weights_without_dividing_by_zero(cfg):
    trend = _series(cfg, {}, default=0.0)
    sigma = _series(cfg, {}, default=0.15)
    cov = _equicorrelated_cov(sigma, 0.4)

    for method, kwargs in (("diagonal", {}), ("full", {"cov": cov})):
        tw = target_weights(DATE, trend, sigma, cfg, covariance=method, **kwargs)
        assert (tw.weights == 0.0).all()
        assert (tw.raw == 0.0).all()
        assert tw.k == 0.0
        assert tw.exante_vol_raw == 0.0
        assert tw.gross == 0.0
        assert tw.gross_cap_binding is False
        assert tw.n_active == 0


def test_all_sigmas_unusable_gives_all_zero_weights(cfg):
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {}, default=np.nan)
    tw = target_weights(DATE, trend, sigma, cfg, covariance="diagonal")
    assert (tw.weights == 0.0).all()
    assert tw.k == 0.0


def test_missing_trend_entries_are_treated_as_no_position(cfg):
    """The signal is undefined until 252 bars exist; section 3 says that means 0."""
    trend = pd.Series({"SPY": 1.0, "GLD": 1.0})  # the other ten are simply absent
    sigma = _series(cfg, {}, default=0.15)
    tw = target_weights(DATE, trend, sigma, cfg, covariance="diagonal")
    assert list(tw.weights.index) == list(cfg.universe)
    assert tw.n_active == 2
    assert tw.weights.drop(["SPY", "GLD"]).abs().sum() == 0.0


# --------------------------------------------------------------------------------------
# covariance method
# --------------------------------------------------------------------------------------


def test_diagonal_and_full_agree_exactly_when_the_gross_cap_binds(cfg):
    """Under a pure rescale to gross 1.0, k cancels, so the two methods coincide.

    Twelve active names at sigma=0.15 with rho=0.30: k_diagonal = 3.4641 and
    k_full = 1.6705 - materially different - yet no weight reaches the 0.25 cap and
    the gross cap binds in both cases, so both collapse to 1/12 each.
    Observed max elementwise difference: 1.4e-17.
    """
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {}, default=0.15)
    cov = _equicorrelated_cov(sigma, 0.30)

    diag = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance="diagonal")
    full = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance="full")

    assert diag.gross_cap_binding and full.gross_cap_binding
    assert diag.capped_instruments == () and full.capped_instruments == ()
    assert diag.k != pytest.approx(full.k, rel=1e-3), "the two k values must really differ"
    assert diag.k == pytest.approx(3.4641016151377544, rel=1e-12)
    assert full.k == pytest.approx(1.6705381391691136, rel=1e-12)

    assert (diag.weights - full.weights).abs().max() < 1e-15
    assert diag.weights.tolist() == pytest.approx([1.0 / 12] * 12, abs=1e-15)


def test_diagonal_and_full_differ_when_the_gross_cap_does_not_bind(cfg):
    """With gross below 1.0 nothing renormalises k away, so correlation changes sizes.

    Four active names at sigma=0.40, rho=0.50: gross is 0.50 diagonal vs 0.316 full.
    """
    active = list(cfg.universe)[:4]
    trend = _series(cfg, {t: 1.0 for t in active})
    sigma = _series(cfg, {}, default=0.40)
    cov = _equicorrelated_cov(sigma, 0.50)

    diag = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance="diagonal")
    full = target_weights(DATE, trend, sigma, cfg, cov=cov, covariance="full")

    assert not diag.gross_cap_binding and not full.gross_cap_binding
    assert diag.capped_instruments == () and full.capped_instruments == ()
    assert diag.gross == pytest.approx(0.5, rel=1e-12)
    assert full.gross == pytest.approx(1.0 / math.sqrt(10.0), rel=1e-12)
    assert (diag.weights - full.weights).abs().max() > 0.04
    # positive correlation means the full estimate sees more portfolio risk, so it
    # takes less of it
    assert full.gross < diag.gross


def test_full_covariance_without_a_matrix_raises(cfg):
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {}, default=0.15)
    with pytest.raises(ValueError, match="covariance matrix required"):
        target_weights(DATE, trend, sigma, cfg, cov=None, covariance="full")


def test_unknown_covariance_method_raises(cfg):
    trend = _series(cfg, {}, default=1.0)
    sigma = _series(cfg, {}, default=0.15)
    with pytest.raises(ValueError, match="unknown covariance method"):
        target_weights(DATE, trend, sigma, cfg, covariance="shrinkage")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# section 5 drift band
# --------------------------------------------------------------------------------------


def test_instrument_inside_the_band_keeps_its_current_weight(cfg):
    target = pd.Series({"SPY": 0.10, "GLD": 0.10})
    current = pd.Series({"SPY": 0.09, "GLD": 0.10})
    out = apply_drift_band(target, current, cfg.drift_band)
    # |0.10 - 0.09| = 0.01, threshold 0.20 * 0.10 = 0.02 -> no trade
    assert out["SPY"] == pytest.approx(0.09)
    assert out["GLD"] == pytest.approx(0.10)


def test_instrument_outside_the_band_moves_all_the_way_to_target(cfg):
    target = pd.Series({"SPY": 0.10, "TLT": -0.10})
    current = pd.Series({"SPY": 0.05, "TLT": 0.00})
    out = apply_drift_band(target, current, cfg.drift_band)
    # a partial move to the edge of the band would be a different rule; section 5
    # says trade, and the target is what you trade to
    assert out["SPY"] == pytest.approx(0.10)
    assert out["TLT"] == pytest.approx(-0.10)


def test_deviation_exactly_equal_to_the_threshold_does_not_trade(cfg):
    """The rule is a strict '>': equality is inside the band.

    0.15 / 0.12 is chosen because |0.15 - 0.12| and 0.20 * 0.15 are the *same*
    float64 (0.03), so this really is the boundary and not a near-miss.
    """
    target = pd.Series({"SPY": 0.15, "EEM": -0.15})
    current = pd.Series({"SPY": 0.12, "EEM": -0.12})
    assert abs(0.15 - 0.12) == 0.20 * 0.15  # exact equality at float64 precision

    out = apply_drift_band(target, current, cfg.drift_band)
    assert out["SPY"] == 0.12
    assert out["EEM"] == -0.12

    # one ulp further out and it does trade
    nudged = pd.Series({"SPY": np.nextafter(0.12, 0.0), "EEM": -0.12})
    out2 = apply_drift_band(target, nudged, cfg.drift_band)
    assert out2["SPY"] == pytest.approx(0.15)


def test_zero_target_always_closes_a_non_zero_position(cfg):
    """Documented behaviour: at w_target == 0 the threshold collapses to zero.

    |0 - w_current| > 0.20 * 0 == 0 is true for any non-zero holding, so a name whose
    trend has switched off is always fully exited, however small the residual.
    """
    target = pd.Series({"SPY": 0.0, "EFA": 0.0, "GLD": 0.0})
    current = pd.Series({"SPY": 0.30, "EFA": 1e-12, "GLD": -0.05})
    out = apply_drift_band(target, current, cfg.drift_band)
    assert out.tolist() == [0.0, 0.0, 0.0]

    # and a band of any size cannot rescue the position
    assert apply_drift_band(target, current, 5.0).tolist() == [0.0, 0.0, 0.0]

    # zero target with zero current is a no-op, not a spurious trade
    flat = apply_drift_band(target, pd.Series({"SPY": 0.0, "EFA": 0.0, "GLD": 0.0}), cfg.drift_band)
    assert flat.tolist() == [0.0, 0.0, 0.0]


def test_drift_band_treats_a_missing_current_position_as_flat(cfg):
    target = pd.Series({"SPY": 0.10, "GLD": 0.10})
    out = apply_drift_band(target, pd.Series({"SPY": 0.10}), cfg.drift_band)
    assert out["SPY"] == pytest.approx(0.10)  # unchanged, inside the band
    assert out["GLD"] == pytest.approx(0.10)  # from flat, |0.10| > 0.02 -> trade


def test_zero_band_trades_on_any_non_zero_deviation():
    target = pd.Series({"SPY": 0.10})
    current = pd.Series({"SPY": 0.10 - 1e-15})
    assert apply_drift_band(target, current, 0.0)["SPY"] == pytest.approx(0.10)
    assert apply_drift_band(target, target, 0.0)["SPY"] == pytest.approx(0.10)


def test_negative_drift_band_raises():
    target = pd.Series({"SPY": 0.10})
    current = pd.Series({"SPY": 0.05})
    with pytest.raises(ValueError, match="drift band cannot be negative"):
        apply_drift_band(target, current, -0.01)


def test_drift_band_can_lift_gross_above_the_section_4_cap(cfg):
    """The band is not cap-preserving, which is why every caller must re-clip.

    Both inputs respect section 4 (target gross 1.00, current gross 0.68) yet the
    banded output is levered at 1.08: SPY drifted up but stayed inside its own band
    so it is not sold back, while EFA is bought all the way to target.
    ``backtest.run_backtest`` and ``runner`` both re-apply the gross cap immediately
    after calling this function; a caller that forgets to is running leverage.
    """
    target = pd.Series({"SPY": 0.50, "EFA": 0.50})
    current = pd.Series({"SPY": 0.58, "EFA": 0.10})
    assert target.abs().sum() <= cfg.gross_exposure_cap
    assert current.abs().sum() <= cfg.gross_exposure_cap

    out = apply_drift_band(target, current, cfg.drift_band)
    assert out["SPY"] == pytest.approx(0.58)  # |0.08| <= 0.20 * 0.50, held
    assert out["EFA"] == pytest.approx(0.50)  # |0.40| >  0.20 * 0.50, traded
    assert out.abs().sum() == pytest.approx(1.08)
    assert out.abs().sum() > cfg.gross_exposure_cap

    # the overshoot is bounded: an untraded leg is at most (1 + band) * |target|
    assert out.abs().sum() <= (1 + cfg.drift_band) * target.abs().sum() + 1e-12


def test_drift_band_can_hold_a_position_above_the_per_instrument_cap(cfg):
    """Same non-preservation for the 0.25 cap, and nothing downstream re-clips it.

    A name sized to exactly the cap that then drifts to 0.28 is inside its band
    (|0.03| <= 0.20 * 0.25 = 0.05) so it is not trimmed. Unlike the gross cap, no
    caller re-applies the per-instrument cap after the band - see the returned
    discrepancy note.
    """
    target = pd.Series({"SPY": cfg.per_instrument_cap})
    current = pd.Series({"SPY": 0.28})
    out = apply_drift_band(target, current, cfg.drift_band)
    assert out["SPY"] == pytest.approx(0.28)
    assert out["SPY"] > cfg.per_instrument_cap

    # worst case is exactly (1 + band) * cap = 0.30
    edge = pd.Series({"SPY": cfg.per_instrument_cap * (1 + cfg.drift_band)})
    held = apply_drift_band(target, edge, cfg.drift_band)
    assert held["SPY"] == pytest.approx(0.30)


def test_drift_band_output_is_elementwise_either_target_or_current(cfg):
    """Whatever the band decides, it never invents a third value (no partial moves)."""
    rng = np.random.default_rng(7)
    universe = list(cfg.universe)
    for _ in range(100):
        target = pd.Series(rng.uniform(-0.25, 0.25, size=12), index=universe)
        current = pd.Series(rng.uniform(-0.25, 0.25, size=12), index=universe)
        out = apply_drift_band(target, current, cfg.drift_band)
        assert ((out == target) | (out == current)).all()
        # and the decision is exactly the documented predicate
        expected_trade = (target - current).abs() > cfg.drift_band * target.abs()
        assert (out[expected_trade] == target[expected_trade]).all()
        assert (out[~expected_trade] == current[~expected_trade]).all()


# --------------------------------------------------------------------------------------
# BUILD STEP 3 GATE
# --------------------------------------------------------------------------------------


def _gate_allocations(cfg) -> tuple[pd.Series, ShareAllocation, ShareAllocation]:
    prices = pd.Series(GATE_PRICES).reindex(list(cfg.universe))
    sigma = pd.Series(GATE_SIGMAS).reindex(list(cfg.universe))
    trend = _series(cfg, {}, default=1.0)
    weights = target_weights(DATE, trend, sigma, cfg, covariance="diagonal").weights
    return (
        weights,
        whole_share_allocation(weights, prices, 1_000),
        whole_share_allocation(weights, prices, 100_000),
    )


def test_gate_small_account_versus_large_account_tracking_error(cfg):
    """BUILD STEP 3 GATE: realised vs target weights at $1,000 and at $100,000.

    Observed with the 2026-08-18 closes and the vols in GATE_SIGMAS:
        $100,000 -> 12/12 holdable, L2 tracking error 0.00644, 98.8% of the risk
                    budget deployed.
        $1,000   -> 5/12 holdable, L2 tracking error 0.20747 (32x worse), 40.8% of
                    the risk budget deployed. SPY, EFA, EEM, GLD, SLV, FXE and VNQ
                    are silently absent: one share costs more than their entire
                    target allocation.
    """
    weights, small, large = _gate_allocations(cfg)
    table = _allocation_table(small) + "\n" + _allocation_table(large)
    print(table)  # visible under `pytest -s`, and in the assertion messages below

    assert weights.abs().sum() == pytest.approx(1.0, abs=1e-9), table

    # --- $100,000: everything is holdable and rounding barely matters ---
    assert large.n_holdable == 12, f"expected all twelve holdable at $100,000{table}"
    assert bool(large.holdable.all()), table
    assert large.tracking_error < 0.02, (
        f"L2 tracking error at $100,000 is {large.tracking_error:.5f}, expected < 0.02{table}"
    )

    # --- $1,000: strictly fewer sleeves and an order of magnitude worse tracking ---
    assert small.n_holdable < 12, f"expected fewer than twelve holdable at $1,000{table}"
    assert small.n_holdable < large.n_holdable, (
        f"$1,000 holds {small.n_holdable} sleeves, $100,000 holds {large.n_holdable}{table}"
    )
    assert small.tracking_error > 10 * large.tracking_error, (
        f"$1,000 tracking error {small.tracking_error:.5f} is not an order of magnitude "
        f"worse than $100,000's {large.tracking_error:.5f}{table}"
    )
    assert small.tracking_error > 0.15, (
        f"L2 tracking error at $1,000 is only {small.tracking_error:.5f}{table}"
    )

    # the sleeves that vanish are exactly those whose target dollars buy no share
    unaffordable = set(small.target_dollars.index[small.target_dollars < small.prices])
    absent = set(small.shares.index[small.shares == 0])
    assert absent == unaffordable, table
    assert {"SPY", "GLD"} <= absent, f"SPY at $767 cannot be held on $1,000{table}"


def test_gate_risk_budget_deployed_collapses_on_a_small_account(cfg):
    _, small, large = _gate_allocations(cfg)
    table = _allocation_table(small) + "\n" + _allocation_table(large)
    # observed: 0.9879 at $100,000, 0.4079 at $1,000
    assert large.risk_budget_deployed > 0.98, table
    assert large.risk_budget_deployed <= 1.0 + 1e-12, table
    assert small.risk_budget_deployed < 0.60, table
    assert small.risk_budget_deployed < large.risk_budget_deployed / 2, table


# --------------------------------------------------------------------------------------
# whole-share rounding
# --------------------------------------------------------------------------------------


def test_rounding_is_toward_zero_so_realised_gross_never_exceeds_target_gross(cfg):
    """This is what keeps section 4's gross cap true after rounding to whole shares."""
    rng = np.random.default_rng(4242)
    universe = list(cfg.universe)
    prices = pd.Series(GATE_PRICES).reindex(universe)

    for _ in range(200):
        raw = rng.normal(size=12) * rng.choice([0.0, 1.0], size=12, p=[0.3, 0.7])
        if not np.any(raw):
            continue
        weights = pd.Series(raw / np.abs(raw).sum(), index=universe)  # gross exactly 1.0
        equity = float(rng.uniform(500, 250_000))
        alloc = whole_share_allocation(weights, prices, equity)

        assert alloc.realised_weights.abs().sum() <= alloc.target_weights.abs().sum() + 1e-12
        assert alloc.realised_dollars.abs().sum() <= alloc.target_dollars.abs().sum() + 1e-9
        # per instrument too, and the sign is never flipped by rounding
        assert (alloc.realised_dollars.abs() <= alloc.target_dollars.abs() + 1e-9).all()
        assert (np.sign(alloc.shares) * np.sign(alloc.target_dollars) >= 0).all()
        assert alloc.realised_weights.abs().sum() <= 1.0 + 1e-12


def test_short_leg_rounds_toward_zero_not_away_from_it():
    weights = pd.Series({"A": -0.37, "B": 0.37})
    prices = pd.Series({"A": 100.0, "B": 100.0})
    alloc = whole_share_allocation(weights, prices, 1_000.0)
    # -$370 / $100 = -3.7 -> -3 shares, not -4
    assert alloc.shares.tolist() == [-3, 3]
    assert alloc.realised_dollars.abs().sum() == pytest.approx(600.0)


def test_target_below_one_share_price_gives_zero_shares_and_is_not_holdable():
    weights = pd.Series({"CHEAP": 0.50, "DEAR": 0.004})
    prices = pd.Series({"CHEAP": 10.0, "DEAR": 100.0})
    alloc = whole_share_allocation(weights, prices, 1_000.0)

    assert alloc.target_dollars["DEAR"] == pytest.approx(4.0)  # $4 will not buy a $100 share
    assert alloc.shares["DEAR"] == 0
    assert alloc.realised_dollars["DEAR"] == 0.0
    assert alloc.realised_weights["DEAR"] == 0.0
    assert bool(alloc.holdable["DEAR"]) is False
    assert alloc.shares["CHEAP"] == 50
    assert bool(alloc.holdable["CHEAP"]) is True
    assert alloc.n_holdable == 1
    assert alloc.n_wanted == 2
    # nothing warns; the sleeve is simply gone
    assert alloc.drift["DEAR"] == pytest.approx(-0.004)


def test_one_cent_short_of_a_share_still_gives_zero_shares():
    weights = pd.Series({"A": 0.0999})
    prices = pd.Series({"A": 100.0})
    alloc = whole_share_allocation(weights, prices, 1_000.0)
    assert alloc.target_dollars["A"] == pytest.approx(99.90)
    assert alloc.shares["A"] == 0
    assert bool(alloc.holdable["A"]) is False


def test_zero_target_weight_counts_as_holdable():
    """A sleeve you did not want is not a sleeve you failed to buy."""
    weights = pd.Series({"A": 0.5, "B": 0.0})
    prices = pd.Series({"A": 10.0, "B": 1e6})
    alloc = whole_share_allocation(weights, prices, 1_000.0)
    assert alloc.shares["B"] == 0
    assert bool(alloc.holdable["B"]) is True
    assert alloc.n_wanted == 1
    assert alloc.n_holdable == 2


def test_cash_left_equals_equity_minus_the_realised_book(cfg):
    rng = np.random.default_rng(99)
    universe = list(cfg.universe)
    prices = pd.Series(GATE_PRICES).reindex(universe)
    for equity in (1_000.0, 7_500.0, 100_000.0):
        raw = rng.uniform(0, 1, size=12)
        weights = pd.Series(raw / raw.sum(), index=universe)
        alloc = whole_share_allocation(weights, prices, equity)
        assert alloc.cash_left == pytest.approx(
            equity - float(alloc.realised_dollars.abs().sum()), abs=1e-9
        )
        assert alloc.cash_left >= 0.0, "gross <= 1.0 must leave a cash account solvent"
        assert alloc.realised_weights.tolist() == pytest.approx(
            (alloc.realised_dollars / equity).tolist(), abs=1e-15
        )


def test_risk_budget_fully_deployed_when_every_target_is_an_exact_share_count(cfg):
    universe = list(cfg.universe)
    weights = pd.Series(1.0 / 12, index=universe)
    prices = pd.Series(1.0, index=universe)
    alloc = whole_share_allocation(weights, prices, 1_200.0)  # $100 per sleeve, $1 shares
    assert alloc.shares.tolist() == [100] * 12
    assert alloc.risk_budget_deployed == pytest.approx(1.0, abs=1e-12)
    assert alloc.tracking_error == pytest.approx(0.0, abs=1e-15)
    assert alloc.cash_left == pytest.approx(0.0, abs=1e-9)
    assert alloc.n_holdable == 12


def test_tracking_error_and_absolute_error_agree_with_the_drift_vector(cfg):
    _, small, _ = _gate_allocations(cfg)
    drift = small.realised_weights - small.target_weights
    assert small.drift.tolist() == pytest.approx(drift.tolist(), abs=1e-15)
    assert small.tracking_error == pytest.approx(float(np.sqrt((drift**2).sum())), rel=1e-12)
    assert small.absolute_error == pytest.approx(float(drift.abs().sum()), rel=1e-12)
    # rounding toward zero can only shrink a position, so every drift is <= 0 here
    assert (small.drift <= 1e-15).all()


def test_tracking_error_decreases_monotonically_with_account_size(cfg):
    prices = pd.Series(GATE_PRICES).reindex(list(cfg.universe))
    sigma = pd.Series(GATE_SIGMAS).reindex(list(cfg.universe))
    weights = target_weights(
        DATE, _series(cfg, {}, default=1.0), sigma, cfg, covariance="diagonal"
    ).weights

    errors = []
    holdable = []
    for equity in (1_000, 5_000, 25_000, 100_000, 1_000_000):
        alloc = whole_share_allocation(weights, prices, equity)
        errors.append(alloc.tracking_error)
        holdable.append(alloc.n_holdable)

    assert errors == sorted(errors, reverse=True), f"tracking error not monotone: {errors}"
    assert holdable == sorted(holdable), f"holdable count not monotone: {holdable}"
    assert holdable[0] < 12 <= holdable[-1]


@pytest.mark.parametrize("equity", [0.0, -1.0, -100_000.0])
def test_non_positive_equity_raises(equity):
    weights = pd.Series({"A": 1.0})
    prices = pd.Series({"A": 10.0})
    with pytest.raises(ValueError, match="equity must be positive"):
        whole_share_allocation(weights, prices, equity)


def test_missing_price_raises():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="missing price for"):
        whole_share_allocation(weights, pd.Series({"A": 10.0}), 1_000.0)
    with pytest.raises(ValueError, match="missing price for"):
        whole_share_allocation(weights, pd.Series({"A": 10.0, "B": np.nan}), 1_000.0)


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_non_positive_price_raises(bad_price):
    weights = pd.Series({"A": 0.5, "B": 0.5})
    prices = pd.Series({"A": 10.0, "B": bad_price})
    with pytest.raises(ValueError, match="non-positive price"):
        whole_share_allocation(weights, prices, 1_000.0)


def test_allocation_ignores_prices_for_instruments_not_in_the_weight_vector():
    weights = pd.Series({"A": 1.0})
    prices = pd.Series({"A": 10.0, "GHOST": -5.0})  # a bad price outside the universe
    alloc = whole_share_allocation(weights, prices, 1_000.0)
    assert list(alloc.prices.index) == ["A"]
    assert alloc.shares.tolist() == [100]
