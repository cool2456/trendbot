from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trendbot.engine.backtest import buy_and_hold, full_universe_start, run_backtest
from trendbot.engine.metrics import sharpe
from trendbot.engine.panel_backtest import panel_buy_and_hold, run_panel_backtest
from trendbot.engine.validation import synthetic_prices
from trendbot.regression import (
    EXPERIMENT_001_BENCHMARK_SHARPE,
    EXPERIMENT_001_GATE_DECIMALS,
    EXPERIMENT_001_NET_SHARPE,
    EXPERIMENT_001_WINDOW,
)
from trendbot.strategies import TimeSeriesTrend

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE_ARTEFACT = REPO_ROOT / "results" / "backtest_headline.json"


def _synthetic_risk_free(index: pd.DatetimeIndex) -> pd.Series:
    t = np.arange(len(index), dtype=float)
    rate = 0.02 + 0.02 * np.sin(t / 250.0)
    return pd.Series(rate, index=index, name="rate")


def _both_engines(prices, cfg, *, cost_bps=None, risk_free=None, covariance="full"):
    old = run_backtest(
        prices, cfg, cost_bps=cost_bps, risk_free=risk_free, covariance=covariance
    )
    new = run_panel_backtest(
        prices,
        cfg.universe,
        TimeSeriesTrend(cfg, covariance=covariance),
        cost_bps=cfg.cost_bps_per_side if cost_bps is None else cost_bps,
        drift_band=cfg.drift_band,
        gross_cap=cfg.gross_exposure_cap,
        per_instrument_cap=cfg.per_instrument_cap,
        risk_free=risk_free,
    )
    return old, new


def _assert_identical(old, new, what: str) -> None:
    for field in ("equity", "returns", "gross_returns", "costs", "rf_daily"):
        a, b = getattr(old, field), getattr(new, field)
        assert a.index.equals(b.index), f"{what}: {field} index differs"
        assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), (
            f"{what}: {field} differs; max |delta| = "
            f"{np.nanmax(np.abs(a.to_numpy() - b.to_numpy())):.3e}"
        )
    for field in ("weights", "targets", "trades", "instrument_pnl"):
        a, b = getattr(old, field), getattr(new, field)
        assert list(a.columns) == list(b.columns), f"{what}: {field} columns differ"
        assert a.index.equals(b.index), f"{what}: {field} index differs"
        assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), (
            f"{what}: {field} differs; max |delta| = "
            f"{np.nanmax(np.abs(a.to_numpy() - b.to_numpy())):.3e}"
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_panel_engine_reproduces_experiment_001_bit_for_bit(cfg, seed):
    prices = synthetic_prices(cfg.universe, seed=seed, n_days=1200)
    old, new = _both_engines(prices, cfg)
    _assert_identical(old, new, f"seed {seed}")


def test_reproduction_survives_a_time_varying_cash_rate(cfg):
    prices = synthetic_prices(cfg.universe, seed=3, n_days=1200)
    rf = _synthetic_risk_free(prices.close.index)
    old, new = _both_engines(prices, cfg, risk_free=rf)
    _assert_identical(old, new, "time-varying cash rate")
    assert sharpe(old.excess_returns) == sharpe(new.excess_returns)


@pytest.mark.parametrize("cost_bps", [0.0, 2.0, 5.0, 10.0, 20.0])
def test_reproduction_holds_across_the_whole_cost_ladder(cfg, cost_bps):
    prices = synthetic_prices(cfg.universe, seed=4, n_days=800)
    old, new = _both_engines(prices, cfg, cost_bps=cost_bps)
    _assert_identical(old, new, f"{cost_bps} bps")


def test_reproduction_is_not_specific_to_the_pre_registered_variant(cfg):
    from dataclasses import replace

    prices = synthetic_prices(cfg.universe, seed=5, n_days=800)
    long_short = replace(cfg, variant="long-short")
    old, new = _both_engines(prices, long_short)
    _assert_identical(old, new, "long-short")

    old, new = _both_engines(prices, cfg, covariance="diagonal")
    _assert_identical(old, new, "diagonal covariance")


@pytest.mark.parametrize("mode", ["none", "monthly", "daily"])
def test_section_8_benchmark_is_the_same_construction(cfg, mode):
    prices = synthetic_prices(cfg.universe, seed=6, n_days=900)
    a = buy_and_hold(prices, cfg, rebalance=mode)
    b = panel_buy_and_hold(prices, cfg.universe, rebalance=mode)
    assert a.index.equals(b.index)
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)

    start = full_universe_start(prices, cfg)
    a = buy_and_hold(prices, cfg, rebalance=mode, start=start)
    b = panel_buy_and_hold(prices, cfg.universe, rebalance=mode, start=start)
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_buy_and_hold_reads_nothing_off_the_config_but_the_universe():
    import ast
    import inspect
    import textwrap

    from trendbot.engine.panel_backtest import UniverseView

    tree = ast.parse(textwrap.dedent(inspect.getsource(buy_and_hold)))
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cfg"
    }
    assert read == {"universe"}, (
        f"buy_and_hold now reads {sorted(read)} off its config, but panel_buy_and_hold "
        "passes a UniverseView that carries only 'universe'. Either widen UniverseView "
        "or stop routing experiment 002's benchmark through this function."
    )
    assert [f.name for f in __import__("dataclasses").fields(UniverseView)] == ["universe"]


def test_the_frozen_constants_are_the_numbers_the_gate_names():
    assert round(EXPERIMENT_001_NET_SHARPE, EXPERIMENT_001_GATE_DECIMALS) == 0.352
    assert round(EXPERIMENT_001_BENCHMARK_SHARPE, EXPERIMENT_001_GATE_DECIMALS) == 0.414
    assert EXPERIMENT_001_WINDOW == ("2008-02-29", "2026-08-18")


def test_the_frozen_constants_match_experiment_001s_own_artefact():
    if not HEADLINE_ARTEFACT.is_file():
        pytest.skip(f"{HEADLINE_ARTEFACT.name} is not on disk (results/ is gitignored)")
    payload = json.loads(HEADLINE_ARTEFACT.read_text())
    assert payload["window"] == list(EXPERIMENT_001_WINDOW)
    assert payload["strategy"]["sharpe"] == EXPERIMENT_001_NET_SHARPE
    assert payload["benchmark"]["sharpe"] == EXPERIMENT_001_BENCHMARK_SHARPE
    assert payload["decision"] == "ABANDON"


def test_experiment_001s_documents_are_untouched():
    import subprocess

    def status_of(name: str) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", name],
            capture_output=True,
            text=True,
        )
        return None if proc.returncode != 0 else proc.stdout.strip()

    if status_of("PREREGISTRATION.md") is None:
        pytest.skip("git is unavailable")

    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "PREREGISTRATION.md"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tracked, (
        "PREREGISTRATION.md is not tracked by git, so this test cannot tell whether it "
        "has been edited. That is the one document experiment 002 must not touch."
    )

    for name in ("PREREGISTRATION.md", "FINDINGS.md"):
        path = REPO_ROOT / name
        if not path.is_file():
            pytest.skip(f"{name} is absent")
        status = status_of(name) or ""
        assert not status.lstrip().startswith("M"), (
            f"{name} is modified relative to git HEAD. Experiment 001's outputs are "
            "immutable; experiment 002 writes to FINDINGS_002.md."
        )
