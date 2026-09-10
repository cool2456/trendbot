from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from trendbot.config import ConfigParseError
from trendbot.config_003 import Config003, load_config_003

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG = REPO_ROOT / "PREREG_003.md"


@pytest.fixture(scope="session")
def text() -> str:
    return PREREG.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def cfg003():
    return load_config_003()


def _parse(text: str, tmp_path: Path) -> Config003:
    path = tmp_path / "PREREG_003.md"
    path.write_text(text, encoding="utf-8")
    return load_config_003(path, use_cache=False)


def test_every_parameter_comes_out_of_the_document(cfg003):
    assert cfg003.configurations_tried == 3, "the cumulative counter, not 2"
    assert cfg003.index_name == "S&P 500"
    assert cfg003.universe_history_required_from == "2008-01-01"
    assert (cfg003.expected_universe_low, cfg003.expected_universe_high) == (300, 400)
    assert cfg003.formation_days == 252
    assert cfg003.skip_days == 21
    assert cfg003.n_quantiles == 10, "deciles, not quintiles"
    assert cfg003.long_only is True
    assert cfg003.gross_exposure_cap == 1.0
    assert cfg003.cost_bps_per_side == 10.0, "10 bps, higher than 002's 5"
    assert cfg003.cost_sensitivity_bps == (0.0, 5.0, 10.0, 20.0, 40.0)
    assert cfg003.sample_start == "2008-01-01"
    assert cfg003.support_min_sharpe == 0.40
    assert cfg003.support_min_sharpe_excess_over_buy_and_hold == 0.15
    assert cfg003.max_inversions == 1, "one inversion tolerated, unlike 002"
    assert cfg003.min_spread_t_stat == 2.0
    assert cfg003.min_alpha_t_stat == 2.0
    assert cfg003.abandon_below_sharpe == 0.15
    assert cfg003.abandon_below_spread_t_stat == 1.0
    assert (cfg003.expected_sharpe_low, cfg003.expected_sharpe_high) == (0.4, 0.8)
    assert cfg003.bug_threshold_sharpe == 1.2
    assert cfg003.turnover_floor == 6.12
    assert cfg003.signed_by == "Pranav"
    assert cfg003.committed_on == "2026-08-19"


def test_the_mandatory_crash_months_expand_to_an_explicit_list(cfg003):
    assert cfg003.required_crash_months == ("2009-03", "2009-04", "2009-05", "2020-04")


def test_the_signal_is_identical_to_experiment_002(cfg003):
    from trendbot.config_002 import load_config_002

    cfg002 = load_config_002()
    assert cfg003.formation_days == cfg002.formation_days
    assert cfg003.skip_days == cfg002.skip_days
    assert cfg003.n_quantiles != cfg002.n_quantiles, "the bucket count IS meant to differ"


def test_the_config_carries_the_hash_of_the_text_it_parsed(cfg003):
    assert cfg003.source_sha256 == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert cfg003.source_path == PREREG


def test_the_config_is_frozen_and_holds_nothing_mutable(cfg003):
    import dataclasses
    from pathlib import Path as _Path

    with pytest.raises(Exception):
        cfg003.formation_days = 200  # type: ignore[misc]

    immutable = (str, bytes, int, float, bool, type(None), tuple, frozenset, _Path)
    fresh = load_config_003(use_cache=False)
    for field in dataclasses.fields(fresh):
        value = getattr(fresh, field.name)
        assert isinstance(value, immutable), f"{field.name} holds a mutable {type(value).__name__}"


@pytest.mark.parametrize(
    "victim",
    [
        "**Configurations tried on this dataset, cumulative:** 3",
        "momentum_i(t) = P_i(t-21) / P_i(t-252) - 1",
        "**Cost assumption: 10 bps per side.**",
        "- **Top decile:** equal weight, long.",
        "Signed: **Pranav**   Date: **2026-08-19**",
        "Expected yield 300–400 names. Report the exact count.",
    ],
)


def test_deleting_any_parameter_is_fatal_rather_than_defaulted(text, tmp_path, victim):
    assert victim in text, f"fixture is stale: {victim!r} is no longer in the document"
    with pytest.raises(ConfigParseError):
        _parse(text.replace(victim, ""), tmp_path)


def test_deleting_a_clause_that_carries_no_number_is_also_fatal(text, tmp_path):
    for clause in (
        "- Fails to beat equal-weight buy-and-hold at all, OR",
        "- Alpha to market is negative, OR",
        "- More than one decile inversion",
        "Names lacking 252 days of history are excluded from that date's ranking.",
        "no volatility targeting",
        "Equal weight within the top decile.",
        "split and dividend adjustment must be point-in-time",
        "Long-only.",
    ):
        assert clause in text, f"fixture is stale: {clause!r}"
        with pytest.raises(ConfigParseError):
            _parse(text.replace(clause, ""), tmp_path)


def test_deleting_section_2s_survivorship_direction_is_fatal(text, tmp_path):
    for clause in (
        "artificially **narrow** and the monotonicity gradient artificially **flat**",
        "a monotonicity pass on this universe is conservative evidence",
    ):
        assert clause in text, f"fixture is stale: {clause!r}"
        with pytest.raises(ConfigParseError):
            _parse(text.replace(clause, ""), tmp_path)


def test_an_ambiguous_parameter_is_fatal_rather_than_resolved(text, tmp_path):
    doubled = text.replace(
        "- **Cost assumption: 10 bps per side.**",
        "- **Cost assumption: 10 bps per side.**\n- **Cost assumption: 5 bps per side.**",
    )
    with pytest.raises(ConfigParseError, match="ambiguous"):
        _parse(doubled, tmp_path)


def test_a_bucket_word_that_disagrees_with_its_own_restatement_is_fatal(text, tmp_path):
    with pytest.raises(ConfigParseError, match="restates the count"):
        _parse(text.replace("(ten buckets, not five", "(five buckets, not five"), tmp_path)


def test_a_formation_window_shorter_than_the_skip_is_fatal(text, tmp_path):
    broken = text.replace(
        "momentum_i(t) = P_i(t-21) / P_i(t-252) - 1", "momentum_i(t) = P_i(t-21) / P_i(t-10) - 1"
    )
    with pytest.raises(ConfigParseError, match="must be longer than the skip"):
        _parse(broken, tmp_path)


def test_a_headline_cost_outside_its_own_ladder_is_fatal(text, tmp_path):
    broken = text.replace("Sensitivity at 0/5/10/20/40.", "Sensitivity at 0/5/15/20/40.")
    with pytest.raises(ConfigParseError, match="absent from the sensitivity"):
        _parse(broken, tmp_path)


def test_an_abandon_threshold_above_the_support_threshold_is_fatal(text, tmp_path):
    broken = text.replace("D1−D10 t-statistic below 1.0", "D1−D10 t-statistic below 3.0")
    with pytest.raises(ConfigParseError, match="no inconclusive band"):
        _parse(broken, tmp_path)


def test_an_expected_universe_range_that_runs_backwards_is_fatal(text, tmp_path):
    with pytest.raises(ConfigParseError, match="runs backwards"):
        _parse(text.replace("Expected yield 300–400 names", "Expected yield 400–300 names"), tmp_path)


def test_prereg_003_is_unmodified_relative_to_git_head():
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "PREREG_003.md"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("git is unavailable")
    assert not proc.stdout.strip().lstrip().startswith("M"), (
        "PREREG_003.md is modified relative to git HEAD. Any change to sections 2-6 "
        "produces experiment 004 with its own document, date and tag."
    )
