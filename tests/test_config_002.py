from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from trendbot.config import ConfigParseError
from trendbot.config_002 import Config002, load_config_002

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG = REPO_ROOT / "PREREG_002.md"


@pytest.fixture(scope="session")
def text() -> str:
    return PREREG.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def cfg002():
    return load_config_002()


def _parse(text: str, tmp_path: Path) -> Config002:
    path = tmp_path / "PREREG_002.md"
    path.write_text(text, encoding="utf-8")
    return load_config_002(path, use_cache=False)


def test_every_parameter_comes_out_of_the_document(cfg002):
    assert cfg002.configurations_tried == 2, "the cumulative counter, not 1"
    assert cfg002.n_universe == 41
    assert cfg002.declared_universe_size == 41
    assert cfg002.universe_history_required_from == "2008-01-01"
    assert cfg002.formation_days == 252
    assert cfg002.skip_days == 21
    assert cfg002.n_quantiles == 5
    assert cfg002.declared_quantile_size == 8
    assert cfg002.long_only is True
    assert cfg002.gross_exposure_cap == 1.0
    assert cfg002.rebalance == "first trading day of each month"
    assert cfg002.has_drift_band is False, "section 5 says 'No drift band'"
    assert cfg002.cost_bps_per_side == 5.0
    assert cfg002.cost_sensitivity_bps == (0.0, 2.0, 5.0, 10.0, 20.0)
    assert cfg002.sample_start == "2008-01-01"
    assert cfg002.support_min_sharpe == 0.40
    assert cfg002.support_min_sharpe_excess_over_buy_and_hold == 0.15
    assert cfg002.abandon_below_sharpe == 0.15
    assert cfg002.requires_quintile_monotonicity is True
    assert (cfg002.expected_sharpe_low, cfg002.expected_sharpe_high) == (0.3, 0.7)
    assert cfg002.bug_threshold_sharpe == 1.0
    assert cfg002.expected_min_drawdown == 0.25
    assert cfg002.signed_by == "Pranav"
    assert cfg002.committed_on == "2026-08-19"


def test_the_sleeves_reconcile_with_the_universe(cfg002):
    flat = [t for tickers in cfg002.sleeves.values() for t in tickers]
    assert sorted(flat) == sorted(cfg002.universe)
    assert len(set(cfg002.universe)) == 41
    assert len(cfg002.sleeves) == 6
    for ticker in cfg002.universe:
        assert cfg002.sleeve_of(ticker) in cfg002.sleeves


def test_the_config_carries_the_hash_of_the_text_it_parsed(cfg002):
    assert cfg002.source_sha256 == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert cfg002.source_path == PREREG


def test_the_config_is_frozen_and_holds_nothing_mutable(cfg002):
    import dataclasses
    from pathlib import Path as _Path
    from types import MappingProxyType

    with pytest.raises(Exception):
        cfg002.formation_days = 200  # type: ignore[misc]
    with pytest.raises(Exception):
        cfg002.sleeves["new"] = ()  # type: ignore[index]

    immutable = (str, bytes, int, float, bool, type(None), tuple, frozenset, _Path, MappingProxyType)
    fresh = load_config_002(use_cache=False)
    for field in dataclasses.fields(fresh):
        value = getattr(fresh, field.name)
        assert isinstance(value, immutable), f"{field.name} holds a mutable {type(value).__name__}"


@pytest.mark.parametrize(
    "victim,what",
    [
        ("**Configurations tried on this dataset, cumulative:** 2", "configuration counter"),
        ("momentum_i(t) = P_i(t-21) / P_i(t-252) - 1", "momentum formula"),
        ("- **Top quintile** (highest ~8 of 41): equal weight, long.", "quantile cut"),
        ("- Cost assumption: **5 bps per side**, charged on turnover.", "cost"),
        ("- Sensitivity reported at 0, 2, 5, 10, 20 bps. Headline is 5 bps.", "cost ladder"),
        ("**Sample window: 2008-01-01 to present.**", "sample window"),
        ("- Gross exposure: **1.0** when fully invested. No leverage.", "gross exposure"),
        ("Signed: **Pranav**   Date: **2026-08-19**", "signature"),
    ],
)


def test_deleting_any_parameter_is_fatal_rather_than_defaulted(text, tmp_path, victim, what):
    assert victim in text, f"fixture is stale: {victim!r} is no longer in the document"
    with pytest.raises(ConfigParseError):
        _parse(text.replace(victim, ""), tmp_path)


def test_deleting_a_clause_that_carries_no_number_is_also_fatal(text, tmp_path):
    for clause in (
        "- It fails to beat equal-weight buy-and-hold at all, OR",
        "- Quintile ordering is non-monotonic, OR",
        "- **Quintile monotonicity holds.**",
        "**No volatility targeting.**",
        "`w_i = 1 / n_selected`",
        "Long-only.",
    ):
        assert clause in text, f"fixture is stale: {clause!r}"
        with pytest.raises(ConfigParseError):
            _parse(text.replace(clause, ""), tmp_path)


def test_an_ambiguous_parameter_is_fatal_rather_than_resolved(text, tmp_path):
    doubled = text.replace(
        "- Cost assumption: **5 bps per side**, charged on turnover.",
        "- Cost assumption: **5 bps per side**, charged on turnover.\n"
        "- Cost assumption: **2 bps per side**, charged on turnover.",
    )
    with pytest.raises(ConfigParseError, match="ambiguous"):
        _parse(doubled, tmp_path)


def test_a_formula_that_disagrees_with_its_own_prose_is_fatal(text, tmp_path):
    with pytest.raises(ConfigParseError, match="prose"):
        _parse(text.replace("P_i(t-21)", "P_i(t-42)"), tmp_path)


def test_a_declared_count_that_disagrees_with_the_table_is_fatal(text, tmp_path):
    with pytest.raises(ConfigParseError, match="declares"):
        _parse(text.replace("## 2. Universe — 41 ETFs, fixed", "## 2. Universe — 40 ETFs, fixed"), tmp_path)


def test_a_quantile_size_stated_out_of_the_wrong_total_is_fatal(text, tmp_path):
    with pytest.raises(ConfigParseError, match="sizes the top bucket"):
        _parse(text.replace("(highest ~8 of 41)", "(highest ~8 of 40)"), tmp_path)


def test_sections_3_and_8_must_agree_on_the_number_of_buckets(text, tmp_path):
    with pytest.raises(ConfigParseError, match="section 8 sorts into"):
        _parse(text.replace("into five quintiles", "into four quintiles"), tmp_path)


def test_a_formation_window_shorter_than_the_skip_is_fatal(text, tmp_path):
    broken = text.replace(
        "momentum_i(t) = P_i(t-21) / P_i(t-252) - 1", "momentum_i(t) = P_i(t-21) / P_i(t-10) - 1"
    )
    with pytest.raises(ConfigParseError, match="must be longer than the skip"):
        _parse(broken, tmp_path)


def test_a_headline_cost_outside_its_own_ladder_is_fatal(text, tmp_path):
    broken = text.replace(
        "- Sensitivity reported at 0, 2, 5, 10, 20 bps. Headline is 5 bps.",
        "- Sensitivity reported at 0, 2, 8, 10, 20 bps. Headline is 5 bps.",
    )
    with pytest.raises(ConfigParseError, match="absent from the sensitivity"):
        _parse(broken, tmp_path)


def test_a_duplicated_ticker_is_fatal(text, tmp_path):
    broken = text.replace("| Real assets | VNQ |", "| Real assets | VNQ, SPY |")
    with pytest.raises(ConfigParseError):
        _parse(broken, tmp_path)


def test_prereg_002_is_unmodified_relative_to_git_head():
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "PREREG_002.md"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("git is unavailable")
    status = proc.stdout.strip()
    assert not status.startswith(" M") and not status.startswith("M"), (
        "PREREG_002.md is modified relative to git HEAD. Any change to sections 2-6 "
        "produces experiment 003 with its own document, date and tag."
    )
