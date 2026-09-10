from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path

import pytest

from trendbot.config import ConfigParseError
from trendbot.config_004 import ACQUISITION, BANKRUPTCY, UNKNOWN, Config004, load_config_004

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG = REPO_ROOT / "PREREG_004.md"


@pytest.fixture(scope="session")
def text() -> str:
    return PREREG.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def cfg004():
    return load_config_004()


def _parse(text: str, tmp_path: Path) -> Config004:
    path = tmp_path / "PREREG_004.md"
    path.write_text(text, encoding="utf-8")
    return load_config_004(path, use_cache=False)


def test_every_parameter_comes_out_of_the_document(cfg004):
    assert cfg004.configurations_tried == 4, "the cumulative counter, not 3"
    assert cfg004.universe_size == 500
    assert cfg004.liquidity_window_days == 60
    assert cfg004.price_floor == 5.00
    assert cfg004.min_history_days == 252
    assert cfg004.adrs_included is True
    assert cfg004.formation_days == 252
    assert cfg004.skip_days == 21
    assert cfg004.n_quantiles == 10
    assert cfg004.long_only is True
    assert cfg004.gross_exposure_cap == 1.0
    assert cfg004.cost_bps_per_side == 10.0
    assert cfg004.cost_sensitivity_bps == (0.0, 5.0, 10.0, 20.0, 40.0)
    assert cfg004.requires_point_in_time_adjustment is True
    assert cfg004.start_rule_min_names == 500
    assert cfg004.must_be_worse_than == 0.699
    assert cfg004.third_failure_invalidates is True
    assert cfg004.signed_by == "Pranav"


def test_section_3s_table_is_parsed_as_a_table(cfg004):
    treatment = cfg004.delisting
    assert treatment.acquisition_return is None, "an acquisition pays its final traded price"
    assert treatment.bankruptcy_return == -1.00
    assert treatment.unknown_return == -0.30
    assert treatment.assigned_return(ACQUISITION) == 0.0
    assert treatment.assigned_return(BANKRUPTCY) == -1.00
    assert treatment.assigned_return(UNKNOWN) == -0.30
    assert treatment.sensitivity_alternatives == (-1.0, 0.0), "the two section 3 names"
    assert treatment.sensitivity_returns == (-1.0, -0.30, 0.0), "plus the headline"


def test_the_sensitivity_ladder_moves_only_the_unknown_bucket(cfg004):
    treatment = cfg004.delisting
    for override in (0.0, -1.0, -0.30):
        assert treatment.assigned_return(UNKNOWN, unknown_override=override) == override
        assert treatment.assigned_return(BANKRUPTCY, unknown_override=override) == -1.00
        assert treatment.assigned_return(ACQUISITION, unknown_override=override) == 0.0


def test_an_unrecognised_bucket_is_an_error_not_a_default(cfg004):
    with pytest.raises(KeyError):
        cfg004.delisting.assigned_return("moved to the moon")


def test_section_6_carries_a_rule_and_no_date(cfg004):
    fields = {f.name for f in dataclasses.fields(cfg004)}
    assert not any("start" in f and "date" in f for f in fields), fields
    assert cfg004.start_rule_min_names == cfg004.universe_size


def test_section_8s_thresholds_are_byte_identical_to_experiment_003(cfg004):
    from trendbot.config_003 import load_config_003

    cfg003 = load_config_003()
    for field in (
        "support_min_sharpe",
        "support_min_sharpe_excess_over_buy_and_hold",
        "max_inversions",
        "min_spread_t_stat",
        "min_alpha_t_stat",
        "abandon_below_sharpe",
        "abandon_below_spread_t_stat",
    ):
        assert getattr(cfg004, field) == getattr(cfg003, field), field
    for field in ("formation_days", "skip_days", "n_quantiles", "cost_bps_per_side"):
        assert getattr(cfg004, field) == getattr(cfg003, field), field


def test_the_alpha_benchmark_is_read_from_the_document_not_assumed(cfg004):
    assert cfg004.market_proxy_symbol == "SPY"

    from trendbot.equities import load_market_proxy

    assert load_market_proxy().symbol == cfg004.market_proxy_symbol, (
        "the declared reporting proxy and the document's named benchmark have diverged"
    )


def test_renaming_the_alpha_benchmark_moves_both_section_8_clauses(text, tmp_path):
    renamed = text.replace("Alpha to SPY", "Alpha to QQQ").replace(
        "alpha to SPY negative", "alpha to QQQ negative"
    )
    assert _parse(renamed, tmp_path).market_proxy_symbol == "QQQ"

    half = text.replace("**Alpha to SPY is positive", "**Alpha to QQQ is positive")
    with pytest.raises(ConfigParseError, match="negative-alpha abandon clause"):
        _parse(half, tmp_path)


def test_the_crash_months_expand_to_an_explicit_list(cfg004):
    assert cfg004.required_crash_months == ("2009-03", "2009-04", "2009-05", "2020-04")


def test_the_config_carries_the_hash_of_the_text_it_parsed(cfg004):
    assert cfg004.source_sha256 == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert cfg004.source_path == PREREG


def test_the_config_is_frozen_and_holds_nothing_mutable(cfg004):
    from pathlib import Path as _Path

    with pytest.raises(Exception):
        cfg004.universe_size = 300  # type: ignore[misc]

    immutable = (str, bytes, int, float, bool, type(None), tuple, frozenset, _Path)
    fresh = load_config_004(use_cache=False)
    for field in dataclasses.fields(fresh):
        value = getattr(fresh, field.name)
        if field.name == "delisting":
            assert dataclasses.is_dataclass(value) and value.__dataclass_params__.frozen
            continue
        assert isinstance(value, immutable), f"{field.name} holds a mutable {type(value).__name__}"


@pytest.mark.parametrize(
    "victim",
    [
        "**Configurations tried on this dataset, cumulative:** 4",
        "momentum_i(t) = P_i(t-21) / P_i(t-252) - 1",
        "- Unadjusted close on date t is at least **$5.00**.",
        "- At least 252 trading days of price history as of date t.",
        "- **Cost assumption: 10 bps per side.**",
        "| Bankruptcy / liquidation | **−100%** |",
        "| Moved to OTC, or reason unknown/missing | **−30%** |",
        "Signed: **Pranav**   Date: **2026-08-19**",
    ],
)


def test_deleting_any_parameter_is_fatal_rather_than_defaulted(text, tmp_path, victim):
    assert victim in text, f"fixture is stale: {victim!r}"
    with pytest.raises(ConfigParseError):
        _parse(text.replace(victim, ""), tmp_path)


def test_deleting_a_clause_that_carries_no_number_is_also_fatal(text, tmp_path):
    for clause in (
        "ADRs are **included**",
        "not yet delisted",
        "no volatility targeting",
        "Long-only.",
        "split-and-dividend adjusted using point-in-time factors",
        "fails to beat buy-and-hold at all",
        "alpha to SPY negative",
        "more than one inversion",
        "reported as invalid rather than as a verdict",
        "required output regardless of verdict",
    ):
        assert clause in text, f"fixture is stale: {clause!r}"
        with pytest.raises(ConfigParseError):
            _parse(text.replace(clause, ""), tmp_path)


def test_removing_the_word_unadjusted_from_the_price_floor_is_fatal(text, tmp_path):
    broken = text.replace(
        "- Unadjusted close on date t is at least **$5.00**.",
        "- Close on date t is at least **$5.00**.",
    )
    with pytest.raises(ConfigParseError, match="price floor"):
        _parse(broken, tmp_path)


def test_an_ambiguous_parameter_is_fatal_rather_than_resolved(text, tmp_path):
    doubled = text.replace(
        "- **Cost assumption: 10 bps per side.**",
        "- **Cost assumption: 10 bps per side.**\n- **Cost assumption: 5 bps per side.**",
    )
    with pytest.raises(ConfigParseError, match="ambiguous"):
        _parse(doubled, tmp_path)


def test_a_start_rule_that_disagrees_with_the_universe_size_is_fatal(text, tmp_path):
    broken = text.replace(
        "the earliest month at which the §2 rule yields at least 500 qualifying",
        "the earliest month at which the §2 rule yields at least 300 qualifying",
    )
    with pytest.raises(ConfigParseError, match="must agree"):
        _parse(broken, tmp_path)


def test_a_section_6_that_names_a_date_is_fatal(text, tmp_path):
    broken = text.replace("**Start:** the earliest month", "**Start:** 2004-01-02 the earliest month")
    with pytest.raises(ConfigParseError, match="rule"):
        _parse(broken, tmp_path)


def test_a_bankruptcy_treatment_other_than_total_loss_is_fatal(text, tmp_path):
    broken = text.replace("| Bankruptcy / liquidation | **−100%** |", "| Bankruptcy / liquidation | **−50%** |")
    with pytest.raises(ConfigParseError, match="not -100%"):
        _parse(broken, tmp_path)


def test_a_headline_haircut_that_collapses_its_own_ladder_is_fatal(text, tmp_path):
    collapsed = text.replace("| Moved to OTC, or reason unknown/missing | **−30%** |",
                             "| Moved to OTC, or reason unknown/missing | **−100%** |")
    with pytest.raises(ConfigParseError, match="collapses"):
        _parse(collapsed, tmp_path)

    moved = text.replace("| Moved to OTC, or reason unknown/missing | **−30%** |",
                         "| Moved to OTC, or reason unknown/missing | **−45%** |")
    assert _parse(moved, tmp_path).delisting.unknown_return == pytest.approx(-0.45)


def test_a_delisting_table_with_a_missing_row_is_fatal(text, tmp_path):
    broken = text.replace("| Moved to OTC, or reason unknown/missing | **−30%** |\n", "")
    with pytest.raises(ConfigParseError, match="expected 3|does not map"):
        _parse(broken, tmp_path)


def test_a_formation_window_shorter_than_the_skip_is_fatal(text, tmp_path):
    broken = text.replace(
        "momentum_i(t) = P_i(t-21) / P_i(t-252) - 1", "momentum_i(t) = P_i(t-21) / P_i(t-10) - 1"
    )
    with pytest.raises(ConfigParseError, match="longer than the skip"):
        _parse(broken, tmp_path)


def test_a_headline_cost_outside_its_own_ladder_is_fatal(text, tmp_path):
    broken = text.replace("Sensitivity at 0/5/10/20/40.", "Sensitivity at 0/5/15/20/40.")
    with pytest.raises(ConfigParseError, match="absent from the sensitivity"):
        _parse(broken, tmp_path)


def test_prereg_004_is_unmodified_relative_to_git_head():
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "PREREG_004.md"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("git is unavailable")
    assert not proc.stdout.strip().lstrip().startswith("M"), (
        "PREREG_004.md is modified relative to git HEAD. Changes to §2-6 create experiment 005."
    )
