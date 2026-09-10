"""PREREG_006.md must be the sole source of every experiment 006 parameter.

Same contract as :mod:`tests.test_config_005`: no parameter has a default, a missing or
unparseable value is fatal, and the parsed object is frozen.

What is unusual about this document is how much of it is a *prohibition* rather than a
parameter — "covariance adapts, nothing else does", "003 is excluded on validity, not
performance", "configs_tried = 5 is a floor". Those clauses carry no number, so nothing
downstream would break if they vanished; the experiment would simply become a different
and worse one, quietly. They are therefore asserted by the parser, and the tests below
delete each one in turn and require the parse to fail.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from trendbot.config import ConfigParseError
from trendbot.config_006 import (
    SLEEVE_SOURCES,
    Config006,
    find_preregistration_006,
    load_config_006,
    _parse_text,
    _phrase,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG = REPO_ROOT / "PREREG_006.md"


@pytest.fixture(scope="module")
def cfg() -> Config006:
    return load_config_006()


@pytest.fixture(scope="module")
def text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _parse(modified: str) -> Config006:
    return _parse_text(modified, PREREG)


def _delete(text: str, phrase: str) -> str:
    """Remove a clause however the markdown happens to have wrapped it.

    The document hard-wraps prose at whatever column the author used, so a literal
    substring would miss a clause that straddles a line break — and a test that silently
    deleted nothing would pass for the wrong reason. This uses the parser's own
    whitespace-flexible matcher and asserts something was actually removed.
    """
    modified, n = re.subn(_phrase(phrase), "", text)
    assert n >= 1, f"the fixture phrase was not found in the document: {phrase!r}"
    return modified


# --------------------------------------------------------------------------------------
# the parameters, against the document
# --------------------------------------------------------------------------------------


def test_document_is_found_and_hashed(cfg):
    assert cfg.source_path == PREREG
    assert cfg.source_sha256 == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert find_preregistration_006() == PREREG


def test_every_section_3_parameter_matches_the_document(cfg):
    assert cfg.covariance_halflife_days == 60
    assert cfg.portfolio_vol_target == 0.10
    assert cfg.gross_exposure_cap == 1.0
    assert cfg.min_sleeve_weight == 0.05
    assert cfg.max_sleeve_weight == 0.60


def test_section_6_execution_matches_the_document(cfg):
    assert cfg.overlay_cost_bps_per_side == 5.0
    assert cfg.cost_sensitivity_bps == (0.0, 2.0, 5.0, 10.0)
    assert cfg.rebalance == "monthly"
    assert cfg.cost_rate_per_side == pytest.approx(0.0005)


def test_section_8_thresholds_match_the_document(cfg):
    assert cfg.min_excess_over_benchmark == 0.15
    assert cfg.min_excess_over_best_sleeve == 0.10
    assert cfg.correlation_clause_required


def test_section_9_expectations_match_the_document(cfg):
    assert (cfg.expected_sharpe_low, cfg.expected_sharpe_high) == (0.3, 0.6)
    assert cfg.bug_threshold_sharpe == 0.9


def test_header_and_signature_match_the_document(cfg):
    assert cfg.configurations_tried == 5
    assert cfg.committed_on == "2026-08-20"
    assert cfg.signed_by == "Pranav"
    assert cfg.signed_date == "2026-08-20"


def test_sleeve_manifest_is_read_from_the_table_not_hardcoded(cfg):
    assert cfg.sleeves == SLEEVE_SOURCES == ("001", "002", "005")
    assert cfg.n_sleeves == 3
    assert cfg.excluded_experiment == "003"
    assert cfg.excluded_experiment not in cfg.sleeves


def test_describe_names_the_parameters_and_the_hash(cfg):
    described = cfg.describe()
    assert "ERC over 3 sleeves 001/002/005" in described
    assert "halflife=60d" in described
    assert cfg.source_sha256[:12] in described


# --------------------------------------------------------------------------------------
# frozen
# --------------------------------------------------------------------------------------


def test_config_rejects_mutation(cfg):
    with pytest.raises(Exception):
        cfg.portfolio_vol_target = 0.20  # type: ignore[misc]
    assert load_config_006().portfolio_vol_target == 0.10


def test_config_holds_no_mutable_parameter(cfg):
    for name in cfg.__slots__:
        value = getattr(cfg, name)
        assert not isinstance(value, (list, dict, set)), f"{name} is mutable"


# --------------------------------------------------------------------------------------
# the prohibitions - each deleted in turn
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase, why",
    [
        (
            "`configs_tried = 5` is therefore a *floor*",
            "the header's admission that the correction is understated",
        ),
        (
            "**A passing verdict here is suggestive, not confirmatory,**",
            "the header's statement that a pass is not confirmatory",
        ),
        (
            "**Covariance adapts. Nothing else does.**",
            "section 3's rule against performance-based allocation",
        ),
        (
            "Realised P&L, drawdown, hit rate and rolling Sharpe are inputs to nothing.",
            "section 3's enumeration of what may not feed the allocation",
        ),
        (
            "an unmeasurable sleeve cannot be allocated to",
            "section 2's reason for excluding 003",
        ),
        (
            "prevents it being reread later as a performance exclusion",
            "section 2's guard against rereading 003's exclusion",
        ),
        (
            "**No sleeve may be added, removed or substituted after this document is",
            "section 2's no-substitution clause",
        ),
        (
            "This is §1's mechanism.",
            "section 8's third clause, the mechanism test",
        ),
        (
            "Sharpe computed as excess of the T-bill rate for both.",
            "section 4's return convention",
        ),
        (
            "Determined by the data, reported, not chosen.",
            "section 5's window rule",
        ),
        (
            "using identical covariance estimation, identical vol target, identical caps",
            "section 4's symmetry requirement",
        ),
    ],
)
def test_deleting_a_load_bearing_clause_is_a_parse_error(text, phrase, why):
    with pytest.raises(ConfigParseError):
        _parse(_delete(text, phrase))


def test_deleting_a_sleeve_row_is_a_parse_error(text):
    without_c = re.sub(r"^\| C \| 005 \|.*$", "", text, flags=re.MULTILINE)
    with pytest.raises(ConfigParseError, match="sleeve set"):
        _parse(without_c)


@pytest.mark.parametrize("label", ["A", "D", "Z"])
def test_adding_a_sleeve_row_is_a_parse_error(text, label):
    """A fourth row must be caught whatever it is labelled.

    A label pattern restricted to the three letters that happen to be in the document
    would make a row labelled D invisible to the parser, so a sleeve could be ADDED and
    silently ignored - the exact failure section 2's "no sleeve may be added" clause
    exists to prevent.
    """
    with_extra = text.replace(
        "| C | 005 |", f"| {label} | 003 | smuggled in |\n| C | 005 |", 1
    )
    assert with_extra != text
    with pytest.raises(ConfigParseError):
        _parse(with_extra)


def test_removing_sleeve_c_carry_correction_is_a_parse_error(text):
    with pytest.raises(ConfigParseError, match="carry-corrected"):
        _parse(_delete(text, "Sleeve C is included despite a negative raw Sharpe."))


# --------------------------------------------------------------------------------------
# the parameters - each removed or made inconsistent in turn
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "**EWMA of daily sleeve returns, halflife 60 trading days**",
        "Gross exposure capped at **1.0**",
        "Rebalance monthly, first trading day",
        "using data through the rebalance date only",
        "applied before renormalisation",
    ],
)
def test_removing_a_parameter_is_a_parse_error_not_a_default(text, phrase):
    with pytest.raises(ConfigParseError):
        _parse(_delete(text, phrase))


def test_infeasible_weight_bounds_are_rejected(text):
    """Three sleeves cannot each hold at most 30% and still sum to one."""
    broken = re.sub(r"maximum \*\*60%\*\*", "maximum **30%**", text)
    assert broken != text
    with pytest.raises(ConfigParseError, match="summing to one"):
        _parse(broken)


def test_a_headline_cost_outside_the_ladder_is_rejected(text):
    broken = re.sub(r"(?<=rebalancing: )5(?= bps per side)", "7", text)
    assert broken != text
    with pytest.raises(ConfigParseError, match="absent from the sensitivity ladder"):
        _parse(broken)


def test_a_non_monthly_rebalance_is_rejected(text):
    broken = text.replace("Rebalance monthly, first trading day", "Rebalance weekly, first trading day")
    with pytest.raises(ConfigParseError, match="not monthly"):
        _parse(broken)


def test_a_zero_configuration_counter_is_rejected(text):
    broken = text.replace("**Configurations tried, cumulative:** 5", "**Configurations tried, cumulative:** 0")
    with pytest.raises(ConfigParseError, match="at least 1"):
        _parse(broken)


def test_a_bug_threshold_inside_the_realistic_range_is_rejected(text):
    broken = text.replace("Above 0.9 means a bug", "Above 0.5 means a bug")
    with pytest.raises(ConfigParseError, match="bug threshold"):
        _parse(broken)


def test_a_missing_document_is_fatal(tmp_path):
    with pytest.raises(ConfigParseError, match="not found"):
        find_preregistration_006(tmp_path / "nowhere" / "deeper")


def test_line_wrapping_does_not_break_a_prose_assertion(text):
    """Reflowing the prose must change no verdict — only meaning may.

    Table rows are left alone: joining them would destroy the markdown structure section
    2's sleeve manifest is read from, which would be a change of meaning rather than of
    layout.
    """
    lines = text.split("\n")
    reflowed_lines = []
    for line in lines:
        stripped = line.strip()
        joinable = (
            reflowed_lines
            and stripped
            and not stripped.startswith(("|", "#", "-", "*", ">", "```"))
            and reflowed_lines[-1].strip()
            and not reflowed_lines[-1].strip().startswith(("|", "#", "```"))
        )
        if joinable:
            reflowed_lines[-1] = reflowed_lines[-1].rstrip() + " " + stripped
        else:
            reflowed_lines.append(line)
    reflowed = "\n".join(reflowed_lines)
    assert reflowed != text, "the reflow did nothing; the test would be vacuous"

    baseline, wrapped = _parse(text), _parse(reflowed)
    for name in baseline.__slots__:
        if name in {"source_sha256", "source_path"}:
            continue
        assert getattr(wrapped, name) == getattr(baseline, name), name
