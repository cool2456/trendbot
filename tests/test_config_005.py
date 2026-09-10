from __future__ import annotations

import dataclasses
import hashlib
import re

import pytest

from trendbot.config import ConfigParseError
from trendbot.config_005 import (
    DISLOCATION_MONTHS,
    Config005,
    find_preregistration_005,
    load_config_005,
)


@pytest.fixture(scope="module")
def cfg005():
    return load_config_005()


@pytest.fixture(scope="module")
def document():
    return find_preregistration_005().read_text(encoding="utf-8")


def reparse(text: str) -> Config005:
    from trendbot.config_005 import _parse_text  # noqa: PLC0415

    return _parse_text(text, find_preregistration_005())


def test_hash_matches_the_document_on_disk(cfg005, document):
    assert cfg005.source_sha256 == hashlib.sha256(document.encode("utf-8")).hexdigest()


def test_signature_and_dates(cfg005):
    assert cfg005.signed_by == "Pranav"
    assert cfg005.committed_on == "2026-08-19"
    assert cfg005.signed_date == "2026-08-19"


def test_signal_is_byte_identical_to_the_earlier_experiments(cfg005):
    from trendbot.config_002 import load_config_002  # noqa: PLC0415

    cfg002 = load_config_002()
    assert (cfg005.formation_days, cfg005.skip_days) == (cfg002.formation_days, cfg002.skip_days)
    assert cfg005.n_quantiles == cfg002.n_quantiles == 5
    assert cfg005.long_only is True


def test_execution_and_costs(cfg005):
    assert cfg005.gross_exposure_cap == 1.0
    assert cfg005.cost_bps_per_side == 5.0
    assert cfg005.cost_sensitivity_bps == (0.0, 2.0, 5.0, 10.0, 25.0)
    assert cfg005.cost_bps_per_side in cfg005.cost_sensitivity_bps
    assert cfg005.rebalance == "first trading day of each month, full rebalance"


def test_sample_window_starts_after_euro_adoption(cfg005):
    assert cfg005.sample_start == "1999-01-01"


def test_universe_specification(cfg005):
    assert cfg005.release_name == "H.10"
    assert (cfg005.expected_universe_low, cfg005.expected_universe_high) == (20, 23)
    assert cfg005.quote_normalisation_required is True


def test_decision_rule_thresholds_are_unchanged_from_002_and_003(cfg005):
    from trendbot.config_003 import load_config_003  # noqa: PLC0415

    cfg003 = load_config_003()
    assert cfg005.support_min_sharpe == cfg003.support_min_sharpe == 0.40
    assert cfg005.support_min_sharpe_excess_over_benchmark == 0.15
    assert cfg005.max_inversions == cfg003.max_inversions == 1
    assert cfg005.min_spread_t_stat == cfg003.min_spread_t_stat == 2.0
    assert cfg005.min_alpha_t_stat == cfg003.min_alpha_t_stat == 2.0
    assert cfg005.abandon_below_sharpe == cfg003.abandon_below_sharpe == 0.15
    assert cfg005.abandon_below_spread_t_stat == cfg003.abandon_below_spread_t_stat == 1.0


def test_expectations_of_record(cfg005):
    assert (cfg005.expected_sharpe_low, cfg005.expected_sharpe_high) == (0.2, 0.6)
    assert cfg005.bug_threshold_sharpe == 1.0
    assert cfg005.mandatory_crash_month_required is False


def test_spot_only_and_the_required_diagnostic(cfg005):
    assert cfg005.spot_only is True
    assert cfg005.interest_rate_diagnostic_required is True


def test_secondary_beta_benchmark_is_parsed_not_inlined(cfg005):
    assert re.fullmatch(r"[A-Z]{1,5}", cfg005.equity_proxy_symbol)


def test_configuration_counter_is_four(cfg005):
    assert cfg005.configurations_tried == 4


def test_every_dislocation_label_can_be_dated(cfg005):
    assert cfg005.dislocation_labels
    for label in cfg005.dislocation_labels:
        assert label in DISLOCATION_MONTHS
        assert cfg005.dislocation_months[label]


@pytest.mark.parametrize(
    "fragment",
    [
        "**Quote convention must be normalised.**",
        "**Spot only.** No interest-rate component.",
        "**Required diagnostic (not a configuration):**",
        "**Experiment 004 does not count, and here is why.**",
        "**Configurations tried, cumulative:** 4",
        "Sort into **quintiles**",
        "Top quintile: equal weight, long",
        "**Cost assumption: 5 bps per side.**",
        "Sensitivity at 0/2/5/10/25.",
        "No substitutions, no additions",
        "Series discontinued mid-sample are **excluded entirely**, not truncated",
        "Absence of any",
        "to check whether currency momentum is a disguised equity beta",
    ],
)


def test_removing_a_required_clause_is_a_parse_error(document, fragment):
    assert fragment in document, f"the test's own fragment is stale: {fragment!r}"
    with pytest.raises(ConfigParseError):
        reparse(document.replace(fragment, ""))


def test_an_unknown_dislocation_label_is_refused(document):
    broken = document.replace("2015 CHF de-peg", "1992 ERM")
    with pytest.raises(ConfigParseError, match="cannot date"):
        reparse(broken)


def test_a_counter_without_its_justification_is_refused(document):
    broken = document.replace("PSR corrects\nfor configurations *tried*", "")
    with pytest.raises(ConfigParseError):
        reparse(broken)


def test_an_abandon_threshold_above_the_support_threshold_is_refused(document):
    broken = document.replace("net Sharpe below **0.15**", "net Sharpe below **0.95**")
    with pytest.raises(ConfigParseError, match="abandon threshold"):
        reparse(broken)


def test_a_headline_cost_outside_the_ladder_is_refused(document):
    broken = document.replace("**Cost assumption: 5 bps per side.**", "**Cost assumption: 7 bps per side.**")
    with pytest.raises(ConfigParseError, match="absent from the sensitivity"):
        reparse(broken)


def test_a_formation_window_shorter_than_the_skip_is_refused(document):
    broken = document.replace("P_i(t-21) / P_i(t-252)", "P_i(t-252) / P_i(t-21)")
    with pytest.raises(ConfigParseError, match="must be longer than the skip"):
        reparse(broken)


def test_config_is_frozen(cfg005):
    assert dataclasses.is_dataclass(cfg005)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg005.cost_bps_per_side = 1.0


def test_config_holds_no_mutable_parameter(cfg005):
    for field in dataclasses.fields(cfg005):
        value = getattr(cfg005, field.name)
        assert not isinstance(value, (list, dict, set)), f"{field.name} is mutable"


def test_describe_names_the_document_hash(cfg005):
    assert cfg005.source_sha256[:12] in cfg005.describe()
