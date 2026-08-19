"""Config tests — build step 1 gate, first half: "config round-trips".

Two jobs here.

1. Assert every parameter against the literal text of PREREGISTRATION.md. If the
   parser ever drifts from the signed document, these fail.
2. Much more important: prove the parser FAILS LOUDLY. A parser that silently
   substitutes a default when a line goes missing would make the pre-registration
   decorative. Each mutation test writes a damaged copy of the document into
   tmp_path and demands ConfigParseError.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import pytest

from trendbot.config import Config, ConfigParseError, find_preregistration, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "PREREGISTRATION.md"
PREREG_TEXT = PREREG_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# helpers for building damaged copies of the document
# --------------------------------------------------------------------------------------


def _write(tmp_path: Path, text: str, name: str = "PREREGISTRATION.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _delete_line(text: str, needle: str) -> str:
    """Drop the single line containing ``needle``; assert it really was unique."""
    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if needle not in line]
    assert len(kept) == len(lines) - 1, (
        f"expected exactly one line containing {needle!r}, found {len(lines) - len(kept)}"
    )
    return "".join(kept)


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"expected {old!r} exactly once, found {text.count(old)}"
    return text.replace(old, new)


def _duplicate_line(text: str, needle: str) -> str:
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert len(hits) == 1, f"expected exactly one line containing {needle!r}"
    i = hits[0]
    return "".join(lines[: i + 1] + [lines[i]] + lines[i + 1 :])


def _load_damaged(tmp_path: Path, text: str) -> Config:
    return load_config(_write(tmp_path, text), use_cache=False)


# --------------------------------------------------------------------------------------
# 1. every parameter, against the literal in the document
# --------------------------------------------------------------------------------------


def test_universe_is_the_twelve_tickers_in_document_order(cfg: Config) -> None:
    assert cfg.universe == (
        "SPY",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "GLD",
        "SLV",
        "DBC",
        "UUP",
        "FXE",
        "FXY",
        "VNQ",
    )
    assert cfg.n_universe == 12
    assert len(set(cfg.universe)) == 12


def test_sleeves_match_the_section_2_table(cfg: Config) -> None:
    assert cfg.sleeves == {
        "Equity": ("SPY", "EFA", "EEM"),
        "Rates": ("TLT", "IEF"),
        "Commodities": ("GLD", "SLV", "DBC"),
        "Currency": ("UUP", "FXE", "FXY"),
        "Real assets": ("VNQ",),
    }
    # five sleeves, in the order the table lists them
    assert list(cfg.sleeves) == ["Equity", "Rates", "Commodities", "Currency", "Real assets"]
    assert [t for tickers in cfg.sleeves.values() for t in tickers] == list(cfg.universe)


def test_sleeve_of_maps_every_ticker_and_rejects_unknown_ones(cfg: Config) -> None:
    assert cfg.sleeve_of("SPY") == "Equity"
    assert cfg.sleeve_of("IEF") == "Rates"
    assert cfg.sleeve_of("DBC") == "Commodities"
    assert cfg.sleeve_of("FXY") == "Currency"
    assert cfg.sleeve_of("VNQ") == "Real assets"
    assert {cfg.sleeve_of(t) for t in cfg.universe} == set(cfg.sleeves)
    with pytest.raises(KeyError):
        cfg.sleeve_of("QQQ")


def test_signal_parameters_are_252_days_long_only(cfg: Config) -> None:
    assert cfg.lookback_days == 252
    assert isinstance(cfg.lookback_days, int)
    assert cfg.variant == "long-only"
    assert cfg.long_only is True


def test_risk_scaling_parameters(cfg: Config) -> None:
    assert cfg.instrument_vol_target == 0.10
    assert cfg.ewma_halflife_days == 30
    assert isinstance(cfg.ewma_halflife_days, int)
    assert cfg.portfolio_vol_target == 0.10
    assert cfg.gross_exposure_cap == 1.0
    assert cfg.per_instrument_cap == 0.25


def test_execution_parameters(cfg: Config) -> None:
    assert cfg.rebalance == "first trading day of each month"
    assert cfg.drift_band == 0.20
    assert cfg.cost_bps_per_side == 5.0
    assert cfg.cost_rate_per_side == pytest.approx(0.0005)
    assert cfg.cost_sensitivity_bps == (0.0, 2.0, 5.0, 10.0, 20.0)
    assert cfg.cost_bps_per_side in cfg.cost_sensitivity_bps


def test_configurations_tried_is_recorded_as_one(cfg: Config) -> None:
    # Section 7 step 5: "If it is ever greater than 1, this document has been violated."
    assert cfg.configurations_tried == 1


def test_section_8_decision_thresholds(cfg: Config) -> None:
    assert cfg.support_min_sharpe == 0.40
    assert cfg.support_min_positive_instruments == 9
    assert cfg.support_min_sharpe_excess_over_buy_and_hold == 0.15
    assert cfg.abandon_below_sharpe == 0.15


def test_section_9_expectations_of_record(cfg: Config) -> None:
    assert cfg.expected_sharpe_low == 0.3
    assert cfg.expected_sharpe_high == 0.6
    assert cfg.bug_threshold_sharpe == 0.8


def test_provenance_fields(cfg: Config) -> None:
    assert cfg.version == "1.0"
    assert cfg.signed_by == "Pranav Tamilselvan"
    assert cfg.signed_date == "Aug 19 2026"
    assert cfg.source_path == PREREG_PATH


def test_every_parsed_parameter_appears_verbatim_in_the_document() -> None:
    """Belt and braces: the numbers above are really in the signed text."""
    for literal in (
        "| Equity | SPY, EFA, EEM |",
        "Lookback: **252 trading days**",
        "**Variant in use:** long-only",
        "Per-instrument vol target: **10%**",
        "EWMA halflife: **30 days**",
        "Portfolio ex-ante vol target: **10%**",
        "Gross exposure cap: **sum(|w_i|) <= 1.0**",
        "Per-instrument cap: **|w_i| <= 0.25**",
        "Rebalance: first trading day of each month.",
        "0.20 * |w_target|",
        "**5 bps per side**",
        "Sensitivity reported at 0, 2, 5, 10, 20 bps",
        "number of configurations tried = 1",
        "exceeds 0.40",
        "at least 9 of 12 instruments",
        "by at least 0.15",
        "net Sharpe is below 0.15",
        "**0.3 – 0.6**",
        "Above 0.8 means a bug",
    ):
        assert literal in PREREG_TEXT, f"document no longer contains {literal!r}"


# --------------------------------------------------------------------------------------
# 2. round-trip and provenance
# --------------------------------------------------------------------------------------


def test_source_sha256_matches_an_independently_computed_digest(cfg: Config) -> None:
    expected = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    assert cfg.source_sha256 == expected
    assert len(cfg.source_sha256) == 64
    assert cfg.describe().endswith(expected[:12])


def test_config_round_trips_through_a_byte_identical_copy(tmp_path: Path, cfg: Config) -> None:
    """Same bytes at a different path -> same parameters and the same content hash."""
    copy = _write(tmp_path, PREREG_TEXT)
    other = load_config(copy, use_cache=False)

    assert other.source_path == copy.resolve()
    assert other.source_sha256 == cfg.source_sha256

    parameters = [f.name for f in dataclasses.fields(Config) if f.name != "source_path"]
    assert len(parameters) >= 25  # guard against fields silently disappearing
    for name in parameters:
        assert getattr(other, name) == getattr(cfg, name), name


def test_one_changed_byte_changes_the_content_hash(tmp_path: Path, cfg: Config) -> None:
    tweaked = PREREG_TEXT + "\n"
    other = load_config(_write(tmp_path, tweaked), use_cache=False)
    assert other.source_sha256 != cfg.source_sha256
    assert other.lookback_days == cfg.lookback_days  # parameters unchanged, provenance not


def test_find_preregistration_locates_the_repo_document() -> None:
    assert find_preregistration() == PREREG_PATH
    assert load_config(PREREG_PATH) == load_config()


def test_find_preregistration_refuses_to_fall_back_when_absent(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    with pytest.raises(ConfigParseError, match="not found"):
        find_preregistration(deep)


# --------------------------------------------------------------------------------------
# 3. the config is genuinely frozen
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", [f.name for f in dataclasses.fields(Config)])
def test_every_field_is_immutable(cfg: Config, field_name: str) -> None:
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        setattr(cfg, field_name, getattr(cfg, field_name))


def test_derived_field_is_immutable_too(cfg: Config) -> None:
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.n_universe = 11


def test_new_attributes_cannot_be_attached(cfg: Config) -> None:
    # slots=True keeps a tuned parameter from being bolted on at runtime. (CPython
    # surfaces this as TypeError rather than AttributeError for frozen+slots classes.)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        cfg.tuned_lookback = 200
    assert not hasattr(cfg, "tuned_lookback")


def test_sleeve_membership_cannot_be_mutated_in_place(cfg: Config) -> None:
    # sleeves is wrapped in a MappingProxyType: freezing the dataclass freezes the
    # binding, not the dict it points at, so the mapping itself has to be read-only.
    with pytest.raises(TypeError):
        cfg.sleeves["Equity"] = ("QQQ",)


def test_universe_is_a_tuple_not_a_mutable_sequence(cfg: Config) -> None:
    assert isinstance(cfg.universe, tuple)
    assert isinstance(cfg.cost_sensitivity_bps, tuple)
    for tickers in cfg.sleeves.values():
        assert isinstance(tickers, tuple)


# --------------------------------------------------------------------------------------
# 4. THE IMPORTANT ONES — the parser must fail loudly, never default
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "Lookback:",
        "**Variant in use:**",
        "Per-instrument cap:",
        "Per-instrument vol target:",
        "EWMA halflife:",
        "Portfolio ex-ante vol target:",
        "Gross exposure cap:",
        "Rebalance:",
        "Drift band:",
        "Cost assumption in backtest:",
        "Sensitivity reported at",
        "number of configurations tried",
        "Realistic net Sharpe:",
        "means a bug",
        "Signed:",
    ],
    ids=lambda n: re.sub(r"\W+", "_", n).strip("_"),
)
def test_deleting_any_parameter_line_raises_rather_than_defaulting(
    tmp_path: Path, needle: str
) -> None:
    damaged = _delete_line(PREREG_TEXT, needle)
    with pytest.raises(ConfigParseError):
        _load_damaged(tmp_path, damaged)


@pytest.mark.parametrize(
    ("old", "new", "what"),
    [
        ("Lookback: **252 trading days**", "Lookback: two hundred and fifty two days", "lookback"),
        ("**Variant in use:** long-only", "**Variant in use:** whatever-works", "variant"),
        ("Per-instrument cap: **|w_i| <= 0.25**", "Per-instrument cap: tight", "per-inst cap"),
        ("EWMA halflife: **30 days**", "EWMA halflife: **thirty days**", "halflife"),
        (
            "Cost assumption in backtest: **5 bps per side**",
            "Cost assumption in backtest: realistic",
            "cost",
        ),
        ("number of configurations tried = 1", "number of configurations tried = many", "n configs"),
    ],
    ids=["lookback", "variant", "per_instrument_cap", "halflife", "cost", "configurations_tried"],
)
def test_corrupting_a_parameter_value_raises(
    tmp_path: Path, old: str, new: str, what: str
) -> None:
    with pytest.raises(ConfigParseError):
        _load_damaged(tmp_path, _replace_once(PREREG_TEXT, old, new))


@pytest.mark.parametrize("needle", ["Lookback:", "**Variant in use:**", "Per-instrument cap:"])
def test_a_duplicated_parameter_line_is_ambiguous_and_raises(tmp_path: Path, needle: str) -> None:
    """Two conflicting statements of a parameter must not be resolved by guessing."""
    with pytest.raises(ConfigParseError, match="ambiguous"):
        _load_damaged(tmp_path, _duplicate_line(PREREG_TEXT, needle))


def test_universe_table_shorter_than_the_declared_count_raises(tmp_path: Path) -> None:
    damaged = _delete_line(PREREG_TEXT, "| Rates | TLT, IEF |")
    with pytest.raises(ConfigParseError, match="declares 12 instruments"):
        _load_damaged(tmp_path, damaged)


def test_universe_table_longer_than_the_declared_count_raises(tmp_path: Path) -> None:
    damaged = _replace_once(
        PREREG_TEXT, "| Real assets | VNQ |", "| Real assets | VNQ |\n| Crypto | BITO |"
    )
    with pytest.raises(ConfigParseError, match="declares 12 instruments"):
        _load_damaged(tmp_path, damaged)


def test_declared_count_disagreeing_with_a_correct_table_raises(tmp_path: Path) -> None:
    damaged = _replace_once(
        PREREG_TEXT,
        "## 2. Universe — 12 instruments, fixed",
        "## 2. Universe — 11 instruments, fixed",
    )
    with pytest.raises(ConfigParseError, match="declares 11 instruments"):
        _load_damaged(tmp_path, damaged)


def test_duplicated_ticker_raises_even_when_the_count_still_says_twelve(tmp_path: Path) -> None:
    # SPY twice, EEM gone: still 12 rows, still reconciles with the sleeves, still wrong.
    damaged = _replace_once(PREREG_TEXT, "| Equity | SPY, EFA, EEM |", "| Equity | SPY, EFA, SPY |")
    with pytest.raises(ConfigParseError, match="duplicate ticker"):
        _load_damaged(tmp_path, damaged)


def test_duplicated_sleeve_name_raises(tmp_path: Path) -> None:
    damaged = _replace_once(PREREG_TEXT, "| Rates | TLT, IEF |", "| Equity | TLT, IEF |")
    with pytest.raises(ConfigParseError, match="duplicate sleeve"):
        _load_damaged(tmp_path, damaged)


def test_unparseable_ticker_in_the_universe_table_raises(tmp_path: Path) -> None:
    damaged = _replace_once(
        PREREG_TEXT, "| Equity | SPY, EFA, EEM |", "| Equity | SPY, EFA, whatever is cheap |"
    )
    with pytest.raises(ConfigParseError, match="unparseable ticker"):
        _load_damaged(tmp_path, damaged)


def test_cost_ladder_missing_the_headline_cost_raises(tmp_path: Path) -> None:
    damaged = _replace_once(
        PREREG_TEXT,
        "Sensitivity reported at 0, 2, 5, 10, 20 bps",
        "Sensitivity reported at 0, 2, 10, 20 bps",
    )
    with pytest.raises(ConfigParseError, match="sensitivity ladder"):
        _load_damaged(tmp_path, damaged)


def test_headline_cost_outside_the_ladder_raises(tmp_path: Path) -> None:
    """Same guard from the other side: move the headline instead of the ladder."""
    damaged = _replace_once(
        PREREG_TEXT,
        "Cost assumption in backtest: **5 bps per side**",
        "Cost assumption in backtest: **7 bps per side**",
    )
    with pytest.raises(ConfigParseError, match="sensitivity ladder"):
        _load_damaged(tmp_path, damaged)


def test_section_8_instrument_count_disagreeing_with_the_universe_raises(tmp_path: Path) -> None:
    damaged = _replace_once(
        PREREG_TEXT, "at least 9 of 12 instruments", "at least 9 of 10 instruments"
    )
    with pytest.raises(ConfigParseError, match="section 8 refers to 10 instruments"):
        _load_damaged(tmp_path, damaged)


@pytest.mark.parametrize("section", [2, 3, 4, 5, 7, 8, 9])
def test_deleting_a_whole_section_raises(tmp_path: Path, section: int) -> None:
    damaged = re.sub(
        rf"^##\s+{section}\.\s+.*?$.*?(?=^##\s+\d+\.)",
        "",
        PREREG_TEXT,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert damaged != PREREG_TEXT
    with pytest.raises(ConfigParseError):
        _load_damaged(tmp_path, damaged)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("empty", ""),
        ("whitespace", "\n\n   \n"),
        ("unrelated_markdown", "# Shopping list\n\n- milk\n- 252 eggs\n"),
        ("headings_only", "## 2. Universe\n## 3. Signal\n## 4. Risk\n"),
        ("truncated", PREREG_TEXT[: PREREG_TEXT.index("## 4.")]),
    ],
)
def test_an_unparseable_document_raises_instead_of_returning_a_partial_config(
    tmp_path: Path, name: str, text: str
) -> None:
    with pytest.raises(ConfigParseError):
        load_config(_write(tmp_path, text, f"{name}.md"), use_cache=False)


def test_absent_file_raises_config_parse_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigParseError, match="cannot read"):
        load_config(tmp_path / "does_not_exist.md", use_cache=False)


def test_a_directory_in_place_of_the_document_raises(tmp_path: Path) -> None:
    directory = tmp_path / "PREREGISTRATION.md"
    directory.mkdir()
    with pytest.raises(ConfigParseError, match="cannot read"):
        load_config(directory, use_cache=False)


def test_a_failed_parse_leaves_nothing_cached(tmp_path: Path) -> None:
    """A damaged document must not poison, or be served from, the cache."""
    path = _write(tmp_path, _delete_line(PREREG_TEXT, "Lookback:"))
    with pytest.raises(ConfigParseError):
        load_config(path)
    path.write_text(PREREG_TEXT, encoding="utf-8")
    assert load_config(path).lookback_days == 252
