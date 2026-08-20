"""STEP 1's GATE — a hand-built fixture with a delisting, a ticker change and a reuse.

PREREG_004.md's build order requires that before any paid data is touched, a fixture
carrying all three of experiment 004's new failure modes resolves correctly and is
verified by hand. Everything asserted here is arithmetic a reader can check.

The three cases, and what each would look like if it were handled wrongly:

**A reused ticker.** ``ZZZ`` belongs to permaticker 100 until it delists in 2011, and
to a different company, permaticker 300, from 2015. Keyed on the ticker string those
two become one security with a continuous price series, and a twelve-month momentum
number gets computed across a four-year gap between two unrelated companies. The tests
assert they stay separate and that the reuse is *reported* rather than silently merged.

**A ticker change.** Permaticker 200 trades as ``OLD`` and later as ``NEW``. Keyed on
the ticker it is two securities, each with half a history, and neither has the 252 days
section 2 requires. Keyed on the permaticker it is one security throughout.

**A delisting.** Permaticker 100 goes bankrupt. Section 3 assigns -100%. A
forward-filled panel would assign 0% and quietly convert a total loss into a flat
month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.config_004 import ACQUISITION, BANKRUPTCY, UNKNOWN, load_config_004
from trendbot.pit_universe import (
    BANKRUPT_PRICE_FLOOR,
    assemble_wide_panels,
    audit_position_exits,
    build_panels,
    build_pit_universe,
    build_security_master,
    resolve_permatickers,
    verify_delisting_returns,
)
from trendbot.sharadar import classify_delist_reason

# --------------------------------------------------------------------------------------
# the fixture, written out so the expected answers are readable
# --------------------------------------------------------------------------------------

CALENDAR = pd.bdate_range("2010-01-04", "2016-12-30")

# permaticker -> (ticker label, first bar, last bar)
#   100  ZZZ   trades 2010-01-04 .. 2011-06-30, then goes bankrupt
#   200  NEW   trades throughout; was called OLD until 2013
#   300  ZZZ   a DIFFERENT company that gets the recycled ticker, from 2015
#   400  AAA   trades throughout, never delists
#   500  BBB   trades throughout but is a preferred share - excluded by section 2
#   600  CCC   trades throughout but is priced under $5 - excluded by the floor
FIXTURE = {
    100: ("ZZZ", "2010-01-04", "2011-06-30", "Domestic Common Stock", True),
    200: ("NEW", "2010-01-04", "2016-12-30", "Domestic Common Stock", False),
    300: ("ZZZ", "2015-01-02", "2016-12-30", "Domestic Common Stock", False),
    400: ("AAA", "2010-01-04", "2016-12-30", "Domestic Common Stock", False),
    500: ("BBB", "2010-01-04", "2016-12-30", "Domestic Preferred Stock", False),
    600: ("CCC", "2010-01-04", "2016-12-30", "Domestic Common Stock", False),
}


def _tickers_table() -> pd.DataFrame:
    rows = []
    for permaticker, (ticker, first, last, category, delisted) in FIXTURE.items():
        rows.append(
            {
                "permaticker": permaticker,
                "ticker": ticker,
                "name": f"Company {permaticker}",
                "category": category,
                "isdelisted": delisted,
                "firstpricedate": pd.Timestamp(first),
                "lastpricedate": pd.Timestamp(last),
            }
        )
    return pd.DataFrame(rows)


def _actions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2011-06-30"),
                "action": "bankruptcyliquidation",
                "ticker": "ZZZ",
                "permaticker": 100,
            },
            {
                "date": pd.Timestamp("2013-05-01"),
                "action": "tickerchangefrom",
                "ticker": "OLD",
                "permaticker": 200,
            },
        ]
    )


def _panels():
    """Prices in which every security's own path is trivially checkable.

    Each name starts at its own base price and rises 0.05% a day, so momentum is
    positive and identical for everyone; the universe filters, not the ranking, are what
    these tests are about. ``CCC`` is priced under the $5 floor throughout.
    """
    n = len(CALENDAR)
    drift = 1.0005 ** np.arange(n)
    bases = {100: 20.0, 200: 30.0, 300: 40.0, 400: 50.0, 500: 60.0, 600: 2.0}
    closeadj = pd.DataFrame(index=CALENDAR, columns=[str(p) for p in FIXTURE], dtype=float)
    volume = pd.DataFrame(index=CALENDAR, columns=[str(p) for p in FIXTURE], dtype=float)
    for permaticker, (_, first, last, _, _) in FIXTURE.items():
        live = (CALENDAR >= pd.Timestamp(first)) & (CALENDAR <= pd.Timestamp(last))
        series = np.where(live, bases[permaticker] * drift, np.nan)
        closeadj[str(permaticker)] = series
        # dollar volume descends with permaticker so the top-N ranking is predictable
        volume[str(permaticker)] = np.where(live, 1e9 / permaticker, np.nan)
    openadj = closeadj.copy()
    closeunadj = closeadj.copy()  # no splits in the fixture, so the two coincide
    return closeadj, openadj, closeunadj, volume


@pytest.fixture(scope="module")
def cfg004():
    return load_config_004()


@pytest.fixture
def master():
    return build_security_master(_tickers_table(), _actions_table())


# --------------------------------------------------------------------------------------
# failure mode 1 - a reused ticker must not merge two companies
# --------------------------------------------------------------------------------------


def test_a_reused_ticker_stays_two_securities(master):
    assert master.label(100) == "ZZZ"
    assert master.label(300) == "ZZZ"
    assert 100 in master.table.index and 300 in master.table.index

    reused = master.reused_tickers()
    assert list(reused["ticker"]) == ["ZZZ"]
    assert reused.iloc[0]["permatickers"] == [100, 300]


def test_the_two_companies_sharing_a_ticker_have_disjoint_price_histories():
    """Merged on the ticker string, 100's 2011 prices would feed 300's 2015 momentum."""
    closeadj, _, _, _ = _panels()
    a, b = closeadj["100"], closeadj["300"]
    assert a.last_valid_index() == pd.Timestamp("2011-06-30")
    assert b.first_valid_index() == pd.Timestamp("2015-01-02")
    overlap = a.notna() & b.notna()
    assert not overlap.any(), "the fixture is meant to have a four-year gap between them"


def test_the_recycled_ticker_is_not_tradeable_under_the_dead_companys_id(master):
    assert not master.tradeable_on(100, pd.Timestamp("2015-06-01"))
    assert master.tradeable_on(300, pd.Timestamp("2015-06-01"))


# --------------------------------------------------------------------------------------
# failure mode 2 - a ticker change must not split one security in two
# --------------------------------------------------------------------------------------


def test_a_ticker_change_leaves_one_security_with_one_history(master):
    assert 200 in master.table.index
    assert "OLD" in master.ticker_history[200]
    assert "NEW" in master.ticker_history[200]

    closeadj, _, _, _ = _panels()
    series = closeadj["200"].dropna()
    assert series.index[0] == pd.Timestamp("2010-01-04")
    assert series.index[-1] == pd.Timestamp("2016-12-30")
    assert len(series) == len(CALENDAR), "one continuous history, not two halves"


# --------------------------------------------------------------------------------------
# failure mode 3 - a delisting is a return, not a missing value
# --------------------------------------------------------------------------------------


def test_the_delisting_is_classified_and_dated(master):
    delisting = master.delistings[100]
    assert delisting.bucket == BANKRUPTCY
    assert delisting.date == pd.Timestamp("2011-06-30")
    assert set(master.delistings) == {100}, "only the delisted name gets a delisting"


def test_bankruptcy_assigns_minus_one_hundred_percent_not_a_flat_month(cfg004, master):
    closeadj, openadj, _, _ = _panels()
    panels, applied = build_panels(cfg004, closeadj=closeadj, openadj=openadj, master=master)

    row = applied[applied["permaticker"] == 100].iloc[0]
    assert row["bucket"] == BANKRUPTCY
    assert row["assigned_return"] == -1.0
    assert row["last_traded_bar"] == pd.Timestamp("2011-06-30")

    checked = verify_delisting_returns(panels, applied)
    assert bool(checked.iloc[0]["agrees"])
    assert checked.iloc[0]["realised_return"] == pytest.approx(-1.0, abs=1e-9)

    # and the counterfactual the treatment rules out
    forward_filled = closeadj["100"].ffill()
    event = row["event_bar"]
    naive = forward_filled.loc[event] / forward_filled.loc[row["last_traded_bar"]] - 1.0
    assert naive == pytest.approx(0.0), "a forward fill would score a total loss as flat"


@pytest.mark.parametrize(
    "bucket,expected",
    [(BANKRUPTCY, -1.0), (UNKNOWN, -0.30), (ACQUISITION, 0.0)],
)
def test_each_section_3_bucket_gets_its_own_return(cfg004, bucket, expected):
    tickers = _tickers_table()
    actions = pd.DataFrame(
        [{"date": pd.Timestamp("2011-06-30"), "action": {
            BANKRUPTCY: "bankruptcyliquidation",
            UNKNOWN: "regulatorydelisting",
            ACQUISITION: "acquisitionby",
        }[bucket], "ticker": "ZZZ", "permaticker": 100}]
    )
    master = build_security_master(tickers, actions)
    assert master.delistings[100].bucket == bucket

    closeadj, openadj, _, _ = _panels()
    panels, applied = build_panels(cfg004, closeadj=closeadj, openadj=openadj, master=master)
    checked = verify_delisting_returns(panels, applied)
    row = checked[checked["permaticker"] == 100].iloc[0]
    assert row["assigned_return"] == pytest.approx(expected)
    assert row["realised_return"] == pytest.approx(expected, abs=1e-9)
    assert bool(row["agrees"])


def test_the_unknown_override_moves_only_the_unknown_bucket(cfg004):
    """Section 3's mandatory ladder must not silently move bankruptcy or acquisition."""
    tickers = _tickers_table().copy()
    tickers.loc[tickers["permaticker"] == 200, "isdelisted"] = True
    tickers.loc[tickers["permaticker"] == 200, "lastpricedate"] = pd.Timestamp("2014-06-30")
    actions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2011-06-30"), "action": "bankruptcyliquidation", "ticker": "ZZZ", "permaticker": 100},
            {"date": pd.Timestamp("2014-06-30"), "action": "regulatorydelisting", "ticker": "NEW", "permaticker": 200},
        ]
    )
    master = build_security_master(tickers, actions)
    closeadj, openadj, _, _ = _panels()

    for override, expected_unknown in ((None, -0.30), (0.0, 0.0), (-1.0, -1.0)):
        _, applied = build_panels(
            cfg004, closeadj=closeadj, openadj=openadj, master=master, unknown_override=override
        )
        by_id = applied.set_index("permaticker")["assigned_return"]
        assert by_id[200] == pytest.approx(expected_unknown)
        assert by_id[100] == pytest.approx(-1.0), "bankruptcy must not move with the ladder"


def test_proceeds_accrue_at_the_cash_rate_after_a_delisting(cfg004, master):
    """Section 3 says proceeds go to cash; section 5 says cash earns the T-bill rate."""
    closeadj, openadj, _, _ = _panels()
    rf = pd.Series(0.04 / 252.0, index=CALENDAR)
    panels, applied = build_panels(
        cfg004, closeadj=closeadj, openadj=openadj, master=master, rf_daily=rf
    )
    # use an acquisition so the terminal value is not ~0 and the accrual is visible
    acquisition_master = build_security_master(
        _tickers_table(),
        pd.DataFrame([{"date": pd.Timestamp("2011-06-30"), "action": "acquisitionby",
                       "ticker": "ZZZ", "permaticker": 100}]),
    )
    panels, applied = build_panels(
        cfg004, closeadj=closeadj, openadj=openadj, master=acquisition_master, rf_daily=rf
    )
    row = applied.iloc[0]
    series = panels.close["100"]
    event = row["event_bar"]
    after = series.loc[event:]
    daily = after.pct_change().dropna()
    assert np.allclose(daily.to_numpy(), 0.04 / 252.0), "proceeds must earn the cash rate"


def test_prices_after_the_delist_date_are_neutralised_not_traded_on(cfg004):
    """A vendor panel that runs past a recorded delisting must not keep trading it.

    The two disagree in real data often enough to matter: the actions table records a
    date, the price table sometimes carries a few more bars. Taking the panel's word
    would let a dead security carry a live price into a momentum ranking.
    """
    tickers = _tickers_table().copy()
    tickers.loc[tickers["permaticker"] == 200, "isdelisted"] = True
    tickers.loc[tickers["permaticker"] == 200, "lastpricedate"] = pd.Timestamp("2014-06-30")
    actions = pd.DataFrame(
        [{"date": pd.Timestamp("2014-06-30"), "action": "regulatorydelisting",
          "ticker": "NEW", "permaticker": 200}]
    )
    master = build_security_master(tickers, actions)
    closeadj, openadj, _, _ = _panels()
    assert closeadj["200"].last_valid_index() == pd.Timestamp("2016-12-30"), (
        "the fixture is meant to carry prices past the delisting"
    )

    panels, applied = build_panels(cfg004, closeadj=closeadj, openadj=openadj, master=master)
    row = applied.set_index("permaticker").loc[200]
    assert row["last_traded_bar"] <= pd.Timestamp("2014-06-30")
    assert row["assigned_return"] == pytest.approx(-0.30)

    checked = verify_delisting_returns(panels, applied)
    assert bool(checked.set_index("permaticker").loc[200, "agrees"])

    # every bar after the event is the frozen terminal value, so nothing later moves
    after = panels.close["200"].loc[row["event_bar"]:]
    assert np.allclose(after.to_numpy(), float(row["terminal_price"]))


def test_a_bankrupt_position_is_worth_essentially_nothing(cfg004, master):
    closeadj, openadj, _, _ = _panels()
    panels, applied = build_panels(cfg004, closeadj=closeadj, openadj=openadj, master=master)
    terminal = float(applied.iloc[0]["terminal_price"])
    assert terminal == pytest.approx(BANKRUPT_PRICE_FLOOR)
    assert terminal < 1e-6


# --------------------------------------------------------------------------------------
# section 2 - the universe rule
# --------------------------------------------------------------------------------------


def _universe(cfg004, master, **kwargs):
    closeadj, openadj, closeunadj, volume = _panels()
    dollar_volume = volume * closeunadj
    rebalances = [d for d in CALENDAR if d.day <= 3 and d.dayofweek < 5]
    months = pd.DatetimeIndex(rebalances).to_period("M")
    first_of_month = pd.DatetimeIndex(rebalances)[np.r_[True, months[1:] != months[:-1]]]
    return build_pit_universe(
        cfg004,
        closeunadj=closeunadj,
        closeadj=closeadj,
        dollar_volume=dollar_volume,
        master=master,
        rebalances=first_of_month,
        **kwargs,
    )


def test_no_security_is_in_the_universe_after_its_delisting(cfg004, master):
    """STEP 2's GATE, on the fixture."""
    universe = _universe(cfg004, master)
    delist = master.delist_date(100)
    for date in universe.membership.index:
        members = universe.members(date)
        if date >= delist:
            assert 100 not in members, f"a delisted security appears on {date.date()}"


def test_the_price_floor_uses_the_unadjusted_close(cfg004, master):
    """CCC trades at ~$2 and must never qualify, whatever its liquidity."""
    universe = _universe(cfg004, master)
    assert 600 not in universe.all_members()


def test_a_preferred_share_is_excluded_by_security_type(cfg004, master):
    eligible = [p for p in FIXTURE if master.is_common_stock(p)]
    assert 500 not in eligible
    universe = _universe(cfg004, master, eligible_ids=eligible)
    assert 500 not in universe.all_members()


def test_a_security_without_enough_history_is_excluded_until_it_has_it(cfg004, master):
    """300 lists in 2015; it cannot be ranked until 252 trading days later."""
    universe = _universe(cfg004, master)
    first_seen = [d for d in universe.membership.index if 300 in universe.members(d)]
    assert first_seen, "300 should eventually qualify"
    listed = pd.Timestamp("2015-01-02")
    trading_days_before = int((CALENDAR >= listed) & (CALENDAR <= first_seen[0])).__index__() if False else \
        int(((CALENDAR >= listed) & (CALENDAR <= first_seen[0])).sum())
    assert trading_days_before >= 252


def test_the_liquidity_ranking_uses_only_data_through_the_rebalance_date(cfg004, master):
    """Replacing every bar after date d must not change membership on or before d."""
    closeadj, openadj, closeunadj, volume = _panels()
    rebalances = pd.DatetimeIndex(
        [d for d in CALENDAR if (CALENDAR.to_period("M") == d.to_period("M")).argmax() >= 0]
    )
    months = CALENDAR.to_period("M")
    first_of_month = CALENDAR[np.r_[True, months[1:] != months[:-1]]]

    base = build_pit_universe(
        cfg004, closeunadj=closeunadj, closeadj=closeadj,
        dollar_volume=volume * closeunadj, master=master, rebalances=first_of_month,
    )
    cut = first_of_month[len(first_of_month) // 2]
    rng = np.random.default_rng(0)
    tampered_volume = volume.copy()
    tampered_volume.loc[cut:] = rng.uniform(1, 1e12, size=tampered_volume.loc[cut:].shape)
    tampered_close = closeunadj.copy()
    tampered_close.loc[cut:] = rng.uniform(1, 500, size=tampered_close.loc[cut:].shape)

    alt = build_pit_universe(
        cfg004, closeunadj=tampered_close, closeadj=closeadj,
        dollar_volume=tampered_volume * tampered_close, master=master, rebalances=first_of_month,
    )
    before = base.membership.loc[:cut].iloc[:-1]
    after = alt.membership.loc[:cut].iloc[:-1]
    assert np.array_equal(before.to_numpy(), after.to_numpy()), (
        "membership before the cut moved when only later bars changed"
    )


def test_the_realised_start_date_is_reported_not_assumed(cfg004, master):
    """Section 6's start is a rule; on a six-name fixture it is never satisfied."""
    universe = _universe(cfg004, master)
    assert universe.cfg_universe_size == 500
    assert universe.start_date is None, "six names cannot yield 500"
    assert universe.counts.max() <= len(FIXTURE)


def test_entries_and_exits_are_counted_per_rebalance(cfg004, master):
    universe = _universe(cfg004, master)
    flow = universe.entries_and_exits()
    assert set(flow.columns) == {"n", "entered", "left"}
    # 100 leaves the universe at the first rebalance on or after its delisting
    delist = master.delist_date(100)
    left_after = flow.loc[flow.index >= delist, "left"].sum()
    assert left_after >= 1


# --------------------------------------------------------------------------------------
# section 3's audit - no position exits without a return
# --------------------------------------------------------------------------------------


def test_every_position_exit_is_priced(cfg004, master):
    closeadj, openadj, _, _ = _panels()
    panels, _ = build_panels(cfg004, closeadj=closeadj, openadj=openadj, master=master)

    weights = pd.DataFrame(0.0, index=CALENDAR, columns=closeadj.columns)
    weights.loc[: pd.Timestamp("2012-01-03"), "100"] = 0.5  # held straight through its death
    weights["400"] = 0.5

    exits = audit_position_exits(weights, panels, master)
    assert len(exits) >= 1
    assert exits["priced"].all(), "a position left the book with no price to leave at"
    assert bool(exits[exits["permaticker"] == 100].iloc[0]["delisted"])


# --------------------------------------------------------------------------------------
# the vendor reason mapping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,bucket",
    [
        ("acquisitionby", ACQUISITION),
        ("merger", ACQUISITION),
        ("bankruptcyliquidation", BANKRUPTCY),
        ("regulatorydelisting", UNKNOWN),
        ("voluntarydelisting", UNKNOWN),
        ("delisted", UNKNOWN),
        ("", UNKNOWN),
        (None, UNKNOWN),
        ("something the vendor invented later", UNKNOWN),
    ],
)
def test_unrecognised_reasons_fall_into_the_haircut_bucket(action, bucket):
    """Guessing generously is how a delisting treatment manufactures a return."""
    assert classify_delist_reason(action) == bucket


# --------------------------------------------------------------------------------------
# the ticker->permaticker join, which is where reuse actually bites
# --------------------------------------------------------------------------------------


def _sep_rows():
    """Vendor-shaped long rows: keyed by (ticker, date), with ZZZ used by two companies."""
    closeadj, openadj, closeunadj, volume = _panels()
    rows = []
    for permaticker, (ticker, first, last, _, _) in FIXTURE.items():
        live = CALENDAR[(CALENDAR >= pd.Timestamp(first)) & (CALENDAR <= pd.Timestamp(last))]
        for date in live:
            key = str(permaticker)
            rows.append(
                {
                    "ticker": ticker,  # NOTE: the vendor gives only this, not the ID
                    "date": date,
                    "open": float(openadj.loc[date, key]),
                    "close": float(closeadj.loc[date, key]),
                    "closeadj": float(closeadj.loc[date, key]),
                    "closeunadj": float(closeunadj.loc[date, key]),
                    "volume": float(volume.loc[date, key]),
                }
            )
    return pd.DataFrame(rows)


def test_a_reused_ticker_is_split_back_onto_two_security_ids(master):
    """The decisive test: two companies, one ticker string, resolved by date window."""
    resolved = resolve_permatickers(_sep_rows(), master)

    zzz = resolved[resolved["ticker"] == "ZZZ"]
    assert set(zzz["permaticker"]) == {100, 300}

    early = zzz[zzz["date"] <= pd.Timestamp("2011-06-30")]
    late = zzz[zzz["date"] >= pd.Timestamp("2015-01-02")]
    assert set(early["permaticker"]) == {100}
    assert set(late["permaticker"]) == {300}
    assert resolved.attrs["reused_tickers"] == 1
    assert resolved.attrs["ambiguous_rows"] == 0


def test_the_wide_panel_keeps_the_two_companies_in_separate_columns(master):
    resolved = resolve_permatickers(_sep_rows(), master)
    panels = assemble_wide_panels(resolved, calendar=CALENDAR)
    closeadj = panels["closeadj"]

    assert "100" in closeadj.columns and "300" in closeadj.columns
    assert closeadj["100"].last_valid_index() == pd.Timestamp("2011-06-30")
    assert closeadj["300"].first_valid_index() == pd.Timestamp("2015-01-02")

    # the counterfactual: keyed on the ticker string these merge into one series whose
    # "twelve-month return" in 2016 spans two unrelated companies
    merged = _sep_rows()
    merged = merged[merged["ticker"] == "ZZZ"].set_index("date")["closeadj"].sort_index()
    assert merged.notna().sum() == closeadj["100"].notna().sum() + closeadj["300"].notna().sum()
    assert merged.index.is_monotonic_increasing


def test_the_derived_open_is_on_the_same_basis_as_the_adjusted_close(master):
    """Trading a split-only open against a split-and-dividend close leaks the dividend."""
    rows = _sep_rows().copy()
    # a 2% dividend adjustment on one name: closeadj sits 2% below close
    rows.loc[rows["ticker"] == "AAA", "close"] = rows.loc[rows["ticker"] == "AAA", "closeadj"] / 0.98
    rows.loc[rows["ticker"] == "AAA", "open"] = rows.loc[rows["ticker"] == "AAA", "close"]

    resolved = resolve_permatickers(rows, master)
    panels = assemble_wide_panels(resolved, calendar=CALENDAR)
    ratio = (panels["openadj"]["400"] / panels["closeadj"]["400"]).dropna()
    assert np.allclose(ratio.to_numpy(), 1.0), (
        "the open was not put on the same adjustment basis as the close"
    )


def test_dollar_volume_uses_unadjusted_prices(master):
    resolved = resolve_permatickers(_sep_rows(), master)
    panels = assemble_wide_panels(resolved, calendar=CALENDAR)
    expected = panels["closeunadj"] * panels["volume"]
    assert np.allclose(
        panels["dollar_volume"].to_numpy(), expected.to_numpy(), equal_nan=True
    )


def test_duplicate_security_date_rows_are_refused(master):
    rows = _sep_rows()
    doubled = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    resolved = resolve_permatickers(doubled, master)
    with pytest.raises(ValueError, match="duplicate"):
        assemble_wide_panels(resolved, calendar=CALENDAR)


# --------------------------------------------------------------------------------------
# the rule itself is unchanged from experiment 003
# --------------------------------------------------------------------------------------


def test_a_full_membership_reproduces_experiment_003s_strategy_exactly(cfg004):
    """PREREG_004.md section 1: "the data is the only variable".

    With every name in the universe on every date, the point-in-time strategy IS
    experiment 003's strategy. If these two ever disagree, 004 has changed the rule as
    well as the data and the comparison to 003 stops meaning anything.
    """
    from trendbot.config_003 import load_config_003
    from trendbot.engine.panel import price_panel
    from trendbot.engine.validation import synthetic_prices
    from trendbot.strategies import EquityCrossSectionalMomentum, PointInTimeMomentum

    names = tuple(f"N{i:03d}" for i in range(60))
    prices = synthetic_prices(names, seed=4, n_days=900)
    panel = price_panel(prices, names)

    membership = pd.DataFrame(True, index=prices.close.index, columns=list(names))
    pit = PointInTimeMomentum(cfg004, membership, "even")(panel)
    eq003 = EquityCrossSectionalMomentum(load_config_003(), names, "even")(panel)

    assert list(pit.columns) == list(eq003.columns)
    assert pit.index.equals(eq003.index)
    assert np.array_equal(pit.to_numpy(), eq003.to_numpy(), equal_nan=True), (
        "the point-in-time wrapper changed the ranking rule"
    )


def test_a_name_out_of_the_universe_is_excluded_from_the_sort_not_zeroed(cfg004):
    """The same distinction sections 2-4 of every prior experiment turned on.

    A name masked out must not be rankable at all. Were it instead given a momentum of
    zero it would sort into the middle of the book and could displace a real name from
    a decile boundary.
    """
    from trendbot.engine.panel import price_panel
    from trendbot.engine.validation import synthetic_prices
    from trendbot.strategies import PointInTimeMomentum

    names = tuple(f"N{i:03d}" for i in range(40))
    prices = synthetic_prices(names, seed=7, n_days=700)
    panel = price_panel(prices, names)

    membership = pd.DataFrame(True, index=prices.close.index, columns=list(names))
    membership[names[0]] = False
    strategy = PointInTimeMomentum(cfg004, membership, "even")

    masked = strategy.eligible_momentum(panel)
    assert masked[names[0]].isna().all(), "an excluded name was still given a momentum"

    weights = strategy(panel)
    assert (weights[names[0]] == 0.0).all()
    ranked = masked.notna().sum(axis=1) >= cfg004.n_quantiles
    held = weights[ranked]
    assert np.allclose(held.sum(axis=1).to_numpy(), 1.0), "the book must stay fully invested"
