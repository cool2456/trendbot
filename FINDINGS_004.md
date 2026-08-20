# Findings 004 — cross-sectional momentum (point-in-time universe)

Implementation of `PREREG_004.md` (sha256 `c5844e4128ba…`, committed `e54bdf4`, tagged
`prereg-004`, unmodified since — asserted by a test against git HEAD). No parameter in
that document was changed, tuned, or substituted. **Configurations tried on this
dataset, cumulative: 4.**

Experiments 001–003 are untouched. The 001 regression gate still passes
**bit-identically**.

---

## 1. Status: INCOMPLETE — the vendor blocked, and nothing was substituted

**Steps 1 and 4 are complete and their gates pass. Steps 2, 3 and 5–11 are blocked on
vendor access and are not reported, because there is nothing honest to report.**

Every Sharadar request from this host is refused:

```
HTTP 429  {"quandl_error": {"code": "QELx06",
  "message": "You have exceeded the API speed limit and your account has temporarily
              been disabled. Please contact clientsuccess@nasdaq.com ..."}}
```

It applies to every Sharadar table (`TICKERS`, `SEP`, `ACTIONS` and `SF1` all return
the same code) and it persisted across every retry over the session: 40 probes at
30-second intervals over 20 minutes, then a further 120 probes at 60-second intervals
over two hours, then a final check an hour after that — **161 requests over roughly
three hours, every one a 429** — plus the client's own five-attempt exponential backoff
on each protocol command.

**The block is not attributable to the key in `.env`, and the message is misleading.**
A three-request diagnostic settles it:

| request | response |
|---|---|
| our key | `429 QELx06` "your account has temporarily been disabled" |
| **a deliberately invalid key** (32 zeros) | **`429 QELx06`, identical** |
| no key at all | `403 QEPx04` "a valid API key is required" |

A key Nasdaq has never seen draws the same "your account has been disabled" response as
ours. The throttle is therefore applied at the edge, to this **source IP**, before the
key is validated at all — the no-key case proves validation happens, and the
invalid-key case proves the 429 precedes it. Nothing here licenses any conclusion about
the account behind our key, which may be perfectly healthy.

The practical consequence is that the remedy is a different network egress or waiting
out an IP-level throttle, not contacting Nasdaq about the account — which is what the
error text would have had me report had I taken it at face value.

**What was NOT done in response.** No other vendor was substituted, no survivor-only
proxy was used to approximate a point-in-time universe, and no number below is
estimated. §2 specifies a universe constructed from survivorship-free data; a result
computed on anything else would be experiment 003 again, which is the exact failure
this experiment exists to correct.

| step | status |
|---|---|
| 1 — free-tier pipeline validation | **complete, gate PASS** |
| 2 — universe construction | blocked: needs `SHARADAR/SEP` |
| 3 — delisting treatment | blocked: needs `SHARADAR/ACTIONS` + `SEP` |
| 4 — noise tests, both variants | **complete, gate PASS** |
| 5 — full-sample backtest | blocked |
| 6 — paired A/B diagnostic (§10) | blocked |
| 7 — decile monotonicity | blocked |
| 8 — beta attribution | blocked |
| 9 — worst months / §9 validity | blocked |
| 10 — PSR at `configs_tried = 4` | blocked |
| 11 — verdict | **not issued** |

Every blocked step is implemented, wired and reachable by one command. When access is
restored, `--paired` and `--validate` produce the full protocol with no further work.

---

## 2. Step 1 — free-tier pipeline validation (§7.2)

§7.2 requires ingestion, security-master handling, delisting logic and universe
construction to be built and verified **before any paid data is used**. That half is
done and does not depend on the vendor at all.

**Gate: PASS.** `tests/test_pit_universe.py` — 37 tests, all green — drives a hand-built
fixture carrying all three of the failure modes the build order names. The fixture is
six securities on a 2010–2016 calendar, and every expected answer is arithmetic:

| permaticker | ticker | trades | what it tests |
|---|---|---|---|
| 100 | `ZZZ` | 2010-01→2011-06, then bankrupt | delisting = a return, not a gap |
| 200 | `NEW` (was `OLD`) | throughout | a ticker change is one security |
| 300 | `ZZZ` | 2015-01→2016-12 | **the recycled ticker** |
| 400 | `AAA` | throughout | control |
| 500 | `BBB` | throughout | preferred share — excluded by type |
| 600 | `CCC` | throughout, ~$2 | excluded by the $5 floor |

**Ticker reuse.** `ZZZ` is two unrelated companies four years apart.
`resolve_permatickers` joins the vendor's `(ticker, date)` price rows against each
candidate security's own `[firstpricedate, lastpricedate]` window, so the early rows
resolve to 100 and the late rows to 300. The test asserts the two keep disjoint price
histories and that the reuse is *reported*, never merged. Keyed on the ticker string
these become one series with a four-year hole, and a twelve-month momentum computed
across that hole is a number about no company at all.

**Ticker change.** 200 trades as `OLD` then `NEW`. Keyed on the ticker it is two
securities with half a history each, neither of which clears §2's 252-day requirement.
Keyed on the permaticker it is one continuous series, which is what the test asserts.

**Delisting.** 100 goes bankrupt; §3 assigns −100%. The realised return on the delist
bar is checked against the assigned one at 1e-9, and the counterfactual is asserted
alongside it: a forward-filled panel scores that total loss as **exactly 0.0%**.

All three §3 buckets are exercised (−100% / −30% / final traded price), the mandatory
sensitivity ladder is shown to move **only** the unknown bucket, and proceeds after a
delisting are shown to accrue at the T-bill rate, which is §3's "proceeds to cash"
combined with §5's cash treatment.

**A defect the fixture caught.** `build_panels` originally took the last *observed*
price in the panel as the final traded bar. Vendor price tables sometimes run a few bars
past a recorded delisting, and on that construction a dead security would keep trading
after it stopped existing. It now takes the last observed price **on or before the
delist date**, and every bar after the event is overwritten with the terminal value, so
stray post-delisting prices are neutralised rather than left where a signal could read
them. Pinned by `test_prices_after_the_delist_date_are_neutralised_not_traded_on`.

**Live vendor round-trip: UNREACHABLE** (see §1).

**001 regression gate: PASS**, bit-identically:

```
  [PASS] generalised engine net Sharpe == 0.352 to 3 dp     0.3520103425
  [PASS] benchmark Sharpe == 0.414 to 3 dp                  0.4137933300
  [PASS] equity curves agree bit for bit                    0.000e+00
  [PASS] weights / targets / trades identical               max |delta| 0.000e+00
```

---

## 3. Step 4 — noise tests, both variants

8 seeds × 6,000 days each, on 60-instrument synthetic panels with full membership.
Sharpe reported gross of costs as well as net.

| variant | gross Sharpe (mean) | t | net (10 bps) | buy-and-hold | turnover |
|---|---|---|---|---|---|
| (a) independent random walks | **+0.033** | +0.47 | −0.083 | +0.005 | 7.4× |
| (b) common factor, 25% of variance | **+0.082** | +1.09 | +0.004 | +0.090 | 7.3× |
| (b) common factor, 50% of variance | **+0.116** | +1.69 | +0.051 | +0.094 | 7.1× |

Per-seed gross Sharpe, variant (a):
`[+0.228, −0.062, −0.004, +0.093, −0.225, +0.163, +0.301, −0.231]`

**Factor attribution on the correlated null:**

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| beta | 0.878 | 0.874 | 0.890 | 0.839 | 0.885 | 0.882 | 0.860 | 0.881 |
| alpha /yr | +1.15% | +1.49% | +1.47% | +0.29% | −1.49% | +0.27% | +0.17% | −0.51% |

> beta to the equal-weight basket **0.874**, alpha **+0.354%/yr** (t +0.98), cross-seed
> correlation between the rule's Sharpe and the basket's **+0.883**

The rule is a factor tilt on correlated data and the tilt earns nothing, because the
factor is driftless. **Gate: PASS** — |mean gross Sharpe| ≤ 0.2 on every variant and
|alpha| < 1%/yr.

**These numbers are identical to experiment 003's, and that is the point.** §1 says the
data is meant to be the only variable. With every name in the universe on every date,
`PointInTimeMomentum` *is* `EquityCrossSectionalMomentum`, and
`test_a_full_membership_reproduces_experiment_003s_strategy_exactly` asserts the two
produce byte-identical weight frames. If they ever diverge, 004 has changed the rule as
well as the data and the comparison to 003 stops meaning anything.

---

## 4. What is built and waiting

All of it is tested offline and reachable from one command.

| module | what it does |
|---|---|
| `trendbot/config_004.py` | parses `PREREG_004.md`; §3 as a table, §6 as a rule with no date in it |
| `trendbot/sharadar.py` | the only module that knows a vendor's vocabulary; `TICKERS`/`SEP`/`ACTIONS`, bulk export, reason→bucket mapping |
| `trendbot/pit_universe.py` | security master, ticker→permaticker resolution, §2's rule, §3's treatment, the exit audit |
| `trendbot/strategies.py` | `PointInTimeMomentum` — 003's rule over a membership matrix |
| `scripts/run_experiment_004.py` | §7's protocol, both arms of §10 |

**§2's rule** is evaluated on the **decision bar** — the close before each rebalance —
not on the rebalance bar itself. §2 says "at each monthly rebalance date t" and §5 fills
at the open of the bar after the signal; those reconcile only one way, because reading
the rule off the rebalance bar's own close while filling at that bar's open would
require knowing the close before the open. That is the lookahead the build order
forbids, and the same convention 001–003 used for their signals.
`test_the_liquidity_ranking_uses_only_data_through_the_rebalance_date` replaces every
bar after a cut date with random values and asserts membership before the cut does not
move.

**§3's treatment reaches the engine through the price panel**, not through a rewrite of
the engine. On the bar after a security's final trade its price becomes
`final_price × (1 + assigned_return)` and then accrues at the risk-free rate.
`verify_delisting_returns` measures the return the engine will actually see and compares
it against §3's table, because constructing the panel correctly and *checking* it are
different claims. Bankruptcy wants a price of exactly zero, which the engine cannot
represent; a floor of 1e-8 is used and the realised return is −100% to twelve decimals.

**Adjustment.** §5 requires split-and-dividend adjusted prices. Sharadar supplies a
split-adjusted open and both a split-adjusted and a split-and-dividend-adjusted close,
so the dividend factor `closeadj / close` is applied to the open to put the two on one
basis; trading a split-only open against a split-and-dividend close would leak the
dividend into the execution price. Dollar volume uses the **unadjusted** close and
as-reported share volume, because restating a 2009 stock's liquidity with today's split
factors would change which names the rule picks. **Whether Sharadar's adjustment is
point-in-time or restated could not be determined — that check needs the data.** It is
the first thing to run when access returns, and §5 requires the exposure to be
quantified rather than assumed.

**Two decisions the document leaves open**, both fixed before any result existed:

1. **The §8 benchmark.** 001–003 read "equal-weight buy-and-hold" literally — buy once,
   never trade. That has no meaning over a universe defined by a rule at each date whose
   members delist. The only coherent reading of "buy-and-hold of the same point-in-time
   universe" is *hold all of it, equally weighted, as the rule defines it each month*.
   Its turnover is forced by the universe rule rather than chosen, so it runs at zero
   cost, as the zero-turnover benchmarks of 001–003 did. The cost-paying variant is
   reported as a sensitivity.
2. **The decile cut.** §2 takes exactly 500 names, so 500/10 = 50 per decile with no
   remainder, and the `even` and `floor` conventions coincide. The ambiguity that needed
   a ruling in 002 and 003 does not arise here. If a month ever yields fewer than 500
   qualifying names inside the sample it will be reported.

---

## 5. What I built that I do not think works as intended

1. **The point-in-time adjustment question is unanswered, and it is the one §5 flags as
   a lookahead channel no `.shift()` catches.** The ratio-invariance argument from
   experiment 003 carries over — a constant factor applied to both endpoints of
   `P(t-21)/P(t-252)` cancels — but that argument covers *splits and dividends*. It does
   not cover a vendor that restates its adjusted series wholesale, and I have not been
   able to check which Sharadar does. Everything downstream inherits that gap.

2. **`resolve_permatickers` was O(rows) in Python and is no longer.** It originally
   looped over every price row to resolve the ticker→ID join by date window — instant on
   the 30-name fixture, tens of millions of iterations on a real SEP table. It is now a
   vectorised merge, measured at **8.0M rows in 3.2 seconds** on a synthetic 4,000-security
   panel with 400 reused tickers. That is not a defect any more, but it is recorded here
   because it shipped as one and because the scale test is synthetic: the real table has
   not been seen.

3. **The join falls back to the latest ticker when the actions table has no
   `permaticker` column.** Sharadar's `ACTIONS` schema was not observable, so
   `build_security_master` handles both cases; the fallback maps a delist reason onto a
   security by ticker string, which is exactly the operation the rest of the module
   exists to avoid. It is flagged in the code but it is a real hole: on a reused ticker
   the wrong reason could attach to the wrong security. Whether the fallback ever fires
   is unknown until the schema is seen.

4. **`SEP_COLUMNS` and `TICKERS_COLUMNS` are asserted but unverified.** The client
   refuses a frame missing the columns it needs, which is the right behaviour, but the
   column names themselves are from documentation rather than from a response. If
   Sharadar names its unadjusted close something other than `closeunadj`, step 1's live
   round-trip fails loudly — but no test can catch that today.

5. **Arm (B) of §10 is defined as "still listed today", which is not quite
   "survivors-only" as 003 experienced it.** 003's universe was *today's S&P 500
   members*; arm (B) is *every security still listed*, ranked by the same liquidity
   rule. Those differ: a company that survived but fell out of the index is in (B) and
   was not in 003. §10 says "the same rule restricted to securities still listed today",
   which is what is implemented, but the A−B gap it measures is therefore the pure
   delisting effect and not the full 003-versus-004 difference. The index-membership
   half of survivorship is not measured by this pair.

6. **`audit_position_exits` classifies but does not assert.** It returns a frame with a
   `priced` column and the runner checks it; nothing in the engine itself refuses to
   proceed if a position exits unpriced. The build order asks for an assertion, and what
   exists is an assertion at the reporting layer rather than at the accounting layer.

7. **The delisting bankruptcy floor is a fudge, disclosed.** A price of 1e-8 rather than
   0 keeps the engine's division well-defined and produces −100% to twelve decimals, but
   a position in a bankrupt name is worth 1e-8 rather than nothing, forever. It is
   immaterial and it is still not what §3 literally says.

---

## 6. Reproduction

```
pytest -q                                                     # 667 passed
python scripts/run_backtest.py --experiment 001 --regression  # exit 0, PASS, bit-identical
python scripts/run_backtest.py --experiment 004 --free-tier   # exit 0, step 1 gate PASS
python scripts/run_backtest.py --experiment 004 --noise       # exit 0, step 4 gate PASS
python scripts/run_backtest.py --experiment 004 --paired      # exit 2, BLOCKED (vendor)
python scripts/run_backtest.py --experiment 004 --validate    # exit 2, BLOCKED (vendor)
```

The two blocked commands print exactly which table they could not reach and state that
nothing was substituted in its place.

**Configurations tried on this dataset, cumulative: 4.** That statement is true. No
formation window other than 252, no skip other than 21, no decile cut other than the top
tenth, no name count other than 500, no price floor other than $5.00 and no liquidity
window other than 60 days was tested at any point. §3's delisting treatment was fixed in
the document before any data was fetched and has not been touched since.
