# Findings 003 — cross-sectional momentum (US equities)

Implementation of `PREREG_003.md` (sha256 `84936a2d7d17…`, committed `746dccb`, tagged
`prereg-003`, unmodified since — asserted by a test against git HEAD). No parameter in
that document was changed, tuned, or substituted. **Configurations tried on this
dataset, cumulative: 3.**

Experiments 001 and 002 are untouched. `PREREGISTRATION.md`, `PREREG_002.md`,
`FINDINGS.md` and `FINDINGS_002.md` are all tracked and byte-identical to their
committed versions, and the 001 regression gate still passes **bit-identically**.

---

## 1. The headline result

Window **2008-01-01 → 2026-08-18** (18.6 years, 4,686 bars). 405 US equities. Net of
10 bps per side. Sharpe is in excess of the 13-week T-bill; the benchmark is treated
identically.

| | strategy | equal-weight buy-and-hold |
|---|---|---|
| CAGR (excess of T-bill) | 14.86% | 13.28% |
| volatility | 23.96% | 20.30% |
| **Sharpe** | **0.699** | **0.716** |
| Sortino | 0.97 | 1.01 |
| max drawdown | −55.4% | −50.2% |
| annual turnover | **7.43×** | 0 |
| skew / excess kurtosis | −0.42 / 7.02 | −0.29 / 10.00 |

### Section 8's pre-committed verdict: **ABANDON**

| support clause (all four required) | result |
|---|---|
| net Sharpe exceeds 0.40 | ✓ 0.699 |
| exceeds buy-and-hold by ≥ 0.15 | ✗ margin **−0.018** |
| ≤1 decile inversion **and** D1−D10 spread positive at t > 2.0 | ✗ **4 inversions, spread −0.40%/mo at t = −0.85** |
| alpha to market positive at t > 2.0 | ✗ +4.94%/yr at **t = 1.74** |

| abandon clause (any one is sufficient) | result |
|---|---|
| **fails to beat equal-weight buy-and-hold at all** | **✗ triggered — 0.699 vs 0.716** |
| **more than one decile inversion** | **✗ triggered — 4 of 9** |
| **D1−D10 t-statistic below 1.0** | **✗ triggered — t = −0.85** |
| alpha to market is negative | not triggered — +4.94%/yr |
| net Sharpe below 0.15 | not triggered — 0.699 |

Three abandon clauses fire independently. **The decile gradient does not merely fail to
appear — it runs backwards.** D10, the past losers, earned the *highest* forward return
of any decile (+1.885%/month against D1's +1.486%).

---

## 2. Step 0 — git tracking status

**Nothing was missing.** Experiment 002 flagged that only four markdown files were
tracked; that was fixed in commit `fe478bd` ("Experiment 002 complete"), which added the
entire source tree. At the start of this experiment `git ls-files` returned 57 tracked files
covering all of `trendbot/`, `tests/`, `scripts/`, `pyproject.toml` and every findings
and pre-registration document, with a clean working tree.

Three directories remain deliberately untracked, and the `.gitignore` entries are
correct rather than oversights:

| path | why it is ignored |
|---|---|
| `data/cache/` | vendor price parquet — regenerable, large, and not a definition of anything |
| `results/` | JSON artefacts each run regenerates |
| `state/`, `.env` | runtime state and secrets |

**What this experiment added to tracking**, because it *is* part of the experiment's
definition rather than a regenerable artefact:

| path | why it must be tracked |
|---|---|
| `data/universe/sp500_constituents_2026-08-19.csv` | §2's universe is a *rule* against a moving index; the resolved membership is only reproducible if the snapshot is pinned |
| `data/universe/sp500_constituents_2026-08-19.meta.json` | source URL, retrieval timestamp, sha256 — the snapshot is refused at load if its bytes no longer match |
| `data/universe/market_proxy.json` | which series §7.5's regression calls "the market", declared before any regression was run |

The immutability guarantee is now enforced rather than assumed: `PREREGISTRATION.md`,
`PREREG_002.md` and `PREREG_003.md` each have a test that fails if the file is modified
relative to git HEAD, and `FINDINGS.md` is checked the same way now that it is tracked.

**The 001 regression gate, re-run:**

```
  [PASS] generalised engine net Sharpe == 0.352 to 3 dp                 0.3520103425
  [PASS] benchmark Sharpe == 0.414 to 3 dp                              0.4137933300
  [PASS] equity curves agree bit for bit (max relative delta == 0)      0.000e+00
  [PASS] weights / targets / trades identical                           max |delta| 0.000e+00
```

---

## 3. Step 1 — universe construction

Source: the current S&P 500 constituent list, retrieved 2026-08-19, pinned to a hashed
CSV. **503 current constituents. 409 have continuous data from 2008-01-01. 94 are
excluded for insufficient history. No substitutions.**

§2 expected 300–400 names; **409 is 9 above that range**. Reported, not adjusted for.

| sector | eligible | excluded |
|---|---|---|
| Industrials | 66 | 17 |
| Financials | 64 | 12 |
| Information Technology | 55 | 18 |
| Health Care | 51 | 8 |
| Consumer Discretionary | 39 | 8 |
| Consumer Staples | 29 | 5 |
| Real Estate | 28 | 2 |
| Utilities | 28 | 3 |
| Materials | 20 | 5 |
| Energy | 15 | 6 |
| Communication Services | 14 | 10 |
| **total** | **409** | **94** |

Latest inception among the eligible: MSCI 2007-11-15, ULTA 2007-10-25, LULU 2007-07-27,
BX 2007-06-22, TEL 2007-06-14, PODD 2007-05-15, IBKR 2007-05-04, DAL 2007-05-03.

The 94 excluded names are dominated by post-2008 IPOs and spin-offs — META (2012),
TSLA (2010), ABBV (2013), V (2008-03), AVGO (2009), PYPL (2015), UBER (2019), PLTR
(2020), COIN (2021) and so on. **This is the survivorship story in miniature and it cuts
both ways**: the exclusions remove the era's biggest winners, while §2's bias removes
its biggest losers.

165 eligible names miss at least one bar inside the window (max 3 of 4,686). As in 001
and 002, "continuous" is read as continuously *listed*, not "the vendor published a bar
every session". Counts are reported so the stricter reading can be applied to a number.

**Window end.** The vendor returned a bar dated 2026-08-19 — today, during the session —
whose "close" is an intraday print. The window ends at **2026-08-18**, the last settled
session. §6's window is "to present"; this is what present means while the market is
open.

---

## 4. Step 2 — corporate action audit

### Adjustment convention, stated plainly

| | |
|---|---|
| source | `yfinance.download(..., auto_adjust=True)` |
| convention | retroactive back-adjustment for splits **and** cash dividends |
| **point-in-time** | **NO** |

The series is the one an observer reconstructs *today*, not the one a trader saw then.
§5 requires this to be stated. What it costs *this* signal is measured rather than
asserted:

> The signal is `P(t-21)/P(t-252)`, a **ratio**. Back-adjustment multiplies every price
> before an action by a constant. An action *after* bar `t` scales both endpoints by the
> same constant, which cancels; an action *between* them scales only the older endpoint,
> which is exactly the correction that makes the ratio a true return.

Tested by injecting an artificial 2:1 split five bars into the future and recomputing
every momentum value dated before it:

> **max absolute difference 0.000e+00 across 1,811,460 momentum values → INVARIANT**

A negative control confirms the harness is not vacuous: the same injection placed
*inside* the formation window moves the signal by more than 0.5 in absolute terms.

So a future corporate action cannot leak into a past signal value through the
adjustment route. **What that does not cover, and is therefore live exposure:** an
action the vendor has *wrong*, which is what the scans below are for; and
delisting/index-membership survivorship, which is §2's problem and is addressed in the
verdict.

### Scan A — every daily move beyond ±35% (the build order's scan)

**97 moves across the 409 eligible names and 1,915,540 observations. 96 are market
moves; 1 coincides with a recorded action.** (Both scans run on all 409, before the
contamination exclusion below.)

Distribution by year — 35 in 2008, 23 in 2009, 8 in 2020 — is exactly where genuine
extreme equity moves belong:

| 2008 | 2009 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2018 | 2019 | 2020 | 2021 | 2022 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 23 | 1 | 2 | 4 | 2 | 1 | 2 | 3 | 2 | 8 | 1 | 2 | 3 | 7 | 1 |

The ten largest are all recognisable events: HIG +102% (2008-12-05), MS +87%
(2008-10-13), LVS +80% (2008-10-29), PCG +75% (2019-01-24, the bankruptcy reversal), UAL
+69%, AIG +66% (2009-03-16), AIG −61% (2008-09-15), STT −59% (2009-01-20), APA −54% and
OXY −52% (both 2020-03-09, the oil price war). **These are handled by doing nothing to
them.** They are the market, not the data.

### Scan B — the adjusted return on *every* recorded action date

Scan A is a *return* filter, so it can only find a broken adjustment large enough to
clear 35%. A 5-for-4 split that failed to adjust is −20%: badly wrong, invisible to scan
A, and quite enough to move a name several deciles. Scan B inverts the question and
checks what the series did on every one of the **240 recorded corporate actions across
162 names**. On a correctly adjusted series an action date is an ordinary trading day.

**Four action dates show an adjusted move beyond ±20%:**

| ticker | action date | adjusted return | semi-raw return | gap | event |
|---|---|---|---|---|---|
| **DHR** | 2016-07-05 | **+61.2%** | +3.6% | **+57.6pp** | Fortive spin-off |
| **EXPE** | 2011-12-21 | **−33.6%** | −51.2% | +17.6pp | TripAdvisor spin-off |
| **VTR** | 2015-08-18 | **−23.6%** | −23.6% | 0 | Care Capital spin-off |
| **AIG** | 2009-07-01 | **−22.1%** | −22.1% | 0 | 1:20 reverse split |

The cause is systematic: **this vendor records spin-offs in the splits table and
back-adjusts by the recorded share ratio, but a spin-off's price ratio is not its share
ratio**, so a discontinuity survives. The next four below the threshold — DD +17.8%
(2019-06-03, Corteva), TMUS −15.8%, LDOS +15.0%, HPQ +13.0% (HPE) — are all the same
class of event, and are reported here so the threshold's position is visible rather than
hidden.

### How each finding was handled

**Rule applied, fixed at ±20% in code before any of these were seen:** a name whose
adjusted series moves beyond that threshold on a recorded action date has an adjustment
this vendor did not get right and is excluded from the universe.

> excluded: **DHR, EXPE, VTR, AIG** → universe 409 − 4 = **405**

DHR is the case that justifies the rule. A spurious **+61%** one-day return inflates
that name's twelve-month momentum for eleven consecutive rebalances, reliably placing it
in the *top* decile and causing it to be bought — precisely the failure §5 warns about,
with the sign reversed.

**Sensitivity — the verdict does not depend on this handling:**

| universe | Sharpe | benchmark | margin |
|---|---|---|---|
| 405 (contaminated names excluded — headline) | 0.6988 | 0.7164 | −0.018 |
| 409 (unfiltered) | 0.6892 | 0.7147 | −0.026 |

### A bug this audit caught in itself

Scan B passed vacuously on its first run — **0 suspicious out of 241 actions**. The
vendor timestamps actions at 09:30 while the price index is midnight-normalised, so
`searchsorted` snapped every action one bar *forward* and the check inspected the
session *after* each ex-date. The DHR discrepancy surfacing in scan A is what exposed it.
Fixed by normalising action timestamps at the source, and pinned by
`test_an_action_stamped_at_the_open_is_checked_against_its_own_bar`, which fails if the
snap ever regresses. A green audit that checks the wrong day is worse than no audit.

**GATE: PASS** — action records retrieved for all 409 eligible names (0 fetch errors),
all 97 extreme moves classified, all 4 contaminated names excluded.

---

## 5. Step 3 — the signal, and its fixture gate

Formula unchanged from 002 by design, so the universe is the only variable. The decile
sort is exercised against a hand-built fixture whose ranks are arithmetic
(`tests/test_xsmom.py`, `tests/test_eq_validation.py`), including the case §3 turns on:
a name with no history must be **excluded from that date's ranking**, not given momentum
zero. In the fixture every real name's momentum is negative, so a name silently zeroed
would rank *first* and be bought at half the book — the test asserts both that this does
not happen and that it *would* have under the reading §3 rules out.

**Decile bucket sizing — a decision the document leaves open.** §3 says "sort into
deciles (ten buckets)" and names no bucket size, unlike §3 of PREREG_002 whose "highest
~8 of 41" pinned the choice. 409 does not divide by ten, and on 95% of dates the
remainder is 9 names. Two conventions were available:

| convention | D1 | D10 | note |
|---|---|---|---|
| **even (used)** | 41 | 40 | ends the same size |
| floor (002's rule) | 40 | 49 | D10 22.5% wider |

**Even was chosen and this was a judgement call, made before the gate was run.** §8's
gate is the D1−D10 spread and §2's survivorship argument is specifically about D10;
widening D10 by 22.5% dilutes the extreme losers toward the middle, biasing the measured
spread in the same direction as the bias §2 already warns about. The alternative is
reported as a sensitivity throughout — under `floor` the gate fails *worse* (6
inversions, t = −0.73), so nothing turns on it.

The convention is used for both the traded position and the gate, so **the traded set
and D1 are the same set by construction** — verified numerically at 1e-12 in
`test_d1_is_exactly_what_the_engine_earns_open_to_open`.

A latent bug was caught here too: `quantile_sizes` and `quantile_labels` disagreed on
the even split, the sizes function putting the short bucket first and the labeller
putting it last. A report would have printed one partition while the engine traded
another. Now pinned by a test sweeping both methods across 25 (n, k) combinations.

---

## 6. Step 4 — noise tests, both variants

8 seeds × 6,000 days each, on 60-instrument synthetic panels. Sharpe reported gross of
costs as well as net: this rule replaces most of its book monthly, so a net figure on
driftless data is expected to sit below zero by the cost drag.

| variant | gross Sharpe (mean) | t | net (10 bps) | buy-and-hold | turnover |
|---|---|---|---|---|---|
| (a) independent random walks | **+0.033** | +0.47 | −0.083 | +0.005 | 7.4× |
| (b) common factor, 25% of variance | **+0.082** | +1.09 | +0.004 | +0.090 | 7.3× |
| (b) common factor, 50% of variance | **+0.116** | +1.69 | +0.051 | +0.094 | 7.1× |

**Factor attribution on the correlated null** (the diagnostic 002 introduced):

> beta to the equal-weight basket **0.874**, alpha **+0.354%/yr** (t +0.98), cross-seed
> correlation between the rule's Sharpe and the basket's **+0.883**

On correlated data the rule is still overwhelmingly a factor tilt, and the tilt earns
nothing because the factor is driftless. Both halves are needed: variant (a) cannot show
either, because independent walks have no factor to tilt onto.

**GATE: PASS** — |mean gross Sharpe| ≤ 0.2 on every variant, and |alpha to the common
factor| < 1%/yr.

---

## 7. Step 5 — backtest, turnover, cost sensitivity

Annualised one-way turnover **7.43×**, above §9's expectation of "higher than 002's
6.12×". Names held: 40–41 every day.

| bps/side | 0 | 5 | **10** | 20 | 40 |
|---|---|---|---|---|---|
| Sharpe | 0.730 | 0.714 | **0.699** | 0.668 | 0.606 |
| CAGR (excess) | 15.72% | 15.29% | **14.86%** | 14.01% | 12.32% |
| vs buy-and-hold | +0.013 | −0.002 | **−0.018** | −0.049 | −0.111 |

Costs matter more than in 002 — 12 bps of Sharpe between free and headline — but they
are not what sinks this. **The strategy beats buy-and-hold only at literally zero
transaction cost, and even there by 0.013, a tenth of the 0.15 margin §8 requires.**

### Benchmark construction sensitivity

| construction | Sharpe | margin | beats it? |
|---|---|---|---|
| buy once, hold (literal — headline) | 0.716 | −0.018 | no |
| rebalanced monthly | 0.722 | −0.023 | no |
| rebalanced daily | 0.746 | −0.048 | no |

It loses to the benchmark under **all three** constructions, so the verdict does not
turn on that unstated detail. Plain SPY over the same window earned an excess Sharpe of
**0.573**, below both.

### Calendar years (total return)

| year | strategy | benchmark | SPY |
|---|---|---|---|
| 2008 | −43.5% | −32.0% | −36.8% |
| 2009 | +17.4% | +34.5% | +26.4% |
| 2010 | +37.6% | +26.4% | +15.1% |
| 2011 | +2.2% | +5.1% | +1.9% |
| 2012 | +27.1% | +20.4% | +16.0% |
| 2013 | +50.6% | +38.7% | +32.3% |
| 2014 | +18.5% | +19.2% | +13.5% |
| 2015 | +10.1% | +8.1% | +1.2% |
| 2016 | +7.8% | +12.1% | +12.0% |
| 2017 | +24.5% | +26.1% | +21.7% |
| 2018 | +0.2% | +0.5% | −4.6% |
| 2019 | +31.4% | +34.2% | +31.2% |
| 2020 | +21.3% | +23.8% | +18.3% |
| 2021 | +18.8% | +29.7% | +28.7% |
| 2022 | +3.0% | −15.9% | −18.2% |
| 2023 | +22.9% | +25.8% | +26.2% |
| 2024 | +31.0% | +20.9% | +24.9% |
| 2025 | +19.1% | +11.0% | +17.7% |
| 2026 | +44.8% | +15.8% | +13.1% |

Beat the benchmark in 8 of 19 years. Maximum drawdown **−55.4%**, trough 2009-03-09 from
a 2008-06-05 peak, recovered 2011-07-05.

---

## 8. Step 6 — decile monotonicity (§8's gate)

Sorted on the momentum known at the previous close; forward return from the open of the
rebalance bar to the open of the next; equal weight within each decile; gross of costs.
223 rebalances, all measured.

| | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean forward return, /month | **1.486%** | 1.090% | 1.174% | 1.094% | 1.260% | 1.236% | 1.336% | 1.298% | 1.291% | **1.885%** |
| annualised | 17.03% | 12.27% | 13.56% | 12.40% | 14.66% | 14.36% | 15.41% | 14.73% | 13.94% | **20.13%** |
| t-statistic | 3.84 | 3.34 | 3.79 | 3.43 | 3.98 | 3.92 | 3.86 | 3.59 | 3.08 | 3.28 |
| monthly volatility | 5.78% | 4.87% | 4.62% | 4.77% | 4.73% | 4.71% | 5.17% | 5.40% | 6.27% | **8.58%** |
| avg names | 41.0 | 40.0 | 41.0 | 40.0 | 40.9 | 40.0 | 40.9 | 40.0 | 40.9 | 40.0 |

Step-by-step change D1→D10, in %/month:
**−0.396**, +0.084, −0.079, **+0.165**, −0.024, **+0.100**, −0.038, −0.007, **+0.593**

### Verdict on the gate: **FAIL, and the gradient is inverted.**

- **4 adjacent inversions of 9.** §8 tolerates at most one.
- **D1−D10 spread = −0.399%/month (−4.68% annualised), t = −0.846, n = 223.** §8
  requires t > 2.0 for support and abandons below t = 1.0. The spread is not merely
  insignificant — it is **negative**. Past losers beat past winners.
- D10 also carries the highest volatility (8.58%/month vs D1's 5.78%) and the lowest hit
  rate (62.3% vs 65.0%), the signature of a distressed-and-recovered bucket.
- Under the alternative bucket convention: 6 inversions, spread −0.332%/month at
  t = −0.729. Same conclusion, slightly worse.

Every decile earns a strongly positive forward return (t between 3.1 and 4.0) because
every decile is long equities in a bull market. The *ordering* is what the hypothesis
predicts, and there is no ordering.

---

## 9. Step 7 — market-beta attribution

Declared in `data/universe/market_proxy.json` before any regression was run: the
cap-weighted market index is the headline and §8's clauses are adjudicated on it; the
equal-weight benchmark is reported alongside.

| regressor | beta | alpha (annualised) | alpha t | p | R² | residual vol |
|---|---|---|---|---|---|---|
| **SPY (cap-weighted) — HEADLINE** | **+1.042** | **+4.94%** | **+1.742** | 0.082 | 0.740 | 12.23% |
| equal-weight B&H of the same 405 | +1.055 | +1.40% | +0.560 | 0.575 | 0.799 | 10.74% |

Alpha is **positive but not significant** — it does not clear §8's t > 2.0, and it does
not trigger the negative-alpha abandon clause either. Beta is essentially 1.0 against
both regressors, so the strategy is very close to a plain long-equity book with a
selection overlay that adds 4.9%/yr of unexplained return at 1.7 standard errors.

Against the equal-weight benchmark alpha falls to +1.40% at t = 0.56 — i.e. **most of
the apparent alpha versus SPY is the equal-weight size tilt, not decile selection.**
Reporting both is what makes that visible; either alone would have been misleading.

---

## 10. Step 8 — worst months

| month | strategy | benchmark | relative |
|---|---|---|---|
| 2008-10 | −16.11% | −18.61% | +2.50% |
| 2008-09 | −14.72% | −8.40% | −6.32% |
| 2008-01 | −13.56% | −3.57% | −9.99% |
| 2020-03 | −13.13% | −12.56% | −0.57% |
| 2018-10 | −12.43% | −9.07% | −3.36% |

### 🚩 FLAG: the implementation is suspect on §9's own terms

§9 requires **March–May 2009 and April 2020** among the worst months. **None of the four
appears.** Where they actually rank, of 224 months:

| month | strategy | benchmark | relative | rank (absolute) | rank (relative) |
|---|---|---|---|---|---|
| 2009-03 | +5.15% | +9.16% | −4.01% | 168 / 224 | **11 / 224** |
| 2009-04 | +2.19% | +12.54% | −10.36% | 119 / 224 | **1 / 224** |
| 2009-05 | −1.01% | +2.68% | −3.69% | 69 / 224 | **16 / 224** |
| 2020-04 | +13.68% | +12.48% | +1.20% | 221 / 224 | 152 / 224 |

§9 states that if a stock-level implementation also fails this test, **the
implementation is wrong, not the anomaly**. I am reporting it as that: a failed
pre-registered prediction and an implementation flagged as suspect. This is now the
**second consecutive experiment** to fail the same check, and the pre-registration
explicitly anticipated that a stock-level rerun would settle it. It did not settle it in
the direction §9 expected.

For completeness rather than as a defence: all three 2009 months are among the worst
*relative* months, with April 2009 the single worst of 224 at −10.36%. April 2020 is not
— it was the eighth-best absolute month. §9 said "worst months", and in absolute terms
the prediction is false; the relative ranking is a different claim and does not rescue
it. I have not investigated further, because doing so under a failed pre-registered
check is where explaining-away starts.

Max drawdown −55.4% comfortably exceeds any crash threshold, but that is not what §9
asked.

---

## 11. Step 9 — PSR and deflated Sharpe at `configs_tried = 3`

All three trials are known, so nothing is estimated: per-period Sharpes of 0.022175
(001), 0.029173 (002) and 0.044017 (003).

| quantity | value |
|---|---|
| annualised Sharpe | +0.699 |
| observations | 4,686 |
| configurations tried | **3** |
| **PSR(0)** | **0.9986** |
| deflated Sharpe at 3 configurations | 0.9902 |

**PSR(0) = 0.9986 clears 95%.** So does the deflated Sharpe.

This changes nothing. PSR asks whether a Sharpe of 0.699 over 18.6 years is
distinguishable from zero, and the answer is yes — but so is the benchmark's 0.716, by
more. A long-only equity book in a 2008–2026 sample has a reliably non-zero Sharpe
whatever it holds. PSR does not ask whether the strategy beats doing nothing, and it
does not ask whether the decile gradient exists. §8 asked both and both answers are no.

---

## 12. Step 10 — the verdict, with both survivorship readings

**ABANDON.**

The hypothesis was that individual stocks outperforming their peers over twelve months
continue to outperform over the following month. On this universe over this window,
against criteria fixed in advance:

- **The strategy does not beat equal-weight buy-and-hold**: 0.699 against 0.716, under
  every benchmark construction and at every cost above zero.
- **The decile gradient is inverted.** Four of nine steps invert and the D1−D10 spread
  is −4.68%/yr at t = −0.85. §8 identified this as the important test.
- **Alpha to the market is positive but insignificant** (+4.94%/yr, t = 1.74), and
  mostly a size tilt: +1.40% at t = 0.56 against the equal-weight benchmark.
- Net Sharpe clears 0.40 and PSR clears 95%. Neither is sufficient; §8 required all four
  support clauses and only one holds.

### Survivorship — both readings, neither selected after the fact

§2 fixed the direction of the bias in advance: survivorship removes failures, failures
concentrate in the *bottom* of a momentum ranking, so D10 is artificially strong, the
D1−D10 spread artificially **narrow** and the gradient artificially **flat**. §2 also
fixed what follows from each outcome:

**Reading 1 — if the gate passes, the pass is conservative evidence.** A real universe
including the bankruptcies would have a weaker D10 and therefore a *wider* spread than
measured. **This reading does not apply: the gate failed.**

**Reading 2 — if the gate fails, the failure is ambiguous.** It is consistent with the
anomaly being absent, *and* with the anomaly being present but flattened below detection
by the bias. **This is the reading that applies.** The measured t of −0.85 cannot
distinguish them, and this experiment cannot be made to.

Two things sharpen how much weight Reading 2 carries, and they point in opposite
directions:

- The observed effect is not flat, it is **inverted** by −4.68%/yr. §2 predicted
  flattening, not sign reversal. Flattening a true positive spread to zero is what the
  bias does; driving it to −4.68%/yr requires the bias to be doing more work than §2
  anticipated. That is possible — D10 contains beaten-down names that *survived to be in
  the index today*, which is close to a definition of "recovered" — but it is a larger
  claim than §2 made.
- Working the other way: 94 of 503 current constituents were excluded for insufficient
  history, and they are disproportionately the era's largest winners. That exclusion is
  a *second* selection effect, not the one §2 described, and its direction on the decile
  gradient is not established here.

The honest summary is that **this universe cannot answer the question §1 asked.** A
point-in-time constituent set with delisted names is required, and PREREG_003 §2 said in
advance that no free source provides one. The verdict is ABANDON on the pre-committed
rule; the *hypothesis* is untested rather than refuted, and saying otherwise in either
direction would be selecting a reading after the fact.

---

## 13. Everything I decided that the document does not fix

None can change a position, and each was fixed before the number it affects existed.

1. **Decile bucket sizes**: even split (D1 41 / D10 40) rather than 002's
   remainder-to-the-bottom. Reasoned in §5 above; the alternative is reported everywhere
   and the gate fails worse under it.
2. **Market proxy**: the cap-weighted index is the headline, the equal-weight benchmark
   reported alongside. Declared in `data/universe/market_proxy.json` before any
   regression was run.
3. **The ±20% action-date threshold** for excluding a contaminated name, fixed in code
   before the four affected names were known.
4. **"Continuous" means continuously listed**, not "the vendor published a bar every
   session" — the same reading as 001 and 002.
5. **Window end 2026-08-18**, the last settled session, because the vendor returned a
   live intraday bar for today.
6. **Pre-window price history forms the first signal**; §2's history requirement only
   makes sense on that reading.
7. **Benchmark is buy-and-hold read literally**; all three constructions reported.
8. **Noise panels are 60 instruments wide**, not 400+. The null is a property of the
   ranking rule, not of universe size, and 400 wide costs hours for an identical answer.
9. **Noise pass criterion is the gross Sharpe**, as in 002.
10. **The D1−D10 t-statistic is a plain one-sample t** on non-overlapping monthly
    spreads; no autocorrelation correction.
11. **Alpha's t-statistic is the plain OLS one.** No Newey-West correction — it would
    widen the standard error, and §8 turns on t > 2. Stated because it matters: the
    headline alpha t of 1.74 already fails, and a correction could only lower it.
12. **Walk-forward windows** 3y/1y/1y, as in 001 and 002.

---

## 14. What I built that I do not think works as intended

1. **`quantile_sizes` and `quantile_labels` disagreed on the even split** until caught
   here — the sizes function put the short bucket first, the labeller put it last. Now
   pinned by a test across 25 (n, k) combinations, but it shipped in a state where a
   report would have described a different partition from the one traded. The general
   lesson is unaddressed: nothing structurally prevents a *reporting* function from
   drifting from the function that acts.

2. **Scan B passed vacuously on its first run** — 0 of 241 actions flagged, because
   09:30 action timestamps sorted after their own midnight-indexed bar. Fixed and pinned.
   It was found by accident, through a discrepancy scan A surfaced, not by the test suite.
   A green data-quality check that inspects the wrong day is the most dangerous artefact
   in this repository and I do not have a general defence against another one.

3. **The contaminated-name exclusion is a blunt instrument.** DHR's defect affects
   roughly eleven months of one name's momentum; the rule discards all 18.6 years of it.
   §3's per-date eligibility machinery could have excluded only the contaminated dates,
   which would have been more faithful, but that is a rule the document does not contain
   and I was not willing to invent one after seeing which names it would affect.

4. **AIG's −22.1% on its reverse-split date may be a genuine market move**, not a broken
   adjustment — the adjusted and semi-raw series agree exactly, unlike DHR's. It was
   excluded anyway because the pre-stated rule is mechanical. That is the right way round
   (the rule was fixed first) but it means one of four exclusions is probably unnecessary.

5. **Four spin-off events sit just under the ±20% threshold** (DD +17.8%, TMUS −15.8%,
   LDOS +15.0%, HPQ +13.0%) and are the same class of defect as the four excluded. The
   threshold is defensible but it is a cliff, and names on the safe side of it carry
   known contamination into the backtest.

6. **`_split_dates` is imported into the runner as a private name.** It is a module
   internal being used across a boundary because the caching lives there; it should be
   public or the caching should move.

7. **The equity path disables the ±50% daily-move check in `load_prices`.** That is
   correct — single stocks legitimately move more — and the corporate-action audit
   replaces it. But the replacement is a *report*, not a *refusal*: `load_prices` will
   now return a badly corrupted equity panel without complaint, and only a caller who
   runs the audit will notice. The 001 and 002 paths keep the original behaviour, so
   nothing regressed, but the equity loader is less safe than the one it borrows from.

8. **Nothing here validates the vendor's dividend adjustments.** Splits are cross-checked
   against recorded actions; dividends are not checked at all. A systematically wrong
   dividend adjustment on a high-yield name would flow straight into a twelve-month
   momentum ranking and nothing in this repository would notice.

---

## 15. Reproduction

```
pytest -q                                                    # 600 passed
python scripts/run_backtest.py --experiment 001 --regression # exit 0, gate PASS, bit-identical
python scripts/run_backtest.py --experiment 003 --noise      # exit 0, gate PASS
python scripts/run_backtest.py --experiment 003 --validate   # exit 0, verdict ABANDON
```

Price data is Yahoo, split- and dividend-adjusted, cached to `data/cache`. The
risk-free series is `^IRX`, the 13-week T-bill, applied symmetrically to idle cash and
to both Sharpe calculations — the same series and convention as 001 and 002.

New code for this experiment:

| file | what it is |
|---|---|
| `trendbot/config_003.py` | parses `PREREG_003.md`; the universe is parsed as a *rule*, not a list |
| `trendbot/equities.py` | universe resolution, both audit scans, the point-in-time report |
| `trendbot/engine/eq_validation.py` | market regression and §8's four-clause rule |
| `trendbot/strategies.py` | `EquityCrossSectionalMomentum` — same rule object, different arguments |
| `trendbot/xsmom.py` | gains the `even` bucket convention; `floor` remains the default |
| `trendbot/engine/xs_validation.py` | `bucket_study` generalises 002's quintile gate to any k |
| `scripts/run_experiment_003.py` | §7's protocol |
| `data/universe/*.csv`, `*.json` | the pinned constituent snapshot and the market proxy |
| `tests/test_config_003.py` | the parser refuses to guess |
| `tests/test_equities.py` | universe resolution and both audit scans, offline |
| `tests/test_eq_validation.py` | decile cut, market regression, decision rule |

`trendbot/signal.py` was not touched; its content hash is still the one
`tests/test_single_signal.py` pins. `trendbot/data.py` gained one optional parameter,
defaulted so that the 001 and 002 paths are byte-identical.

**Configurations tried on this dataset, cumulative: 3.** That statement is true. No
formation window other than 252, no skip other than 21, no decile cut other than the top
tenth, and no universe other than §2's rule was tested at any point. No short leg, no
beta neutralisation, and no residual-momentum variant was built or run.
