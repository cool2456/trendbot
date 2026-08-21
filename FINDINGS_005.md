# Findings 005 — cross-sectional momentum (currencies)

Implementation of `PREREG_005.md` (sha256 `6b890589479c…`, committed `0472120`, tagged
`prereg-005`, unmodified since). No parameter in that document was changed, tuned or
substituted. **Configurations tried, cumulative: 4.**

Experiments 001–004 are untouched: no document, config parser, engine module or runner
belonging to them was modified, and the 001 regression gate passes **bit-identically**
(equity curves, weights, targets and trades all agree to `0.000e+00`).

---

## 1. The headline result

Window **1999-01-04 → 2026-08-14** (27.5 years, 6,927 publication days). 22 currencies
against the US dollar, from Federal Reserve H.10 via FRED. Net of 5 bps per side.
Sharpe is in excess of the 13-week T-bill; the benchmark is treated identically.

| | strategy | dollar factor |
|---|---|---|
| CAGR (excess of T-bill) | −2.08% | −2.52% |
| volatility | 6.86% | 5.56% |
| **Sharpe** | **−0.271** | **−0.431** |
| Sortino | −0.37 | −0.59 |
| max drawdown (of the excess-return stream) | −50.0% | −51.8% |
| annual turnover | **6.14×** | 0 |
| skew / excess kurtosis | −0.48 / 5.04 | +0.02 / 4.40 |
| correlation to the dollar factor | 0.772 | — |

### Section 8's pre-committed verdict: **ABANDON**

| support clause (all four required) | result |
|---|---|
| net Sharpe exceeds 0.40 | ✗ **−0.271** |
| exceeds the dollar factor by ≥ 0.15 | ✓ margin **+0.159** |
| ≤ 1 quintile inversion **and** Q1−Q5 spread positive at t > 2.0 | ✗ 1 inversion, spread +0.084%/mo at **t = +0.623** |
| alpha to the dollar factor positive at t > 2.0 | ✗ +0.42%/yr at **t = +0.503** |

| abandon clause (any one is sufficient) | result |
|---|---|
| fails to beat the dollar factor at all | no (−0.271 vs −0.431) |
| more than one quintile inversion | no (1 of 4) |
| **Q1−Q5 t-statistic below 1.0** | **YES — t = +0.623** |
| alpha to the dollar factor negative | no (+0.42%/yr) |
| **net Sharpe below 0.15** | **YES — −0.271** |

Two abandon clauses fire independently. The verdict is not marginal.

### What the result actually says

The one support clause that passes is the one comparing the strategy to the benchmark:
selecting the top quintile of twelve-month winners *does* beat holding all 22 currencies,
by 0.16 of a Sharpe, consistently enough that the strategy beat the dollar factor in 16
of 28 calendar years. Every clause about whether the result is *worth anything* fails.

Both numbers are negative because of what §4 and §5 jointly define, and this is the most
important thing to understand about the headline: the measured quantity is
**spot change − US T-bill**. The book is fully invested in foreign currency, so it earns
no foreign interest, and §5 charges the T-bill rate against it anyway. Over a window in
which the T-bill averaged 2.03%, that is a −2%/yr construction before any currency moves.
The strategy's CAGR is −2.08%. §4 anticipates this and calls it an approximation; step 8
measures it and finds it worth **+0.48 of Sharpe** — more than three times §8's margin
threshold. See §9 below.

The verdict does not change under that correction: on carry-adjusted returns the Sharpe
is +0.209 against the dollar factor's +0.090, so the 0.40 support threshold and the 0.15
margin are both still missed, and the Q1−Q5 t-statistic — which is computed on the spot
panel and is carry-free on both legs — is unaffected.

---

## 2. Step 1 — the quote-convention audit (hard gate): **PASS**

§2 names this "the single most likely way this experiment produces a wrong answer", and
the reason is that an inverted series raises nothing at all. It is a well-formed positive
price series with a plausible volatility. Every downstream computation succeeds. Only the
rank is backwards.

**The direction is therefore never inferred from the values.** It is read out of FRED's
own `units` string, which states the convention in words, and cross-checked against the
title. `"<A> to One <B>"` means the value is the number of A that buys one B. A series
whose two strings disagree, whose units do not parse, or whose units do not name exactly
one US dollar leg is **refused**, not guessed at (`trendbot/fx.py:parse_quote_convention`).

### The direction determined for every series

18 of 22 are quoted foreign-per-USD and are inverted; 4 are already USD-per-foreign and
are left alone.

| series | FRED units | direction | normalise by | currency |
|---|---|---|---|---|
| DEXBZUS | Brazilian Reals to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Brazilian Reals |
| DEXCAUS | Canadian Dollars to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Canadian Dollars |
| DEXCHUS | Chinese Yuan Renminbi to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Chinese Yuan Renminbi |
| DEXDNUS | Danish Kroner to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Danish Kroner |
| DEXHKUS | Hong Kong Dollars to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Hong Kong Dollars |
| DEXINUS | Indian Rupees to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Indian Rupees |
| DEXJPUS | Japanese Yen to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Japanese Yen |
| DEXKOUS | South Korean Won to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | South Korean Won |
| DEXMAUS | Malaysian Ringgit to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Malaysian Ringgit |
| DEXMXUS | Mexican Pesos to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Mexican Pesos |
| DEXNOUS | Norwegian Kroner to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Norwegian Kroner |
| DEXSDUS | Swedish Kronor to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Swedish Kronor |
| DEXSFUS | South African Rand to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | South African Rand |
| DEXSIUS | Singapore Dollars to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Singapore Dollars |
| DEXSLUS | Sri Lankan Rupees to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Sri Lankan Rupees |
| DEXSZUS | Swiss Francs to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Swiss Francs |
| DEXTAUS | Taiwan Dollars to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Taiwan Dollars |
| DEXTHUS | Thai Baht to One U.S. Dollar | FOREIGN_PER_USD | 1 / x | Thai Baht |
| **DEXUSAL** | U.S. Dollars to One Australian Dollar | USD_PER_FOREIGN | **x** | Australian Dollar |
| **DEXUSEU** | U.S. Dollars to One Euro | USD_PER_FOREIGN | **x** | Euro |
| **DEXUSNZ** | U.S. Dollars to One New Zealand Dollar | USD_PER_FOREIGN | **x** | New Zealand Dollar |
| **DEXUSUK** | U.S. Dollars to One U.K. Pound Sterling | USD_PER_FOREIGN | **x** | U.K. Pound Sterling |

(DEXVZUS, Venezuelan Bolivares, parses as FOREIGN_PER_USD but is excluded from the
universe — see §3.)

### Verification 1 — normalised level on a named date, against known market history

Ten checks. **Each band was chosen so that the inverted value falls outside it**; a band
that both the correct and the reciprocal value satisfy tests nothing, so the code asserts
that each band is discriminating and reports it per row. All 10 pass; all 10 discriminate.

| series | date | normalised (USD per unit) | expected band | inverted would be | basis |
|---|---|---|---|---|---|
| DEXJPUS | 2013-12-31 | 0.0095012 | [0.007, 0.013] | 105.25 | yen near 105/USD at end-2013 |
| DEXUSUK | 2007-12-31 | 1.9843 | [1.70, 2.20] | 0.50396 | sterling near its multi-decade high ≈ 2.00 |
| DEXSZUS | 2011-08-09 | 1.3674 | [1.15, 1.60] | 0.73130 | franc's record high, 2nd week of Aug 2011 |
| DEXKOUS | 2010-12-30 | 0.00088449 | [0.0005, 0.0015] | 1130.6 | won near 1,130/USD at end-2010 |
| DEXMXUS | 2015-12-31 | 0.058156 | [0.030, 0.100] | 17.195 | peso near 17/USD at end-2015 |
| DEXUSEU | 2008-07-15 | 1.5923 | [1.40, 1.65] | 0.62802 | euro's all-time high, just above 1.60 |
| DEXCAUS | 2002-01-31 | 0.62834 | [0.55, 0.70] | 1.5915 | CAD record low ≈ 1.61/USD, Jan 2002 |
| DEXINUS | 2013-08-28 | 0.014535 | [0.010, 0.020] | 68.80 | rupee's then-record low ≈ 68/USD |
| DEXCHUS | 2004-06-30 | 0.12082 | [0.110, 0.130] | 8.2766 | yuan pegged at 8.2765 until July 2005 |
| DEXBZUS | 2015-09-24 | 0.24017 | [0.15, 0.32] | 4.1638 | real past 4/USD, late Sept 2015 |

The table covers **both quote directions** — a table of only inverted series would leave
the four untouched ones unverified.

### Verification 2 — structural identities that hold by monetary policy

Stronger than any single-series band, because these are commitments made by central banks
rather than facts about this repository.

| identity | statistic | policy band | observed range | days inside |
|---|---|---|---|---|
| Hong Kong linked exchange rate | USD per HKD | [0.12658, 0.12987] | 0.12739 … 0.12973 | **100.00%** |
| Danish krone ERM II central rate | implied DKK per EUR = (USD per EUR)/(USD per DKK) | 7.46038 ± 2.25% | 7.0752 … 7.5480 | **99.97%** |

The Danish check is a **cross-rate between two separate FRED series**, so inverting either
one breaks it and it cannot be passed by luck. It is the single most informative check
here, and it earned its place: see §11.

### Verification 3 — implied long-run drift

Inversion flips the sign of a currency's drift against the dollar. The weakest of the
three checks, reported rather than relied on. All 22 inside ±25%/yr; the extremes are
Sri Lankan rupees −5.75%/yr and Swiss francs +1.90%/yr, both in the direction 27 years of
history say they should be.

### Access path

Both transports were used and **cross-checked against each other**. The universe was first
built from `fred.stlouisfed.org` (no key was present in `.env` when the build started); a
`FRED_KEY` appeared mid-build, and every series was then re-fetched through
`api.stlouisfed.org` and compared:

* release 17 enumeration: **33 series from each, identical sets** — so the web listing was
  not silently truncated, which was the main risk of the keyless path;
* all 23 daily FX series: titles identical, units identical, **max absolute disagreement
  across every shared observation `0.000e+00`**, zero NaN-pattern mismatches, zero
  date-coverage mismatches.

---

## 3. Step 2 — universe construction (§2)

§2's rule is executed, not paraphrased: the H.10 release is enumerated **from FRED**
(release id 17), and each member is filtered by **its own FRED metadata**.

**Final count: 22 currencies.** §2 expected roughly 20–23. ✓

Ten of the release's 33 series are excluded as not being daily bilateral USD exchange
rates — by their units, not by name:

| series | why |
|---|---|
| DTWEXAFEGS, DTWEXBGS, DTWEXEMEGS | daily, but units are `Index Jan 2006=100` |
| DTWEXB, DTWEXO | daily, but units are `Index Jan 1997=100` |
| DTWEXM | daily, but units are `Index Mar 1973=100` |
| TWEX, TWEXB, TWEXM, TWEXO | weekly, ending Wednesday (all DISCONTINUED) |

### The universe, with the largest gap in each

| series | currency | first | last | published / scheduled | largest gap |
|---|---|---|---|---|---|
| DEXBZUS | Brazilian Reals | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXCAUS | Canadian Dollars | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXCHUS | Chinese Yuan Renminbi | 1999-01-04 | 2026-08-14 | 6927 / 6927 | **0** |
| DEXDNUS | Danish Kroner | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXHKUS | Hong Kong Dollars | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXINUS | Indian Rupees | 1999-01-04 | 2026-08-14 | 6925 / 6927 | 1 |
| DEXJPUS | Japanese Yen | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXKOUS | South Korean Won | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXMAUS | Malaysian Ringgit | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXMXUS | Mexican Pesos | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXNOUS | Norwegian Kroner | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXSDUS | Swedish Kronor | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXSFUS | South African Rand | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXSIUS | Singapore Dollars | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXSLUS | Sri Lankan Rupees | 1999-01-04 | 2026-08-14 | 6925 / 6927 | 1 |
| DEXSZUS | Swiss Francs | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXTAUS | Taiwan Dollars | 1999-01-04 | 2026-08-14 | 6923 / 6927 | 1 |
| DEXTHUS | Thai Baht | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXUSAL | Australian Dollar | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXUSEU | Euro | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXUSNZ | New Zealand Dollar | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |
| DEXUSUK | U.K. Pound Sterling | 1999-01-04 | 2026-08-14 | 6926 / 6927 | 1 |

### Gap gate: **PASS**

The allowance is **10 scheduled publication days** — two calendar weeks, longer than the
longest national market closure among these currencies (Chinese New Year ≈ 8 business
days, Japan's Golden Week ≈ 5) and far shorter than any discontinuation. **It was fixed
in code before the data was inspected.** The largest gap anywhere in the universe turned
out to be **1 day**, so the threshold never bound; had it been set anywhere from 2 to
several hundred the universe would be the same.

The single-day gaps are almost all the same date — 2005-09-05, US Labor Day, on which
FRED carries a Chinese yuan observation and nothing else. See §4.

### Excluded: 1

| series | currency | reason |
|---|---|---|
| **DEXVZUS** | Venezuelan Bolivares | first observation **2000-01-03**, after the window opens on 1999-01-04 — no data for the start of the sample |

Excluded **entirely**, not truncated, per §2. Nothing was substituted for it. (It is worth
noting what this exclusion buys: the bolívar underwent redenominations that would have
given it a drift no plausibility check could accept, and §3's 1999 boundary removes it
mechanically rather than by judgement.)

---

## 4. Step 3 — returns, and how missing observations are handled

§4's return is the log change in the USD value of the foreign currency, spot only.

**One rate per day, and therefore no open.** FRED publishes a single H.10 rate per day —
the New York Fed noon buying rate. There is no intraday open, so the panel's `open` and
`close` are the same frame. The separation between decision and fill is the engine's
single execution shift and nothing else: the signal is formed on the rate of bar *t* and
the position is taken at the rate of bar *t+1*, which is §5 verbatim. No fill anywhere
uses a price that was not published before the decision.

### Two kinds of missing day, handled differently

| kind | what it means | handling |
|---|---|---|
| **US market holiday** | no currency has a rate | not a trading day at all. The calendar is built as the dates on which *at least one* H.10 series published, so these dates are simply absent from the panel. |
| **foreign holiday** | every other currency published, this one did not | the market was shut; the last published rate is the only defensible mark. The series is **forward-filled**. |

**Forward-filled cells inside the window: 26 of 152,394 (0.0171%).** The handling is
almost inert in practice — FRED prints a rate for nearly every currency on nearly every
US business day — but it had to be decided rather than discovered, because a hole landing
exactly on a `t−21` or `t−252` anchor would otherwise drop that currency out of a
rebalance's ranking for a reason that has nothing to do with momentum.

### 2005-09-05, reported rather than removed

US Labor Day. FRED carries a Chinese yuan observation and nothing else. The pre-declared
calendar rule ("a date on which at least one H.10 series published is a bar") therefore
admits it, and the other 21 currencies are forward-filled across it. That is one bar in
6,927. It is left in, because the calendar rule was fixed before the data was looked at
and changing it afterwards would be selection. Its effect on any reported number is below
the resolution of every figure in this document.

### No lookahead — computed, not asserted

A forward fill is easy to *claim* is backward-looking. The runner demonstrates it:

* every observation from **2012-08-28** onward multiplied by 3, the fill, the momentum and
  the quintile weights all recomputed from scratch;
* **max |change| in any target weight on any bar before that date: `0.000e+00`.**
* no series is back-filled before its own first observation: **PASS**.

Both momentum shifts are positive and there is no negative shift anywhere in the package
(enforced by `tests/test_repo_invariants.py`). `tests/test_fx.py` asserts the same two
properties on fixtures.

### The sample window and the euro

The panel carries each series' **full published history** (back to 1971 for the oldest) so
that the 252-day formation window at the first bar of 1999 reaches into 1998. Everything
is **measured** from 1999-01-01 and nothing before it is reported — the same arrangement
experiments 002 and 003 use. The euro begins on 1999-01-04 and so has no defined momentum
until roughly 2000-01; on those dates it is simply excluded from the ranking, which is
§4's own rule for insufficient history rather than a special case. The book is 5
currencies on every one of the 6,927 bars regardless, because 21 valid currencies still
split into five buckets.

---

## 5. Step 4 — noise tests, both variants: **PASS**

16 seeds per variant (the build order's floor is 8), 6,000 days, panels 22 instruments
wide — the real universe's width, so the null is tested at the bucket geometry the
strategy actually trades. Volatility is set to 10%/yr rather than the generator's equity
default of 16%, because major FX runs near that and the cost drag's size *in Sharpe terms*
depends on it. Neither is a strategy parameter and neither can change a position.

Sharpe is reported **gross** as well as net: this rule replaces a quarter of its book
every month, so a net figure on driftless data is expected to sit below zero by the cost
drag, and testing only the net number would conflate "the rule finds nothing" with "the
costs are large".

### Three independent seed blocks

Because the first two blocks disagreed in *sign*, a third was run at a different seed
offset. All three are reported.

| variant | seeds 0–15, 23 wide | seeds 0–15, 22 wide | seeds 100–115, 22 wide |
|---|---|---|---|
| (a) independent walks, gross | −0.090 (t −2.41) | +0.095 (t +2.17) | −0.019 (t −0.36) |
| (b) common factor, 25% of variance, gross | +0.046 (t +1.44) | +0.115 (t +2.81) | −0.065 (t −1.45) |
| (b) common factor, 50% of variance, gross | +0.062 (t +1.47) | +0.119 (t +2.53) | −0.060 (t −1.46) |
| buy-and-hold, mean | +0.035 … +0.081 | +0.052 … +0.085 | −0.088 … −0.066 |

Every value lies inside ±0.12 Sharpe. **The sign is not stable across a one-column change
in panel width or a change of seed block, which is exactly what one expects of an estimate
whose true value is zero.** The gate (|mean gross Sharpe| ≤ 0.2 for the strategy *and* for
buy-and-hold, on every variant) passes on all three blocks.

The across-seed t-statistics reaching ±2.4 on individual blocks are not evidence of a real
effect: they measure whether a mean of ~0.1 Sharpe is distinguishable from zero given a
small across-seed standard error, and the blocks disagree about the sign.

### Factor attribution on the correlated null — the diagnostic 002 introduced

A Sharpe near zero is necessary but not sufficient: the rule could be a leveraged bet on
the common factor in a period when the factor happened to go nowhere. Regressing each
seed's daily return on the equal-weight basket separates the two.

| block | beta to the basket | alpha | alpha t | corr(strategy Sharpe, basket Sharpe) |
|---|---|---|---|---|
| seeds 0–15, 23 wide | 0.918 | −0.080%/yr | −0.51 | +0.880 |
| seeds 0–15, 22 wide | 0.921 | +0.339%/yr | +2.61 | +0.927 |
| seeds 100–115, 22 wide | 0.913 | −0.022%/yr | −0.18 | +0.926 |

**This is the most useful single result in the whole experiment**, because it tells you
what the rule does when there is nothing to find: it becomes a ≈0.92-beta bet on the
common factor with an alpha that is zero to within about ±0.3%/yr of sampling noise, and
whose Sharpe tracks the factor's at correlation ≈0.9.

Compare that to what the rule does on the real panel (§8): **beta 0.952 to the dollar
factor, alpha +0.42%/yr**. The real-data beta is barely distinguishable from the null's
0.913–0.921, and the real-data alpha of +0.42%/yr is the same order as the ±0.3%/yr
sampling spread of the null's own alpha. On the real data it is also not distinguishable
from zero on its own t-statistic (t = +0.50, p = 0.61).

Gate: |mean gross Sharpe| ≤ 0.2 on every variant **and** |alpha to the common factor| <
1%/yr → **PASS** on all three blocks.

---

## 6. Step 5 — backtest, turnover, cost sensitivity

One run. Annualised one-way turnover **6.14×** — the book is 5 currencies rebalanced
monthly, and about a quarter of it changes hands each month.

| bps/side | CAGR | Sharpe | vs dollar factor | max drawdown |
|---|---|---|---|---|
| 0 | −1.78% | **−0.227** | +0.204 | −47.0% |
| 2 | −1.90% | −0.245 | +0.186 | −48.3% |
| **5 (headline)** | **−2.08%** | **−0.271** | **+0.159** | **−50.0%** |
| 10 | −2.38% | −0.316 | +0.115 | −52.8% |
| 25 | −3.27% | −0.449 | −0.019 | −61.2% |

§5 calls the ladder central rather than decorative, and it is: across 0→25 bps the Sharpe
moves 0.223, and **the margin over the dollar factor — the one support clause that passes
at 5 bps — is extinguished by 25 bps.** At 25 bps the strategy no longer beats the
benchmark at all, which is itself an abandon clause. So the single surviving support
clause survives only because §5 pre-committed to 5 bps rather than to a retail FX spread.

Menkhoff et al. state that transaction costs partially explain the published spread. On
this universe and window, costs are not the reason the effect fails — it fails gross too
— but they are the reason the *relative* result would fail as well at a realistic retail
cost.

### Benchmark construction sensitivity

The headline dollar factor is the **daily equal-weighted mean of the normalised currency
returns**, fixed before any number existed (see §12). The alternatives:

| construction | Sharpe | margin vs strategy |
|---|---|---|
| **daily-rebalanced equal weight (headline)** | **−0.431** | **+0.159** |
| monthly-rebalanced equal weight | −0.440 | +0.169 |
| buy-and-hold (001–003 house convention) | −0.421 | +0.150 |

The three span 0.019 of Sharpe. §8's margin clause is decided at 0.15, and all three
constructions clear it — buy-and-hold only exactly. Nothing in the verdict turns on the
choice, which is stated here rather than discovered later. The benchmark bears no cost, as
in 001–003; that makes it harder to beat, so the margin is conservative.

### Calendar years (excess of T-bill)

The strategy beat the dollar factor in **16 of 28** calendar years. Its best years relative
to the factor are 2002 (+9.5pp) and 2003 (+7.4pp); its worst are 2017 (−5.6pp), 2009
(−5.6pp) and 2025 (−3.8pp). Only 9 of 28 years are positive in absolute terms.

### In-sample / out-of-sample and walk-forward

Nothing is fitted, so both halves are out-of-sample by construction. Split at 2012-10-04:
first half −0.016, second half **−0.628**, gap +0.612. Walk-forward (3y window, 1y
forward, 1y step; window lengths not pre-registered and unable to change a position): 24
windows, train −0.329, live −0.295, **42% of live windows positive**.

The strategy is worse in the second half than the first, and the deterioration is monotone
enough to be visible in the walk-forward table: 5 of the first 8 live windows are positive,
against 3 of the last 12.

---

## 7. Step 6 — quintile monotonicity (§8's gate)

331 rebalance intervals, all measured. Sorted on momentum known at the previous close;
forward return from the rate at the rebalance bar to the rate at the next rebalance bar —
the same fill convention the strategy gets; equal weight within each bucket; gross of
costs. Quintile split `"even"`, giving sizes **[5, 4, 5, 4, 4]**.

| bucket | mean monthly | annualised | t-stat | monthly vol | hit rate | avg names |
|---|---|---|---|---|---|---|
| **Q1** (winners) | **+0.034%** | +0.19% | +0.32 | 1.94% | 53.2% | 5.00 |
| Q2 | +0.017% | −0.03% | +0.16 | 1.99% | 49.5% | 4.00 |
| Q3 | −0.074% | −1.10% | −0.71 | 1.90% | 50.2% | 4.96 |
| Q4 | −0.089% | −1.35% | −0.75 | 2.17% | 50.2% | 4.00 |
| **Q5** (losers) | **−0.049%** | −1.00% | −0.34 | 2.63% | 46.8% | 4.00 |

Step-by-step change Q1→Q5, %/month: **[−0.0174, −0.0910, −0.0154, +0.0401]**

* **adjacent inversions: 1 of 4** (Q4→Q5). §8 tolerates at most 1 — this clause passes.
* **Q1−Q5 spread: +0.0837% per month, t = +0.623.**

§8 requires **t > 2.0** to support and abandons below **t = 1.0**. At **t = +0.623** the
abandon clause fires.

**The gradient has the right sign and is far too weak to matter.** Q1 beats Q5 by about
1.0%/yr gross. Menkhoff et al. report up to 10% p.a. for a 40+ currency long-short over
1976–2010 with institutional execution; §9 predicted materially less from a post-1999,
22-currency, long-only, retail-cost test, and materially less is what appeared. The point
estimate is consistent with a real but small effect; the t-statistic says 27 years of
monthly data cannot distinguish it from nothing.

Note also that Q5 is not the worst bucket — Q4 is — and that Q5 has by far the highest
volatility (2.63% vs Q1's 1.94%), which is the losers bucket collecting the currencies in
crisis.

### Sensitivity to the bucket convention

Under `"floor"` (002's convention, which at 22 currencies makes Q5 half again as wide as
Q1): **3 of 4 steps invert**, spread +0.0804%/month at t = +0.62. The spread and its
t-statistic are essentially unchanged; the *inversion count* is not. Under `"floor"` the
inversion clause would also have fired, so the choice of convention makes the verdict
*less* severe, not more. See §12 for why `"even"` was fixed in advance.

---

## 8. Step 7 — beta attribution

| regression | beta | alpha | alpha t | p | R² | residual vol | n |
|---|---|---|---|---|---|---|---|
| **vs the dollar factor (headline)** | **+0.952** (t +101.0) | **+0.42%/yr** | **+0.503** | 0.615 | 0.596 | 4.36% | 6,927 |
| vs SPY (secondary) | +0.050 (t +11.6) | −2.25%/yr | −1.73 | 0.084 | 0.019 | 6.81% | 6,885 |

**Against the dollar factor.** Beta 0.952 with R² 0.596: currency momentum, as specified
here, is very nearly a long position in the dollar factor. What is left over is +0.42%/yr
and indistinguishable from zero. §8's alpha clause requires t > 2.0 and gets +0.50.

Set beside §5's noise result — the same rule on driftless synthetic data produces beta
0.913–0.921 and alpha within ±0.3%/yr of zero — the honest reading is that **the factor
loading and the residual are both about what ranking noise would give you.** This is
precisely the failure mode experiment 002's factor-attribution diagnostic was built to
detect, and it detects it here.

**Against SPY.** This is the check the 002 diagnostic could not perform, because its
common factor is synthetic and contains no equity market. The answer is that currency
momentum is **not** a disguised equity beta: R² is 1.9% and beta is +0.050. The loading is
small but very precisely estimated (t = +11.6) — being long the top-quintile currencies is
mildly pro-cyclical, which is what the FX literature would predict of a book that tends to
hold high-yielding, risk-sensitive currencies. Alpha to SPY is −2.25%/yr at t = −1.73;
that is the T-bill financing drag of §1 showing up again, not an equity-adjusted skill
measure.

The SPY regression uses the 6,885-day intersection of the H.10 and NYSE calendars; the two
are close but not identical.

---

## 9. Step 8 — the interest-rate approximation diagnostic (§4)

**This is a diagnostic, not a configuration.** `configs_tried` stays at 4, and that is
enforced arithmetically rather than by intention: the **same target weights, computed from
the same spot signal**, are replayed against a different price panel. No signal is
recomputed, so there is nothing to select between.

### The arithmetic

The headline book is fully invested, so the engine's cash is zero and its excess return is
already `spot change − i_US`: a spot position financed at the US rate, earning nothing on
the foreign leg. The true currency excess return is `spot change + i_foreign − i_US`. The
two differ by exactly `i_foreign`, so holding a foreign money-market **deposit** instead of
the bare currency supplies the missing term:

```
P_total(t) = P_spot(t) * exp( sum_{u <= t} i_foreign(u) / 252 )
```

The accrual at bar *t* uses the rate known at bar *t−1*; a currency with no rate accrues
nothing and its total-return series is its spot series unchanged.

### Coverage: 16 of 22 currencies

| covered | FRED series | window mean rate |
|---|---|---|
| Brazilian Reals | IRSTCI01BRM156N | 12.97% |
| Canadian Dollars | IR3TIB01CAM156N | 2.18% |
| Chinese Yuan Renminbi | IR3TIB01CNM156N | 3.15% |
| Danish Kroner | IR3TIB01DKM156N | 1.89% |
| Indian Rupees | IRSTCI01INM156N | 6.57% |
| Japanese Yen | IR3TIB01JPM156N | 0.26% |
| South Korean Won | IR3TIB01KRM156N | 3.34% |
| Mexican Pesos | IR3TIB01MXM156N | 8.18% |
| Norwegian Kroner | IR3TIB01NOM156N | 3.22% |
| Swedish Kronor | IR3TIB01SEM156N | 1.63% |
| South African Rand | IR3TIB01ZAM156N | 7.53% |
| Swiss Francs | IR3TIB01CHM156N | 0.44% |
| Australian Dollar | IR3TIB01AUM156N | 3.80% |
| Euro | IR3TIB01EZM156N | 1.69% |
| New Zealand Dollar | IR3TIB01NZM156N | 4.11% |
| U.K. Pound Sterling | IR3TIB01GBM156N | 2.78% |

**Uncovered: Hong Kong Dollars, Malaysian Ringgit, Singapore Dollars, Sri Lankan Rupees,
Taiwan Dollars, Thai Baht.** FRED carries no short-term interest rate for any of the six —
this was established by searching the FRED API, not by trying a few identifiers and giving
up, and the rejected candidates are printed by the runner. All six are non-OECD, and the
OECD Main Economic Indicators series that supply the other 16 do not cover them.

They accrue **zero** carry, which biases the diagnostic **downward**. How far down is
bounded by how often they were actually held:

| uncovered currency | days held (of 6,927) | share of all weight allocated |
|---|---|---|
| Thai Baht | 1,931 | 5.58% |
| Hong Kong Dollars | 1,777 | 5.13% |
| Malaysian Ringgit | 1,203 | 3.47% |
| Taiwan Dollars | 1,172 | 3.38% |
| Singapore Dollars | 999 | 2.88% |
| Sri Lankan Rupees | 954 | 2.75% |
| **total** | | **23.20%** |

Nearly a quarter of the traded book earns no carry in the diagnostic. The true correction
is larger than the one below.

The US leg is reported for the differential only (DTB3, window mean **2.03%**); it is not
used to build the carry, because the engine already subtracts a T-bill rate from strategy
and benchmark alike. Note that the foreign legs are interbank/money-market rates while the
US leg is a bill rate, so the differential carries a small basis.

### The delta

| | spot-only (headline) | with foreign carry | delta |
|---|---|---|---|
| strategy Sharpe | **−0.271** | **+0.209** | **+0.480** |
| strategy CAGR | −2.08% | +1.20% | +3.28% |
| dollar factor Sharpe | −0.431 | +0.090 | +0.521 |

**The approximation is worth about half a Sharpe — more than three times §8's 0.15 margin
threshold, and larger than the entire gap between the strategy and its benchmark.** It is
the single largest number in this experiment. §4 was right to demand it be measured and
right to call the spot-only definition an approximation; what §4 could not know in advance
is that the approximation dominates the *level* of the result.

What it does **not** do is change the verdict:

* carry-adjusted Sharpe **+0.209** still fails §8's "exceeds 0.40";
* the carry-adjusted margin over the carry-adjusted dollar factor is **+0.119**, which
  fails §8's "by at least 0.15" — the margin is *narrower* with carry, not wider;
* the Q1−Q5 spread and its t-statistic are computed on the spot panel with both legs
  financed identically, and are untouched.

So the ABANDON verdict is robust to the one approximation the pre-registration flagged.
That is worth more than the headline number itself.

---

## 10. Step 9 — the five worst months, and FX-dislocation clustering

§9 imposes no mandatory month here — "currency momentum has no single canonical crash
date". It asks instead whether the worst months cluster around known FX dislocations, and
states that **the absence of any clustering is a warning sign about the implementation**.

| month | return | named dislocation |
|---|---|---|
| **2011-09** | **−9.20%** | **2011 CHF** ✓ |
| 2008-09 | −6.34% | — |
| 2007-08 | −5.93% | — |
| 2010-05 | −5.44% | — |
| 2008-08 | −5.27% | — |

Windows searched (fixed in `trendbot/config_005.py` before the run, from §9's labels):
2008 Q4 = Oct–Dec 2008; 2011 CHF = Aug–Sep 2011; 2015 CHF de-peg = Jan 2015; 2020 March =
Mar 2020; 2022 GBP = Sep–Oct 2022.

**Result: 1 of 5. §9's warning condition is not triggered** — there is clustering, and
2011-09 is the SNB's imposition of the 1.20 franc floor on 6 September 2011, which is
precisely the sort of event that destroys a long-winners currency book.

### Adjacency, reported and deliberately not counted

Four months did not match. Their distance to the nearest named window:

| month | return | distance |
|---|---|---|
| 2008-09 | −6.34% | **1 month** from 2008-10 |
| 2008-08 | −5.27% | **2 months** from 2008-10 |
| 2007-08 | −5.93% | 14 months |
| 2010-05 | −5.44% | 15 months |

§9 names "2008 Q4" and Q4 is October–December, so August and September 2008 are outside
it and are **not** counted as matches. The window was not widened to absorb them; doing so
after seeing where the months landed would be exactly the selection §8 exists to prevent.

Stated as an observation rather than as a result: all four unmatched months are
recognisable FX dislocations. 2008-08 and 2008-09 are the immediate run-up to Lehman —
the global financial crisis' FX phase began a quarter before the quarter §9 names.
2007-08 is the quant crisis and the first big yen-carry unwind. 2010-05 is the euro
sovereign-debt crisis month. **Every one of the five worst months is a currency
dislocation**, which is what the check was really asking, and the implementation is not
suspect on §9's terms.

---

## 11. Data integrity — two bad prints the ERM II check caught

Not a configuration and not a cleaning step. The headline runs on the data exactly as FRED
publishes it, because §§2–6 are frozen and authorise no outlier rule.

Step 1's Danish krone cross-rate check found **two days on which the implied DKK/EUR rate
sits outside the ERM II ±2.25% band**:

| date | DEXDNUS (DKK per USD) | implied DKK per EUR | band |
|---|---|---|---|
| 2000-10-30 | 8.630 | 7.2768 | [7.2925, 7.6282] |
| 2000-11-28 | 8.278 | **7.0752** | [7.2925, 7.6282] |

Neighbouring days sit at 7.444, 7.442, 7.457, 7.460. Denmark did not breach its ERM II
band on two isolated days in late 2000 and return to it the next morning; these are bad
prints in `DEXDNUS`, and the second creates a spurious +5.8% one-day appreciation followed
by a −4.9% reversal. No threshold was fitted to find them — a policy band is a commitment,
not a percentile.

**Sensitivity, with those two observations held flat:** Sharpe −0.2714 → −0.2711, delta
**+0.0002**. Immaterial. The headline remains the data as published; this is disclosure,
not selection.

This is the clearest argument for the cross-rate check earning its place: it is the only
check here that can find a wrong *value* rather than a wrong *direction*.

---

## 12. Step 10 — PSR and deflated Sharpe at `configs_tried = 4`

The counter is **4, not 5**. PREREG_005.md's header argues that blocked experiment 004
does not advance it: an IP-level vendor throttle stopped it before it reached data, so it
produced nothing from which a winner could have been selected. The config parser asserts
that argument is still in the document — a future edit that bumps the number without the
reasoning, or strips the reasoning while keeping the number, fails to parse
(`tests/test_config_005.py`).

Per-period Sharpes of the four configurations tried, all recorded outputs, nothing
estimated:

| configuration | annualised net Sharpe | per-period |
|---|---|---|
| 001 time-series trend, 12 ETFs | 0.35201 | 0.022175 |
| 002 cross-sectional momentum, 41 ETFs | 0.46310 | 0.029173 |
| 003 cross-sectional momentum, 409 equities | 0.69880 | 0.044020 |
| **005 currency momentum, 22 FX** | **−0.27140** | **−0.017094** |

* **PSR(0) = 0.0766** — does **not** clear 95%. Read directly: given the observed Sharpe,
  skew (−0.48) and kurtosis over 6,927 observations, the probability that the true Sharpe
  exceeds zero is **7.7%**.
* **Deflated Sharpe at 4 configurations = 0.0001** — not significant at 95%.

§9 warned that "the PSR correction is now materially larger" and that "a raw Sharpe that
would have cleared in 002 may not clear here". That warning is untested by this result:
the Sharpe is negative, so PSR(0) fails long before the deflation term is reached. The
deflation is not what killed this experiment.

---

## 13. Step 11 — the verdict

**ABANDON**, per §8, on two independently sufficient clauses:

1. **Q1−Q5 t-statistic below 1.0** — observed **+0.623**.
2. **net Sharpe below 0.15** — observed **−0.271**.

Three of four support clauses fail. The one that passes — beating the dollar factor by at
least 0.15 — passes at 5 bps and fails at 25 bps.

### What this adds to the accumulated question

Across four cross-sectional experiments the signal has been held constant while the
universe changed, so the results answer one coherent question: *where, if anywhere, does
`P(t−21)/P(t−252) − 1`, sorted cross-sectionally, long the top bucket, work?*

| experiment | universe | verdict | why |
|---|---|---|---|
| 002 | 41 ETFs | ABANDON | quintile monotonicity failed; the universe collapsed to one market factor |
| 003 | 409 survivor equities | ABANDON | decile gradient inverted; survivorship made the universe unable to answer the question |
| 004 | point-in-time equities | *pending* | vendor throttle; never reached data |
| **005** | **22 H.10 currencies** | **ABANDON** | **gradient correct in sign, far too weak to distinguish from zero; the book is a 0.95-beta dollar bet** |

005 was chosen because it fixes 003's contamination structurally — currencies do not
delist — and because §1 expected low cross-correlation to avoid 002's single-factor
collapse. The first worked: there is no survivorship bias here and the universe *can*
answer the question. The second **did not**. The dollar is as dominant a factor for
currencies as the market was for the 41 ETFs: R² of 0.596 and beta 0.952, against noise
values of ≈0.92. 002 failed because 41 ETFs collapsed into one market factor; 005 fails in
substantially the same way, with the dollar in place of the market.

The result is not surprising on the pre-registration's own terms. §1 recorded in advance
that Menkhoff et al. "conclude that limits to arbitrage prevent these returns from being
easily exploitable" and that "a null result is consistent with the literature and is not a
surprising outcome". The measured Q1−Q5 spread of ≈1.0%/yr gross, against a published
10%/yr for a 40+ currency long-short with institutional execution over 1976–2010, is
exactly the "materially less" §9 predicted.

---

## 14. Everything I decided that the document does not fix

Every one of these was fixed **before** the number it affects existed.

| decision | choice | why, and what turns on it |
|---|---|---|
| **dollar factor construction** | daily equal-weighted mean of normalised currency returns | §5 names it "the dollar factor", which is the DOL of Lustig/Roussanov/Verdelhan and Menkhoff et al. Being path-independent, §5's benchmark and §8's alpha regressor are literally the same object. All three candidate constructions span 0.019 of Sharpe and all three clear §8's 0.15 margin; nothing turns on it. **Both alternatives reported (§6).** |
| **quintile split** | `"even"` (sizes 5,4,5,4,4) | §4 names no bucket size and §8's gate is the Q1−Q5 spread, so the two gated ends are kept the same size. This is the rule `trendbot/xsmom.py` already documents and experiment 003 established. `"floor"` reported as a sensitivity, and it is **more** severe (3 inversions vs 1), so this choice does not flatter the result. |
| **open vs close** | `open = close = the single published rate` | Forced by the data: FRED publishes one H.10 rate per day. The decision/fill separation is the engine's single execution shift, which is §5 verbatim. |
| **missing-value handling** | forward fill on the shared calendar; never back-fill | Delegated explicitly by the build order. Affects 0.0177% of cells. Demonstrated lookahead-free by tampering with the future and recomputing (§4). |
| **trading calendar** | dates on which ≥1 H.10 series published | Distinguishes a US market holiday (not a bar) from a foreign holiday (a bar, forward-filled). Admits 2005-09-05, reported in §4. |
| **gap allowance** | 10 scheduled publication days | Two calendar weeks, longer than Chinese New Year. Fixed before the data was seen; never bound (largest actual gap: 1 day). |
| **short-rate sources** | 3-month interbank, else immediate/call, else policy rate; FRED only | §4 says "from FRED" and names no series. Priority order fixed in advance; every source and every uncovered currency reported (§9). |
| **noise panel width / vol** | 22 columns, 10%/yr | The real universe's width and FX's actual volatility. Reporting choices; three seed blocks reported because two disagreed in sign. |
| **§9 dislocation windows** | 2008 Q4 = Oct–Dec 2008; 2011 CHF = Aug–Sep 2011; 2015 = Jan 2015; 2020 = Mar 2020; 2022 GBP = Sep–Oct 2022 | §9 gives labels, not months. Fixed in `config_005.py` before the run. Taken literally afterwards: windows were **not** widened to absorb Aug/Sep 2008 (§10). |
| **risk-free rate** | `^IRX`, the 13-week T-bill, as in 001–003 | §5 says "the T-bill rate". Same series, same convention, same treatment of strategy and benchmark as the previous three experiments. |
| **max drawdown convention** | of the excess-return stream | Matches the Sharpe convention. The raw-return drawdown is −38.8%; the excess-return one is −50.0%. Both computed; the excess one reported, for consistency with everything else. |

---

## 15. What I built that I do not think works as intended

Stated plainly, as the build order requires.

1. **The interest-rate diagnostic is missing 23.2% of the traded book.** FRED carries no
   short rate for Hong Kong, Malaysia, Singapore, Sri Lanka, Taiwan or Thailand, and the
   six together account for 23.2% of all weight the strategy ever allocated. Those
   currencies accrue zero carry, so the reported +0.480 Sharpe delta is a **lower bound**
   of unknown tightness. This is the largest known defect in the experiment, and it sits
   underneath its largest number.

2. **The foreign and US legs of the differential are not the same instrument.** The
   foreign legs are 3-month interbank rates (or call money, for Brazil and India); the US
   leg the engine actually charges is `^IRX`, a T-bill discount rate. Interbank minus bill
   is not zero — it is roughly the TED spread, and it blows out in exactly the crises that
   dominate the worst months. The diagnostic therefore slightly overstates the carry in
   calm periods and misstates it in stressed ones. Quantifying that would need a
   consistent set of either bill or interbank rates for 22 economies, which FRED does not
   have.

3. **Two foreign short rates are monthly call-money rates, not term rates.** Brazil
   (IRSTCI01BRM156N) and India (IRSTCI01INM156N) fall through to the immediate-rate
   fallback. Brazil is also the highest-rate currency in the panel at 12.97% and the
   second-most-held. An overnight rate is a defensible proxy for a one-month carry but it
   is not the same thing.

4. **`DEXDNUS` contains two values that are certainly wrong and they were left in.**
   2000-10-30 and 2000-11-28 breach the ERM II band (§11). Correct behaviour under a
   frozen pre-registration, and the sensitivity is +0.0002 of Sharpe, but the headline
   number is computed on data I know contains two errors.

5. **The 2005-09-05 bar is an artefact.** One currency published, 21 were forward-filled
   across a US public holiday. The pre-declared calendar rule admits it. It is one bar in
   6,927 and cannot move any reported figure, but it is not a real trading day.

6. **The noise test's across-seed t-statistics are not trustworthy at face value.**
   Individual 16-seed blocks produced |t| up to 2.8 on a mean Sharpe of ~0.1, and the sign
   flipped between blocks. The *gate* (|Sharpe| ≤ 0.2) is robust; the t-statistics printed
   beside it are not, and should not be read as evidence of a real effect in either
   direction. I report three blocks for exactly this reason.

7. **`fetch_short_rates` caches a negative result.** Once FRED has been observed not to
   carry a candidate, that outcome is cached in
   `data/cache/fred_short_rate_resolution.json` and not re-probed. If FRED later adds a
   Thai or Singaporean short rate, the diagnostic will not notice until the cache is
   cleared or `--refresh` is passed. This was a deliberate trade against re-probing a
   courtesy service on every run, but it is a staleness bug waiting to happen.

8. **The web-scraping path in `trendbot/fred.py` is fragile by construction.** It parses
   the FRED series page's `og:title` and `Units:` block with regular expressions. It is
   currently verified byte-identical to the API path across all 23 series, and it fails
   loudly rather than silently if the page changes — but a FRED redesign breaks it, and it
   only exists for the case where no `FRED_KEY` is configured. Now that a key is present,
   the API path is what runs.

9. **I triggered FRED's IP-level throttle during this build.** Probing ~140 candidate
   series identifiers concurrently earned a 403 across *both* `fred.stlouisfed.org` and
   `api.stlouisfed.org` for several minutes — the same class of failure that blocked
   experiment 004. The client now paces requests at 1.5s and treats 403 as retryable, and
   all data is cached, but the incident is worth recording: the courtesy budget on free
   public data is smaller than it looks.

10. **The strategy trades from 1971 in the artefact, and nothing reports it.** The panel
    carries full history so the formation window can reach behind 1999. The engine
    therefore runs a backtest from 1971 whose results are never measured, reported, or
    able to affect any figure here (every statistic is sliced from 1999-01-01, including
    turnover). It is wasted computation and a trap for anyone who reads
    `result.equity` directly instead of `result.stats_from(start)`.

11. **Not a defect, but the most likely thing to mislead a reader:** the headline Sharpe of
    −0.271 is *mostly a definition*, not a finding. §4's spot-only return plus §5's
    subtraction of the T-bill produces a book that is charged US financing and earns no
    foreign interest. §9's diagnostic says that is worth +0.48 of Sharpe. Anyone quoting
    "currency momentum had a Sharpe of −0.27" without that sentence attached is quoting a
    construction, not a market fact.

---

## 16. Reproduction

```
pytest -q                                                  # 759 passed
python scripts/run_backtest.py --experiment 001 --regression   # bit-identical, PASS
python scripts/run_backtest.py --experiment 005 --noise        # step 4, both variants
python scripts/run_backtest.py --experiment 005 --validate     # the whole §7 protocol
```

`results/backtest_005.json` carries the machine-readable artefact: the universe, the quote
convention determined for each series, the exclusions, the cost ladder, the quintile table,
both regressions, the interest-rate diagnostic, the worst months, PSR and the verdict.

New modules: `trendbot/fred.py` (FRED client, both transports), `trendbot/fx.py` (quote
normalisation, universe rule, verification checks), `trendbot/config_005.py` (PREREG_005
parser), `trendbot/engine/fx_validation.py` (§8's rule, §9's clustering, §4's carry
arithmetic), `trendbot/strategies.py::CurrencyCrossSectionalMomentum`,
`scripts/run_experiment_005.py`.

The engine, panel protocol, metrics and validation modules from 001–004 were **extended,
never rewritten**: no file belonging to an earlier experiment was modified except
`scripts/run_backtest.py` (one dispatch branch) and `trendbot/strategies.py` (one appended
class).
