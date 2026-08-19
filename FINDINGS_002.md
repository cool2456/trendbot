# Findings 002 — cross-sectional momentum (41 ETFs)

Implementation of `PREREG_002.md` (sha256 `e8432ca4ccbf…`, committed `c82bbe3`, tagged
`prereg-002`, unmodified since — asserted by a test against git HEAD). No parameter in
that document was changed, tuned, or substituted. **Configurations tried on this
dataset, cumulative: 2.**

Experiment 001's documents and results are untouched. `PREREGISTRATION.md` is
byte-identical to its committed version (commit `dccdea4`) and two tests fail if it
is modified. `FINDINGS.md` and the experiment 001 source tree are **not tracked by
git in this repository** — only the four markdown documents are — so no test can
prove `FINDINGS.md` is unedited, and I am not claiming one does. What is true is that
nothing in this experiment writes to it, its modification time predates this work, and
experiment 002's results were written to `FINDINGS_002.md` and `results/backtest_002.json`.

---

## 1. The headline result

Window **2008-01-01 → 2026-08-18** (18.6 years, 4,686 bars) — section 6's
pre-committed sample window, fixed before any result existed. Net of 5 bps per side.
Sharpe is in excess of the 13-week T-bill; the benchmark is treated identically.

| | strategy | equal-weight buy-and-hold |
|---|---|---|
| CAGR (excess of T-bill) | 6.56% | 6.59% |
| volatility | 16.78% | 15.64% |
| **Sharpe** | **0.463** | **0.487** |
| Sortino | 0.63 | 0.68 |
| max drawdown | −37.0% | −45.8% |
| Calmar | 0.177 | 0.144 |
| annual turnover | **6.12×** | 0 |
| average gross exposure | 100% | 100% |
| skew / excess kurtosis | −0.59 / 5.65 | −0.31 / 11.92 |

### Section 8's pre-committed verdict: **ABANDON**

| support clause (all three required) | result |
|---|---|
| net Sharpe exceeds 0.40 | ✓ 0.463 |
| exceeds buy-and-hold by ≥ 0.15 | ✗ margin **−0.023** |
| quintile monotonicity holds | ✗ **ordering is not monotonic** |

| abandon clause (any one is sufficient) | result |
|---|---|
| **fails to beat equal-weight buy-and-hold at all** | **✗ triggered — 0.463 vs 0.487** |
| **quintile ordering is non-monotonic** | **✗ triggered — 2 of 4 steps invert** |
| net Sharpe below 0.15 | not triggered — 0.463 |

Two of the three abandonment clauses fire independently. The verdict is **ABANDON**.

---

## 2. Step 1 — universe verification

All 41 tickers have price history beginning well before section 2's required date of
2008-01-01. **None was dropped. The final count is 41**, so section 3's top quintile
is **8 of 41**, which is what section 3 declares.

| sleeve | ticker and first available bar |
|---|---|
| US sectors | XLB 1998-12-22 · XLE 1998-12-22 · XLF 1998-12-22 · XLI 1998-12-22 · XLK 1998-12-22 · XLP 1998-12-22 · XLU 1998-12-22 · XLV 1998-12-22 · XLY 1998-12-22 |
| Country / region | EWJ 1996-03-18 · EWG 1996-03-18 · EWU 1996-03-18 · EWC 1996-03-18 · EWA 1996-03-18 · EWY 2000-05-12 · EWT 2000-06-23 · EWZ 2000-07-14 · EWH 1996-03-18 · EWS 1996-03-18 · EWW 1996-03-18 · EWL 1996-03-18 |
| Broad equity | SPY 1993-01-29 · QQQ 1999-03-10 · IWM 2000-05-26 · DIA 1998-01-20 · EFA 2001-08-27 · EEM 2003-04-14 · IWD 2000-05-26 · IWF 2000-05-26 |
| Real assets | VNQ 2004-09-29 |
| Rates / credit | TLT 2002-07-30 · IEF 2002-07-30 · SHY 2002-07-30 · LQD 2002-07-30 · HYG 2007-04-11 · TIP 2003-12-05 |
| Commodities | GLD 2004-11-18 · SLV 2006-04-28 · DBC 2006-02-06 · USO 2006-04-10 · DBA 2007-01-05 |

Latest inception in the universe is **HYG, 2007-04-11**, comfortably inside the
requirement. Source is Yahoo, split- and dividend-adjusted.

**One reading of "continuous" that I decided against, stated so it can be overruled.**
Thirteen tickers are missing at least one *bar* inside the window:

| ticker | XLB | XLI | EWG | EWU | EWC | EWT | EWH | EWW | EFA | IWD | IWF | LQD | DBC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missing bars | 2 | 3 | 3 | 3 | 3 | 1 | 3 | 3 | 2 | 2 | 3 | 3 | 3 |

Every one of those gaps falls on **three dates — 2026-07-21, 2026-07-22 and
2026-07-31** — on which the vendor published nothing for a batch of ten to twelve
symbols at once. That is a feed artefact, not a discontinuity in the instrument, so I
read "continuous data from 2008-01-01" as *continuously listed* and dropped nobody. A
literal reading of "continuous" would drop 13 of 41 tickers on 1–3 missing bars in
4,686 and destroy the experiment. The counts are printed above so anyone who prefers
the strict reading can apply it to a number rather than to a description. Those three
dates are late enough in the sample that no momentum value inside the window reads
them; they affect only position marks, which the engine forward-fills.

**Survivorship bias, per section 2.** The universe is selected from ETFs that exist
today, so it excludes funds that launched before 2008 and have since closed. This
biases the result upward by an unknown amount. It is smaller than for an equity
universe but it is not zero, and it is not corrected for anywhere below.

---

## 3. The interface problem, and the gate that had to pass first

Experiment 001's strategy protocol is per-instrument: `DataFrame -> Series` of target
position. A cross-sectional rule cannot be written that way, because instrument *i*'s
position on date *t* depends on where *i* ranks against every other instrument on that
date, and no amount of per-instrument information determines it.

The protocol was widened by exactly one step, to `dict[str, DataFrame] -> DataFrame`
of target weights, indexed by date and columned by symbol (`trendbot/engine/panel.py`).
The returned frame means *target weights as of the close of each row's own bar, before
execution*: a strategy never lags its own output, and the single execution shift stays
where experiment 001 put it — `lag_for_execution`, called once, by the engine. The
frame must also be a **pure function of the panel**; everything path-dependent (the
drift band, the re-application of the exposure caps to weights that have drifted, the
cost charged on the trade actually done) stays in the engine, exactly as before.

That split is what made the regression possible: experiment 001's sizing pipeline up
to and including both caps was already pure, so it lifted into the new protocol
unchanged.

### The gate

`python scripts/run_backtest.py --experiment 001 --regression`

```
  [PASS] window matches the one experiment 001 reported                 2008-02-29 -> 2026-08-18
  [PASS] generalised engine net Sharpe == 0.352 to 3 dp                 0.3520103425
  [PASS] benchmark Sharpe == 0.414 to 3 dp                              0.4137933300
  [PASS] original engine still produces its own recorded number         0.3520103425
  [PASS] benchmark is literally the same function in both paths         0.413793330034 vs 0.413793330034
  [PASS] equity curves agree bit for bit (max relative delta == 0)      0.000e+00
  [PASS] weights identical                                              max |delta| 0.000e+00
  [PASS] targets identical                                              max |delta| 0.000e+00
  [PASS] trades identical                                               max |delta| 0.000e+00
```

The gate asks for three decimal places. What it got was **exact equality** — the two
engines produce bit-identical equity curves, weights, targets and trades on the real
twelve-ETF history. This is committed as a permanent regression in
`tests/test_experiment_001_regression.py`, which repeats the comparison offline on
synthetic panels across seeds, across the whole cost ladder, with a time-varying cash
rate, and for the long-short and diagonal-covariance readings as well, so the
equivalence is not a coincidence of one configuration.

The recorded numbers live in `trendbot/regression.py` as frozen constants and are
labelled there as outputs, not parameters. If that gate ever fails, editing them is
not the fix.

---

## 4. Step 2 — the signal, and its fixture gate

```
momentum_i(t) = P_i(t-21) / P_i(t-252) - 1
```

Ranked cross-sectionally at each rebalance; top quintile equal-weighted long; all
others zero. Instruments without a defined momentum are **excluded from that date's
ranking**, not assigned zero momentum.

`tests/test_xsmom.py` checks the rule against a six-name fixture whose ranks are
arithmetic done by hand. The decisive case is the exclusion rule: in that fixture every
real name's momentum is negative, so a name silently given momentum 0 would rank
*first* of seven and be bought at half the book. The test asserts both that this does
not happen and that it *would* have happened under the reading the document rules out.

**Bucket sizing.** 41 does not divide by 5. Section 3 says "Top quintile (highest ~8 of
41)" and `41 // 5 == 8`, so every bucket but the last holds 8 and the leftover name
falls into the **bottom** bucket: sizes **8, 8, 8, 8, 9**. This is the only partition of
all 41 in which the top bucket is exactly the 8 that section 3 names. The same
convention is used for the traded position (section 3) and the five-bucket gate
(section 8), so **the traded set and Q1 are the same set by construction**, and that
identity is verified numerically in section 6 below. This was the one genuine ambiguity
in the document and it was resolved by asking rather than by choosing.

Inside the window the book holds **8 names on 4,665 days and 7 on 21 days**. The 21 are
in January and February 2008, when HYG (listed 2007-04-11) and DBA (2007-01-05) did not
yet have a 252-day formation window, so 39 instruments were ranked and `39 // 5 == 7`.
That is section 3's exclusion rule operating, not a defect.

---

## 5. Step 3 — noise tests, both variants

16 seeds × 6,000 days each. Sharpe is reported **gross of costs as well as net**,
because this rule replaces most of its book every month; a net figure on driftless data
is expected to sit below zero by the cost drag, and testing only the net number would
conflate "the rule finds nothing" with "the costs are large". Only the first is what
section 7 step 1 asks about.

| variant | gross Sharpe (mean) | sd | t | median | min | max | net (5 bps) | buy-and-hold |
|---|---|---|---|---|---|---|---|---|
| (a) independent random walks | **−0.042** | 0.210 | −0.81 | −0.005 | −0.387 | +0.403 | −0.097 | −0.020 |
| (b) common factor, 25% of variance | **+0.049** | 0.195 | +0.99 | +0.043 | −0.242 | +0.395 | +0.015 | +0.075 |
| (b) common factor, 50% of variance | **+0.070** | 0.178 | +1.58 | +0.079 | −0.209 | +0.375 | +0.044 | +0.078 |

Per-seed gross Sharpe, variant (a):
`[-0.232, -0.102, -0.379, -0.024, 0.018, 0.144, -0.387, -0.054, -0.140, 0.050, 0.032, 0.014, -0.289, 0.106, 0.164, 0.403]`

Per-seed gross Sharpe, variant (b) at 50%:
`[-0.109, -0.122, 0.025, 0.085, 0.225, 0.375, 0.219, -0.177, -0.126, 0.072, -0.209, 0.149, 0.035, 0.187, 0.199, 0.297]`

Both are indistinguishable from zero. Annual turnover on noise is 5.9–6.1×, the same
order as on the real data.

### The factor-attribution half of variant (b)

A Sharpe near zero is necessary but not sufficient. The rule could be a leveraged bet
on the common factor, measured in a period when the factor happened to go nowhere.
Regressing each seed's daily strategy return on the equal-weight basket separates the
two — beta is how much of the book *is* the factor, alpha is what is left:

> beta to the equal-weight basket **0.913**, alpha **−0.025%/yr** (t −0.12), cross-seed
> correlation between the rule's Sharpe and the basket's **+0.922**

So on correlated data the rule **is** overwhelmingly a factor tilt — which is exactly
the failure mode variant (a) cannot see, because independent walks have no factor to
tilt onto — and **the tilt earns nothing**, because the factor is driftless. Both halves
are needed; either alone would have been consistent with a broken engine.

The generator's beta spread (0.6 to 1.4, linear across the universe) and the share of
variance carried by the factor are reporting choices, not pre-registered parameters,
which is why the test is run at more than one loading. Idiosyncratic vol is set per
instrument so that **every instrument has the same total volatility**; without that, the
betas would move total vol too and a cross-sectional rank would be sorting partly on
volatility, making any result ambiguous.

**Gate: PASS** — |mean gross Sharpe| ≤ 0.2 for the strategy and for buy-and-hold on
every variant, and |alpha to the common factor| < 1%/yr.

---

## 6. Step 4 — quintile monotonicity (section 8's pass/fail gate)

All instruments are sorted into five quintiles at each rebalance on the momentum known
at the **previous close** — the same information the execution lag allows the strategy
to trade on — and each quintile's forward return runs from the **open of the rebalance
bar to the open of the next rebalance bar**, which is the fill convention section 5
gives the strategy. Equal weight within each quintile, gross of costs. 223 rebalances.

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| mean forward return, per month | **0.7490%** | **0.7501%** | 0.6944% | 0.5227% | **0.6632%** |
| annualised | 8.10% | 8.22% | 7.35% | 5.10% | 6.52% |
| t-statistic | 2.54 | 2.66 | 2.33 | 1.70 | 1.91 |
| monthly volatility | 4.41% | 4.20% | 4.45% | 4.59% | 5.19% |
| hit rate | 64.6% | 66.4% | 62.3% | 60.1% | 60.5% |
| average names | 8.0 | 8.0 | 8.0 | 8.0 | 9.0 |

Step-by-step change in mean forward return, Q1→Q5, in %/month:
**+0.0010**, −0.0556, −0.1717, **+0.1405**

### Verdict on the gate: **the ordering is NOT monotonic.**

**2 of the 4 steps invert.** Q2 out-earns Q1, and Q5 out-earns Q4. The gradient the
hypothesis predicts is not there.

- **Q1 − Q5 spread: +0.0859% per month (+1.04% annualised), t = +0.260, n = 223.**
  Positive, and statistically indistinguishable from zero. A t of 0.26 on 223
  independent monthly observations is not weak evidence for the hypothesis; it is an
  absence of evidence.
- Q1 − Q4, the widest gap involving Q1, is +0.226%/month at t = 0.85. Also not
  significant. (Q2 − Q4 is fractionally wider still, at +0.227%/month.)

This is not sensitive to where the window starts. Recomputed from 2008-04-10 (the first
date all 41 instruments have a formation window) the ordering still inverts once with a
spread t of +0.30; recomputed from 2009-01-02 it inverts twice with a spread t of +0.07.

**Cross-check that this measures the strategy that was actually run.** Section 3's
traded portfolio *is* Q1, so at zero cost the book's own equity measured
open-of-rebalance to open-of-next-rebalance must equal Q1's series exactly. Over the
same 223 months the **maximum absolute difference is 2.9×10⁻¹⁶**. Q1 is the traded
portfolio, not a proxy for it, so the failure of monotonicity is a statement about the
strategy and not about the diagnostic.

---

## 7. Step 5 — full-sample backtest, turnover, cost sensitivity

One run. Annualised one-way turnover is **6.12×**, against experiment 001's 3.23× —
section 5 predicted that costs would bite harder here, and they do, though not enough
to matter to the verdict. The average name survives **4.15 months** in the portfolio;
1.93 of the 8 holdings change at a typical rebalance.

| bps/side | 0 | 2 | **5** | 10 | 20 |
|---|---|---|---|---|---|
| Sharpe | 0.481 | 0.474 | **0.463** | 0.445 | 0.408 |
| CAGR (excess) | 6.89% | 6.76% | **6.56%** | 6.24% | 5.59% |
| vs buy-and-hold | −0.005 | −0.013 | **−0.023** | −0.042 | −0.078 |

**The strategy does not clear its thresholds only at 0 bps — it fails at 0 bps too.**
At literally zero transaction cost it still loses to buy-and-hold by 0.005 and still
falls 0.155 short of the required +0.15 margin, and the monotonicity clause fails
regardless of costs. Costs are not what sinks this.

Gross of costs the Sharpe is 0.481; the whole cost drag is worth 0.018 of Sharpe.

### Benchmark construction sensitivity

Section 8 says "equal-weight buy-and-hold" and does not say how often, if ever, it is
rebalanced. Read literally that means never, which is the headline and is the same
construction experiment 001 used.

| construction | Sharpe | CAGR | margin | strategy beats it? |
|---|---|---|---|---|
| **buy once, hold (literal — headline)** | **0.487** | 6.59% | **−0.023** | **no** |
| rebalanced monthly | 0.447 | 5.98% | +0.017 | yes |
| rebalanced daily | 0.460 | 6.28% | +0.003 | yes |

The strategy beats the two rebalanced constructions by 0.003–0.017 and loses to the
literal one by 0.023. Under **none** of the three does it clear the +0.15 margin the
support clause requires, and under none of the three does the monotonicity clause pass,
so the verdict is ABANDON on every reading of the benchmark. Plain SPY over the same
window earned an excess Sharpe of **0.573**, ahead of both.

### The pre-committed window, and what it costs

Section 6 fixes the sample at 2008-01-01, and section 2 requires history beginning on
or before that date, so the formation window at the start of the sample reaches back
into 2007 and the book is fully invested from the first bar of 2008.

| variant | first traded | Sharpe | benchmark | margin |
|---|---|---|---|---|
| pre-window history used to form the signal (**headline**) | 2008-01-02 | 0.463 | 0.487 | −0.023 |
| cold start: no data before 2008-01-01 at all | 2008-12-31 | 0.596 | 0.487 | +0.109 |

The cold start sits in cash through the whole of 2008, which is why it looks better by
0.13 of Sharpe. Reporting it as the headline would be choosing a window after seeing
the result. It still fails the +0.15 margin and it still fails monotonicity.

### Calendar years (total return)

| year | strategy | benchmark | SPY |
|---|---|---|---|
| 2008 | −27.3% | −31.0% | −36.8% |
| 2009 | +14.6% | +27.3% | +26.4% |
| 2010 | +19.0% | +17.9% | +15.1% |
| 2011 | −11.9% | −1.9% | +1.9% |
| 2012 | +12.1% | +13.2% | +16.0% |
| 2013 | +24.1% | +10.8% | +32.3% |
| 2014 | +7.1% | +5.2% | +13.5% |
| 2015 | +0.2% | −3.4% | +1.2% |
| 2016 | −2.7% | +9.0% | +12.0% |
| 2017 | +16.1% | +18.9% | +21.7% |
| 2018 | −14.6% | −6.5% | −4.6% |
| 2019 | +22.8% | +24.2% | +31.2% |
| 2020 | +20.3% | +16.0% | +18.3% |
| 2021 | +18.7% | +18.4% | +28.7% |
| 2022 | +1.7% | −16.5% | −18.2% |
| 2023 | +7.6% | +19.8% | +26.2% |
| 2024 | +12.8% | +13.9% | +24.9% |
| 2025 | +34.6% | +21.1% | +17.7% |
| 2026 | +15.1% | +14.6% | +13.1% |

The strategy beat the benchmark in 10 of 19 calendar years and lost on Sharpe anyway:
it wins more often and by less, loses less often and by more, and carries 1.1 points
more volatility.

### Portfolio composition

Average weight by sleeve over the window: US sectors 30.6%, country/region 23.7%, broad
equity 16.7%, commodities 16.0%, rates/credit 9.7%, real assets 3.3%. Most-held names
are XLK (in the book on 42.9% of rebalances), QQQ (42.4%), GLD (36.2%), SLV (35.3%),
IWF (34.4%); least-held are EWU (1.3%), EFA (4.0%), HYG (4.5%), IWD (4.5%), SHY (7.6%).

### Section 7 steps 5 — walk-forward and split

Window lengths are not pre-registered and cannot change a position; they are reporting
choices, stated here: 3 years in-window, 1 year forward, stepping 1 year.

- 50/50 split at 2017-04-24: first half **+0.236**, second half **+0.677**, gap −0.440.
- Walk-forward, 15 windows: mean in-window **+0.489**, mean forward **+0.761**, gap
  −0.272, **73%** of forward windows positive. The worst forward window is 2011
  (−0.507); the best is 2019 (+2.148).

Nothing is fitted, so neither half is privileged and the asymmetry is a statement about
the periods, not about overfitting. The benchmark splits the same way over the same two
halves — **+0.336** then **+0.649** — so the asymmetry is a property of the window, not
of the strategy. The strategy trails the benchmark by 0.100 in the first half and leads it by
0.028 in the second; the whole of the full-sample shortfall comes from the earlier
period.

---

## 8. Step 6 — behavioural validation

Section 9 names **March 2009** and **April 2020** in advance as the canonical momentum
crashes and predicts at least one drawdown above 25%.

The five worst months by absolute net return:

| month | strategy | benchmark | relative |
|---|---|---|---|
| 2011-09 | −14.55% | −9.85% | −4.70% |
| 2008-10 | −12.48% | −17.81% | +5.33% |
| 2008-09 | −10.81% | −9.48% | −1.33% |
| 2026-03 | −9.66% | −6.06% | −3.60% |
| 2010-05 | −9.33% | −6.53% | −2.80% |

### 🚩 FLAG: the prediction did not come true

**Neither March 2009 nor April 2020 is among the five worst months.** Where they
actually rank, out of 224 months:

| month | strategy | benchmark | relative | rank by absolute | rank by relative |
|---|---|---|---|---|---|
| 2009-03 | **+2.26%** | +6.79% | −4.53% | 141 / 224 | 12 / 224 |
| 2020-04 | **+9.08%** | +9.39% | −0.31% | **217 / 224** | 96 / 224 |

April 2020 was the **eighth-best** month of the 224 in the sample. March 2009 was
mid-pack. Section 9 predicted both by name and both predictions failed. I am recording
that as a failed prediction rather than resolving it, and it is the item I would look at
first if this experiment were ever reopened.

What is true, and is a different claim from the one section 9 made: the strategy's worst
months *relative to the benchmark* do cluster around the March 2009 reversal —
**April 2009 is the single worst relative month in the sample at −8.31%**, May 2009 is
third at −5.98%, and March 2009 itself is twelfth. The five worst relative months:

| month | strategy | benchmark | relative |
|---|---|---|---|
| 2009-04 | −0.57% | +7.74% | **−8.31%** |
| 2026-06 | −6.78% | −0.46% | −6.32% |
| 2009-05 | +2.50% | +8.48% | −5.98% |
| 2016-11 | −5.60% | +0.09% | −5.69% |
| 2022-11 | +1.53% | +6.93% | −5.40% |

I am not claiming this rescues section 9's prediction. Section 9 said the strategy
"loses badly" in those months, and in absolute terms it did not.

**Drawdowns.** Maximum drawdown **−37.0%**, trough 2008-11-20 from a peak on
2008-07-01, recovered 2011-04-20. Longest underwater run 706 trading days (2.8 years),
94.1% of days spent below a prior peak. Section 9's expectation of at least one
drawdown above 25% is met.

---

## 9. Step 7 — PSR and deflated Sharpe at `configs_tried = 2`

The deflation term needs the variance of the per-period Sharpe ratios across the
configurations actually tried. There are exactly two and both are known, so nothing is
estimated: experiment 001's recorded per-period Sharpe of **0.022175** and this one's
**0.029173**.

| quantity | value |
|---|---|
| annualised Sharpe | +0.463 |
| observations | 4,686 |
| skewness / kurtosis | −0.586 / 8.65 |
| configurations tried | **2** |
| benchmark SR\* = E[max SR] over 2 trials | 0.002572 (per period) |
| **PSR(0)** | **0.9761** |
| deflated Sharpe at 2 configurations | 0.9644 |

**PSR(0) = 0.9761 clears 95%.** So does the deflated Sharpe at 0.9644.

Section 9 warned that PSR might again fail to clear 95%. It did not fail. That result
does not change the verdict and does not license anything: PSR asks whether a Sharpe of
0.463 measured over 18.6 years is distinguishable from zero, and the answer is yes. It
does not ask whether the strategy beats the benchmark, and it does not ask whether the
quintile gradient exists. Both of those questions were asked in section 8 and both were
answered no. A strategy can be reliably non-zero and still be worse than doing nothing.

---

## 10. Step 8 — the verdict

**ABANDON.**

The hypothesis was that instruments outperforming their peers over the past twelve
months continue to outperform them over the following month. On this universe over this
window, measured by the criteria fixed in advance:

- The strategy **does not beat equal-weight buy-and-hold**: 0.463 against 0.487. That
  alone triggers abandonment.
- **The quintile ordering is not monotonic.** Q2 out-earns Q1 and Q5 out-earns Q4, and
  the Q1−Q5 spread of +1.04% a year carries a t-statistic of 0.26. That alone triggers
  abandonment. Section 8 identified this as the important test in advance, on the
  grounds that a clean gradient across all five buckets is much harder to produce by
  chance than a good Sharpe in one bucket. There is no gradient.
- Net Sharpe of 0.463 does clear the 0.40 support threshold, and PSR(0) clears 95%.
  Neither is sufficient; section 8 required all three support clauses and only one holds.

---

## 11. Everything I decided that the document does not fix

None of these can change a position, and none was chosen after seeing a result except
where explicitly stated.

1. **Quintile bucket sizes when the count does not divide.** 41 into five: 8/8/8/8/9,
   remainder to the bottom bucket. This was the one genuine ambiguity and it was
   resolved by asking before the signal was written, not by choosing. It is the only
   partition of all 41 whose top bucket is the 8 that section 3 names.
2. **"Continuous data" means continuously listed**, not "the vendor published a bar
   every session". See section 2 above; the alternative reading would drop 13 tickers.
3. **The engine sees price history from before the sample window** in order to form the
   first signal. Section 2's requirement that history begin on or before 2008-01-01
   only makes sense on this reading. The alternative is reported in section 7 and is
   worth +0.13 of Sharpe in the flattering direction.
4. **The benchmark is buy-and-hold read literally** — bought once, then left alone —
   which is the same construction experiment 001 used. Both alternatives are reported.
5. **The noise-test pass criterion is the gross Sharpe**, not the net. Justified in
   section 5; the net figure is reported alongside it.
6. **The common-factor generator's parameters**: betas spread linearly from 0.6 to 1.4,
   factor carrying 25% and 50% of the average instrument's variance, total volatility
   held equal across instruments. Run at two loadings so the answer is not a point
   estimate.
7. **Quintile forward returns are open-to-open across consecutive rebalances**, so Q1's
   series is exactly what the traded book earns (verified to 2.9×10⁻¹⁶).
8. **The Q1−Q5 t-statistic is a plain one-sample t** on the monthly spread, with no
   autocorrelation correction. The monthly windows do not overlap.
9. **Walk-forward window lengths**: 3y / 1y / 1y step, matching experiment 001.
10. **Ties in the momentum ranking are broken by universe order.** No tie occurs in the
    real sample; the rule exists so the label is deterministic.
11. **Reporting conventions** (252-day annualisation, ddof=1, Sortino's full-sample
    denominator) are inherited unchanged from experiment 001.

---

## 12. What I built that I do not think works as intended

Stated plainly, including the things that pass their tests.

1. **The panel protocol cannot express a path-dependent target, and I did not make it
   able to.** `dict[str, DataFrame] -> DataFrame` is a pure function of prices, so a
   rule whose target weight depends on what the book currently holds — experiment 001's
   drift band is exactly such a rule — cannot live in a strategy. It lives in the
   engine, which is where it already lived, so nothing regressed. But the generalisation
   is narrower than it looks: it generalises *across instruments*, not *across the
   book's state*. A future experiment needing a genuinely path-dependent target (a
   turnover budget, a no-trade band tied to realised drift, anything with hysteresis)
   would need the protocol widened again. I would rather say this now than have someone
   discover it while writing experiment 003.

2. **`panel_buy_and_hold` passes a duck-typed object where a `Config` is annotated.**
   Experiment 002's benchmark must be experiment 001's function or the two could
   silently disagree by more than the 0.15 the decision rule turns on, so
   `UniverseView` — a frozen object carrying only `universe` — is handed to
   `buy_and_hold`, whose signature says `Config`. That is a type lie that happens to be
   true. I added a static test asserting `buy_and_hold` reads no attribute off its
   config but `universe`, which converts the fragility into a checked invariant, but the
   annotation is still wrong and a type checker would be right to complain.

3. **`TimeSeriesTrend` computes experiment 001's sizing on every bar when it only needs
   ~220 of them.** This is what makes the regression exact — the strategy is a total
   function of the panel and the engine picks the rows it trades on — but it is roughly
   ten times more work than the original engine did, about 13 seconds on the full
   history. Correct and slow. If the panel engine ever becomes the only engine, this is
   the thing to fix, and fixing it will require care not to break the exactness the gate
   depends on.

4. **The bucket rule has a degenerate branch that never fires here.** When fewer than
   `n_quantiles` instruments have a defined momentum the bucket size is zero, nothing is
   ranked, and the book sits in cash. That is the honest extension of "insufficient
   history → excluded from the ranking" to the case where the ranking cannot exist, but
   it is untested against reality: on this data the count is 39, 40 or 41. I would not
   trust it on a universe that actually shrinks.

5. **The quintile study silently drops rebalances where any bucket comes out empty**, as
   well as the last rebalance, which has no forward return. On this data that removes
   exactly one month of 224. On sparser data it could remove many without saying so, and
   the count printed is the count *kept*, which reads as coverage.

6. **The noise test's pass criterion changed between experiments and the two are not
   comparable.** Experiment 001's `NoiseTestResult.passes()` tests the net Sharpe;
   experiment 002's `PanelNoiseResult.passes()` tests the gross. The reason is given in
   section 5 and I stand by it, but two functions with the same name and different
   meanings is how a future reader gets this wrong.

7. **Nothing here validates the vendor's adjustments.** The `_sanity_check` inherited
   from experiment 001 rejects non-positive prices and one-day moves beyond ±50%, and
   the 41-ticker panel passes it, but a systematically wrong dividend adjustment on a
   high-yield name — HYG, LQD, TIP, VNQ are the candidates — would flow straight into a
   twelve-month momentum ranking and nothing in this repository would notice.

---

## 13. Reproduction

```
pytest -q                                                   # 534 passed
python scripts/run_backtest.py --experiment 001 --regression # exit 0, gate PASS
python scripts/run_backtest.py --experiment 002 --noise      # exit 0, gate PASS
python scripts/run_backtest.py --experiment 002 --validate   # exit 0, verdict ABANDON
```

Price data is Yahoo, split- and dividend-adjusted, cached to `data/cache`. The
risk-free series is `^IRX`, the 13-week T-bill, forward-filled across holidays and
divided by 252 for a daily accrual — the same series and convention experiment 001
used, applied symmetrically to the strategy's idle cash and to both Sharpe
calculations.

New code for this experiment:

| file | what it is |
|---|---|
| `trendbot/config_002.py` | parses `PREREG_002.md`; every parameter pulled, none defaulted |
| `trendbot/xsmom.py` | THE signal: momentum, quantile labels, top-bucket weights |
| `trendbot/strategies.py` | both strategies in the panel protocol |
| `trendbot/engine/panel.py` | the generalised protocol and its input contract |
| `trendbot/engine/panel_backtest.py` | the engine, generalised |
| `trendbot/engine/xs_validation.py` | universe check, correlated null, quintile gate, section 8 |
| `trendbot/regression.py` | experiment 001's recorded result, frozen |
| `scripts/run_experiment_002.py` | section 7's protocol |
| `tests/test_experiment_001_regression.py` | the gate, offline and permanent |
| `tests/test_panel_engine.py` | protocol contract, the shift, causality |
| `tests/test_xsmom.py` | the hand-built fixture gate |
| `tests/test_xs_validation.py` | correlated null, quintile gate, decision rule |
| `tests/test_config_002.py` | the parser refuses to guess |

`trendbot/signal.py` was not touched; its content hash is still the one
`tests/test_single_signal.py` pins.

**Configurations tried on this dataset, cumulative: 2.** That statement is true. No
formation window other than 252, no skip other than 21, no quintile cut other than the
top fifth, and no universe other than the 41 in section 2 was tested at any point.
