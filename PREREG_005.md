# Pre-registration 005 — cross-sectional currency momentum

**Committed on:** 2026-08-19

**Configurations tried, cumulative:** 4
(001 time-series trend / 12 ETFs. 002 cross-sectional momentum / 41 ETFs.
003 cross-sectional momentum / 409 survivor equities. 005 is the fourth.

**Experiment 004 does not count, and here is why.** 004 was built but blocked — an
IP-level vendor throttle prevented Steps 2, 3 and 5–11 from ever reaching data. No
verdict was issued, nothing was estimated, nothing was substituted. PSR corrects
for configurations *tried*, meaning variants evaluated against data such that a
winner could have been selected. A run that never reached the data produced nothing
to select from. Experiment numbers are identifiers; this counter tracks completed
evaluations, and the two need not match.

004 remains **pending**, not failed. If the data ever becomes reachable it runs its
full protocol unattended and becomes configuration 5.

**Note for future experiments:** had 004 completed — even with an uninterpretable
result — it would have counted. The multiple-testing family is not "tests on one
dataset" but *the set of experiments from which a success might be reported*, and
that selection spans datasets.)

Frozen on commit. Changes to §2–6 create experiment 006. Results of 001–003 stand; 004 is pending.

---

## 1. Hypothesis

**Currencies that have outperformed other currencies over the past twelve months
continue to outperform them over the following month.**

The signal is byte-identical to experiments 002, 003 and 004. Only the universe
changes. Across four cross-sectional experiments the signal is held constant while
the universe varies, so the accumulated results answer one coherent question:
*where, if anywhere, does this signal work?*

**Why currencies:**

1. **No survivorship bias, structurally.** Currencies do not delist or go bankrupt.
   The contamination that made 003's verdict uninterpretable cannot occur here.
2. **Low cross-correlation.** Menkhoff et al. note that even small currency
   portfolios yield large diversification gains because currencies co-move less than
   stocks. 002 failed because 41 ETFs collapsed into a single market factor; this
   universe does not have that defect.
3. **The data is free and authoritative** — Federal Reserve H.10, via FRED.

Reference: Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Currency Momentum
Strategies," JFE 106(3), 660–684. Reported cross-sectional spread up to 10% p.a.
between past winner and loser currencies over 1976–2010, >40 currencies.

**The paper's own caveat, recorded here in advance:** it concludes that limits to
arbitrage prevent these returns from being easily exploitable, and that transaction
costs partially explain the spread. A null result is consistent with the
literature and is not a surprising outcome.

**This is a research question, not a tradeable strategy.** FX is not accessible in
the author's account. Feasibility is explicitly not what this experiment asks.

## 2. Universe — a rule

**Every daily USD exchange rate series published in the Federal Reserve H.10
release and available via FRED that has continuous daily data over the full sample
window.**

- Expected yield: roughly 20–23 pairs. Report the exact count and the list.
- Series discontinued mid-sample are **excluded entirely**, not truncated. Record
  which and why.
- No substitutions, no additions.

**Quote convention must be normalised.** FRED mixes conventions — some series are
USD per foreign unit, others foreign units per USD. All series must be converted to
a common convention (foreign currency value expressed in USD) before ranking.
Getting a subset inverted scrambles every rank silently without raising an error.
This is the single most likely way this experiment produces a wrong answer.

## 3. Sample window

**1999-01-01 to present.** Fixed now.

Chosen to begin after euro adoption, so that DEM, FRF, ITL and the other legacy
currencies merging into EUR do not need special handling. This is a
data-cleanliness boundary, not a performance-based choice, and it is committed
before any result exists.

## 4. Returns and signal

**Return definition:** log change in the USD value of the foreign currency.
**Spot only.** No interest-rate component.

This is an approximation. True currency excess return is the spot change plus the
interest differential. Menkhoff et al. find the momentum effect is largely driven
by spot rate changes and is not dominated by the interest-rate component, which is
why the approximation is acceptable — but it *is* an approximation and must be
labelled as one in the findings.

**Required diagnostic (not a configuration):** repeat the headline result using
short-term interest rate differentials from FRED to approximate excess returns, and
report the difference. This measures the size of the approximation rather than
resolving it.

**Signal:**

```
momentum_i(t) = P_i(t-21) / P_i(t-252) - 1
```

Sort into **quintiles**. Top quintile: equal weight, long. All others: zero.
Long-only — which in FX means long the foreign currency against USD.

With ~20 pairs the traded book is 4–5 currencies. That is thin, partially mitigated
by the low cross-correlation noted in §1, and recorded here as a known weakness
rather than discovered later.

## 5. Execution

- Gross exposure 1.0 when invested. No leverage, no volatility targeting.
- Signal on close of bar t → position taken at bar t+1.
- Rebalance: first trading day of each month, full rebalance.
- **Cost assumption: 5 bps per side.** Major FX pairs are tighter than equities.
  Sensitivity at 0/2/5/10/25. Menkhoff et al. state costs partially explain the
  spread, so the cost ladder is central, not decorative.
- Benchmark: **equal-weight long all currencies in the universe against USD**
  (the "dollar factor"). Sharpe computed as excess of the T-bill rate for both
  strategy and benchmark.

## 6. What is frozen

Universe rule · quote normalisation requirement · sample window · return definition
(spot) · formation window (252) · skip (21) · quintile cut · equal weighting ·
long-only · rebalance schedule · cost assumption · benchmark.

## 7. Test protocol

1. **Quote-convention audit** — §2. Verify every series direction against a known
   reference value before any ranking. Gate on this.
2. **Noise tests, both variants** — independent and common-factor correlated random
   walks, ≥8 seeds each, with factor attribution. Both ≈ 0.
3. **Universe construction audit** — final count, list, and exclusions.
4. **Full-sample backtest** — one run, with the §5 cost ladder.
5. **Quintile monotonicity** — §8.
6. **Beta attribution** — against the dollar factor (headline) and against SPY
   (secondary, to check whether currency momentum is a disguised equity beta).
7. **Interest-rate approximation diagnostic** — §4.
8. **PSR / deflated Sharpe at `configs_tried = 4`.**
9. **Verdict** per §8.

The 001 regression test must continue to pass bit-identically throughout.

## 8. Pre-committed decision rule

**Supported** only if all four hold:

- Net Sharpe (excess of T-bill, 5 bps) exceeds **0.40**, AND
- Net Sharpe exceeds the equal-weight dollar-factor benchmark by at least **0.15**,
  AND
- **Quintile monotonicity:** forward returns decrease monotonically Q1 → Q5 with at
  most **one** adjacent inversion, and Q1−Q5 spread positive at **t > 2.0**, AND
- **Alpha to the dollar factor is positive with t > 2.0.**

**Abandon** if any of: fails to beat the benchmark at all; more than one inversion,
or Q1−Q5 t below 1.0; alpha to the dollar factor negative; net Sharpe below **0.15**.

Anything else is **inconclusive** and is not traded.

Thresholds are unchanged from 002 and 003 on purpose.

## 9. Expectations of record

- Realistic net Sharpe: **0.2 – 0.6**. Above 1.0 means a bug, most likely an
  inverted quote convention or a lookahead in the ranking.
- The published 10% p.a. spread is a 1976–2010, 40+ currency, long-short figure with
  institutional execution. This is a post-1999, ~20 currency, long-only,
  retail-cost test. Expect materially less.
- Costs may eliminate the effect entirely. The paper says transaction costs
  partially explain the spread; the cost ladder will show whether that is decisive
  here.
- A 4–5 currency traded book is thin. Individual months will be dominated by single
  positions.
- **Momentum crashes:** unlike equities, currency momentum has no single canonical
  crash date, so §9 imposes no specific month requirement here. Instead: report the
  five worst months and whether they cluster around known FX dislocations
  (2008 Q4, 2011 CHF, 2015 CHF de-peg, 2020 March, 2022 GBP). Absence of any
  clustering is a warning sign about the implementation.
- The spot-only approximation biases the result by an unknown amount; §4's
  diagnostic quantifies it.
- **The PSR correction is now materially larger.** At `configs_tried = 4` the
  deflated Sharpe haircut exceeds every prior experiment's. A raw Sharpe that would
  have cleared in 002 may not clear here. That is the accumulated cost of five
  attempts on one hypothesis space, and it is not a reason to reset the counter.

## 10. Signature

Running §7 without modifying §2–6.

Signed: **Pranav**   Date: **2026-08-19**
