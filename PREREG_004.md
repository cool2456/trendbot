# Pre-registration 004 — cross-sectional momentum (point-in-time universe)

**Committed on:** 2026-08-19

**Configurations tried on this dataset, cumulative:** 4
(001 time-series trend / 12 ETFs. 002 cross-sectional momentum / 41 ETFs.
003 cross-sectional momentum / 409 survivor equities. Every PSR and deflated
Sharpe here uses `configs_tried = 4`. The counter only rises.)

Frozen on commit. Changes to §2–6 create experiment 005. Results of 001–003 stand.

---

## 1. Hypothesis

**Individual stocks that have outperformed other stocks over the past twelve
months continue to outperform them over the following month.**

Identical to experiment 003's hypothesis, deliberately.

**Why re-testing is legitimate rather than data mining:** 003 did not refute this
hypothesis. Its own verdict was that the universe *could not answer the question* —
the decile gradient inverted (D10 +1.885%/mo vs D1 +1.486%/mo), which is the
signature of a bottom bucket populated by distressed-and-recovered names in a
dataset from which distressed-and-died names had been deleted. 003 measured a
survivorship artifact, not the anomaly.

004 changes exactly one thing: the universe is constructed point-in-time from
survivorship-free data. The signal, the decile cut, the weighting, the rebalance
schedule and every threshold in §8 are unchanged from 003, so that the data is the
only variable. This is the first valid test of the hypothesis in this project.

Reference: Jegadeesh & Titman (1993), JF 48(1).

## 2. Universe — a rule, not a list

At each monthly rebalance date t, the eligible universe is:

**The 500 US common stocks with the highest median dollar volume over the trailing
60 trading days**, subject to:

- Security type is common stock. Exclude ETFs, ETNs, CEFs, closed-end debt, units,
  warrants, preferred shares. ADRs are **included**.
- Unadjusted close on date t is at least **$5.00**.
- At least 252 trading days of price history as of date t.
- The stock was actually tradeable on date t (not yet delisted).

This rule is computable from price and volume alone at each date, so the universe
is point-in-time by construction. It requires no index membership data and depends
on no vendor's index reconstruction.

**A stock that was liquid in 2011 and bankrupt in 2013 is in the universe in 2011
and absent in 2013.** That is the entire point.

## 3. Delisting treatment — pre-committed, because it is a large degree of freedom

When a held position delists mid-month, the return assigned is determined by the
vendor's delist reason:

| Reason | Treatment |
|---|---|
| Acquisition / merger | Final traded price, proceeds to cash for remainder of month |
| Bankruptcy / liquidation | **−100%** |
| Moved to OTC, or reason unknown/missing | **−30%** |

The −30% figure follows Shumway (1997), the standard haircut for missing delisting
returns. **Sensitivity is mandatory:** report the headline result plus the same
result with the unknown bucket set to 0% and to −100%, so the size of this choice
is visible rather than buried.

Choosing this treatment after seeing results would be the single easiest way to
manufacture a passing number. It is fixed here.

## 4. Signal

```
momentum_i(t) = P_i(t-21) / P_i(t-252) - 1
```

Identical to 002 and 003. Sort eligible names into **deciles**. Top decile: equal
weight, long. All others: zero. Long-only.

## 5. Risk scaling and execution

- Equal weight within the top decile. Gross exposure 1.0 when invested. No
  leverage, no volatility targeting.
- Cash earns the T-bill rate; Sharpe is excess of the same rate for strategy and
  benchmark alike.
- Signal on close of bar t → fill at open of bar t+1.
- Rebalance: first trading day of each month, full rebalance.
- **Cost assumption: 10 bps per side.** Sensitivity at 0/5/10/20/40.
- Prices must be **split-and-dividend adjusted using point-in-time factors**. If
  the vendor's adjustment is not point-in-time, say so and quantify the exposure.

## 6. Sample window — a rule, not a date

**Start:** the earliest month at which the §2 rule yields at least 500 qualifying
names, each with 252 trading days of prior history. **End:** present.

Stated as a rule because the answer depends on vendor coverage neither the author
nor the implementer has seen at the time of writing. The rule is fixed now; the
resulting date is whatever it is. Report it.

**Frozen:** universe rule · liquidity ranking window (60d) · price floor ($5) ·
name count (500) · delisting treatment · formation window (252) · skip (21) ·
decile cut · equal weighting · long-only · rebalance schedule · cost assumption ·
sample window rule.

## 7. Test protocol

1. **Noise tests, both variants** — independent and common-factor correlated random
   walks, ≥8 seeds each, with factor attribution. Both ≈ 0.
2. **Pipeline validation on the free tier first.** Build and verify ingestion,
   delisting handling and universe construction on the 30-name free dataset before
   any paid data is used. The 001 regression test must still pass bit-identically.
3. **Universe construction audit** — names per rebalance, count of delistings by
   reason, and the realised start date under §6.
4. **Full-sample backtest** — one run, with the §3 sensitivity ladder.
5. **THE PAIRED DIAGNOSTIC — see §10.**
6. **Decile monotonicity** — full D1→D10 table, inversion count, D1−D10 spread
   with t-statistic.
7. **Market-beta attribution** — against SPY (headline) and against equal-weight
   universe buy-and-hold (secondary). Report beta, alpha, alpha t-stat for both.
8. **PSR / deflated Sharpe at `configs_tried = 4`.**
9. **Verdict** per §8.

## 8. Pre-committed decision rule — unchanged from 003

**Supported** only if all four hold:

- Net Sharpe (excess of T-bill, 10 bps) exceeds **0.40**, AND
- Net Sharpe exceeds equal-weight buy-and-hold of the same point-in-time universe
  by at least **0.15**, AND
- **Decile monotonicity:** forward returns decrease monotonically D1 → D10 with at
  most **one** adjacent inversion, and D1−D10 spread positive at **t > 2.0**, AND
- **Alpha to SPY is positive with t > 2.0.**

**Abandon** if any of: fails to beat buy-and-hold at all; more than one inversion,
or D1−D10 t below 1.0; alpha to SPY negative; net Sharpe below **0.15**.

Anything else is **inconclusive** and is not traded.

Thresholds are byte-identical to 003 on purpose. The data is the only variable.

Positive alpha against SPY but negative against the equal-weight universe is not an
automatic abandon, but must be reported prominently in the verdict as evidence of a
size effect rather than momentum.

## 9. Expectations of record

- Realistic net Sharpe: **0.3 – 0.7**. Above 1.0 means a bug or a delisting
  treatment that is quietly too generous.
- Results will be **worse** than 003's 0.699 headline. If they are better, the
  point-in-time universe has not been constructed correctly — survivorship inflates
  results, and removing it should hurt.
- **Momentum crashes are mandatory.** March–May 2009 and April 2020 must appear
  among the worst months. This check has now failed twice. A third failure on a
  correctly constructed universe means the implementation is wrong, not the
  anomaly, and the result must be reported as invalid rather than as a verdict.
- Published momentum decayed roughly 58% post-publication (McLean & Pontiff).
  J&T's ~1%/month is a 1965–1989 figure and is not the expectation.
- Turnover will be high; costs will be the dominant drag.
- Not implementable at $1,000. Feasibility is not this experiment's question.

## 10. The paired diagnostic — required output regardless of verdict

Run the identical strategy, identical code, identical dates, **twice**:

- **(A)** the §2 point-in-time universe including delisted securities.
- **(B)** the same rule restricted to securities still listed today.

Report every §8 metric for both, and the difference.

**The A−B gap is the measured size of survivorship bias**, and it is a finding of
this experiment whether or not the strategy passes. It tells this project
permanently whether free survivor-only data can ever answer a cross-sectional
question, which has now blocked two experiments.

If (B) reproduces something close to 003's inverted gradient while (A) does not,
that confirms the 003 diagnosis directly.

## 11. Signature

Running §7 without modifying §2–6.

Signed: **Pranav**   Date: **2026-08-19**
