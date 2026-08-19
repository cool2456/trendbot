# Pre-registration 003 — cross-sectional momentum (US equities)

**Committed on:** 2026-08-19

**Configurations tried on this dataset, cumulative:** 3
(001 = time-series trend, 12 ETFs. 002 = cross-sectional momentum, 41 ETFs.
Every PSR and deflated Sharpe here uses `configs_tried = 3`. Counter only rises.)

Frozen on commit. Changes to §2–6 create experiment 004. Results of 001 and 002
stand and are not revised.

---

## 1. Hypothesis

**Individual stocks that have outperformed other stocks over the past twelve
months continue to outperform them over the following month.**

Same mechanism as 002 — underreaction to asset-specific information, extended by
extrapolative trading. What changes is the universe, and the change is motivated
by a specific finding rather than by a desire for a better number.

**Why this is a new hypothesis, not a retuned 002:**
002's factor attribution showed that on correlated instruments the ranking rule
degenerates into a market-beta tilt: beta 0.913, cross-seed Sharpe correlation
+0.922, alpha −0.025%/yr. Forty-one ETFs share too much common variance for a
cross-sectional sort to express anything but beta. Jegadeesh & Titman documented
this anomaly on **individual equities**, where idiosyncratic variance dominates.
002 tested the anomaly on a universe structurally incapable of expressing it.

This is a correction of a design error identified by 002's diagnostics, not a
search for a configuration that passes.

Reference: Jegadeesh & Titman (1993), JF 48(1). Also Asness, Moskowitz & Pedersen
(2013) for out-of-sample persistence across markets and asset classes.

## 2. Universe

**Current S&P 500 constituents with continuous daily data from 2008-01-01 to
present.** Expected yield 300–400 names. Report the exact count.

Verification required: per-ticker first available date, and the list of current
constituents excluded for insufficient history. Do not substitute replacements.

**Survivorship bias — stated in advance, and the direction matters.**
This universe contains only companies that exist today and are in the index today.
Firms that went bankrupt, were acquired, or fell out of the index are absent. This
is a severe bias and there is no free point-in-time source to correct it.

Its *direction* is what makes the experiment still worth running. Survivorship
removes failures, and failures concentrate in the bottom of a momentum ranking.
The bottom decile is therefore artificially strong, which makes the D1−D10 spread
artificially **narrow** and the monotonicity gradient artificially **flat**.

So: a monotonicity pass on this universe is conservative evidence. A monotonicity
failure is ambiguous — it could be the anomaly's absence or the bias. This
asymmetry must be stated in the verdict, in both directions, and neither reading
may be selected after the fact.

## 3. Signal

```
momentum_i(t) = P_i(t-21) / P_i(t-252) - 1
```

Trailing twelve-month return skipping the most recent month. Identical to 002 —
deliberately unchanged, so that the universe is the only variable.

At each monthly rebalance, rank all eligible names and sort into **deciles**
(ten buckets, not five — breadth permits the finer sort that J&T used).

- **Top decile:** equal weight, long.
- All others: zero.

Long-only. Names lacking 252 days of history are excluded from that date's ranking.

## 4. Risk scaling

Equal weight within the top decile. Gross exposure 1.0 when invested, no leverage,
no volatility targeting. Cash earns the T-bill rate; Sharpe is excess of the same
rate for strategy and benchmark alike.

## 5. Execution

- Signal on close of bar t → fill at open of bar t+1.
- Rebalance: first trading day of each month, full rebalance to the new decile.
- **Cost assumption: 10 bps per side.** Higher than 002's 5 bps, because single
  stocks carry wider spreads than large ETFs. Sensitivity at 0/5/10/20/40.
- Corporate actions: split and dividend adjustment must be point-in-time. A stock
  universe makes this materially more dangerous than an ETF universe — an
  unadjusted split reads as a −50% return and will be ranked as a momentum loser.
  Verify explicitly.

## 6. What is frozen

Universe definition · formation window (252) · skip (21) · decile cut · equal
weighting · long-only · rebalance schedule · cost assumption · sample window
(2008-01-01 to present).

## 7. Test protocol

1. **Noise tests, both variants.** Independent random walks and common-factor
   correlated walks, ≥8 seeds each. Both must return ≈ 0.
2. **Universe verification** and corporate-action audit per §2 and §5.
3. **Full-sample backtest** — one run.
4. **Decile monotonicity** — §8.
5. **Market-beta attribution — required.** Regress strategy excess returns on
   market excess returns. Report beta, alpha, and alpha's t-statistic. This is the
   diagnostic that diagnosed 002 and it is now standard equipment.
6. **PSR / deflated Sharpe at `configs_tried = 3`.**
7. **Verdict** against §8.

## 8. Pre-committed decision rule

**Supported** only if all four hold:

- Net Sharpe (excess of T-bill, 10 bps) exceeds **0.40**, AND
- Net Sharpe exceeds equal-weight buy-and-hold of the same universe by at least
  **0.15**, AND
- **Decile monotonicity:** forward returns decrease monotonically D1 → D10,
  allowing at most **one** adjacent inversion, with D1−D10 spread positive at
  **t > 2.0**, AND
- **Alpha to market beta is positive with t > 2.0.** A result that is purely a
  beta tilt does not count, whatever its Sharpe.

**Abandon** if any of:

- Fails to beat equal-weight buy-and-hold at all, OR
- More than one decile inversion, or D1−D10 t-statistic below 1.0, OR
- Alpha to market is negative, OR
- Net Sharpe below **0.15**.

Anything else is **inconclusive** and is not traded.

Note the two changes from 002's rule, both pre-committed: one inversion is
tolerated because ten buckets are noisier than five, and the spread now carries a
t-statistic requirement because 002 passed +1.04%/yr at t = 0.26 — a spread with no
statistical content. Adding the alpha clause means a beta tilt can no longer
produce a pass.

## 9. Expectations of record

- Realistic net Sharpe: **0.4 – 0.8** before bias adjustment. Above 1.2 means a bug
  or an uncorrected corporate action.
- Turnover higher than 002's 6.12×. Costs will be the dominant drag.
- **Momentum crashes are mandatory.** March–May 2009 and April 2020 must appear
  among the worst months. 002 failed this and it was diagnostic. If a stock-level
  implementation also fails it, the implementation is wrong, not the anomaly.
- Survivorship inflates the level and flattens the gradient, per §2.
- Published momentum has decayed post-publication (McLean & Pontiff: ~58% lower
  after publication). J&T's ~1%/month is a 1965–1989 figure and is not the
  expectation here.
- **Not implementable at $1,000.** Top decile of ~350 names is ~35 positions;
  feasibility is not the question this experiment asks.

## 10. Signature

Running §7 without modifying §2–6.

Signed: **Pranav**   Date: **2026-08-19**
