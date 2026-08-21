# Pre-registration 006 — multi-strategy risk allocation

**Committed on:** 2026-08-20

**Configurations tried, cumulative:** 5
(001 TS trend / 12 ETFs — abandon. 002 XS momentum / 41 ETFs — abandon.
003 XS momentum / 409 survivor equities — invalid, data could not answer.
004 — built, blocked, never run, does not count. 005 XS currency momentum /
22 pairs — abandon. 006 is the fifth completed configuration.)

**The correction here is understated, and that is recorded in advance.** This
experiment reuses return series from 001, 002 and 005, whose results are already
known. Even with no cherry-picking (§3), the sleeve set is not independent of prior
findings. `configs_tried = 5` is therefore a *floor*, not the true multiple-testing
burden. **A passing verdict here is suggestive, not confirmatory,** and would
require out-of-sample confirmation on data not used in 001–005 before it meant
anything. This clause exists so that a favourable result cannot later be read as
stronger than it is.

Frozen on commit. Changes to §2–7 create experiment 007. Results of 001–005 stand.

---

## 1. Hypothesis

**A portfolio of imperfectly-correlated weak strategies, allocated by risk
contribution and never by performance, achieves a higher risk-adjusted return than
the corresponding combination of those strategies' own benchmarks.**

This is a different question from every prior experiment. 001, 002, 003 and 005 each
asked whether one signal beat one universe. Each answered no or marginally. None
asked whether the *combination* beats the combination of benchmarks.

Mechanism, stated as arithmetic:

```
S_combined = s × √N / √(1 + (N−1)ρ)
```

Combining N streams of individual Sharpe `s` and average pairwise correlation `ρ`
raises portfolio Sharpe whenever `ρ < 1`. The gain is entirely in the correlation
term. The claim under test is that the *strategy* sleeves are less correlated with
each other than their *benchmarks* are with each other — plausibly true, because
001's findings recorded crisis-diversifier behaviour (outperforming SPY in 2008,
2011 and 2022) that a long-only benchmark cannot have.

This is what multi-strategy managed futures funds actually run. It is not a new
signal; it is a portfolio construction claim about signals already measured.

Reference: for the risk-parity construction, Maillard, Roncalli & Teïletche (2010),
"The Properties of Equally Weighted Risk Contribution Portfolios."

## 2. Sleeves — all completed experiments, no selection

| sleeve | source | return series |
|---|---|---|
| A | 001 | time-series trend, 12 ETFs, long-only, as specified in `PREREGISTRATION.md` |
| B | 002 | cross-sectional momentum, 41 ETFs, top quintile |
| C | 005 | cross-sectional currency momentum, 22 pairs, **carry-corrected** returns per §4's diagnostic |

**Every completed sleeve is included, including those that lost to their
benchmarks.** There is no selection step, so there is no opportunity to select on
performance. Sleeve C is included despite a negative raw Sharpe.

**003 is excluded on validity, not performance.** Its own findings state the
universe could not answer its question; an unmeasurable sleeve cannot be allocated
to. Recording the reason here prevents it being reread later as a performance
exclusion.

**No sleeve may be added, removed or substituted after this document is committed.**

## 3. Allocation — equal risk contribution

At each monthly rebalance, weights solve for equal risk contribution across the
three sleeves:

```
w_i × (Σw)_i  equal for all i
```

- Covariance `Σ` estimated from an **EWMA of daily sleeve returns, halflife 60
  trading days**, using data through the rebalance date only.
- Weights normalised so ex-ante portfolio volatility targets **10% annualised**.
- Gross exposure capped at **1.0**. No leverage. (This will bind, as it did in 001;
  report the fraction of days it binds.)
- Minimum weight per sleeve **5%**, maximum **60%**, applied before renormalisation.

**Covariance adapts. Nothing else does.** No sleeve receives more capital because it
performed well recently — not over any window, at any frequency. Realised P&L,
drawdown, hit rate and rolling Sharpe are inputs to nothing. This is the
"adapt the risk, never adapt the edge" rule applied at portfolio level, and it is
the single clause that separates this experiment from automated data mining.

## 4. Benchmark

**The same equal-risk-contribution construction applied to the three sleeves' own
benchmarks** — 001's equal-weight 12-ETF buy-and-hold, 002's equal-weight 41-ETF
buy-and-hold, 005's daily-rebalanced dollar factor — using identical covariance
estimation, identical vol target, identical caps.

Like-for-like. The only difference between portfolio and benchmark is whether the
sleeves are the strategies or the markets they trade.

Sharpe computed as excess of the T-bill rate for both.

## 5. Sample window

**The intersection of all three sleeves' available histories.** Determined by the
data, reported, not chosen. Expected to be governed by 002's ETF universe and
005's post-1999 start.

## 6. Execution

- Sleeve returns are taken from each experiment's existing, committed
  implementation. No re-implementation, no re-parameterisation.
- Sleeve-level costs are already charged inside each sleeve's returns.
- **Additional cost for portfolio rebalancing: 5 bps per side on the change in
  sleeve weight.** Reallocating between sleeves means trading their underlying
  positions and is not free. Sensitivity at 0/2/5/10 bps.
- Rebalance monthly, first trading day.

## 7. What is frozen

Sleeve set · exclusion of 003 · ERC construction · covariance estimator and
halflife · vol target · exposure caps · min/max sleeve weights · benchmark
construction · rebalance schedule · cost assumption · sample window rule.

## 8. Pre-committed decision rule

**Supported** only if all three hold:

- Net Sharpe exceeds the §4 benchmark by at least **0.15**, AND
- Net Sharpe exceeds **the best individual sleeve's Sharpe** by at least **0.10** —
  if the combination does not beat its own best component, diversification bought
  nothing and the portfolio is redundant, AND
- The **realised average pairwise correlation between sleeves is lower than the
  realised average pairwise correlation between their benchmarks.** This is §1's
  mechanism. If it is false, any outperformance came from somewhere other than the
  stated cause and does not support the hypothesis.

**Abandon** if any of: fails to beat the §4 benchmark at all; fails to beat the best
individual sleeve; sleeve correlations are not lower than benchmark correlations.

Anything else is **inconclusive** and is not traded.

The third clause is the important one. The first two can be satisfied by luck in a
three-sleeve portfolio. The correlation clause tests the *mechanism* rather than the
*outcome*, which is the same reason 002 and 003 gated on monotonicity rather than
Sharpe alone.

## 9. Expectations of record

- **N = 3 is thin.** √3 = 1.73 is a modest multiplier, and the formula in §1 assumes
  equal Sharpes and a single correlation, which is an idealisation.
- **Sleeves A and B share an ETF universe.** Their correlation will not be low.
  Effective N is closer to 2 than 3.
- Realistic combined Sharpe: **0.3 – 0.6**. Above 0.9 means a bug or a covariance
  estimate using future data.
- **The benchmark improves too.** Diversifying three benchmarks also raises the
  benchmark's Sharpe. The margin may not widen even if both numbers rise. This is
  the most likely way this experiment produces an inconclusive.
- The gross cap will bind, as in 001, suppressing the vol target.
- Sleeve C's carry correction relies on FRED short rates missing for 6 currencies
  covering 23.2% of that book; sleeve C's returns inherit that gap.
- **Most likely verdict: inconclusive.** A pass would still require out-of-sample
  confirmation per the header before meaning anything.

## 10. Signature

Running §7 without modifying §2–7.

Signed: **Pranav**   Date: **2026-08-20**
