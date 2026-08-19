# Pre-registration 002 — cross-sectional momentum (ETF universe)

**Committed on:** 2026-08-19

**Configurations tried on this dataset, cumulative:** 2
(Experiment 001, diversified time-series trend, was configuration 1. Every PSR and
deflated Sharpe in this document must be computed with `configs_tried = 2`, not 1.
This counter only ever increases.)

Frozen on commit. Any change to sections 2–6 produces experiment 003 with its own
document, date and tag. Experiment 001's result stands and is not revised.

---

## 1. Hypothesis

**Instruments that have outperformed their peers over the past twelve months
continue to outperform them over the following month.**

Mechanism: investors underreact to asset-specific information, so relative
performance persists; extrapolative trading by momentum-followers extends the
divergence before it eventually reverses at longer horizons. This is a *relative*
effect, not a directional one — it does not require the market to go up, down, or
anywhere in particular.

**Why this is a different hypothesis, not a retuned version of 001:**
Experiment 001 asked "is this asset rising in absolute terms." Between 2008 and
2026 that question was dominated by a single equity bull market, so a long-only
absolute rule converged toward simply holding everything — which is why it lost to
buy-and-hold. This experiment asks "is this asset rising *relative to its peers*,"
which is orthogonal to market direction and cannot collapse into buy-and-hold.

Reference: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling
Losers," Journal of Finance 48(1). Reported premium approximately 1% per month
on US equities, 1965–1989. Survived subsequent out-of-sample scrutiny that most
documented anomalies failed.

**Statistical note:** cross-sectional strategies yield N×T observations rather
than T. With ~40 instruments this is a materially larger effective sample than
experiment 001, which is the specific weakness — insufficient breadth — that
experiment 001's result exposed.

## 2. Universe — 41 ETFs, fixed

All must have price history beginning on or before 2008-01-01.

| Sleeve | Tickers |
|---|---|
| US sectors | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY |
| Country / region | EWJ, EWG, EWU, EWC, EWA, EWY, EWT, EWZ, EWH, EWS, EWW, EWL |
| Broad equity | SPY, QQQ, IWM, DIA, EFA, EEM, IWD, IWF |
| Real assets | VNQ |
| Rates / credit | TLT, IEF, SHY, LQD, HYG, TIP |
| Commodities | GLD, SLV, DBC, USO, DBA |

**Verification required before any backtest:** confirm each ticker has continuous
data from 2008-01-01. Drop any that does not, record which were dropped and why in
`FINDINGS.md`, and report the final count. Do not substitute replacements.

**Survivorship note:** this universe is selected from ETFs that exist today, so it
excludes funds that launched before 2008 and have since closed. This biases results
upward. The bias is smaller than for an equity universe — ETF closures are rarer
than stock delistings — but it is not zero and must be stated in `FINDINGS.md`.

## 3. Signal

```
momentum_i(t) = P_i(t-21) / P_i(t-252) - 1
```

Trailing twelve-month return **skipping the most recent month** (21 trading days).
The skip is standard and removes contamination from short-term reversal; it is not
a tuned parameter.

Each rebalance date, rank all instruments by `momentum_i`. Position:

- **Top quintile** (highest ~8 of 41): equal weight, long.
- All others: zero.

Long-only. Insufficient history for any instrument → excluded from that ranking.

## 4. Risk scaling

Equal weight within the selected quintile. `w_i = 1 / n_selected` for selected
instruments, 0 otherwise.

- Gross exposure: **1.0** when fully invested. No leverage.
- Cash earns the T-bill rate. Sharpe computed as excess of the same T-bill rate
  for both strategy and benchmark.
- **No volatility targeting.** Experiment 001 established that the gross cap binds
  89–100% of the time in a cash account, rendering vol targeting inoperative.
  Removing it eliminates a component that cannot function rather than pretending
  it does.

## 5. Execution

- Signal on close of bar t → fill at **open of bar t+1**.
- Rebalance: first trading day of each month.
- No drift band. Full rebalance to the new quintile each month.
- Cost assumption: **5 bps per side**, charged on turnover.
- Sensitivity reported at 0, 2, 5, 10, 20 bps. Headline is 5 bps.

Expected turnover is materially higher than experiment 001. If the strategy only
survives at 0 bps, that is a failure, not a technicality.

## 6. What is frozen

Universe · formation window (252) · skip (21) · quintile cut · equal weighting ·
long-only · rebalance schedule · cost assumption · sample window.

**Sample window: 2008-01-01 to present.** Fixed now, before any result exists.
Experiment 001 demonstrated that window choice can flip a verdict; it is therefore
pre-committed rather than selected.

## 7. Test protocol

1. **Noise test** — synthetic random walks, expected Sharpe ≈ 0. A cross-sectional
   rule on independent noise series must earn nothing. Also run on correlated noise
   with a common factor, which must also earn nothing.
2. **Universe verification** — data availability per section 2.
3. **Full-sample backtest** — one run.
4. **Quintile monotonicity** — see section 8.
5. **Walk-forward** — rolling windows, nothing fitted.
6. **PSR / deflated Sharpe** with `configs_tried = 2`.
7. **Decision** against section 8.

## 8. Pre-committed decision rule

The hypothesis is **supported** only if all three hold:

- Net Sharpe (excess of T-bill, 5 bps) exceeds **0.40**, AND
- Net Sharpe exceeds equal-weight buy-and-hold of the same 41 ETFs by at least
  **0.15**, AND
- **Quintile monotonicity holds.** Sorting all instruments into five quintiles by
  `momentum_i` and computing each quintile's forward return, the ordering must be
  monotonically decreasing from Q1 to Q5, and the Q1−Q5 spread must be positive.

**Abandon** if any of:

- It fails to beat equal-weight buy-and-hold at all, OR
- Quintile ordering is non-monotonic, OR
- Net Sharpe is below **0.15**.

Anything else is **inconclusive** and is not traded.

The monotonicity test is the important one. Sharpe can be produced by luck in one
quintile; a clean monotonic gradient across all five is much harder to produce by
chance and is the standard validation in the asset-pricing literature.

## 9. Expectations of record

Stated in advance so they cannot be revised upward:

- Realistic net Sharpe: **0.3 – 0.7**. Above 1.0 means a bug.
- Turnover will be high and costs will hurt materially more than in 001.
- Momentum crashes: this strategy loses badly at sharp market reversals
  (March 2009 and April 2020 are the canonical cases). Expect at least one
  drawdown above 25%.
- The survivorship bias in section 2 inflates the result by an unknown amount.
- **PSR may again fail to clear 95%.** With `configs_tried = 2` the correction is
  larger than in 001. An inconclusive verdict on statistical power grounds is a
  likely and legitimate outcome, and does not license a third configuration.

## 10. Signature

Running the protocol in section 7 without modifying sections 2 through 6.

Signed: **Pranav**   Date: **2026-08-19**
