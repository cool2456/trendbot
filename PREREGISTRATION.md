# Pre-registration — diversified trend following v1.0

**Committed on:** ____________  (fill in before running any backtest)

This document is frozen. Any change to any number below produces a **new strategy
with a new version number and a new date**, and its results may not be compared
to, or substituted for, this one. Editing this file after seeing a result is the
single most common way a research project turns into self-deception.

---

## 1. Hypothesis

Instruments that have risen over the trailing twelve months continue to rise, and
instruments that have fallen continue to fall, for long enough and across enough
markets that a mechanical rule capturing it earns a positive risk-adjusted return
net of costs.

Mechanism: slow diffusion of information causes initial underreaction, followed by
delayed overreaction from herding, which is why the effect persists for roughly a
year and then partially reverses.

Reference: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum," JFE 104(2).

## 2. Universe — 12 instruments, fixed

| Sleeve | Tickers |
|---|---|
| Equity | SPY, EFA, EEM |
| Rates | TLT, IEF |
| Commodities | GLD, SLV, DBC |
| Currency | UUP, FXE, FXY |
| Real assets | VNQ |

No additions, removals, or substitutions. If an instrument delists, the position
goes to cash and the universe shrinks; it is not replaced.

## 3. Signal

```
trend_i(t) = sign( P_i(t) / P_i(t-252) - 1 )
```

- Lookback: **252 trading days**. Not 200. Not 189. Not "optimised."
- Computed on the close of bar t using only data through bar t.
- Undefined (insufficient history) → position 0.
- Long-only variant: `trend_i ∈ {0, 1}`. Declare which variant below.

**Variant in use:** long-only

## 4. Risk scaling — the only adaptive component

```
sigma_i(t) = ewm_std(daily_returns_i, halflife=30) * sqrt(252)
x_i(t)     = trend_i(t) * (0.10 / sigma_i(t))
w_i(t)     = x_i(t) / 12
w(t)       = k(t) * w(t)      # k scales ex-ante portfolio vol to 10% annualised
```

- Per-instrument vol target: **10%**
- EWMA halflife: **30 days**
- Portfolio ex-ante vol target: **10%**
- Gross exposure cap: **sum(|w_i|) <= 1.0** — no leverage, cash account
- Per-instrument cap: **|w_i| <= 0.25**

Volatility is the only quantity permitted to change position sizes. Nothing
adapts to realised P&L, drawdown, hit rate, or recent performance of the signal.

## 5. Execution

- Signal on close of bar t → fill at **open of bar t+1**.
- Rebalance: first trading day of each month.
- Drift band: only trade instrument i if `|w_target − w_current| > 0.20 * |w_target|`.
- Cost assumption in backtest: **5 bps per side**, charged on turnover.
- Sensitivity reported at 0, 2, 5, 10, 20 bps. The 5 bps figure is the headline.

## 6. What is frozen

Lookback · sign rule · universe · vol target · halflife · portfolio vol target ·
exposure caps · rebalance schedule · drift band · cost assumption.

None of these update on performance. Not in backtest, not in paper, not live.

## 7. Test protocol, in order

1. **Noise test** — run on synthetic random-walk data. Expected Sharpe ≈ 0.
   A materially positive result means the engine has a bug. Stop and find it.
2. **Full-sample backtest** — one run. Record the result before doing anything else.
3. **IS/OOS split** — fit-free, so both halves are out-of-sample by construction.
   Report both. Large asymmetry means a data problem, not an edge.
4. **Walk-forward** — rolling windows, no re-optimisation (nothing to optimise).
5. **Deflated Sharpe** — number of configurations tried = 1. Record it as 1.
   If it is ever greater than 1, this document has been violated.
6. **Paper trading** — minimum 6 months, divergence log on every run.
7. **Decision point** — only after step 6.

## 8. Pre-committed decision rule

Fill this in *now*, before any result exists.

- I will consider the hypothesis supported if net Sharpe over the full sample
  exceeds 0.40 AND the sign is positive in at least 9 of 12 instruments
  AND net Sharpe exceeds equal-weight buy-and-hold of the same 12 ETFs
  by at least 0.15.
- I will abandon it if net Sharpe is below 0.15, or if it fails to beat
  equal-weight buy-and-hold at all.
- Between those, the result is inconclusive and I will not trade it.

Writing these numbers after seeing the backtest is the failure this document exists
to prevent.

## 9. Expectations of record

Stated in advance so they cannot be revised upward afterwards:

- Realistic net Sharpe: **0.3 – 0.6**. Above 0.8 means a bug.
- Roughly one year in three is a losing year.
- Underwater periods of 3–5 years are normal, not evidence of failure.
- This will underperform a plain index fund in most calendar years.
- At Sharpe 0.5, distinguishing this edge from luck requires ~16 years of live data.
  Nothing observed in the first year means anything.

## 10. Signature

By dating this document I commit to running the protocol in section 7 without
modifying sections 2 through 6.

Signed: Pranav Tamilselvan   Date: Aug 19 2026
