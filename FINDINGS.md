# Findings — diversified trend following v1.0

Implementation of `PREREGISTRATION.md` (sha256 `353849ef77c8…`, committed `dccdea4`,
2026-08-19 04:43:46 −0400, unmodified since — verified with `git log --follow` and
asserted by a test). No parameter in that document was changed, tuned, or
substituted. **Configurations tried: 1.**

---

## 1. The headline result

Window **2008-02-29 → 2026-08-18** (18.4 years) — the first bar on which all twelve
instruments have a complete 252-day lookback. Net of 5 bps per side. Sharpe is in
excess of the 13-week T-bill; idle cash accrues at that same rate; the benchmark is
treated identically.

| | strategy | equal-weight buy-and-hold |
|---|---|---|
| CAGR (excess) | 2.38% | 3.73% |
| volatility | 7.48% | 10.09% |
| **Sharpe** | **0.352** | **0.414** |
| Sortino | 0.48 | 0.57 |
| max drawdown | −20.7% | −27.3% |
| annual turnover | 3.23× | 0 |
| average gross exposure | 93.6% | 100% |

### Section 8's pre-committed verdict: **ABANDON**

| clause | result |
|---|---|
| net Sharpe exceeds 0.40 | ✗ 0.352 |
| sign positive in ≥ 9 of 12 instruments | ✓ 10/12 standalone, 11/12 by P&L contribution |
| exceeds buy-and-hold by ≥ 0.15 | ✗ −0.062 |
| **fails to beat buy-and-hold at all** | **✗ 0.352 vs 0.414 → triggers abandonment** |

The strategy is not broken. It did almost exactly what section 9 said it would:

| expectation of record | observed |
|---|---|
| realistic net Sharpe 0.3–0.6 | 0.352 ✓ |
| above 0.8 means a bug | 0.352, no bug indicated ✓ |
| roughly one year in three is a losing year | 5 of 19 = 26% ✓ |
| underwater periods of 3–5 years are normal | longest run 4.2 years ✓ |
| will underperform a plain index fund in most years | underperformed SPY in **16 of 19** years (84%) ✓ |

Every prediction the document made in advance came true, including the one that kills
it. Plain SPY over the same window earned an excess Sharpe of **0.596**, comfortably
ahead of both.

The three years the strategy *did* beat SPY are worth naming: **2008 (+2.9% vs
−32.5%), 2011 (+4.9% vs +1.9%) and 2022 (−7.2% vs −18.2%)**. That is the entire case
for trend following, arriving exactly on schedule — it is a crisis diversifier and it
behaved like one. But section 8 asked whether it beats equal-weight buy-and-hold on
standalone risk-adjusted return over the whole sample, and the answer is no. A rule
asking "does this improve a portfolio I already hold?" would have been answered
differently by the same data. That rule was not the one pre-registered, and rewriting
it now is exactly what section 8 exists to prevent.

**Cost sensitivity** (section 5's ladder; 5 bps is the headline):

| bps/side | 0 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Sharpe | 0.374 | 0.365 | **0.352** | 0.330 | 0.287 |
| vs buy-and-hold | −0.040 | −0.049 | **−0.062** | −0.083 | −0.127 |

Costs are not what sinks it. At literally zero transaction cost the strategy still
loses by 0.040, so trading costs account for about a third of the shortfall and
eliminating them entirely would not produce a win. The verdict is ABANDON at every
rung of the ladder, including the free one.

### The one choice that does change the verdict

Section 8 turns on "net Sharpe over the full sample" and never defines the sample.
The twelve ETFs list fourteen years apart, and the two defensible windows disagree:

| window | benchmark: buy & hold | rebalanced monthly | rebalanced daily |
|---|---|---|---|
| **2008-02-29 → (all 12 listed)** | **−0.062 ABANDON** | −0.039 ABANDON | −0.072 ABANDON |
| 1993-01-29 → (full history) | +0.048 INCONCLUSIVE | +0.051 INCONCLUSIVE | +0.030 INCONCLUSIVE |

Strategy Sharpe is 0.352 on the short window and 0.540 on the long one. The long
window is not a better test: 1994–2007 averages about **one** live instrument — SPY
alone, pinned at the 0.25 per-instrument cap — and scores 1.13 there, which is a
levered-down index fund with a momentum filter, not the twelve-instrument diversified
system section 2 describes.

How the benchmark is constructed turns out **not** to matter (all three constructions
agree within 0.033 and all three abandon on the headline window). An earlier draft of
this document claimed it spanned the whole decision space, including one SUPPORTED
cell. That claim was an artifact of a bug in my own benchmark code — it deleted the
market's return on every re-levelling day, 403 days of the full history — found by
the adversarial review and fixed. See defect 11 below. The verdict-fragility story is
real but narrower than I first reported: **one** unspecified choice moves the answer,
not three.

A third choice, the covariance estimator inside `k(t)`, moves the Sharpe by 0.031
(0.352 full-covariance against 0.383 zero-correlation) without flipping this verdict.
That is the same size as the margin, so on slightly different data it would have
decided the outcome.

---

## 2. The three numbers the brief asked for

**Noise-test Sharpe.** The real engine on independent driftless geometric random
walks: **−0.055** across 8 seeds × 6000 days (sd 0.112); equal-weight buy-and-hold on
the same data, −0.027. Both inside the ±0.2 gate. The counterpart check matters as
much: on random walks with +8%/yr drift the same engine earns **+1.216**, so it is
honest rather than merely dead.

**Walk-forward train-vs-live Sharpe gap.** 15 windows, 3y in-window / 1y forward,
stepping 1y: mean in-window **+0.307**, mean forward **+0.384**, **gap −0.077**.
Negative — the forward periods were marginally *better* than the windows preceding
them, which is the expected shape when nothing is fitted; its absence would have been
the alarming result. 12 of 15 forward windows were positive, and the spread across
them (−1.88 to +1.70) is the honest picture of how little a single year tells you.

**Feasibility at $1,000.** It does not work, and not marginally:

- **6 of 12 instruments are holdable.** SPY, EFA, EEM, GLD, SLV and VNQ get **zero
  shares**, because one share costs more than their entire target allocation.
- **The whole equity sleeve disappears.** A trend-following portfolio with no SPY, no
  EFA and no EEM is not the strategy in section 2. Three of five sleeves (Equity,
  Commodities, Real assets) are wholly or partly absent.
- **56.8%** of the intended risk budget is deployed; **43.2%** of the account sits in
  cash it cannot use.
- Weight tracking error against the design: **0.1494** (L2), against 0.0077 at
  $100,000 — roughly twenty times worse.
- **The minimum account at which all twelve become holdable is $10,163**, set by GLD
  at $398.55 a share on a 3.92% target weight.

| account | holdable | risk budget deployed | tracking error | missing |
|---|---|---|---|---|
| $1,000 | 6/12 | 56.8% | 0.1494 | SPY, EFA, EEM, GLD, SLV, VNQ |
| $5,000 | 10/12 | 79.4% | 0.0921 | SPY, GLD |
| $25,000 | 12/12 | 96.1% | 0.0173 | — |
| $100,000 | 12/12 | 98.6% | 0.0077 | — |

Full tables in `FEASIBILITY.md`.

---

## 3. Things I built that I do not think will work as intended

### 3.1 The notional guard makes the paper path non-functional at $100,000

The selected guard profile caps a single run at **$25,000** of notional. The paper
account holds **$100,000**. Building the initial book is ~**$98,922** and is refused
outright, and it is not only the first trade:

| account | median rebalance | 90th pct | max | blocked by the $25k cap |
|---|---|---|---|---|
| $100,000 | $23,186 | $56,238 | $97,115 | **49% of the 222 monthly rebalances** |
| $25,000 | $5,797 | $14,060 | $24,279 | **0%** |

The guard does exactly what it was told. What it was told is sized for a $25,000
account, not the $100,000 one it is guarding. As built, about half of all rebalances
would refuse to trade and the book would drift arbitrarily far from target. Two clean
fixes: run the paper account at $25,000, or raise `max_gross_notional` to ~$150,000.
I left the chosen value in place rather than quietly retuning a threshold;
`tests/test_runner.py::test_conservative_notional_cap_blocks_a_hundred_thousand_dollar_account`
pins the conflict so it cannot be forgotten.

The other three guards are fine **after** the staleness bug in defect 13 was fixed —
before that, a second, independent, roughly-coin-flip blocker was stacked on top of
this one and I had described the guard as "fine" without checking.

Related, and by design rather than error: the 15% drawdown halt is **tighter than the
strategy's own 20.7% worst historical drawdown**. A faithful paper run is therefore
*expected* to halt at some point, and that halt will be a normal bad stretch rather
than a malfunction.

### 3.2 The anti-lookahead test the brief specifies does not, on its own, prove anything

The brief asks for "a test proving a strategy fed `close.shift(-1)` earns no abnormal
return". It exists and passes — but it passes on driftless noise **whether or not the
engine lags anything**, because a 252-day momentum signal shifted by one bar is nearly
the same signal and noise contains no return to earn either way. Taken literally it
also cannot pass for a *correct* engine on real data: a signal that genuinely knows
`close(t+1)` while the fill happens at `open(t+1)` captures the intraday move, and no
amount of correct lagging can remove lookahead baked into the signal before the engine
saw it.

Two tests do the real work, and both were sharpened after the adversarial review found
the first of them was blind to exactly the leak it was supposed to catch (defect 14):

- **`test_no_future_bar_can_change_a_past_decision`** — rewrite every price from bar
  *j* onward and assert the past is bit-identical. Realised quantities are compared
  strictly before *j*; **target weights are compared up to and including *j***,
  because the target for a rebalance on *j* is computed from the signal at *j−1* and
  a one-bar leak moves precisely that row and nothing earlier.
- **`test_the_harness_detects_the_engines_own_lag_switch`** — the negative control.
  The identical harness pointed at the engine with its lag removed **must fail**.
  Without it, a blind harness and a clean engine look the same from outside.

The load-bearing check on the lag itself uses the rebalance-day return rather than the
full-sample Sharpe, because monthly rebalancing dilutes one bar of foresight across
twenty bars of noise: unlagged, the rebalance-day mean return is **+0.73% (t ≈ +25 to
+31)**; lagged it is −0.07% (t ≈ −1.4 to −2.3, and negative because that is the day
the 5 bps is paid). The brief's suggested "Sharpe > 3" is unreachable through the
full-sample statistic on this engine, and asserting it would be asserting something
false.

### 3.3 The result is not statistically significant even taken entirely at face value

With `configurations_tried = 1` there is no selection bias to deflate, so the deflated
Sharpe collapses to the probabilistic Sharpe against zero: **PSR(0) = 0.9333**, below
the 95% threshold. Over 18.4 years and 4,646 observations a Sharpe of 0.352 still
cannot be distinguished from zero at conventional confidence. Section 9 said so in
advance ("at Sharpe 0.5, distinguishing this edge from luck requires ~16 years"); this
is that sentence arriving as a number. Nothing changes if the strategy had beaten the
benchmark — it would have been an insignificant win rather than an insignificant loss.

The validation module demonstrates its own point: sweeping 100 configurations
(lookback × EWMA halflife × long-only/long-short) on pure noise manufactured in-sample
Sharpes above 1.0 on 3 of 6 independent datasets, peaking at **+1.805**. The deflated
Sharpe called none of them significant, and on 3 of those 6 an **uncorrected** PSR
would have declared noise significant (up to 0.9991). That gap is the whole value of
the correction. Worth noting: an earlier version of this sweep varying only the
lookback produced deflated Sharpes above 0.95 on pure noise. The deflation term
depends on how much the trials differ from one another, so a hundred nearly-identical
lookbacks under-corrects. On a narrow sweep, this statistic will lie to you.

### 3.4 The per-instrument cap silently switches off the strategy's only adaptive component

Section 4 says "volatility is the only quantity permitted to change position sizes".
In practice **40% of live positions sit pinned at the 0.25 cap**, and in **12% of
rebalances every single active instrument is pinned** — at which point clipping them
all to 0.25 and rescaling to gross 1.0 gives `1/n_active` each, *identical whatever
the volatilities were*. On those months the vol targeting does nothing at all and the
book is simply equal-weight. This is a faithful consequence of the pre-registered
rule, not a bug, but it means the document's "only adaptive component" is inert about
an eighth of the time and partially overridden most of the time.

### 3.5 Both caps hold at trade time; only gross holds continuously

Target weights at every rebalance satisfy both caps exactly (max |wᵢ| =
0.250000000, max gross = 1.000000000). Between monthly rebalances weights drift with
prices, so the *held* |wᵢ| exceeds 0.25 on 7.1% of days, peaking at 0.265 — inherent
to monthly rebalancing and unremovable without daily trading, which section 5 forbids.
Gross is different: gross above 1.0 would mean negative cash, which a cash account
genuinely cannot do, so it is enforced continuously and never exceeds 1.0.

### 3.6 Live data is much shorter than research data

The backtest uses Yahoo total-return history (1993–2026). The **broker's** own market
data reaches back only to 2016-01-04 on this plan, and the live path reads from
Alpaca. The 252-day signal and 30-day EWMA are unaffected, but backtest and live are
not computing from byte-identical inputs and I could not verify the two vendors agree
over the full sample. The last closes agree exactly on 2026-08-18 (SPY 767.45, GLD
398.55, all twelve) — reassuring, but that is one day.

The cached Yahoo panel also contains six missing closes (EFA, EEM and DBC, July 2026).
They are handled correctly now (defect 10) but they are real vendor holes in the data
the headline number is computed from.

---

## 4. Decisions the pre-registration does not make

Each was settled before the headline run; none is a strategy parameter in section 6's
frozen list.

| # | What the document leaves open | What I did | Why, and what it costs |
|---|---|---|---|
| 1 | **Sample window.** §7 says "full-sample" and never defines the sample. | Headline = 2008-02-29 onward, the first bar all twelve have a full lookback. | Operator's decision, made before any result existed. **This is the one choice that changes the verdict**: abandon on the short window, inconclusive on the long one. |
| 2 | **Risk-free / cash yield.** §4 mandates a cash account holding idle cash; nothing says whether it earns anything or whether "net Sharpe" is an excess return. | Idle cash accrues at the 13-week T-bill (`^IRX`); Sharpe is in excess of the same rate, applied identically to strategy and benchmark. | Operator's decision. The strategy runs at 7.5% vol and the benchmark at 10.1%, so this does **not** cancel — worth ~0.2–0.4 of Sharpe against a 0.15 margin. A convention chosen after the document was signed, and the largest unpinned lever in the result. |
| 3 | **Ex-ante portfolio vol in k(t).** §4 gives no covariance estimator. | Full EWMA covariance at the halflife 30 §4 already specifies; `k` unbounded. | **Not free — I initially said it was, and was wrong.** Zero-correlation gives median k 4.54 against 3.12 and Sharpe 0.383 against 0.352. It does not flip this verdict but is the same size as the margin. |
| 4 | **Order of k and the two caps.** | k → clip each \|wᵢ\| to 0.25 → scale gross to 1.0. | The only order in which both caps hold at the end. Clipping is not scale-homogeneous, which is why `k` does **not** cancel even when the gross cap binds. |
| 5 | **Prices: adjusted or not.** §3 never defines *P*. | Split- and dividend-adjusted (total return), for signal, vol and P&L alike. | Required by the cited Moskowitz–Ooi–Pedersen reference, and unavoidable: IEF and UUP are the two largest positions and both have roughly a 1-in-4 chance of the 12-month sign flipping on a 4% dividend yield. |
| 6 | **Monthly rebalance timing.** §5 says both "fill at open of t+1" and "rebalance on the first trading day of the month". | Trade at the open of the first trading day; signal from the previous close. | The only reading consistent with both sentences. |
| 7 | **Drift band vs the caps.** The band can leave a book above gross 1.0. | Re-apply both §4 caps to the post-band weights. | Without it the held book reaches gross **1.11, above 1.0 on 34% of days** — negative cash in an account §4 calls a cash account. The cost is that scaling touches instruments the band had decided not to trade. |
| 8 | **Benchmark construction.** §8 never says how often, if ever, the basket is rebalanced. | Literally: equal dollars placed once at the **start of the reported window**, then left to drift. | Cadence is worth at most 0.033 here and changes no verdict. *When* it is bought matters more: a basket bought in 1993 and sliced to 2008 scores 0.384 against 0.414 for one actually bought in 2008. |
| 9 | **"Sign positive in 9 of 12."** Three defensible readings. | Standalone per-instrument trend return (10/12), P&L contribution reported alongside (11/12). | Both clear the ≥9 hurdle, so the choice does not affect the verdict. |
| 10 | **Walk-forward window lengths.** §7 says "rolling windows" with no sizes. | 3y in-window / 1y forward / 1y step, exposed as CLI arguments. | Not in §6's frozen list, and nothing is fitted, so no window choice can change a position. |
| 11 | **Deflated Sharpe at N=1.** The Bailey–López de Prado E[max SR] term contains Z⁻¹(0) and diverges. | SR\* = 0, so DSR collapses to PSR(0). | With one configuration there is no selection bias to deflate. Implemented explicitly rather than papered over. |
| 12 | **Guard thresholds.** Neither document contains one. | Operator's "conservative" profile: 15% drawdown, $25k notional, 12 orders/day, 1 trading day staleness. `GuardConfig` has **no defaults** — a missing threshold cannot be constructed. | See 3.1: the notional limit is incompatible with the account size. |

---

## 5. Defects found and fixed during the build

Every one was caught by a test or by adversarial review, and every one was real.

**Found while building, by the tests written alongside each module:**

1. **The held book breached the gross cap.** The drift band left `sum|wᵢ|` above 1.0
   on **34% of days**, peaking at **1.110** — negative cash in an account section 4
   calls a cash account. The per-instrument cap leaked the same way, reaching 0.282 on
   13% of days. Both §4 caps are now re-applied to the post-band weights.
2. **Transaction costs were financed by borrowing.** Costs were deducted *after*
   allocating, leaving cash at −(cost) and gross at 1.000482: invisible leverage at
   exactly the cap. Costs are now paid before the book is established.
3. **`sharpe()` returned 7.3 × 10¹⁶ for a constant series.** `pd.Series([0.001]*252).std()`
   is 2.2 × 10⁻¹⁹, not 0.0, because 0.001 is not exactly representable, so an exact
   `sd == 0` guard missed it. Now scale-relative, in `sharpe` and `sortino` alike.
4. **A 15.00% drawdown tripped a 15% limit** — `1 − 85/100` is 0.15000000000000002.
5. **`Config.sleeves` was mutable.** A frozen dataclass freezes the binding, not the
   dict; `cfg.sleeves["Equity"] = ("QQQ",)` silently rewrote the universe for every
   later reader, process-wide. Now a `MappingProxyType`.
6. **A stale price history would have rebalanced daily.** If the history stopped
   before the session's month, that month contained no earlier bar, so *every* day
   looked like the first trading day.
7. **Guard refusals latched the halt flag.** A stale-data refusal over a holiday
   weekend would have needed manual clearing, training an operator to clear the flag
   reflexively and destroying its value. Guard violations now stop the run without
   latching; only `DrawdownBreach` is sticky.
8. **§8's second abandonment clause was parsed away.** Config held the 0.15 threshold
   but not "or if it fails to beat equal-weight buy-and-hold at all" — the limb that
   actually decides this result. Its presence is now asserted at parse time.
9. **The §8 benchmark was not buy-and-hold at all.** It recomputed equal weights
   *every day*, with a dead `monthly_rebalance` argument whose two branches were
   identical. Rewritten as a true buy-and-hold with drifting weights.

**Found by adversarial review, after all of the above were already fixed.** An
independent reference implementation was written from scratch and hunted for
disagreements; four further reviewers attacked the invariants, the guards and this
document's own numbers.

10. **A missing close re-marked the position to its start-of-month price.** The
    vectorised drift fell back to a growth factor of 1.0 wherever the close was NaN,
    marking the position at whatever it was worth at the *open of the current month*.
    The six holes in the cached data each produced a spurious ±0.5% one-day return,
    and the one landing before a rebalance became permanent, because DBC was then
    *traded* at a stale 26.58 instead of its actual 29.32. Positions are now valued
    off a forward-filled series while the signal keeps using the raw one.
11. **The benchmark deleted the market's return on every re-levelling day.** It
    referenced the re-level bar's own close instead of the previous one, so that day's
    move was never applied — 403 zeroed days across the full history. This was what
    manufactured the SUPPORTED cell in the earlier draft of section 1.
12. **The benchmark was bought in 1993 and only then sliced to the reported window,**
    so the "equal-weight" basket reported over 2008–2026 actually held 5.51%–10.63%
    per instrument. Worth 0.030 of Sharpe (0.384 against 0.414), in the direction
    that flattered the strategy. Every caller now passes the reporting window.
13. **CRITICAL: the staleness guard counted calendar days and would have blocked 48%
    of all monthly rebalances.** The strategy trades on the first trading day of the
    month, a Monday about half the time, whose newest close is Friday's — three
    calendar days old, one *trading* day old. Staleness is now counted in trading days
    against the broker's own `/v2/calendar`, with a business-day fallback that
    over-counts across holidays and therefore errs towards refusing.
14. **CRITICAL: the live runner read a parquet cache nothing ever refreshed,** so the
    paper bot would have frozen on day one's data for ever, with no CLI to recover.
    It now re-fetches every run.
15. **The no-lookahead harness could not detect a one-bar leak** — including the
    repo's own `_unsafe_disable_execution_lag`. A strict `< cut` comparison excluded
    the one row a one-bar leak moves. Fixed, with a negative control that fails if the
    harness ever goes blind again.
16. **Invariant 5 was violated:** `run_once` built its state objects *outside*
    `fail_closed`, so a truncated `high_water.json` from an earlier crash raised
    without setting the halt flag — and that file was written non-atomically while
    `halt.json` was written atomically. Both fixed.
17. **A broker failure mid-batch lost every prediction.** Predictions were written
    after all submissions succeeded, so a failure on order five of ten left four real
    fills at the broker and nothing in the divergence log — the one mechanism whose
    job is to notice exactly that. Each prediction is now written before its order.
18. **Reported turnover measured price drift, not trading.** Differencing the *daily
    held* weight matrix counts days when nothing is traded; a two-asset book bought
    once and never touched scored 2.4. The reported 4.54× was 1.40× actual trading
    (22.7 bps/yr implied against 16.6 observed). Turnover is now computed from
    executed trades (**3.23×**, implying 16.2 bps/yr — consistent) and the old
    quantity survives as `annual_weight_churn`.
19. **A one-day vendor hole liquidated an instrument out of the benchmark.**
    "No close printed today" was read as "not investable", so on 2026-07-21 the basket
    sold EFA, EEM and DBC and bought them back two days later.
20. **Cached price data was never validated.** `load_prices` sanity-checked a fresh
    fetch but returned cached parquet unchecked — and the cache is the path every
    backtest actually takes.
21. **A corrupt halt flag could not be cleared.** `is_halted()` is a file-existence
    test but `record()` parsed the JSON, so a flag truncated by a crash blocked
    trading *and* crashed both `--status` and `--reset-halt`. Recovery meant deleting
    a file by hand — exactly the reflexive manual intervention defect 7 exists to
    avoid. An unreadable flag now reports itself as `"unreadable halt flag"`, still
    blocks, and still clears.
22. **The daily order cap silently half-disabled on a conforming broker.**
    `get_orders_since` was not on the `Broker` ABC, and both callers guarded it with
    `hasattr` and degraded to zero — so an adapter that did not implement it would let
    a re-run submit a second full set of orders on top of a correct book. It is now on
    the ABC returning `None`, and `None` refuses rather than reads as zero.
23. **A non-paper broker was retried silently for ever.** `check_broker_is_paper`
    raised a non-sticky `GuardViolation`, so the one condition hard invariant 4 exists
    for left no persistent record and retried on every schedule. It now raises
    `NotAPaperAccount`, which latches the halt flag.
24. **The divergence reconciler used a fixed 45-day wall-clock lookback**, so a
    prediction outstanding longer became permanently unreconcilable and emitted a
    false "not found at broker" note on every subsequent run. Reachable in normal
    operation, since the 15% drawdown halt is tighter than the strategy's own 20.7%
    historical drawdown. The lookback now reaches back to the oldest outstanding
    prediction.

---

## 6. Where I think the pre-registration itself is weak

Left unchanged, as instructed.

- **Line 3, `Committed on: ____________`, is blank**, and the document forbids running
  any backtest until it is filled. Mitigating: git shows the file was committed once
  (`dccdea4`) and never modified, which is stronger evidence of a freeze than a
  self-typed date. But the document's own gate was not satisfied on its own terms.
- **The headline metric is undefined.** "Net Sharpe over the full sample" (§8) with no
  sample bounds, no risk-free convention and no data source is not a pre-registered
  metric. The document pins the lookback to the day — "Not 200. Not 189. Not
  'optimised.'" — and then leaves the evaluation window open, which is the choice that
  actually decides the answer. It protected a parameter that was never in danger and
  left the evaluation criteria unguarded.
- **§8's benchmark is underspecified.** "Equal-weight buy-and-hold of the same 12
  ETFs" gives no cadence and no handling of instruments that did not exist yet.
  Measured here the cadence is worth only 0.033, but *when* the basket is bought is
  worth 0.030 — and neither is stated.
- **§4's caps quietly override §4's own thesis.** "Volatility is the only quantity
  permitted to change position sizes" is contradicted in practice by the 0.25 cap,
  which pins 40% of positions and makes the book exactly equal-weight in 12% of
  months. The document does not acknowledge that its cap and its adaptive component
  compete.
- **§9's "above 0.8 means a bug" is a good instinct pointed at the wrong failure.** It
  guards against a coding error, but the thing that most inflates this strategy's
  Sharpe is the sample window: 0.540 on the full history against 0.352 on the
  all-twelve one. A threshold on the headline number cannot tell a bug from a badly
  chosen window, and here the window was the larger effect.

---

## 7. Verification

```
pytest -q                                            442 passed
python scripts/feasibility.py --equity 1000          6/12 holdable, 56.8% deployed
python scripts/run_backtest.py --synthetic --seed 0  strategy −0.055, B&H −0.027, PASS
python scripts/run_backtest.py --validate            all four protocol steps, gate PASS
python scripts/run_live.py --dry-run                 plans, submits nothing
```

Every test is offline and deterministic; the suite imports no network library and
never touches the vendor modules. No orders have been submitted to the paper account.

**Independent re-derivation.** The accounting was checked against a reference
implementation written from scratch as a naive day-by-day loop over shares and cash —
its own rebalance calendar, drift band, EWMA estimates, cash accrual and cost timing,
sharing nothing with the engine but the signal and sizing functions. On hole-free data
the two agree to **3.6 × 10⁻¹⁵** relative on the equity curve across all 8,445 bars,
3.9 × 10⁻¹⁶ on held weights, and bit-identically on target weights at all 403
rebalances, with the excess Sharpe matching at every rung of the cost ladder. That
disagreement-hunt is what surfaced defects 10–12 and 18–19 above.
