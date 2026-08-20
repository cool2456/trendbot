# Claude Code build prompt — experiment 004

Run from the `trendbot` repo with `PREREG_004.md` committed and tagged, and a
Sharadar API key in `.env`.

---

Implement experiment 004, specified in `PREREG_004.md`. Read it first.

The engine, panel protocol, metrics and validation modules from experiments 001–003
are trusted. The 001 regression test must continue to pass bit-identically. Do not
rewrite the engine. Extend it.

`PREREG_004.md` is the sole source of truth. Do not invent, tune or substitute any
parameter. Stop and ask on genuine ambiguity — that behaviour has caught the sample
window, the cash rate, the weight pipeline, the quantile splits and the market proxy
across three experiments, and every one of them moved a verdict.

Experiments 001–003 outputs are immutable. Write to `FINDINGS_004.md`.

## What is genuinely new here

Every prior experiment used a static ticker list. This one uses a **rule evaluated
at each rebalance date over a universe that includes securities which no longer
exist.** That introduces failure modes the prior three could not have surfaced:

**Ticker reuse and ticker changes.** Symbols are recycled after delisting. A ticker
string is not a stable identifier — use the vendor's permanent security ID
throughout, and map ticker changes explicitly. If you key anything on the ticker
string, a 2009 company and a 2019 company can silently merge into one price series.

**Delisting is not a missing value.** A delisted stock has no price after its final
date. Forward-filling it, dropping it silently, or treating the gap as a zero return
all produce different and wrong answers. §3 specifies the treatment; implement it
literally and assert that no held position ever exits without an assigned return.

**Look-ahead through the universe rule.** The trailing-60-day dollar volume ranking
at date t must use only data through t. The `$5.00` floor uses the **unadjusted**
close at t, not today's adjusted price.

**Adjustment factors.** Confirm whether the vendor's adjusted series is
point-in-time or restated with current factors. If restated, that is a lookahead
channel no `.shift()` catches — quantify the exposure rather than proceeding.

## Build order

**Step 1 — free tier first.** Build ingestion, security-master handling, delisting
logic and universe construction against the free 30-name dataset. No paid data until
this works. Confirm the 001 regression test still passes.
*Gate:* a hand-built fixture with a synthetic delisting, a ticker change and a
reused ticker resolves correctly, verified by hand.

**Step 2 — universe construction.** Implement §2 literally. Report names per
rebalance date, the realised §6 start date, and the count of names entering and
leaving the universe each month.
*Gate:* no security appears in the universe on a date after its delisting date.

**Step 3 — delisting treatment.** Implement §3's table. Report the count of
delistings by reason across the sample.
*Gate:* assert every position exit has an assigned return; no silent drops.

**Step 4 — noise tests, both variants**, ≥8 seeds each, with factor attribution.

**Step 5 — full-sample backtest**, one run, with §3's delisting sensitivity ladder
(unknown bucket at 0% / −30% / −100%) and §5's cost ladder.

**Step 6 — the paired diagnostic, §10.** Run (A) point-in-time with delistings and
(B) survivors-only, identical code and dates. Report every §8 metric for both and
the A−B difference. This is a required output regardless of the verdict.

**Step 7 — decile monotonicity.** Full D1→D10 table, inversion count, D1−D10 spread
with t-statistic. Report for both (A) and (B).

**Step 8 — beta attribution** against SPY (headline) and equal-weight universe
(secondary). Beta, alpha, alpha t-stat for each.

**Step 9 — worst months.** §9 requires March–May 2009 and April 2020 to appear.
This check has failed twice. **If it fails a third time on a correctly constructed
universe, report the result as invalid rather than as a verdict** — per §9, that
indicates an implementation fault, and a verdict computed on a faulty
implementation is not a verdict.

**Step 10 — PSR / deflated Sharpe at `configs_tried = 4`.** Not 3.

**Step 11 — verdict** per §8, stated plainly, with the A−B gap reported alongside.

## Do not

- Do not tune. `configs_tried = 4` must remain true.
- Do not test additional formation windows, decile cuts, name counts, price floors
  or liquidity windows. Each is experiment 005.
- Do not choose the delisting treatment after seeing a result. §3 is fixed.
- Do not modify experiments 001–003 outputs.
- Do not add a short leg, beta neutralisation or residual-momentum variant.
- Do not report a gate as passed without running it.

## Verify before reporting done

```
pytest -q
python scripts/run_backtest.py --experiment 001 --regression
python scripts/run_backtest.py --experiment 004 --noise
python scripts/run_backtest.py --experiment 004 --paired
python scripts/run_backtest.py --experiment 004 --validate
```

`FINDINGS_004.md` must contain: the free-tier validation result, universe
construction audit with realised start date, delisting counts by reason plus the
sensitivity ladder, both noise distributions with factor attribution, the paired
A/B table with the A−B gap called out explicitly, decile tables for both arms with
inversion counts and t-statistics, both beta attributions, cost ladder, worst
months with the §9 validity determination, PSR(0) at `configs_tried = 4`, the §8
verdict, and a plain list of anything you built that you believe does not work as
intended.
