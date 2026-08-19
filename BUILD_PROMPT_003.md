# Claude Code build prompt — experiment 003

Run from the `trendbot` repo with `PREREG_003.md` committed and tagged.

---

Implement experiment 003, specified in `PREREG_003.md`. Read it first.

The engine, metrics, validation module and panel protocol from experiments 001 and
002 are trusted. The 001 regression test must continue to pass bit-identically.
Do not rewrite the engine. Extend it.

`PREREG_003.md` is the sole source of truth. Do not invent, tune or substitute any
parameter. Stop and ask on genuine ambiguity — that behaviour has now caught the
sample window, the cash rate, the weight pipeline and the quintile split, and
every one of them affected the verdict.

Experiment 001 and 002 outputs are immutable. Write to `FINDINGS_003.md`.

**Track everything in git.** Experiment 002 flagged that only four markdown files
were tracked, so the immutability guarantee was unenforced. Fix that first: verify
`trendbot/`, `tests/`, `scripts/` and all findings documents are tracked, and say
what was missing.

## The new risk in this experiment

001 and 002 ran on large, liquid ETFs. This one runs on ~350 individual equities,
which introduces failure modes the prior experiments could not have surfaced:

**Corporate actions are now dangerous.** An unadjusted 2:1 split reads as a −50%
return and lands the name in the bottom decile. An unadjusted spinoff or special
dividend does the same. Audit this explicitly before trusting any result: scan for
single-day moves beyond ±35% and reconcile each against a known corporate action.
Report the count found and how each was handled.

**Adjustment must be point-in-time.** A price series adjusted with today's factors
encodes future splits into past prices — the lookahead bug that no `.shift()`
catches. State plainly which adjustment convention the data source uses and whether
it is point-in-time. If it is not, say so and quantify the exposure rather than
proceeding quietly.

**Eligibility is per-date, not per-universe.** A name lacking 252 days of history at
date t is excluded from *that date's* ranking, not from the universe.

## Build order

**Step 0 — git hygiene**, as above. Confirm the 001 regression test still passes.

**Step 1 — universe construction.** Current S&P 500 constituents with continuous
data from 2008-01-01. Report exact count, per-ticker first date, and the list
excluded for insufficient history. Do not substitute.

**Step 2 — corporate action audit** per the section above. This gates everything
downstream.

**Step 3 — signal.** Identical formula to 002, decile sort instead of quintile.
*Gate:* hand-built fixture with known ranks matches by hand.

**Step 4 — noise tests, both variants**, ≥8 seeds each, with the factor attribution
that 002 introduced. Both must return ≈ 0.

**Step 5 — full-sample backtest.** One run. 10 bps headline, sensitivity at
0/5/10/20/40. Report annualised turnover.

**Step 6 — decile monotonicity.** Full D1→D10 forward return table, count of
adjacent inversions, D1−D10 spread with **t-statistic**. §8 requires t > 2.0 for
support and abandons below t = 1.0. Report the number, not an adjective.

**Step 7 — market-beta attribution.** Regress strategy excess return on market
excess return. Report beta, alpha, alpha's t-statistic. §8 abandons on negative
alpha regardless of Sharpe. This is now standard equipment, not a bonus.

**Step 8 — worst months.** §9 requires March–May 2009 and April 2020 to appear.
If they do not, report the implementation as suspect rather than explaining it away.

**Step 9 — PSR / deflated Sharpe at `configs_tried = 3`.** Not 2.

**Step 10 — verdict** against §8's four support clauses and four abandon clauses.

**Survivorship framing is mandatory in the verdict.** §2 states the bias flattens
the gradient, so a pass is conservative evidence and a failure is ambiguous. State
both readings. Do not select the flattering one after seeing the result.

## Do not

- Do not tune. `configs_tried = 3` must stay a true statement.
- Do not test additional formation windows, decile cuts, or universes.
- Do not add a short leg, a beta neutralisation, or a residual-momentum variant —
  each is a separate experiment with its own pre-registration.
- Do not modify 001 or 002 outputs.
- Do not report a gate as passed without running it.

## Verify before reporting done

```
pytest -q
python scripts/run_backtest.py --experiment 001 --regression
python scripts/run_backtest.py --experiment 003 --noise
python scripts/run_backtest.py --experiment 003 --validate
```

`FINDINGS_003.md` must contain: git tracking status, universe table, corporate
action audit, both noise distributions with factor attribution, full decile table
with inversion count and t-statistic, beta/alpha regression, turnover and cost
ladder, worst months, PSR(0) at `configs_tried = 3`, the §8 verdict with both
survivorship readings, and a plain list of anything you built that you believe does
not work as intended.
