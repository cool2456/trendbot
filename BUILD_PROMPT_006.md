# Claude Code build prompt — experiment 006 (multi-strategy risk allocation)

Run from the `trendbot` repo with `PREREG_006.md` committed and tagged.

---

Implement experiment 006, specified in `PREREG_006.md`. Read it first, including the
header note about the understated multiple-testing correction.

The engine, panel protocol, metrics and validation modules from 001–005 are trusted.
The 001 regression test must continue to pass bit-identically.

`PREREG_006.md` is the sole source of truth. Do not invent, tune or substitute any
parameter. Stop and ask on genuine ambiguity — that behaviour has caught the sample
window, the cash rate, the weight pipeline, the quantile splits, the market proxy and
the dollar-factor construction across five experiments, and every one moved a number.

Experiments 001–005 outputs are immutable. Write to `FINDINGS_006.md`.

## What is different about this experiment

There is **no new signal**. Every sleeve return series already exists in this repo,
produced by committed code that has already been validated. This experiment is
entirely portfolio construction on top of results you already have.

That makes the failure modes different from every prior run:

**Sleeve returns must come from the committed implementations, unmodified.** Do not
re-implement, re-parameterise, or "clean up" any sleeve. If a sleeve's returns are
not reproducible bit-identically from its own committed code, stop and report that
rather than proceeding — it would mean a prior result has drifted.
*Gate:* re-derive each sleeve's headline Sharpe and assert it matches its findings
document exactly (001: 0.352, 002: 0.463, 005: carry-corrected +0.209).

**Covariance estimation is the lookahead surface.** §3's EWMA must use only data
through each rebalance date. A full-sample covariance would be a lookahead that no
`.shift()` catches, and it would inflate the result substantially — an ERC portfolio
built with hindsight correlations is not a portfolio, it is an optimisation.
*Gate:* assert the covariance matrix at date t is unchanged when future data is
appended to the input.

**Date alignment across sleeves.** Three sleeves on three calendars — US equity
sessions for A and B, FX observation dates for C. Misalignment silently shifts one
sleeve's returns relative to the others, which corrupts the correlation estimate,
which is what §8's third clause gates on. Decide and state the alignment rule, and
confirm it introduces no lookahead.

**The ERC solve must be checked, not assumed.** Equal risk contribution has no
closed form for N > 2. Whatever solver you use, assert the realised risk
contributions are equal to within tolerance at every rebalance, and report the worst
deviation across the sample. A solver that silently fails to converge produces
plausible-looking weights that are not ERC.

## Build order

**Step 1 — sleeve reproduction.** Load all three sleeve return series from committed
code. Gate as above.

**Step 2 — date alignment**, per the section above. Report the rule, the intersection
window realised under §5, and the number of dates dropped per sleeve.

**Step 3 — covariance + ERC solver.** Gates as above: point-in-time assertion and
convergence assertion.

**Step 4 — noise test.** Run the ERC construction on **synthetic sleeve returns with
zero mean and a known covariance structure.** The portfolio Sharpe must be ≈ 0. Also
verify the construction recovers the known correlation structure. ≥8 seeds.

**Step 5 — benchmark construction**, §4. Identical machinery, benchmark inputs.
*Gate:* assert the benchmark uses the same solver, halflife, vol target and caps —
any asymmetry between portfolio and benchmark construction invalidates the
comparison.

**Step 6 — full-sample backtest**, one run, with §6's cost ladder. Report the
fraction of days the gross cap binds.

**Step 7 — §8's three clauses**, each reported as a number:
  (a) portfolio Sharpe minus benchmark Sharpe;
  (b) portfolio Sharpe minus best individual sleeve Sharpe;
  (c) realised mean pairwise sleeve correlation vs realised mean pairwise benchmark
      correlation, with the full correlation matrices for both.

Clause (c) is the mechanism test. Report both matrices in full, not just the means.

**Step 8 — attribution.** Realised risk contribution per sleeve over time, and the
weight path. If one sleeve dominates the risk budget, say so — an ERC portfolio that
is effectively single-sleeve is not testing §1's hypothesis.

**Step 9 — PSR / deflated Sharpe at `configs_tried = 5`**, with the header's caveat
restated: the correction is a floor, not the true burden.

**Step 10 — verdict** per §8, stated plainly. If it passes, state explicitly that
per the header a pass is suggestive only and requires out-of-sample confirmation on
data not used in 001–005.

## Do not

- Do not add, remove or substitute a sleeve. §2 is fixed.
- Do not re-implement or re-parameterise any sleeve.
- Do not use any performance-based input to allocation — not rolling Sharpe, not
  drawdown, not hit rate, not momentum of sleeve returns. §3 permits covariance and
  nothing else.
- Do not optimise the covariance halflife, vol target, or caps.
- Do not include 003.
- Do not modify experiments 001–005 outputs.
- Do not report a gate as passed without running it.

## Verify before reporting done

```
pytest -q
python scripts/run_backtest.py --experiment 001 --regression
python scripts/run_backtest.py --experiment 006 --sleeves
python scripts/run_backtest.py --experiment 006 --noise
python scripts/run_backtest.py --experiment 006 --validate
```

`FINDINGS_006.md` must contain: sleeve reproduction check with the three asserted
Sharpes, date alignment rule and dropped-date counts, covariance point-in-time and
ERC convergence assertions with worst deviation, noise distribution, benchmark
symmetry assertion, cost ladder, gross-cap binding fraction, §8's three clauses as
numbers with both full correlation matrices, risk-contribution attribution and weight
path, PSR at `configs_tried = 5` with the floor caveat, the §8 verdict, and a plain
list of anything you built that you believe does not work as intended.
