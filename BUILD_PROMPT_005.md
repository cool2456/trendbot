# Claude Code build prompt — experiment 005 (currency momentum)

Run from the `trendbot` repo with `PREREG_005.md` committed and tagged, and a
FRED API key in `.env` as `FRED_KEY`.

---

Implement experiment 005, specified in `PREREG_005.md`. Read it first.

The engine, panel protocol, metrics and validation modules from experiments 001–004
are trusted. The 001 regression test must continue to pass bit-identically. Do not
rewrite the engine. Extend it.

`PREREG_005.md` is the sole source of truth. Do not invent, tune or substitute
any parameter. Stop and ask on genuine ambiguity — that behaviour has caught the
sample window, the cash rate, the weight pipeline, the quantile splits and the
market proxy across three experiments, and every one moved a verdict.

Experiments 001–004 outputs are immutable (004 is pending, not failed). Write to `FINDINGS_005.md`.

## The one thing most likely to produce a wrong answer

**FRED mixes quote conventions.** Some H.10 series are quoted as USD per foreign
unit (e.g. USD per euro), others as foreign units per USD (e.g. yen per USD). They
are not distinguishable from the series values alone.

If a subset is left inverted, every momentum rank involving those pairs is
backwards. **Nothing throws an error.** The backtest runs, the metrics compute, and
the answer is garbage.

Handle it explicitly, as Step 1, before anything else:

- Parse the direction from each series' FRED title/metadata, not from a hardcoded
  list you wrote from memory.
- Normalise everything to a single convention: the value of the foreign currency
  expressed in USD.
- **Verify each series against an independently known reference.** For at least
  three pairs with well-known levels, assert the normalised value on a specific
  historical date falls in a plausible range. State which pairs and which dates.
- Assert that no normalised series has a mean daily log return implying an
  implausible long-run drift (a sign of inversion).

Report the direction determined for every series in `FINDINGS_005.md`.

## Build order

**Step 1 — FRED ingestion and quote normalisation.** As above. This is a hard gate.
Do not proceed until every series direction is verified and reported.

**Step 2 — universe construction** per §2. Report final count, the full list, and
every series excluded for discontinuity with the reason. Series that stop mid-sample
are excluded entirely, not truncated.
*Gate:* no series in the universe has a gap longer than a plausible holiday run;
report the largest gap per series.

**Step 3 — returns.** Spot log returns per §4. Note that FX series have missing
values on US holidays and on foreign holidays that differ by country — decide and
state how these are handled, and confirm the handling introduces no lookahead.

**Step 4 — noise tests, both variants**, ≥8 seeds each, with the factor attribution
introduced in 002.

**Step 5 — full-sample backtest.** One run, with §5's cost ladder
(0/2/5/10/25 bps). Report annualised turnover.

**Step 6 — quintile monotonicity.** Full Q1→Q5 table, inversion count, Q1−Q5 spread
with **t-statistic**. §8 requires t > 2.0 for support and abandons below t = 1.0.
Report the number, not an adjective.

**Step 7 — beta attribution.** Against the equal-weight dollar factor (headline) and
against SPY (secondary — this checks whether currency momentum is a disguised equity
beta, which the 002 diagnostic would have missed).

**Step 8 — interest-rate diagnostic** per §4. Approximate excess returns using
short-term rate differentials from FRED and report how much the headline moves. This
is a diagnostic measuring the spot-only approximation, **not** a second
configuration — `configs_tried` stays at 4.

**Step 9 — worst months.** Report the five worst and whether they cluster around
known FX dislocations (2008 Q4, 2011 CHF, 2015 CHF de-peg, March 2020, 2022 GBP).
Per §9 there is no mandatory month here, but absence of any clustering is a warning
about the implementation and must be flagged as such.

**Step 10 — PSR / deflated Sharpe at `configs_tried = 4`.** Not 3, not 5 — see
`PREREG_005.md`'s header for why blocked experiment 004 does not count.

**Step 11 — verdict** per §8, stated plainly.

## Do not

- Do not tune. `configs_tried = 4` must remain true.
- Do not test additional formation windows, quantile cuts, or universes.
- Do not add carry, value, or a short leg — each is experiment 006.
- Do not modify experiments 001–004 outputs.
- Do not report a gate as passed without running it.

## Verify before reporting done

```
pytest -q
python scripts/run_backtest.py --experiment 001 --regression
python scripts/run_backtest.py --experiment 005 --noise
python scripts/run_backtest.py --experiment 005 --validate
```

`FINDINGS_005.md` must contain: the quote-convention table with the verification
method and reference checks, universe list and exclusions, holiday/missing-data
handling, both noise distributions with factor attribution, the full quintile table
with inversion count and t-statistic, both beta attributions, the cost ladder and
turnover, the interest-rate diagnostic delta, the five worst months with the
clustering assessment, PSR(0) at `configs_tried = 4`, the §8 verdict, and a plain
list of anything you built that you believe does not work as intended.
