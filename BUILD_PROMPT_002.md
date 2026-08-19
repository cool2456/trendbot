# Claude Code build prompt — experiment 002

Run from the existing `trendbot` repo, with `PREREG_002.md` committed and tagged.

---

Implement experiment 002, specified in `PREREG_002.md` in this repo. Read it first.

This is **not a new project**. The engine, metrics, validation module, broker layer,
guards and runner from experiment 001 already exist and are trusted — they survived
a from-scratch reimplementation that agreed to 3.6×10⁻¹⁵ on the equity curve, plus
24 defects found and fixed. Do not rewrite any of it. Do not re-litigate its
correctness. Extend it.

`PREREG_002.md` is the sole source of truth for every parameter. Do not invent,
tune, substitute or improve any number in it. If something is ambiguous, stop and
ask rather than choosing — that is what surfaced the sample-window and cash-rate
problems in experiment 001, and both mattered.

**Experiment 001's outputs are immutable.** Do not edit `PREREGISTRATION.md`,
`FINDINGS.md`, or any 001 result. Write to `FINDINGS_002.md`.

## The interface problem — handle this first

Experiment 001's strategy protocol is per-instrument: `DataFrame -> Series` of
target position. Cross-sectional momentum cannot be expressed that way, because a
position depends on where an instrument ranks *against the others* at that date.

Generalise the protocol to operate on a panel: `dict[str, DataFrame] -> DataFrame`
of target weights, indexed by date, columned by symbol. Then re-express experiment
001's time-series trend signal in the new interface.

**Gate (mandatory):** re-run experiment 001 through the generalised engine. It must
reproduce net Sharpe **0.352** and benchmark **0.414** over 2008-02-29 → 2026-08-18
to at least three decimal places. If it does not, the refactor broke something —
find it before writing any new signal. Commit this as a permanent regression test.

## Build order

**Step 1 — universe verification.** For each of the 41 tickers in §2, confirm
continuous daily data from 2008-01-01. Report per-ticker first available date. Drop
any that fail and record which and why. Do not substitute replacements. Report the
final count; the quintile size in §3 depends on it.

**Step 2 — the signal.** `momentum_i(t) = P_i(t-21) / P_i(t-252) - 1`, ranked
cross-sectionally at each monthly rebalance, top quintile equal-weighted long, rest
zero. Instruments lacking history are excluded from that date's ranking, not
assigned zero momentum.
*Gate:* on a hand-built fixture with known ranks, the selected set matches by hand.

**Step 3 — noise tests, both variants required by §7.**
(a) Independent random walks. Expected Sharpe ≈ 0.
(b) **Correlated random walks driven by a shared common factor.** Also ≈ 0. This
second test is the important one — a cross-sectional rule applied to correlated
series can manufacture apparent skill by loading on the common factor, and (a)
alone will not catch it. Run at least 8 seeds each and report the distribution.

**Step 4 — quintile monotonicity, per §8.** Sort all instruments into five
quintiles by momentum at each rebalance. Compute each quintile's equal-weighted
forward one-month return. Report the full Q1→Q5 series, whether the ordering is
monotonically decreasing, and the Q1−Q5 spread with its t-statistic.

This is a pass/fail gate on the hypothesis, not a diagnostic. Report it plainly.

**Step 5 — full-sample backtest.** One run. 5 bps headline, sensitivity at
0/2/5/10/20. Also report **annualised turnover**, since §5 flags that costs should
bite harder here than in 001. If the strategy only clears its thresholds at 0 bps,
say so explicitly.

**Step 6 — behavioural validation.** Identify the five worst months. Momentum
crashes are a known signature of this strategy and §9 predicts March 2009 and
April 2020 specifically. If those do not appear among the worst months, the
implementation may not be doing what it claims — flag it rather than explaining it
away.

**Step 7 — PSR / deflated Sharpe with `configs_tried = 2`.** Not 1. Report PSR(0)
and state plainly whether it clears 95%.

**Step 8 — verdict.** Evaluate against §8's three support clauses and three
abandon clauses. State supported / inconclusive / abandon. Do not soften it, do not
add mitigating commentary, do not suggest a variant that might do better.

## Do not

- Do not tune anything. `configs_tried = 2` must remain a true statement.
- Do not test additional formation windows, quintile cuts, or universes.
- Do not modify or reinterpret experiment 001's result.
- Do not add a live trading path, dashboard, or ML component.
- Do not report a gate as passed without running it.

## Verify before reporting done

```
pytest -q
python scripts/run_backtest.py --experiment 001 --regression
python scripts/run_backtest.py --experiment 002 --noise
python scripts/run_backtest.py --experiment 002 --validate
```

Write `FINDINGS_002.md` containing: the universe verification table, both noise
test distributions, the full quintile table with the monotonicity verdict, turnover
and cost sensitivity, the five worst months, PSR(0) at `configs_tried = 2`, the §8
verdict, and a plain-language statement of anything you built that you think does
not work as intended.
