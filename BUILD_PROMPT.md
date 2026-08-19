# Claude Code build prompt — diversified trend following

Put `PREREGISTRATION.md` in an empty directory. Paste everything below the line into Claude Code in that directory.

---

Build a complete, tested implementation of the trend-following strategy specified in `PREREGISTRATION.md`, which is in this directory. Read it first.

**`PREREGISTRATION.md` is the sole source of truth for every parameter.** Do not invent, tune, substitute, or "improve" any number in it. If a parameter you need is missing, stop and ask me — do not pick one. If you believe a parameter is wrong, say so in your final report and leave it unchanged.

Work autonomously through the build order below. Do not stop for approval between steps. Do stop if an acceptance gate fails: report the failure and what you think caused it rather than working around it.

## Hard invariants

1. **No lookahead, structurally enforced.** Signal from bar `t`'s close fills at bar `t+1`'s open, via a `.shift()` inside the engine. Write a test proving a strategy fed `close.shift(-1)` earns no abnormal return.
2. **One signal implementation.** Backtest and live import the same function object. Test that hashes the signal module and fails if two definitions exist.
3. **Broker is the only source of truth for positions.** Never infer from local state, logs, or believed fills. Query every run.
4. **Paper only.** No live code path. Live requires a config file that does not exist in this repo and that you must not create.
5. **Fail closed.** Unhandled exception in the execution path sets a persistent halt flag and stops. Clearing requires an explicit CLI command.
6. **Offline tests.** No network calls in the test suite. Fixtures only.

## Architecture

```
trendbot/
├── PREREGISTRATION.md          # read-only input, never edit
├── pyproject.toml
├── trendbot/
│   ├── config.py               # parses PREREGISTRATION.md into a frozen dataclass
│   ├── signal.py               # THE strategy. imported by both paths.
│   ├── engine/
│   │   ├── backtest.py         # vectorised, shift-enforced, cost-aware
│   │   ├── metrics.py          # CAGR, Sharpe, Sortino, maxDD, turnover, exposure
│   │   └── validation.py       # noise test, IS/OOS, walk-forward, deflated Sharpe
│   ├── brokers/
│   │   ├── base.py             # ABC: get_positions, get_equity, submit, get_clock
│   │   ├── alpaca.py           # paper endpoint, hardcoded paper=True
│   │   └── schwab.py           # stub only. raise NotImplementedError with a note
│   ├── sizing.py               # vol estimate, weights, caps, whole-share rounding
│   ├── guards.py               # halt state, drawdown, notional, order cap, stale data
│   ├── runner.py               # scheduled entrypoint
│   └── monitor/divergence.py   # predicted vs realised fill, every run
├── scripts/
│   ├── feasibility.py
│   ├── run_backtest.py
│   └── run_live.py             # --dry-run, --reset-halt
└── tests/
```

`config.py` parsing the pre-registration file directly is deliberate. Parameters live in one human-readable place, and changing them means editing a signed document.

## Build order — each gate must pass before the next step

**Step 1 — config + signal.** Parse the pre-registration into a frozen dataclass. Implement the signal exactly as written.
*Gate:* config round-trips; signal returns the documented values on a hand-built fixture.

**Step 2 — engine + metrics.**
*Gate:* on a synthetic zero-drift random walk, buy-and-hold Sharpe within ±0.2 of zero, and a `close.shift(-1)` strategy earns nothing.

**Step 3 — sizing, including the small-account problem.** Vol estimate, weights, caps per the spec. Then whole-share rounding, which is where a small account breaks: compute integer share counts, and for any instrument where the target dollar allocation is less than one share, the position is zero and that sleeve is silently absent from the portfolio.
*Gate:* a test at $1,000 and at $100,000 showing the realised portfolio weights versus target weights, and the tracking error between them.

**Step 4 — `scripts/feasibility.py`.** Takes an account size and reports, using recent prices:

| column | meaning |
|---|---|
| ticker | instrument |
| price | last close |
| target $ | allocation at equal risk weight |
| shares | integer shares affordable |
| realised $ | what you'd actually hold |
| drift % | realised vs target |
| holdable | yes/no |

Then a summary: how many of the 12 sleeves are actually holdable at this account size, what fraction of the intended risk budget is deployable, and the minimum account size at which all 12 become holdable. Run it at $1,000, $5,000, $25,000 and $100,000 and put the tables in `FEASIBILITY.md`.

Do not soften this output. If $1,000 cannot run the strategy, say so in plain language with the numbers.

**Step 5 — validation module.** Noise test, IS/OOS, walk-forward, deflated Sharpe. Configurations tried is a required argument with no default; for this strategy it is 1.
*Gate:* sweeping 100 parameter combinations on pure noise produces a best in-sample Sharpe above 1.0 while the deflated Sharpe is not significant. This module exists to tell me my results are fake.

**Step 6 — broker abstraction + Alpaca paper adapter.** `paper=True` hardcoded. Schwab adapter raises `NotImplementedError` with a comment noting that Schwab's API has no paper environment and that its 7-day refresh token makes unattended operation impractical.

**Step 7 — guards, divergence log, runner.** Deterministic `client_order_id`. Open orders block new ones. Drawdown, notional, daily order cap, stale-data refusal, halt-on-exception.
*Gate:* mocked-broker tests proving (a) a repeated run submits nothing, (b) an exception sets the halt flag, (c) a halted state blocks all subsequent runs, (d) positions are read from the broker, never from local state.

## Verify before reporting done

```
pytest -q
python scripts/feasibility.py --equity 1000
python scripts/run_backtest.py --synthetic --seed 0
python scripts/run_backtest.py --validate
python scripts/run_live.py --dry-run
```

Write `FINDINGS.md` containing: the noise-test Sharpe, the walk-forward train-vs-live Sharpe gap, the feasibility summary at $1,000, and a plain-language statement of anything you built that you think will not work as intended.

## Do not

- Do not tune any parameter. The deflated Sharpe assumes exactly one configuration was tried; make that assumption true.
- Do not add a live trading path, a dashboard, a web UI, or an ML model.
- Do not use `try/except: pass` anywhere in the execution path.
- Do not report success on a gate you did not actually run.
- Do not edit `PREREGISTRATION.md`.
