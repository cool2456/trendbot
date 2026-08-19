"""Experiment 001's recorded result, frozen as constants.

These are **outputs, not parameters.** They were produced by
``PREREGISTRATION.md`` run through ``trendbot.engine.backtest`` and are written down
in ``FINDINGS.md`` and ``results/backtest_headline.json``. Experiment 002 generalises
the engine those numbers came out of, so they become a regression target: if the
generalised engine cannot reproduce them, the generalisation broke something and no
new signal may be written until it is found.

Nothing here may be edited to make a test pass. A disagreement means the engine
changed, and the engine changing is the finding.

The full-precision values are kept because the gate is only stated to three decimal
places but the refactor is expected to be exact - it reuses the same sizing, the same
shift, the same calendar and the same accounting - and an exact match is much stronger
evidence than a rounded one.
"""

from __future__ import annotations

__all__ = [
    "EXPERIMENT_001_WINDOW",
    "EXPERIMENT_001_NET_SHARPE",
    "EXPERIMENT_001_BENCHMARK_SHARPE",
    "EXPERIMENT_001_GATE_DECIMALS",
    "EXPERIMENT_001_HEADLINE",
]

# The all-twelve window: the first bar on which every instrument in section 2 has a
# complete 252-day lookback, through the last bar of the dataset.
EXPERIMENT_001_WINDOW = ("2008-02-29", "2026-08-18")

# Net of 5 bps per side, in excess of the 13-week T-bill.
EXPERIMENT_001_NET_SHARPE = 0.35201034245670926

# Equal-weight buy-and-hold of the same twelve ETFs, bought once and held, measured
# in excess of the same rate.
EXPERIMENT_001_BENCHMARK_SHARPE = 0.41379333003397456

# The build gate is stated as "to at least three decimal places": 0.352 and 0.414.
EXPERIMENT_001_GATE_DECIMALS = 3

EXPERIMENT_001_HEADLINE = {
    "window": EXPERIMENT_001_WINDOW,
    "net_sharpe": EXPERIMENT_001_NET_SHARPE,
    "benchmark_sharpe": EXPERIMENT_001_BENCHMARK_SHARPE,
}
