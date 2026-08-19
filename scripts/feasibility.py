#!/usr/bin/env python3
"""Can this account actually hold the strategy?

Whole-share rounding is where a small account stops running the strategy that was
designed and starts running a different, smaller one - silently. If the target
dollar allocation for an instrument is less than the price of one share, the
position is zero, and that sleeve is simply absent from the portfolio. Nothing warns
you. This script is the warning.

The target weights used here are the section 4 risk-weighted allocation with every
instrument assumed to be trending - the fully deployed portfolio the pre-registration
describes. That is the right basis for an affordability question: the point is
whether the account can hold the designed book at all, not whether it can hold
whichever subset happens to be switched on this month.

Usage:
    python scripts/feasibility.py --equity 1000
    python scripts/feasibility.py --equity 1000 5000 25000 100000 --markdown FEASIBILITY.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trendbot.config import load_config  # noqa: E402
from trendbot.data import DataError, last_trade_prices, load_prices  # noqa: E402
from trendbot.sizing import (  # noqa: E402
    annualised_vol,
    ewma_covariance,
    target_weights,
    whole_share_allocation,
)


def designed_weights(cfg, history):
    """Section 4 weights with every instrument trending - the fully deployed book."""
    close = history.close[list(cfg.universe)]
    returns = close.pct_change(fill_method=None)
    sigma = annualised_vol(returns, cfg.ewma_halflife_days)
    cov = ewma_covariance(returns, cfg.ewma_halflife_days)
    date = close.index[-1]
    all_on = pd.Series(1.0, index=list(cfg.universe))
    return target_weights(
        date, all_on, sigma.loc[date], cfg, cov=cov.xs(date, level=0), covariance="full"
    ), sigma.loc[date]


def feasibility_table(weights: pd.Series, prices: pd.Series, equity: float) -> pd.DataFrame:
    alloc = whole_share_allocation(weights, prices, equity)
    drift = np.where(
        alloc.target_weights.abs() > 0,
        (alloc.realised_weights - alloc.target_weights) / alloc.target_weights.where(alloc.target_weights != 0) * 100.0,
        0.0,
    )
    return pd.DataFrame(
        {
            "ticker": weights.index,
            "price": prices.reindex(weights.index).to_numpy(),
            "target $": alloc.target_dollars.to_numpy(),
            "shares": alloc.shares.to_numpy(),
            "realised $": alloc.realised_dollars.to_numpy(),
            "drift %": drift,
            "holdable": np.where(alloc.shares.to_numpy() != 0, "yes", "NO"),
        }
    ).set_index("ticker"), alloc


def minimum_viable_equity(weights: pd.Series, prices: pd.Series) -> float:
    """Smallest account at which every instrument gets at least one share.

    ``shares_i = floor(w_i * E / p_i) >= 1``  requires  ``E >= p_i / w_i`` for every i,
    so the binding instrument is the one with the largest price-to-weight ratio.
    """
    active = weights[weights.abs() > 0]
    ratios = prices.reindex(active.index) / active.abs()
    return float(ratios.max())


def binding_instrument(weights: pd.Series, prices: pd.Series) -> tuple[str, float]:
    active = weights[weights.abs() > 0]
    ratios = prices.reindex(active.index) / active.abs()
    return str(ratios.idxmax()), float(ratios.max())


def report(cfg, weights, prices, sigma, equity, out=sys.stdout, markdown=False) -> dict:
    table, alloc = feasibility_table(weights, prices, equity)
    n_holdable, n_wanted = alloc.n_holdable, alloc.n_wanted
    deployed = alloc.risk_budget_deployed
    min_equity = minimum_viable_equity(weights, prices)
    worst, _ = binding_instrument(weights, prices)

    missing = [t for t in weights.index if alloc.shares[t] == 0 and abs(weights[t]) > 0]
    sleeves_lost = sorted({cfg.sleeve_of(t) for t in missing})
    sleeves_total = sorted(cfg.sleeves)

    fmt = table.copy()
    fmt["price"] = fmt["price"].map("{:,.2f}".format)
    fmt["target $"] = fmt["target $"].map("{:,.2f}".format)
    fmt["realised $"] = fmt["realised $"].map("{:,.2f}".format)
    fmt["drift %"] = fmt["drift %"].map("{:+.1f}".format)

    body = fmt.to_markdown() if markdown else fmt.to_string()
    header = f"account equity ${equity:,.0f}"
    print(f"\n{'## ' if markdown else ''}{header}\n" if markdown else f"\n=== {header} ===", file=out)
    print(body, file=out)

    lines = [
        "",
        f"- **Holdable sleeves:** {n_holdable} of {n_wanted} instruments hold at least one share.",
        f"- **Risk budget deployed:** {deployed:.1%} of the intended gross exposure "
        f"({alloc.realised_weights.abs().sum():.3f} of a designed {weights.abs().sum():.3f}).",
        f"- **Cash left unused:** ${alloc.cash_left:,.2f} ({alloc.cash_left / equity:.1%} of the account).",
        f"- **Weight tracking error vs design:** {alloc.tracking_error:.4f} (L2), "
        f"{alloc.absolute_error:.4f} (L1).",
        f"- **Minimum account for all {n_wanted} to be holdable:** ${min_equity:,.0f} "
        f"(binding instrument: {worst} at ${prices[worst]:,.2f} on a "
        f"{weights[worst]:.2%} target weight).",
    ]
    if missing:
        lines.append(
            f"- **Missing entirely:** {', '.join(missing)} — "
            f"{len(sleeves_lost)} of {len(sleeves_total)} sleeves affected ({', '.join(sleeves_lost)})."
        )
    else:
        lines.append("- **Missing entirely:** none. Every designed position is holdable.")
    print("\n".join(lines), file=out)

    verdict = plain_verdict(equity, n_holdable, n_wanted, deployed, min_equity, missing, sleeves_lost, sleeves_total)
    print(f"\n> {verdict}" if markdown else f"\n{verdict}", file=out)

    return {
        "equity": equity,
        "n_holdable": n_holdable,
        "n_wanted": n_wanted,
        "deployed": deployed,
        "min_equity": min_equity,
        "tracking_error": alloc.tracking_error,
        "missing": missing,
        "verdict": verdict,
    }


def plain_verdict(equity, n_holdable, n_wanted, deployed, min_equity, missing, sleeves_lost, sleeves_total) -> str:
    """Say what is true, in plain language, without softening it."""
    if not missing:
        return (
            f"**${equity:,.0f} can run this strategy.** All {n_wanted} instruments are holdable and "
            f"{deployed:.0%} of the intended risk budget is deployed."
        )
    fraction_lost = 1 - n_holdable / n_wanted
    severity = (
        "cannot run this strategy at all"
        if fraction_lost >= 0.5 or deployed < 0.5
        else "runs a materially degraded version of this strategy"
    )
    return (
        f"**${equity:,.0f} {severity}.** Only {n_holdable} of {n_wanted} instruments are holdable; "
        f"{', '.join(missing)} get zero shares because one share costs more than their entire target "
        f"allocation. {len(sleeves_lost)} of the {len(sleeves_total)} sleeves "
        f"({', '.join(sleeves_lost)}) are wholly or partly absent, so the diversification the strategy "
        f"depends on is not present. Only {deployed:.0%} of the intended risk budget is deployed. "
        f"You need ${min_equity:,.0f} for the designed portfolio to exist."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--equity", type=float, nargs="+", required=True, help="account size(s) in dollars")
    parser.add_argument("--markdown", type=Path, default=None, help="also write a markdown report here")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the cached research history's last close instead of querying the broker",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    history = load_prices(cfg.universe, source="yahoo")
    weights_obj, sigma = designed_weights(cfg, history)
    weights = weights_obj.weights

    if args.offline:
        prices = history.close[list(cfg.universe)].ffill().iloc[-1]
        price_source = f"cached research history, last close {history.close.index[-1].date()}"
    else:
        try:
            prices = last_trade_prices(cfg.universe)
            asof = sorted(set(prices.attrs.get("asof", {}).values()))
            price_source = f"Alpaca last completed session ({', '.join(asof)}), unadjusted"
        except DataError as exc:
            print(f"broker prices unavailable ({exc}); falling back to cached history", file=sys.stderr)
            prices = history.close[list(cfg.universe)].ffill().iloc[-1]
            price_source = f"cached research history, last close {history.close.index[-1].date()}"

    summaries = []
    for equity in args.equity:
        summaries.append(report(cfg, weights, prices, sigma, equity, markdown=False))

    if args.markdown:
        with args.markdown.open("w") as handle:
            write_markdown(handle, cfg, weights, prices, sigma, args.equity, price_source)
        print(f"\nwrote {args.markdown}", file=sys.stderr)
    return 0


def write_markdown(handle, cfg, weights, prices, sigma, equities, price_source) -> None:
    min_equity = minimum_viable_equity(weights, prices)
    worst, _ = binding_instrument(weights, prices)
    print("# Feasibility — diversified trend following v1.0\n", file=handle)
    print(
        f"Generated from `scripts/feasibility.py`. Prices: {price_source}. "
        f"Volatilities: EWMA halflife {cfg.ewma_halflife_days}d on the cached research history.\n",
        file=handle,
    )
    print(
        "Target weights are the section 4 risk-weighted allocation with **every instrument assumed "
        "to be trending** — the fully deployed book the pre-registration describes. Whole-share "
        "rounding is toward zero, so the realised portfolio never exceeds the target and the gross "
        "cap continues to hold after rounding.\n",
        file=handle,
    )

    print("## The designed portfolio\n", file=handle)
    design = pd.DataFrame(
        {
            "sleeve": [cfg.sleeve_of(t) for t in weights.index],
            "ann. vol": sigma.reindex(weights.index).map("{:.1%}".format),
            "target weight": weights.map("{:.2%}".format),
            "price": prices.reindex(weights.index).map("${:,.2f}".format),
            "$ for 1 share at min. size": (prices.reindex(weights.index) / weights.abs()).map("${:,.0f}".format),
        }
    )
    design.index.name = "ticker"
    print(design.to_markdown(), file=handle)
    print(
        f"\nThe binding instrument is **{worst}**: at a {weights[worst]:.2%} target weight, one "
        f"${prices[worst]:,.2f} share is not affordable until the account reaches "
        f"**${min_equity:,.0f}**.\n",
        file=handle,
    )

    rows = []
    for equity in equities:
        rows.append(report(cfg, weights, prices, sigma, equity, out=handle, markdown=True))

    print("\n## Summary\n", file=handle)
    summary = pd.DataFrame(
        [
            {
                "account": f"${r['equity']:,.0f}",
                "holdable": f"{r['n_holdable']}/{r['n_wanted']}",
                "risk budget deployed": f"{r['deployed']:.1%}",
                "tracking error (L2)": f"{r['tracking_error']:.4f}",
                "missing": ", ".join(r["missing"]) or "—",
            }
            for r in rows
        ]
    ).set_index("account")
    print(summary.to_markdown(), file=handle)
    print(
        f"\n**Minimum account size at which all {len(weights)} sleeves become holdable: "
        f"${min_equity:,.0f}.**\n",
        file=handle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
