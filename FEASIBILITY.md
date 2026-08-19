# Feasibility — diversified trend following v1.0

Generated from `scripts/feasibility.py`. Prices: Alpaca last completed session (2026-08-18), unadjusted. Volatilities: EWMA halflife 30d on the cached research history.

Target weights are the section 4 risk-weighted allocation with **every instrument assumed to be trending** — the fully deployed book the pre-registration describes. Whole-share rounding is toward zero, so the realised portfolio never exceeds the target and the gross cap continues to hold after rounding.

## The designed portfolio

| ticker   | sleeve      | ann. vol   | target weight   | price   | $ for 1 share at min. size   |
|:---------|:------------|:-----------|:----------------|:--------|:-----------------------------|
| SPY      | Equity      | 13.4%      | 7.59%           | $767.45 | $10,115                      |
| EFA      | Equity      | 16.0%      | 6.35%           | $107.27 | $1,690                       |
| EEM      | Equity      | 30.7%      | 3.31%           | $65.34  | $1,972                       |
| TLT      | Rates       | 9.3%       | 11.00%          | $81.66  | $743                         |
| IEF      | Rates       | 4.8%       | 14.27%          | $92.93  | $651                         |
| GLD      | Commodities | 25.9%      | 3.92%           | $398.55 | $10,163                      |
| SLV      | Commodities | 49.8%      | 2.04%           | $57.44  | $2,809                       |
| DBC      | Commodities | 23.4%      | 4.35%           | $30.48  | $701                         |
| UUP      | Currency    | 5.2%       | 14.27%          | $28.14  | $197                         |
| FXE      | Currency    | 4.9%       | 14.27%          | $106.84 | $749                         |
| FXY      | Currency    | 8.7%       | 11.76%          | $57.48  | $489                         |
| VNQ      | Real assets | 14.8%      | 6.88%           | $97.62  | $1,419                       |

The binding instrument is **GLD**: at a 3.92% target weight, one $398.55 share is not affordable until the account reaches **$10,163**.


## account equity $1,000

| ticker   |   price |   target $ |   shares |   realised $ |   drift % | holdable   |
|:---------|--------:|-----------:|---------:|-------------:|----------:|:-----------|
| SPY      |  767.45 |      75.87 |        0 |         0    |    -100   | NO         |
| EFA      |  107.27 |      63.47 |        0 |         0    |    -100   | NO         |
| EEM      |   65.34 |      33.13 |        0 |         0    |    -100   | NO         |
| TLT      |   81.66 |     109.97 |        1 |        81.66 |     -25.7 | yes        |
| IEF      |   92.93 |     142.69 |        1 |        92.93 |     -34.9 | yes        |
| GLD      |  398.55 |      39.21 |        0 |         0    |    -100   | NO         |
| SLV      |   57.44 |      20.45 |        0 |         0    |    -100   | NO         |
| DBC      |   30.48 |      43.45 |        1 |        30.48 |     -29.9 | yes        |
| UUP      |   28.14 |     142.69 |        5 |       140.7  |      -1.4 | yes        |
| FXE      |  106.84 |     142.69 |        1 |       106.84 |     -25.1 | yes        |
| FXY      |   57.48 |     117.59 |        2 |       114.96 |      -2.2 | yes        |
| VNQ      |   97.62 |      68.78 |        0 |         0    |    -100   | NO         |

- **Holdable sleeves:** 6 of 12 instruments hold at least one share.
- **Risk budget deployed:** 56.8% of the intended gross exposure (0.568 of a designed 1.000).
- **Cash left unused:** $432.43 (43.2% of the account).
- **Weight tracking error vs design:** 0.1494 (L2), 0.4324 (L1).
- **Minimum account for all 12 to be holdable:** $10,163 (binding instrument: GLD at $398.55 on a 3.92% target weight).
- **Missing entirely:** SPY, EFA, EEM, GLD, SLV, VNQ — 3 of 5 sleeves affected (Commodities, Equity, Real assets).

> **$1,000 cannot run this strategy at all.** Only 6 of 12 instruments are holdable; SPY, EFA, EEM, GLD, SLV, VNQ get zero shares because one share costs more than their entire target allocation. 3 of the 5 sleeves (Commodities, Equity, Real assets) are wholly or partly absent, so the diversification the strategy depends on is not present. Only 57% of the intended risk budget is deployed. You need $10,163 for the designed portfolio to exist.

## account equity $5,000

| ticker   |   price |   target $ |   shares |   realised $ |   drift % | holdable   |
|:---------|--------:|-----------:|---------:|-------------:|----------:|:-----------|
| SPY      |  767.45 |     379.37 |        0 |         0    |    -100   | NO         |
| EFA      |  107.27 |     317.34 |        2 |       214.54 |     -32.4 | yes        |
| EEM      |   65.34 |     165.67 |        2 |       130.68 |     -21.1 | yes        |
| TLT      |   81.66 |     549.85 |        6 |       489.96 |     -10.9 | yes        |
| IEF      |   92.93 |     713.45 |        7 |       650.51 |      -8.8 | yes        |
| GLD      |  398.55 |     196.07 |        0 |         0    |    -100   | NO         |
| SLV      |   57.44 |     102.23 |        1 |        57.44 |     -43.8 | yes        |
| DBC      |   30.48 |     217.26 |        7 |       213.36 |      -1.8 | yes        |
| UUP      |   28.14 |     713.45 |       25 |       703.5  |      -1.4 | yes        |
| FXE      |  106.84 |     713.45 |        6 |       641.04 |     -10.1 | yes        |
| FXY      |   57.48 |     587.96 |       10 |       574.8  |      -2.2 | yes        |
| VNQ      |   97.62 |     343.92 |        3 |       292.86 |     -14.8 | yes        |

- **Holdable sleeves:** 10 of 12 instruments hold at least one share.
- **Risk budget deployed:** 79.4% of the intended gross exposure (0.794 of a designed 1.000).
- **Cash left unused:** $1,031.31 (20.6% of the account).
- **Weight tracking error vs design:** 0.0921 (L2), 0.2063 (L1).
- **Minimum account for all 12 to be holdable:** $10,163 (binding instrument: GLD at $398.55 on a 3.92% target weight).
- **Missing entirely:** SPY, GLD — 2 of 5 sleeves affected (Commodities, Equity).

> **$5,000 runs a materially degraded version of this strategy.** Only 10 of 12 instruments are holdable; SPY, GLD get zero shares because one share costs more than their entire target allocation. 2 of the 5 sleeves (Commodities, Equity) are wholly or partly absent, so the diversification the strategy depends on is not present. Only 79% of the intended risk budget is deployed. You need $10,163 for the designed portfolio to exist.

## account equity $25,000

| ticker   |   price |   target $ |   shares |   realised $ |   drift % | holdable   |
|:---------|--------:|-----------:|---------:|-------------:|----------:|:-----------|
| SPY      |  767.45 |    1896.83 |        2 |      1534.9  |     -19.1 | yes        |
| EFA      |  107.27 |    1586.69 |       14 |      1501.78 |      -5.4 | yes        |
| EEM      |   65.34 |     828.33 |       12 |       784.08 |      -5.3 | yes        |
| TLT      |   81.66 |    2749.26 |       33 |      2694.78 |      -2   | yes        |
| IEF      |   92.93 |    3567.23 |       38 |      3531.34 |      -1   | yes        |
| GLD      |  398.55 |     980.37 |        2 |       797.1  |     -18.7 | yes        |
| SLV      |   57.44 |     511.15 |        8 |       459.52 |     -10.1 | yes        |
| DBC      |   30.48 |    1086.3  |       35 |      1066.8  |      -1.8 | yes        |
| UUP      |   28.14 |    3567.23 |      126 |      3545.64 |      -0.6 | yes        |
| FXE      |  106.84 |    3567.23 |       33 |      3525.72 |      -1.2 | yes        |
| FXY      |   57.48 |    2939.79 |       51 |      2931.48 |      -0.3 | yes        |
| VNQ      |   97.62 |    1719.6  |       17 |      1659.54 |      -3.5 | yes        |

- **Holdable sleeves:** 12 of 12 instruments hold at least one share.
- **Risk budget deployed:** 96.1% of the intended gross exposure (0.961 of a designed 1.000).
- **Cash left unused:** $967.32 (3.9% of the account).
- **Weight tracking error vs design:** 0.0173 (L2), 0.0387 (L1).
- **Minimum account for all 12 to be holdable:** $10,163 (binding instrument: GLD at $398.55 on a 3.92% target weight).
- **Missing entirely:** none. Every designed position is holdable.

> **$25,000 can run this strategy.** All 12 instruments are holdable and 96% of the intended risk budget is deployed.

## account equity $100,000

| ticker   |   price |   target $ |   shares |   realised $ |   drift % | holdable   |
|:---------|--------:|-----------:|---------:|-------------:|----------:|:-----------|
| SPY      |  767.45 |    7587.33 |        9 |      6907.05 |      -9   | yes        |
| EFA      |  107.27 |    6346.77 |       59 |      6328.93 |      -0.3 | yes        |
| EEM      |   65.34 |    3313.32 |       50 |      3267    |      -1.4 | yes        |
| TLT      |   81.66 |   10997    |      134 |     10942.4  |      -0.5 | yes        |
| IEF      |   92.93 |   14268.9  |      153 |     14218.3  |      -0.4 | yes        |
| GLD      |  398.55 |    3921.48 |        9 |      3586.95 |      -8.5 | yes        |
| SLV      |   57.44 |    2044.59 |       35 |      2010.4  |      -1.7 | yes        |
| DBC      |   30.48 |    4345.19 |      142 |      4328.16 |      -0.4 | yes        |
| UUP      |   28.14 |   14268.9  |      507 |     14267    |      -0   | yes        |
| FXE      |  106.84 |   14268.9  |      133 |     14209.7  |      -0.4 | yes        |
| FXY      |   57.48 |   11759.2  |      204 |     11725.9  |      -0.3 | yes        |
| VNQ      |   97.62 |    6878.41 |       70 |      6833.4  |      -0.7 | yes        |

- **Holdable sleeves:** 12 of 12 instruments hold at least one share.
- **Risk budget deployed:** 98.6% of the intended gross exposure (0.986 of a designed 1.000).
- **Cash left unused:** $1,374.76 (1.4% of the account).
- **Weight tracking error vs design:** 0.0077 (L2), 0.0137 (L1).
- **Minimum account for all 12 to be holdable:** $10,163 (binding instrument: GLD at $398.55 on a 3.92% target weight).
- **Missing entirely:** none. Every designed position is holdable.

> **$100,000 can run this strategy.** All 12 instruments are holdable and 99% of the intended risk budget is deployed.

## Summary

| account   | holdable   | risk budget deployed   |   tracking error (L2) | missing                      |
|:----------|:-----------|:-----------------------|----------------------:|:-----------------------------|
| $1,000    | 6/12       | 56.8%                  |                0.1494 | SPY, EFA, EEM, GLD, SLV, VNQ |
| $5,000    | 10/12      | 79.4%                  |                0.0921 | SPY, GLD                     |
| $25,000   | 12/12      | 96.1%                  |                0.0173 | —                            |
| $100,000  | 12/12      | 98.6%                  |                0.0077 | —                            |

**Minimum account size at which all 12 sleeves become holdable: $10,163.**

