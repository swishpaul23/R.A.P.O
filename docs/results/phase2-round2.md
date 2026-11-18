# Phase 2, round 2: tuned on 2010-2019, tested once on 2020+

Protocol: `phase2-round2-protocol.md` (committed before this run).
Development: 2010-07-30 to 2019-12-31. Holdout: 2020-01-02 to 2026-11-17.

## Development (all 6 candidates)

| Signal | q | Avg Sharpe gain | EW gain | HRP gain | Avg tilt | Whipsaws |
|---|---|---|---|---|---|---|
| logistic | 0.80 | -0.015 | -0.020 | -0.011 | 0.8% | 0 |
| logistic | 0.90 | -0.007 | -0.013 | -0.001 | 0.4% | 0 |
| hmm | 0.80 | -0.005 | -0.002 | -0.008 | 0.7% | 4 |
| hmm **(chosen)** | 0.90 | -0.002 | -0.002 | -0.002 | 0.1% | 1 |
| vol_21 | 0.80 | -0.033 | -0.014 | -0.051 | 1.6% | 6 |
| vol_21 | 0.90 | -0.017 | -0.007 | -0.026 | 0.3% | 3 |

## Holdout: hmm at q = 0.90

Average tilt 1.5%, 1 whipsaws.

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| SPY | 14.3% | 20.0% | 0.63 | -33.7% | — |
| 60/40 | 8.9% | 12.4% | 0.52 | -20.9% | 0.13 |
| Equal weight | 8.1% | 13.2% | 0.44 | -25.8% | 0.31 |
| Equal weight + regime | 7.7% | 12.6% | 0.43 | -25.9% | 0.59 |
| HRP | 3.5% | 8.3% | 0.11 | -23.8% | 0.78 |
| HRP + regime | 3.3% | 8.2% | 0.09 | -23.9% | 1.04 |

### 2020 COVID crash

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -3.2% | -33.7% | 0.07 |
| 60/40 | 0.3% | -20.9% | 0.15 |
| Equal weight | 2.3% | -20.3% | 0.33 |
| Equal weight + regime | 1.7% | -18.3% | 0.28 |
| HRP | 4.2% | -10.7% | 0.88 |
| HRP + regime | 3.8% | -10.2% | 0.82 |

### 2022 rate shock

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -18.2% | -24.5% | -0.79 |
| 60/40 | -16.0% | -20.6% | -1.15 |
| Equal weight | -19.7% | -25.8% | -1.41 |
| Equal weight + regime | -19.9% | -25.9% | -1.44 |
| HRP | -16.1% | -21.7% | -1.65 |
| HRP + regime | -16.2% | -21.8% | -1.67 |

### Overlay minus base, 95% block bootstrap

| Base | Metric | Difference | 95% interval | Significant |
|---|---|---|---|---|
| Equal weight | sharpe | -0.01 | -0.07 to 0.04 | no |
| Equal weight | max_drawdown | -0.1% | -0.9% to 2.8% | no |
| HRP | sharpe | -0.02 | -0.05 to 0.01 | no |
| HRP | max_drawdown | -0.1% | -0.8% to 0.6% | no |
