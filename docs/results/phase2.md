# Phase 2 results: regime detector

Data through 2026-11-17. Out-of-sample from 2010-07-30. Universe: SPY, QQQ, EFA, EEM, TLT, IEF, GLD, VNQ.
Stress label: SPY falls 8% or more within the next 21 trading days. Tilt: 0.40 x clip((EMA5(p) - 0.20) / 0.80, 0, 1) into {'SHY': 0.5, 'GLD': 0.5}. Monthly rebalance, 10 bps costs.

## Forecast quality (same out-of-sample days for every model)

| Model | AUC | Brier | Skill vs climatology | Days |
|---|---|---|---|---|
| climatology (historical stress rate) | — | 0.055 | 0% | 4080 |
| vix_rule | 0.51 | 0.105 | -89.6% | 4080 |
| hmm | 0.67 | 0.232 | -319.8% | 4080 |
| logistic (chosen) | 0.61 | 0.062 | -11.2% | 4080 |
| boosting | 0.36 | 0.062 | -12.2% | 4080 |

Stress base rate in this period: 5.8%. AUC 0.5 = no better than chance at ranking days; Brier skill below 0% = worse than always forecasting the historical stress rate. The model with the lowest Brier drives the overlay.

## Strategies, full out-of-sample period

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr | Whipsaws |
|---|---|---|---|---|---|---|
| SPY | 14.3% | 16.9% | 0.78 | -33.7% | — | — |
| 60/40 | 9.6% | 10.2% | 0.79 | -20.9% | 0.12 | — |
| Equal weight | 8.5% | 11.0% | 0.65 | -25.8% | 0.30 | — |
| Equal weight + regime | 8.4% | 11.0% | 0.65 | -25.8% | 0.34 | 0 |
| HRP | 5.0% | 6.9% | 0.52 | -23.8% | 0.81 | — |
| HRP + regime | 5.0% | 6.9% | 0.52 | -23.8% | 0.83 | 0 |

## 2020 COVID crash

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -3.2% | -33.7% | 0.07 |
| 60/40 | 0.3% | -20.9% | 0.15 |
| Equal weight | 2.3% | -20.3% | 0.33 |
| Equal weight + regime | 2.3% | -20.3% | 0.33 |
| HRP | 4.2% | -10.7% | 0.88 |
| HRP + regime | 4.2% | -10.7% | 0.88 |

## 2022 rate shock

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -18.2% | -24.5% | -0.79 |
| 60/40 | -16.0% | -20.6% | -1.15 |
| Equal weight | -19.7% | -25.8% | -1.41 |
| Equal weight + regime | -19.7% | -25.8% | -1.41 |
| HRP | -16.1% | -21.7% | -1.65 |
| HRP + regime | -16.1% | -21.7% | -1.65 |

## Is the overlay's edge more than luck? (95% block bootstrap)

| Base | Metric | Overlay − base | 95% interval | Significant |
|---|---|---|---|---|
| Equal weight | sharpe | -0.00 | -0.01 to 0.00 | no |
| Equal weight | max_drawdown | -0.0% | -0.3% to 0.3% | no |
| HRP | sharpe | -0.00 | -0.01 to 0.00 | no |
| HRP | max_drawdown | -0.0% | -0.3% to 0.1% | no |

Latest reading (2026-11-17): stress probability 10.3%, tilt 0.0%.
