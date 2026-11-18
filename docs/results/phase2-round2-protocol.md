# Phase 2, round 2: pre-registered protocol

Written and committed **before** any round-2 run touches 2020 onward. Round 1 (fixed settings, `phase2.md`) found no model beats climatology and the overlay almost never activated. Round 2 tunes on a development period and tests once on a holdout.

## Periods

- **Development:** first out-of-sample date (2010-07-30) to 2019-12-31. All choices are made here.
- **Holdout (final test):** 2020-01-01 to the end of the snapshot. Run once with the locked choice. Contains the 2020 COVID crash and the 2022 rate shock.

The signals are still produced walk-forward (expanding window), so holdout-period values only use data available at the time.

## Candidate rules (6, kept small to limit multiple comparisons)

Each turns a signal into a defensive tilt using the signal's **expanding percentile rank** (data ≤ that day, at least 252 days of history):

tilt = 0.40 × clip((rank − q) / (1 − q), 0, 1), smoothed with a 5-day EMA.

| Signal | What it is |
|---|---|
| `logistic` | Round 1's calibrated logistic stress probability |
| `hmm` | Round 1's HMM stress-state probability (best ranking, AUC 0.67) |
| `vol_21` | SPY 21-day realized volatility (volatility targeting) |

q ∈ {0.80, 0.90}, so each rule tilts only when its signal is in the top 20% or top 10% of its own history.

Fixed, not tuned: cap 0.40; defensive sleeve 50% SHY / 50% GLD; monthly rebalance; 10 bps costs; base strategies HRP and equal weight on SPY, QQQ, EFA, EEM, TLT, IEF, GLD, VNQ.

## Selection rule

Pick the candidate with the highest **average Sharpe improvement over its base strategy** (HRP and equal weight, averaged) in the development period. Ties go to the earlier row in the table above, then the higher q.

## Holdout report

For the chosen rule only, on the holdout: the full metrics table vs HRP, equal weight, 60/40 and SPY; the 2020 and 2022 windows; 95% block-bootstrap intervals (21-day blocks, 1,000 resamples, seed 0) for the Sharpe and max-drawdown differences versus each base. The development results for all 6 candidates are reported too, so nothing is hidden.

A holdout difference counts as a win only if its 95% interval excludes zero.
