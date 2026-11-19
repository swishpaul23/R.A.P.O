# Phase 2, backward test: pre-registered protocol

Written and committed **before** any 2000–2009 data is downloaded or run.

## Question

Does the rule locked in round 2 (`phase2-round2.md`) help in the two slow bear markets that weren't testable before: the dot-com bust and the 2008 financial crisis?

## The locked rule (no changes)

- Signal: HMM stress-state probability (2-state Gaussian HMM on SPY daily return and 21-day volatility), walk-forward, refit every 63 days, filtered probabilities only.
- Trigger: tilt = 0.40 × clip((rank − 0.90) / 0.10, 0, 1), where rank is the signal's expanding percentile rank (≥ 252 days of history), smoothed with a 5-day EMA.
- Defensive sleeve: 50% short Treasuries, 50% gold. Monthly rebalance, 10 bps costs. Bases: HRP and equal weight.

The rule was chosen on 2010–2019 data, so this is a backward test: the test period is earlier than the selection period. Nothing about 2000–2009 influenced the choice.

## Data (long-history stand-ins for ETFs that didn't exist yet)

| Role | Used here | Round 1–2 ETF |
|---|---|---|
| US stocks, label, HMM | SPY (1993) | SPY |
| Nasdaq | QQQ (1999; eligible after its lookback) | QQQ |
| International | VGTSX | EFA |
| Emerging markets | VEIEX | EEM |
| Long Treasuries | VUSTX | TLT |
| Intermediate Treasuries | VFITX | IEF |
| REITs | VGSIX | VNQ |
| Gold | GC=F (futures, from 2000) | GLD |
| Short Treasuries (defensive) | VFISX | SHY |
| Bonds for 60/40 | VBMFX | AGG |

Data starts 1993-01-29. The HMM needs 750 days of training, so it first predicts in early 1996; the percentile rank needs another 252 days, so the rule is live well before 2000. Before gold data exists, the defensive sleeve is short Treasuries only.

## Test period and report

- **Test:** 2000-01-01 to 2009-12-31, run once.
- Stress windows: dot-com bust (2000-03-24 to 2002-10-09) and 2008 financial crisis (2007-10-09 to 2009-03-09), peak to trough of the S&P 500.
- Report the same table as the round-2 holdout (vs plain HRP, equal weight, 60/40, SPY), the two windows, and 95% block-bootstrap intervals (21-day blocks, 1,000 resamples, seed 0) for the Sharpe and max-drawdown differences. A difference counts only if its interval excludes zero.
- Stand-in funds differ slightly from the ETFs (fees, pricing). That noise is noted, not corrected.
