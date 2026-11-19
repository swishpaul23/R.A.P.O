# Phase 2, backward test: locked rule on 2000-2009

Protocol: `phase2-backward-protocol.md` (committed before this run).
Rule: hmm at q = 0.90, chosen on 2010-2019. Test: 2000-01-03 to 2009-12-31. Signal first available 1996-02-16.
Average tilt 1.4%, 1 whipsaws.

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| SPY | -1.0% | 22.4% | -0.05 | -55.2% | — |
| 60/40 | 2.1% | 12.7% | 0.02 | -35.2% | 0.12 |
| Equal weight | 5.8% | 14.6% | 0.27 | -39.4% | 0.44 |
| Equal weight + regime | 6.1% | 13.7% | 0.30 | -36.7% | 0.64 |
| HRP | 7.6% | 5.4% | 0.88 | -9.2% | 0.48 |
| HRP + regime | 7.5% | 5.4% | 0.88 | -9.7% | 0.68 |

## 2000–02 dot-com bust

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -47.2% | -47.5% | -1.11 |
| 60/40 | -24.1% | -25.5% | -1.01 |
| Equal weight | -29.3% | -29.5% | -1.16 |
| Equal weight + regime | -28.9% | -29.4% | -1.17 |
| HRP | 24.2% | -3.9% | 0.97 |
| HRP + regime | 24.3% | -3.9% | 0.99 |

## 2008 financial crisis

| Strategy | Return | Max DD | Sharpe |
|---|---|---|---|
| SPY | -54.8% | -55.2% | -1.32 |
| 60/40 | -34.8% | -35.2% | -1.44 |
| Equal weight | -37.2% | -39.4% | -1.26 |
| Equal weight + regime | -33.6% | -36.7% | -1.28 |
| HRP | 7.1% | -9.2% | 0.51 |
| HRP + regime | 7.3% | -9.7% | 0.53 |

## Overlay minus base, 95% block bootstrap

| Base | Metric | Difference | 95% interval | Significant |
|---|---|---|---|---|
| Equal weight | sharpe | 0.03 | -0.02 to 0.08 | no |
| Equal weight | max_drawdown | 2.7% | -0.0% to 5.3% | no |
| HRP | sharpe | -0.00 | -0.06 to 0.07 | no |
| HRP | max_drawdown | -0.5% | -1.0% to 1.1% | no |

Stand-ins: VGTSX (intl), VEIEX (EM), VUSTX (long Treasuries), VFITX (intermediate), VGSIX (REITs), GC=F (gold futures), VFISX (short Treasuries), VBMFX (bonds for 60/40).
