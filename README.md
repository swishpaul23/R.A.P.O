# R.A.P.O: Robo-Advisor Portfolio Optimizer

A walk-forward portfolio backtester and research project. It compares five long-only allocation strategies after trading costs, and tests whether a machine-learning market-stress detector can protect a portfolio, across four crises from 2000 to 2026.

**Live demo:** _add your Vercel URL here_ · **Backtest** page: run any portfolio · **Research** page: the findings below

## Key findings

**1. Diversification did the heavy lifting.** Hierarchical Risk Parity (HRP) had a far smaller worst-case loss than the S&P 500 in all four crises:

| Crisis | S&P 500 (SPY) max drawdown | HRP max drawdown | HRP return in window |
|---|---|---|---|
| 2000–02 dot-com bust | −47.5% | −3.9% | +24.2% |
| 2008 financial crisis | −55.2% | −9.2% | +7.1% |
| 2020 COVID crash | −33.7% | −10.7% | +4.2% |
| 2022 rate shock | −24.5% | −21.7% | −16.1% |

**2. That protection depends on bonds.** HRP leans on Treasuries. They rallied in the first three crises, but in 2022 stocks and bonds fell together (rolling stock–bond correlation went from about −0.5 to 0), and most of the protection disappeared.

**3. Predicting crashes didn't help.** Four models (VIX rule, Hidden Markov Model, logistic regression, XGBoost) tried to forecast an 8%+ S&P 500 drop within 21 trading days. None beat simply forecasting the historical 6% stress rate (Brier skill below 0% for every model). Shifting up to 40% of the portfolio into short Treasuries and gold on their signals made **no statistically significant difference in any of 8 tests** (95% block-bootstrap intervals all include zero).

Takeaway: choose a robust mix and understand which kind of crash it is exposed to; don't try to time the market.

## How the research was kept honest

- **Walk-forward everywhere.** Every forecast and portfolio decision uses only data available at that date. Tests change future data and check that past results don't move.
- **Calibrated, and compared to a baseline.** Model probabilities are isotonic-calibrated inside each training fold and scored against climatology (always predicting the historical rate).
- **Pre-registered tests.** Round 2 chose its rule on 2010–2019, then ran **once** on 2020–2026. A backward test then ran that locked rule once on 2000–2009. Each protocol was committed to the repo before its run (see [`docs/results/`](docs/results)).
- **Uncertainty reported.** Paired 21-day block bootstrap (1,000 resamples) for every strategy difference.
- **Negative results kept.** Round 1's fixed settings found nothing; that write-up stays in the repo.

Results: [round 1](docs/results/phase2.md) · [round 2](docs/results/phase2-round2.md) · [backward test](docs/results/phase2-backward.md)

## How it works

| Module | Job |
|---|---|
| `rapo/data.py` | Adjusted prices from Yahoo Finance with a CSV cache; falls back to the committed snapshot in `data/snapshot/`. Risk-free rate from the 13-week T-bill. |
| `rapo/returns.py`, `rapo/estimators.py` | Simple/log returns (a newer asset is NaN before it listed; history is never truncated), annualized means, Ledoit-Wolf covariance. |
| `rapo/strategies.py`, `rapo/hrp.py` | Equal weight, inverse vol, min variance, max Sharpe (capped, min-variance fallback), HRP, fixed weights (60/40). |
| `rapo/backtest.py` | Walk-forward engine: weights drift between rebalances, costs on turnover, assets enter once they have a full lookback window. |
| `rapo/metrics.py` | CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, VaR, turnover, rebound capture, whipsaws. |
| `rapo/montecarlo.py`, `rapo/report.py` | Vectorized efficient-frontier cloud; runs every strategy through the same backtest. |
| `rapo/regime/` | Features (volatility, VIX, stock–bond correlation, credit spread, rate momentum), forward-drawdown labels, walk-forward models, filtered HMM, overlay, evaluation and the study runners. |
| `api/main.py` | FastAPI: `/api/backtest` runs a backtest; `/api/regime` serves the pre-computed study. |
| `web/` | React + TypeScript dashboard (Vite, Tailwind, lightweight-charts). |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn api.main:app --reload                       # API on :8000

cd web && npm install && npm run dev                 # UI on http://localhost:5173
```

`http://localhost:5173/?demo=1` runs the UI on bundled data without the API.

Reproduce the research (a few minutes each):

```bash
python scripts/make_snapshot.py        # refresh prices (needs access to Yahoo Finance)
python scripts/run_regime_study.py     # rounds 1 and 2
python scripts/run_backward_test.py    # 2000-2009 backward test
```

## Deploy (Vercel)

One Vercel project serves both parts from the same domain, using Vercel Services (`vercel.json`): the React app at `/` and the FastAPI app at `/api`. Create the project from this repo with **Root Directory `./`**. Models train offline; the site serves the saved results in `data/regime/`.

Yahoo Finance often blocks cloud servers. When live downloads fail, the API falls back to the price snapshot in `data/snapshot/` and the UI says so. Refresh it from your own machine with `python scripts/make_snapshot.py`, then commit the CSVs.

## Tests

```bash
python -m pytest          # 137 tests: package, regime research, API
cd web && npm test        # 38 tests: frontend
```

## Limitations

- Pre-2005 tests use mutual-fund stand-ins (e.g. VUSTX for TLT, VFISX for SHY) and gold futures, because most ETFs didn't exist yet.
- The 2000s were an unusually strong decade for bonds, which flatters HRP there.
- Six years of holdout data and four crises still leave wide confidence intervals.
- Research tool, not investment advice.
