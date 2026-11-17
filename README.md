# R.A.P.O: Robo-Advisor Portfolio Optimizer

Walk-forward backtester that compares five long-only allocation strategies (equal weight, inverse volatility, minimum variance, max Sharpe, and Hierarchical Risk Parity) against a benchmark, after trading costs, with separate results for the 2020 and 2022 stress periods.

> v2 is in progress. Results and write-up land with Phase 2 (market-regime detector). Design doc and plans are in `docs/superpowers/`.

## How it works

| Module | Job |
|---|---|
| `rapo/data.py` | Adjusted prices from Yahoo Finance, cached to Parquet. Risk-free rate from the 13-week T-bill. |
| `rapo/returns.py` | Simple and log returns. A newer asset is NaN before it listed; history is never truncated. |
| `rapo/estimators.py` | Annualized means and Ledoit-Wolf shrunk covariance. |
| `rapo/strategies.py`, `rapo/hrp.py` | The five strategies. Each takes a returns window and returns long-only weights that sum to 1. |
| `rapo/backtest.py` | Walk-forward engine: decisions only see past data; weights drift between rebalances; costs are charged on turnover. |
| `rapo/metrics.py` | CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, historical VaR, turnover. |
| `rapo/montecarlo.py` | Vectorized random-portfolio cloud for the efficient frontier. |
| `rapo/report.py` | Runs every strategy through the same backtest and builds the comparison tables. |
| `api/main.py` | FastAPI endpoint the web app calls. |
| `web/` | React + TypeScript research dashboard. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload                       # API on :8000

cd web && npm install && npm run dev                 # UI on http://localhost:5173
```

`http://localhost:5173/?demo=1` runs the UI on bundled synthetic data without the API.

## Tests

```bash
python -m pytest          # Python package and API
cd web && npm test        # frontend
```

Research tool, not investment advice.
