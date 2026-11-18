"""HTTP API over the rapo package. The React app in web/ talks to this.

Run locally:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from rapo.data import default_cache_dir, fetch_prices, risk_free_daily, snapshot_downloader
from rapo.estimators import annual_mean, ledoit_wolf_cov
from rapo.hrp import hrp_linkage
from rapo.metrics import STRESS_WINDOWS
from rapo.montecarlo import random_portfolios
from rapo.report import build_report
from rapo.returns import simple_returns

BENCH_COL = "__benchmark__"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot"
FRONTIER_POINTS = 3000


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=25)
    benchmark: str = "SPY"
    start: date = date(2015, 1, 1)
    end: date | None = None
    lookback: int = Field(252, ge=20, le=1260)
    rebalance_every: int = Field(63, ge=1, le=252)
    cost_bps: float = Field(10.0, ge=0, le=200)

    @field_validator("tickers")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for t in v:
            t = t.strip().upper()
            if t and t not in out:
                out.append(t)
        if not out:
            raise ValueError("Add at least one ticker.")
        return out

    @field_validator("benchmark")
    @classmethod
    def _bench(cls, v: str) -> str:
        return v.strip().upper()


def _clean(x):
    """Recursively turn NaN/inf into None so the response is valid JSON."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _rows(table: pd.DataFrame, rename: dict[str, str]) -> list[dict]:
    return [{"name": rename.get(name, name), **row.to_dict()} for name, row in table.iterrows()]


def _dates(req: BacktestRequest) -> tuple[str, str]:
    return req.start.isoformat(), (req.end or date.today()).isoformat()


def _load(req: BacktestRequest, prices_loader, rf_loader) -> tuple[pd.DataFrame, pd.Series]:
    start, end = _dates(req)
    needed = req.tickers + ([req.benchmark] if req.benchmark not in req.tickers else [])
    raw = prices_loader(needed, start, end)
    return raw, rf_loader(start, end)


def _run(req: BacktestRequest, raw: pd.DataFrame, rf: pd.Series, data_source: str) -> dict:
    start, end = _dates(req)
    prices = raw[req.tickers].copy()
    prices[BENCH_COL] = raw[req.benchmark]

    report = build_report(prices, rf, benchmark=BENCH_COL, lookback=req.lookback,
                          rebalance_every=req.rebalance_every, cost_bps=req.cost_bps)
    bench_label = f"{req.benchmark} (benchmark)"
    rename = {BENCH_COL: bench_label}

    series = {n: r.returns for n, r in report.results.items()}
    series[bench_label] = report.benchmark
    dates = report.benchmark.index
    equity, drawdown = {}, {}
    for name, r in series.items():
        wealth = (1.0 + r).cumprod()
        equity[name] = wealth.round(6).tolist()
        drawdown[name] = (wealth / wealth.cummax() - 1.0).round(6).tolist()

    weights = {name: {t: float(w) for t, w in res.weights.iloc[-1].items() if t != BENCH_COL}
               for name, res in report.results.items()}

    # Snapshot views use the most recent lookback window with complete data.
    universe = simple_returns(prices[req.tickers])
    recent = universe.iloc[-req.lookback:].dropna(axis=1, how="any")
    correlation = clusters = frontier = None
    if recent.shape[0] >= 20 and recent.shape[1] >= 1:
        cols = list(recent.columns)
        correlation = {"tickers": cols, "matrix": recent.corr().round(4).to_numpy().tolist()}
        if len(cols) >= 2:
            link = hrp_linkage(recent)
            import scipy.cluster.hierarchy as sch
            clusters = {"order": [cols[i] for i in sch.leaves_list(link)],
                        "linkage": link.round(6).tolist()}
        mu, cov = annual_mean(recent), ledoit_wolf_cov(recent)
        rf_annual = float(rf.mean() * 252)
        cloud = random_portfolios(mu, cov, n=FRONTIER_POINTS, rf=rf_annual)
        strat_points = []
        for name, w in weights.items():
            wv = pd.Series(w).reindex(cols).fillna(0.0).to_numpy()
            ret = float(wv @ mu.to_numpy())
            vol = float(np.sqrt(wv @ cov.to_numpy() @ wv))
            strat_points.append({"name": name, "ret": ret, "vol": vol,
                                 "sharpe": (ret - rf_annual) / vol if vol > 0 else None})
        frontier = {"points": cloud[["vol", "ret", "sharpe"]].round(5).to_dict("records"),
                    "strategies": strat_points}

    stress = []
    for wname, table in report.stress_tables.items():
        s, e = STRESS_WINDOWS[wname]
        stress.append({"window": wname, "start": s, "end": e, "rows": _rows(table, rename)})

    return _clean({
        "meta": {"tickers": req.tickers, "benchmark": req.benchmark, "start": start, "end": end,
                 "lookback": req.lookback, "rebalance_every": req.rebalance_every,
                 "cost_bps": req.cost_bps, "first_trade_date": dates[0].date().isoformat(),
                 "data_source": data_source},
        "strategies": list(report.results),
        "benchmark_label": bench_label,
        "table": _rows(report.table, rename),
        "stress": stress,
        "series": {"dates": [d.date().isoformat() for d in dates], "equity": equity, "drawdown": drawdown},
        "weights": weights,
        "correlation": correlation,
        "clusters": clusters,
        "frontier": frontier,
    })


def snapshot_loaders(snapshot_dir: Path = SNAPSHOT_DIR) -> tuple[Callable, Callable] | None:
    """Loaders that read the committed price snapshot, or None if there isn't one."""
    if not snapshot_dir.is_dir():
        return None
    dl = snapshot_downloader(snapshot_dir)
    cache = default_cache_dir() / "snapshot"
    return (lambda t, s, e: fetch_prices(t, s, e, cache, dl),
            lambda s, e: risk_free_daily(s, e, cache, dl))


_DEFAULT = object()


def create_app(prices_loader: Callable = fetch_prices,
               rf_loader: Callable = risk_free_daily,
               fallback: tuple[Callable, Callable] | None | object = _DEFAULT) -> FastAPI:
    """`fallback` = (prices_loader, rf_loader) used when live data fails (default: the committed snapshot)."""
    if fallback is _DEFAULT:
        fallback = snapshot_loaders()
    app = FastAPI(title="R.A.P.O API", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/backtest")
    def backtest(req: BacktestRequest):
        try:
            try:
                raw, rf = _load(req, prices_loader, rf_loader)
                source = "live"
            except ValueError as live_error:
                if fallback is None:
                    raise
                try:
                    raw, rf = _load(req, *fallback)
                    source = "snapshot"
                except ValueError:
                    raise live_error from None
            return _run(req, raw, rf, source)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


app = create_app()
