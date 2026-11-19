"""Run every strategy through the same backtest and tabulate the results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rapo.backtest import BacktestResult, run_backtest
from rapo.hrp import hrp
from rapo.metrics import STRESS_WINDOWS, summary, window
from rapo.returns import simple_returns
from rapo.strategies import Strategy, equal_weight, inverse_vol, make_max_sharpe, min_variance


def default_strategies(rf_annual: float = 0.0) -> dict[str, Strategy]:
    return {
        "Equal weight": equal_weight,
        "Inverse vol": inverse_vol,
        "Min variance": min_variance,
        "Max Sharpe": make_max_sharpe(rf_annual=rf_annual),
        "HRP": hrp,
    }


DEFAULT_STRATEGIES = default_strategies()


@dataclass
class Report:
    results: dict[str, BacktestResult]
    benchmark: pd.Series                      # buy-and-hold daily returns, same dates
    table: pd.DataFrame                       # full-period metrics, one row per strategy + benchmark
    stress_tables: dict[str, pd.DataFrame]    # same metrics inside each stress window


def build_report(prices: pd.DataFrame, rf: pd.Series, benchmark: str,
                 strategies: dict[str, Strategy] | None = None,
                 extra_benchmarks: dict[str, Strategy] | None = None, **backtest_kwargs) -> Report:
    """`extra_benchmarks` (e.g. 60/40) run through the same backtest and are reported like strategies."""
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} has no price data.")
    if strategies is None:
        strategies = default_strategies(rf_annual=float(rf.mean() * 252))

    universe = simple_returns(prices.drop(columns=benchmark))
    results = {name: run_backtest(universe, s, **backtest_kwargs)
               for name, s in {**strategies, **(extra_benchmarks or {})}.items()}

    dates = next(iter(results.values())).returns.index
    bench = simple_returns(prices[[benchmark]])[benchmark].reindex(dates).fillna(0.0)
    bench.name = benchmark

    series = {name: res.returns for name, res in results.items()}
    series[benchmark] = bench
    turnovers = {name: res.turnover for name, res in results.items()}

    table = pd.DataFrame({n: summary(r, rf, turnovers.get(n)) for n, r in series.items()}).T

    stress = {}
    for wname in STRESS_WINDOWS:
        sliced = {n: window(r, wname) for n, r in series.items()}
        if all(len(s) > 1 for s in sliced.values()):
            stress[wname] = pd.DataFrame({
                n: summary(s, rf, window(turnovers[n], wname) if n in turnovers else None)
                for n, s in sliced.items()
            }).T

    return Report(results=results, benchmark=bench, table=table, stress_tables=stress)
