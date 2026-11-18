import pandas as pd
import pytest

from rapo.metrics import STRESS_WINDOWS
from rapo.report import DEFAULT_STRATEGIES, Report, build_report

KW = dict(lookback=60, rebalance_every=20)


def test_build_report_runs_all_strategies(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", **KW)
    assert isinstance(rep, Report)
    assert set(rep.table.index) == set(DEFAULT_STRATEGIES) | {"BENCH"}
    assert "max_drawdown" in rep.table.columns
    assert set(rep.stress_tables) == set(STRESS_WINDOWS)
    for t in rep.stress_tables.values():
        assert set(t.index) == set(DEFAULT_STRATEGIES) | {"BENCH"}


def test_benchmark_not_in_strategy_universe(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", **KW)
    assert "BENCH" not in rep.results["HRP"].weights.columns


def test_benchmark_is_buy_and_hold_on_same_dates(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", **KW)
    hrp_dates = rep.results["HRP"].returns.index
    assert rep.benchmark.index.equals(hrp_dates)
    expected = synthetic_prices["BENCH"].pct_change().loc[hrp_dates]
    assert rep.benchmark.to_numpy() == pytest.approx(expected.to_numpy())


def test_missing_benchmark_raises(synthetic_prices, flat_rf):
    with pytest.raises(ValueError, match="SPY"):
        build_report(synthetic_prices, flat_rf, benchmark="SPY", **KW)


def test_stress_tables_include_turnover_inside_the_window(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", **KW)
    t = rep.stress_tables["2022 rate shock"]
    assert t.loc["HRP", "annual_turnover"] > 0
    assert pd.isna(t.loc["BENCH", "annual_turnover"])
