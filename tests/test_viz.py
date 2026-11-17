import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from rapo import viz  # noqa: E402
from rapo.estimators import annual_mean, ledoit_wolf_cov  # noqa: E402
from rapo.montecarlo import random_portfolios  # noqa: E402
from rapo.report import build_report  # noqa: E402
from rapo.returns import simple_returns  # noqa: E402


def test_figures_render(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", lookback=60, rebalance_every=20)
    rets = simple_returns(synthetic_prices.drop(columns="BENCH"))
    mc = random_portfolios(annual_mean(rets), ledoit_wolf_cov(rets), n=500)
    figs = [
        viz.equity_curves(rep),
        viz.drawdowns(rep),
        viz.correlation_heatmap(rets),
        viz.dendrogram(rets),
        viz.frontier(mc, {"HRP": (0.1, 0.05)}),
    ]
    assert all(isinstance(f, Figure) for f in figs)
