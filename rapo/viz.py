"""Matplotlib figures (used for the README and offline analysis).

The web frontend renders its own charts from the API; these exist so results
can be exported as static images.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.figure import Figure

from rapo.hrp import hrp_linkage

BG, FG, GRID = "#0b0f14", "#c9d1d9", "#1f2933"
PALETTE = ["#00d1b2", "#4ea1ff", "#f5a524", "#c084fc", "#ff5d73", "#8b949e"]


def _style(ax, title: str):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.set_title(title, color=FG, loc="left", fontsize=11, fontfamily="monospace")
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)


def _series(report) -> dict[str, pd.Series]:
    out = {n: r.returns for n, r in report.results.items()}
    out[report.benchmark.name] = report.benchmark
    return out


def equity_curves(report) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (name, r), c in zip(_series(report).items(), PALETTE):
        ax.plot((1 + r).cumprod(), label=name, color=c, linewidth=1.2)
    _style(ax, "GROWTH OF $1")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    return fig


def drawdowns(report) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for (name, r), c in zip(_series(report).items(), PALETTE):
        wealth = (1 + r).cumprod()
        ax.plot(wealth / wealth.cummax() - 1, label=name, color=c, linewidth=1.0)
    _style(ax, "DRAWDOWN")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(returns.corr(), ax=ax, cmap="vlag", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    _style(ax, "CORRELATION")
    ax.grid(False)
    return fig


def dendrogram(returns: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sch.dendrogram(hrp_linkage(returns), labels=list(returns.columns), ax=ax,
                   color_threshold=0, above_threshold_color=PALETTE[0])
    _style(ax, "HRP CLUSTER TREE")
    return fig


def frontier(mc: pd.DataFrame, highlight: dict[str, tuple[float, float]]) -> Figure:
    """Random-portfolio cloud coloured by Sharpe; highlight = {name: (vol, ret)}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(mc["vol"], mc["ret"], c=mc["sharpe"], cmap="viridis", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Sharpe")
    for (name, (vol, ret)), c in zip(highlight.items(), PALETTE[3:] + PALETTE):
        ax.scatter([vol], [ret], marker="D", s=60, color=c, edgecolor=FG, zorder=5)
        ax.annotate(name, (vol, ret), color=FG, fontsize=8, xytext=(5, 5), textcoords="offset points")
    _style(ax, "EFFICIENT FRONTIER")
    ax.set_xlabel("Volatility", color=FG)
    ax.set_ylabel("Return", color=FG)
    return fig
