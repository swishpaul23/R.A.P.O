"""How good are the stress forecasts, and is a strategy difference more than luck?"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def _aligned(proba: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat({"p": proba, "y": labels}, axis=1).dropna()
    return df["p"], df["y"]


def forecast_scores(proba: pd.Series, labels: pd.Series) -> dict:
    """AUC (ranking quality), Brier score (probability accuracy, lower is better), base rate."""
    p, y = _aligned(proba, labels)
    auc = float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")
    return {"auc": auc, "brier": float(brier_score_loss(y, p)), "base_rate": float(y.mean()), "n": int(len(y))}


def calibration_table(proba: pd.Series, labels: pd.Series, bins: int = 5) -> pd.DataFrame:
    """When the model says X%, how often was it actually stress? Equal-width probability bins."""
    p, y = _aligned(proba, labels)
    cut = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y}).groupby(cut, observed=True).agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
    return t.reset_index(drop=True)


def block_bootstrap_ci(a: pd.Series, b: pd.Series, metric: Callable[[pd.Series], float],
                       block: int = 21, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """metric(a) - metric(b), with a 95% circular block-bootstrap interval.

    Whole blocks of consecutive days are resampled (keeping volatility clustering), and
    the same blocks are used for a and b so the comparison stays paired.
    """
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    point = float(metric(df["a"]) - metric(df["b"]))
    m = len(df)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    A, B = df["a"].to_numpy(), df["b"].to_numpy()
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m, n_blocks)
        idx = ((starts[:, None] + np.arange(block)) % m).ravel()[:m]
        diffs[i] = metric(pd.Series(A[idx])) - metric(pd.Series(B[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)
