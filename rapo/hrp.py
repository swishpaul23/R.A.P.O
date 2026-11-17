"""Hierarchical Risk Parity (Lopez de Prado, 2016).

1. Cluster assets by correlation distance d = sqrt((1 - corr) / 2).
2. Reorder the covariance so similar assets sit next to each other.
3. Split the ordered list in half repeatedly, giving each half capital in
   inverse proportion to its variance.

No matrix inversion is needed, so HRP stays stable when assets are highly
correlated, which is where mean-variance optimizers break down.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def _ivp(cov: pd.DataFrame) -> np.ndarray:
    """Inverse-variance weights within a cluster."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    w = _ivp(sub)
    return float(w @ sub.to_numpy() @ w)


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Leaf order of the dendrogram, so neighbours are similar assets."""
    return list(sch.leaves_list(link))


def _recursive_bisection(cov: pd.DataFrame, order: list) -> pd.Series:
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left, v_right = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hrp_linkage(returns: pd.DataFrame) -> np.ndarray:
    """Single-linkage tree on correlation distance (also used for the dendrogram)."""
    corr = returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform
    return sch.linkage(squareform(dist, checks=False), method="single")


def hrp(returns: pd.DataFrame) -> pd.Series:
    """HRP weights, indexed in the input column order."""
    cols = list(returns.columns)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    cov = returns.cov()
    order = [cols[i] for i in _quasi_diag(hrp_linkage(returns))]
    w = _recursive_bisection(cov, order)
    w = w[cols]
    return w / w.sum()
