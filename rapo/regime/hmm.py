"""Hidden Markov Model baseline: the classic finance regime model.

A 2-state Gaussian HMM is fitted to [daily return, 21-day volatility]. The state
with the higher mean volatility is "stress". Unlike the classifiers, it learns the
regimes itself instead of being told which days were stressful.

To avoid look-ahead, the model is refitted on data <= d, and probabilities come from
the forward (filtering) pass only, so the value at t never uses observations after t.
(hmmlearn's predict_proba is the smoothed posterior, which does use the future.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def _observations(spy_returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ret": spy_returns, "vol": spy_returns.rolling(21).std()}).dropna()


def _filtered(model, X: np.ndarray) -> np.ndarray:
    """Forward-algorithm state probabilities P(state_t | obs_0..t)."""
    k = model.n_components
    log_b = np.column_stack([
        multivariate_normal(model.means_[j], model.covars_[j], allow_singular=True).logpdf(X)
        for j in range(k)
    ])
    log_a = np.log(model.transmat_ + 1e-300)
    out = np.empty_like(log_b)
    alpha = np.log(model.startprob_ + 1e-300) + log_b[0]
    out[0] = alpha - logsumexp(alpha)
    for t in range(1, len(X)):
        alpha = logsumexp(out[t - 1][:, None] + log_a, axis=0) + log_b[t]
        out[t] = alpha - logsumexp(alpha)
    return np.exp(out)


def hmm_stress_proba(spy_returns: pd.Series, min_train: int = 750, refit_every: int = 63,
                     seed: int = 0) -> pd.Series:
    from hmmlearn.hmm import GaussianHMM  # dev-only dependency: training runs offline

    obs = _observations(spy_returns)
    X = obs.to_numpy()
    scale = X[:min_train].std(axis=0)  # fixed scaling from the first training window only
    Xs = X / scale
    proba = pd.Series(np.nan, index=spy_returns.index, name="hmm")

    for d in range(min_train, len(Xs), refit_every):
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=seed)
        model.fit(Xs[: d + 1])
        stress = int(np.argmax(model.means_[:, 1]))
        end = min(d + refit_every, len(Xs))
        filt = _filtered(model, Xs[:end])
        proba.loc[obs.index[d:end]] = filt[d:end, stress]
    return proba
