import numpy as np
import pandas as pd

from rapo.regime.hmm import hmm_stress_proba


def regimes(seed=0, block=250, n_blocks=8):
    rng = np.random.default_rng(seed)
    parts, stressed = [], []
    for b in range(n_blocks):
        sigma = 0.03 if b % 2 else 0.005
        parts.append(rng.normal(0, sigma, block))
        stressed += [b % 2 == 1] * block
    idx = pd.bdate_range("2000-01-01", periods=block * n_blocks)
    return pd.Series(np.concatenate(parts), index=idx), pd.Series(stressed, index=idx)


def test_finds_high_volatility_blocks():
    r, stressed = regimes()
    p = hmm_stress_proba(r, min_train=750, refit_every=63)
    live = p.notna()
    # Skip the first 30 days of each block so the 21-day vol window has caught up.
    settled = pd.Series(np.tile(np.r_[np.zeros(30, bool), np.ones(220, bool)], 8), index=r.index)
    assert p[live & stressed & settled].mean() > 0.7
    assert p[live & ~stressed & settled].mean() < 0.3


def test_nan_before_min_train():
    r, _ = regimes()
    p = hmm_stress_proba(r, min_train=750, refit_every=63)
    assert p.iloc[:740].isna().all()


def test_no_lookahead():
    r, _ = regimes()
    base = hmm_stress_proba(r, min_train=750, refit_every=63)
    shocked = r.copy()
    shocked.iloc[1001:] = shocked.iloc[1001:] * 5
    after = hmm_stress_proba(shocked, min_train=750, refit_every=63)
    assert np.allclose(base.iloc[:1001].to_numpy(), after.iloc[:1001].to_numpy(), equal_nan=True)
