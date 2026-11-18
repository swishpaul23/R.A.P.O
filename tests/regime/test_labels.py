import numpy as np
import pandas as pd

from rapo.regime.labels import stress_label


def s(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_hand_path():
    lab = stress_label(s([100, 100, 91, 100, 100]), horizon=2, threshold=-0.08)
    assert lab.iloc[:3].tolist() == [1.0, 1.0, 0.0]
    assert lab.iloc[3:].isna().all()


def test_exact_threshold_counts_as_stress():
    assert stress_label(s([100, 92, 100]), horizon=1, threshold=-0.08).iloc[0] == 1.0


def test_no_lookahead_beyond_horizon():
    base = stress_label(s(np.linspace(100, 120, 50)), horizon=5)
    shocked = s(np.r_[np.linspace(100, 120, 50)[:30], np.full(20, 10.0)])
    lab = stress_label(shocked, horizon=5)
    # Row 24 looks at rows 25..29 only, so the crash at row 30 cannot affect it.
    assert lab.iloc[:25].equals(base.iloc[:25])
    assert lab.iloc[25] == 1.0
