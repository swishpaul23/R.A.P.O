"""Turn a stress probability into a gradual, capped shift toward defensive assets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rapo.strategies import Strategy

DEFAULT_DEFENSIVE = {"SHY": 0.5, "GLD": 0.5}


def tilt_from_proba(p: pd.Series, cap: float = 0.40, floor: float = 0.20, span: int = 5) -> pd.Series:
    """Share of the portfolio to move to defensive assets.

    Smooth the probability (EMA) so day-to-day noise doesn't trigger trades, ignore it
    below `floor`, then scale linearly up to `cap` at probability 1. Missing = no tilt.
    """
    smooth = p.fillna(0.0).ewm(span=span, adjust=False).mean()
    return (cap * np.clip((smooth - floor) / (1.0 - floor), 0.0, 1.0)).rename("tilt")


def percentile_tilt(signal: pd.Series, q: float, cap: float = 0.40, min_history: int = 252,
                    span: int = 5) -> pd.Series:
    """Tilt when the signal is in the top (1 - q) of its own history so far.

    The percentile rank at d uses only values dated <= d (expanding window), so the
    rule adapts to each signal's scale: calibrated probabilities, HMM state
    probabilities and raw volatility can all be compared on the same footing.
    """
    rank = signal.expanding(min_periods=min_history).rank(pct=True)
    raw = cap * np.clip((rank - q) / (1.0 - q), 0.0, 1.0)
    return raw.fillna(0.0).ewm(span=span, adjust=False).mean().rename("tilt")


def make_tilt_overlay(base: Strategy, tilt: pd.Series, defensive: dict[str, float] | None = None,
                      base_assets: list[str] | None = None) -> Strategy:
    """Wrap `base`: weights = (1 - tilt) x base + tilt x defensive.

    `base_assets` limits what the base strategy sees (so defensive-only tickers like
    SHY aren't also traded by HRP). The tilt is read as of the window's last day, which
    is the day before the rebalance, so nothing from the future leaks in.
    """
    defensive = defensive or DEFAULT_DEFENSIVE

    def strategy(window: pd.DataFrame) -> pd.Series:
        cols = list(window.columns)
        base_cols = [c for c in cols if base_assets is None or c in base_assets] or cols
        w_base = base(window[base_cols]).reindex(cols).fillna(0.0)

        d = pd.Series({c: v for c, v in defensive.items() if c in cols}, dtype=float)
        t = tilt.asof(window.index[-1]) if len(tilt) else np.nan
        t = 0.0 if pd.isna(t) or d.sum() <= 0 else float(t)
        if t == 0.0:
            return w_base / w_base.sum()
        w_def = (d / d.sum()).reindex(cols).fillna(0.0)
        w = (1.0 - t) * w_base + t * w_def
        return w / w.sum()

    return strategy


def make_regime_overlay(base: Strategy, proba: pd.Series, defensive: dict[str, float] | None = None,
                        base_assets: list[str] | None = None, cap: float = 0.40, floor: float = 0.20,
                        span: int = 5) -> Strategy:
    """Round-1 overlay: tilt from an absolute probability floor (see `tilt_from_proba`)."""
    return make_tilt_overlay(base, tilt_from_proba(proba, cap, floor, span), defensive, base_assets)
