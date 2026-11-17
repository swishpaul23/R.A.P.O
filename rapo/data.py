"""Price data with a local Parquet cache.

All network access goes through a `Downloader`, so tests (and offline runs)
can swap in a fake. Each ticker is cached in its own Parquet file plus a small
JSON sidecar recording the date range that has already been requested, so
re-running a backtest doesn't hit Yahoo again and extending the end date only
downloads the missing tail.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

Downloader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]

DEFAULT_CACHE = Path(".cache/prices")
RISK_FREE_TICKER = "^IRX"  # 13-week US T-bill yield, quoted in annual percent


def yf_downloader(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Adjusted daily close for one ticker from Yahoo Finance (empty if unknown)."""
    import yfinance as yf

    # yfinance's `end` is exclusive, so ask for one extra day.
    raw = yf.download(
        ticker,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def _paths(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    safe = ticker.replace("^", "_").replace("/", "_")
    return cache_dir / f"{safe}.parquet", cache_dir / f"{safe}.json"


def _load_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
              cache_dir: Path, downloader: Downloader) -> pd.Series:
    data_path, meta_path = _paths(cache_dir, ticker)
    cached = None
    meta = None
    if data_path.exists() and meta_path.exists():
        cached = pd.read_parquet(data_path)["price"]
        meta = {k: pd.Timestamp(v) for k, v in json.loads(meta_path.read_text()).items()}

    if meta and meta["from"] <= start and meta["through"] >= end:
        series = cached
    elif meta and meta["from"] <= start:
        tail = downloader(ticker, meta["through"] + pd.Timedelta(days=1), end)
        series = pd.concat([cached, tail])
        series = series[~series.index.duplicated(keep="last")].sort_index()
        meta["through"] = end
    else:
        lo = min(start, meta["from"]) if meta else start
        hi = max(end, meta["through"]) if meta else end
        series = downloader(ticker, lo, hi).sort_index()
        meta = {"from": lo, "through": hi}

    if series is None or series.empty:
        return pd.Series(dtype=float, name=ticker)

    cache_dir.mkdir(parents=True, exist_ok=True)
    series.to_frame("price").to_parquet(data_path)
    meta_path.write_text(json.dumps({k: v.isoformat() for k, v in meta.items()}))

    out = series[(series.index >= start) & (series.index <= end)]
    out.name = ticker
    return out


def fetch_prices(tickers: list[str], start: str, end: str,
                 cache_dir: Path = DEFAULT_CACHE,
                 downloader: Downloader = yf_downloader) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker in input order.

    Dates are outer-joined: an asset that listed later is NaN before its
    first trading day. Raises ValueError naming any ticker with no data.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir = Path(cache_dir)
    columns = {t: _load_one(t, s, e, cache_dir, downloader) for t in tickers}
    missing = [t for t, col in columns.items() if col.empty]
    if missing:
        raise ValueError(f"No price data for: {', '.join(missing)}. Check the symbol and your internet connection.")
    df = pd.concat(columns.values(), axis=1, join="outer").sort_index()
    df.columns = list(tickers)
    df.index.name = "date"
    return df


def risk_free_daily(start: str, end: str, cache_dir: Path = DEFAULT_CACHE,
                    downloader: Downloader = yf_downloader) -> pd.Series:
    """Daily risk-free rate from the 13-week T-bill yield (annual % / 100 / 252)."""
    irx = fetch_prices([RISK_FREE_TICKER], start, end, cache_dir, downloader)[RISK_FREE_TICKER]
    rf = (irx / 100.0 / 252.0).ffill()
    rf.name = "rf"
    return rf
