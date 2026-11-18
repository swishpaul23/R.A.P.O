"""Save a snapshot of real daily prices to data/snapshot/ for the deployed demo.

Yahoo Finance often blocks requests from cloud servers, so the deployed API falls
back to this snapshot when live downloads fail (the UI shows a banner when it does).

Run from the repo root on your own machine:
    python scripts/make_snapshot.py
then commit data/snapshot/.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapo.data import RISK_FREE_TICKER, yf_downloader  # noqa: E402

TICKERS = [
    "SPY", "QQQ", "VTI", "EFA", "EEM",          # equities
    "TLT", "IEF", "AGG",                        # bonds
    "GLD", "VNQ",                               # gold, real estate
    "XLP", "XLU", "XLK",                        # sectors
    "NVDA", "AAPL", "MSFT",                     # single stocks
    "HYG", "SHY", "BIL",                        # credit, short T-bills
    "^VIX", "^TNX",                             # volatility index, 10-year yield
    RISK_FREE_TICKER,                           # risk-free rate
]
START = "2005-01-01"
OUT = Path(__file__).resolve().parent.parent / "data" / "snapshot"


def main() -> None:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(date.today())
    for t in TICKERS:
        s = yf_downloader(t, pd.Timestamp(START), end)
        if s.empty:
            print(f"  skipped {t}: no data")
            continue
        s.round(6).rename("price").rename_axis("date").to_csv(OUT / f"{t.replace('^', '_')}.csv")
        print(f"  saved {t}: {s.index[0].date()} to {s.index[-1].date()} ({len(s)} days)")
    (OUT / "AS_OF.txt").write_text(f"{end.date()}\n")
    print(f"Snapshot written to {OUT}")


if __name__ == "__main__":
    main()
