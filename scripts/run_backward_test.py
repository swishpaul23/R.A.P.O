"""Backward test of the locked round-2 rule on 2000-2009 (dot-com bust and 2008).

    python scripts/run_backward_test.py

Needs the 1993+ snapshot with the stand-in funds (re-run scripts/make_snapshot.py).
Writes data/regime/backward.json and docs/results/phase2-backward.md.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapo.data import fetch_prices, risk_free_daily, snapshot_downloader  # noqa: E402
from rapo.regime.backward import DEFENSIVE, SIXTY, UNIVERSE, run_backward  # noqa: E402
from scripts.run_regime_study import num, pct  # noqa: E402

SNAPSHOT = ROOT / "data" / "snapshot"


def markdown(res: dict) -> str:
    r = res["rule"]
    out = [
        "# Phase 2, backward test: locked rule on 2000-2009",
        "",
        "Protocol: `phase2-backward-protocol.md` (committed before this run).",
        f"Rule: {r['signal']} at q = {r['q']:.2f}, chosen on 2010-2019. "
        f"Test: {res['series']['dates'][0]} to {res['series']['dates'][-1]}. "
        f"Signal first available {res['signal_first_date']}.",
        f"Average tilt {pct(res['avg_tilt'])}, {res['whipsaws']} whipsaws.",
        "",
        "| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr |",
        "|---|---|---|---|---|---|",
    ]
    for row in res["table"]:
        out.append(f"| {row['name']} | {pct(row['cagr'])} | {pct(row['ann_vol'])} | {num(row['sharpe'])} | "
                   f"{pct(row['max_drawdown'])} | {num(row['annual_turnover'])} |")
    for w, rows in res["stress_windows"].items():
        out += ["", f"## {w}", "", "| Strategy | Return | Max DD | Sharpe |", "|---|---|---|---|"]
        for row in rows:
            out.append(f"| {row['name']} | {pct(row['total_return'])} | {pct(row['max_drawdown'])} | {num(row['sharpe'])} |")
    out += ["", "## Overlay minus base, 95% block bootstrap", "",
            "| Base | Metric | Difference | 95% interval | Significant |", "|---|---|---|---|---|"]
    for b in res["bootstrap"]:
        f = pct if b["metric"] == "max_drawdown" else num
        out.append(f"| {b['base']} | {b['metric']} | {f(b['diff'])} | {f(b['lo'])} to {f(b['hi'])} | "
                   f"{'yes' if b['significant'] else 'no'} |")
    out += ["", "Stand-ins: VGTSX (intl), VEIEX (EM), VUSTX (long Treasuries), VFITX (intermediate), "
            "VGSIX (REITs), GC=F (gold futures), VFISX (short Treasuries), VBMFX (bonds for 60/40).", ""]
    return "\n".join(out)


def main() -> None:
    tickers = sorted(set(UNIVERSE + list(DEFENSIVE) + list(SIXTY)))
    dl = snapshot_downloader(SNAPSHOT)
    cache = Path(tempfile.mkdtemp())
    prices = fetch_prices(tickers, "1993-01-29", "2009-12-31", cache, dl)
    rf = risk_free_daily("1993-01-29", "2009-12-31", cache, dl)
    print("First dates:", prices.apply(lambda s: s.first_valid_index().date().isoformat()).to_dict())
    res = run_backward(prices, rf)
    (ROOT / "data" / "regime").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "regime" / "backward.json").write_text(json.dumps(res, allow_nan=False, separators=(",", ":")))
    (ROOT / "docs" / "results" / "phase2-backward.md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
