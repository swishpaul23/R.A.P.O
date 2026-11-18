"""Run the Phase 2 regime study on the committed price snapshot.

    python scripts/run_regime_study.py

Writes data/regime/latest.json (served by the API) and docs/results/phase2.md.
Takes a few minutes: every model is refitted hundreds of times, walk-forward.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapo.data import fetch_prices, risk_free_daily, snapshot_downloader  # noqa: E402
from rapo.regime.features import REQUIRED  # noqa: E402
from rapo.regime.study import run_study  # noqa: E402

UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ"]
START = "2005-01-01"
SNAPSHOT = ROOT / "data" / "snapshot"


def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def markdown(res: dict) -> str:
    s = res["settings"]
    out = [
        "# Phase 2 results: regime detector",
        "",
        f"Data through {res['as_of']}. Out-of-sample from {s['oos_start']}. Universe: {', '.join(s['universe'])}.",
        f"Stress label: {s['label']}. Tilt: {s['tilt']} into {s['defensive']}. "
        f"Monthly rebalance, {s['cost_bps']:.0f} bps costs.",
        "",
        "## Forecast quality (same out-of-sample days for every model)",
        "",
        "| Model | AUC | Brier | Skill vs climatology | Days |",
        "|---|---|---|---|---|",
    ]
    ref = res["forecast_reference"]
    out.append(f"| climatology (historical stress rate) | — | {num(ref['brier'], 3)} | 0% | {ref['n']} |")
    for m, f in res["forecast"].items():
        mark = " (chosen)" if m == res["chosen_model"] else ""
        out.append(f"| {m}{mark} | {num(f['auc'])} | {num(f['brier'], 3)} | {pct(f['brier_skill'])} | {f['n']} |")
    out += ["", f"Stress base rate in this period: {pct(ref['base_rate'])}. AUC 0.5 = no better than chance at ranking days; "
            "Brier skill below 0% = worse than always forecasting the historical stress rate. "
            "The model with the lowest Brier drives the overlay.", "",
            "## Strategies, full out-of-sample period", "",
            "| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr | Whipsaws |", "|---|---|---|---|---|---|---|"]
    for r in res["backtests"]:
        out.append(f"| {r['name']} | {pct(r['cagr'])} | {pct(r['ann_vol'])} | {num(r['sharpe'])} | "
                   f"{pct(r['max_drawdown'])} | {num(r['annual_turnover'])} | {r.get('whipsaws', '—')} |")
    for w, rows in res["stress_windows"].items():
        out += ["", f"## {w}", "", "| Strategy | Return | Max DD | Sharpe |", "|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r['name']} | {pct(r['total_return'])} | {pct(r['max_drawdown'])} | {num(r['sharpe'])} |")
    out += ["", "## Is the overlay's edge more than luck? (95% block bootstrap)", "",
            "| Base | Metric | Overlay − base | 95% interval | Significant |", "|---|---|---|---|---|"]
    for b in res["bootstrap"]:
        f = pct if b["metric"] == "max_drawdown" else num
        out.append(f"| {b['base']} | {b['metric']} | {f(b['diff'])} | {f(b['lo'])} to {f(b['hi'])} | "
                   f"{'yes' if b['significant'] else 'no'} |")
    c = res["current"]
    out += ["", f"Latest reading ({c['date']}): stress probability {pct(c['proba'])}, tilt {pct(c['tilt'])}.", ""]
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    tickers = sorted(set(UNIVERSE + REQUIRED + ["SHY", "AGG", "^IRX"]))
    dl = snapshot_downloader(SNAPSHOT)
    cache = Path(tempfile.mkdtemp())
    as_of = (SNAPSHOT / "AS_OF.txt").read_text().strip()
    prices = fetch_prices(tickers, START, as_of, cache, dl)
    rf = risk_free_daily(START, as_of, cache, dl)
    res = run_study(prices, rf, UNIVERSE, START)

    (ROOT / "data" / "regime").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "regime" / "latest.json").write_text(json.dumps(res, allow_nan=False, separators=(",", ":")))
    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "phase2.md").write_text(markdown(res))
    print(markdown(res))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
