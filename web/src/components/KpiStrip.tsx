import type { MetricRow } from "../lib/api";
import { fmtNum, fmtPct, toneOf } from "../lib/format";
import { Skeleton } from "./Panel";

interface Props {
  row?: MetricRow;
  bench?: MetricRow;
  loading: boolean;
}

const TONE = { pos: "text-gain", neg: "text-loss", neutral: "text-fg" } as const;

export default function KpiStrip({ row, bench, loading }: Props) {
  const tiles = [
    { label: "CAGR", value: fmtPct(row?.cagr), tone: toneOf(row?.cagr),
      sub: row && bench ? `${fmtPct((row.cagr ?? 0) - (bench.cagr ?? 0))} vs benchmark` : undefined },
    { label: "Sharpe", value: fmtNum(row?.sharpe), tone: "neutral" as const,
      sub: bench ? `Benchmark ${fmtNum(bench.sharpe)}` : undefined },
    { label: "Max drawdown", value: fmtPct(row?.max_drawdown, 1), tone: toneOf(row?.max_drawdown),
      sub: bench ? `Benchmark ${fmtPct(bench.max_drawdown, 1)}` : undefined },
    { label: "VaR 95% (1 day)", value: fmtPct(row?.var_95), tone: toneOf(row?.var_95),
      sub: "Historical, daily" },
    { label: "Turnover / yr", value: row?.annual_turnover == null ? "—" : `${fmtNum(row.annual_turnover, 2)}×`,
      tone: "neutral" as const, sub: "Traded value ÷ NAV" },
  ];
  return (
    <section aria-label="Key metrics" className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {tiles.map((t) => (
        <div key={t.label} className="rounded-xl border border-line bg-surface px-4 py-3">
          <div className="text-xs font-medium text-fg-3">{t.label}</div>
          {loading ? (
            <Skeleton className="mt-2 h-7 w-24" />
          ) : (
            <div className={`num mt-1 text-2xl leading-tight ${row ? TONE[t.tone] : "text-fg-3"}`}>{row ? t.value : "—"}</div>
          )}
          <div className="num mt-1 min-h-4 text-xs text-fg-3">{!loading && row ? t.sub : ""}</div>
        </div>
      ))}
    </section>
  );
}
