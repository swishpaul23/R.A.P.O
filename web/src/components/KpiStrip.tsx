import type { MetricRow } from "../lib/api";
import { fmtNum, fmtPct, toneOf } from "../lib/format";
import { GLOSSARY } from "../lib/glossary";
import InfoTip from "./InfoTip";
import { Skeleton } from "./Panel";

interface Props {
  row?: MetricRow;
  bench?: MetricRow;
  loading: boolean;
  isWindow: boolean;
}

const TONE = { pos: "text-gain", neg: "text-loss", neutral: "text-fg" } as const;

export default function KpiStrip({ row, bench, loading, isWindow }: Props) {
  const ret = isWindow ? row?.total_return : row?.cagr;
  const benchRet = isWindow ? bench?.total_return : bench?.cagr;
  const tiles = [
    { label: isWindow ? "Return in window" : "CAGR", value: fmtPct(ret), tone: toneOf(ret),
      sub: row && bench ? `${fmtPct((ret ?? 0) - (benchRet ?? 0))} vs benchmark` : undefined },
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
          <div className="flex items-center justify-between gap-1 text-xs font-medium text-fg-3">
            <span>{t.label}</span>
            <InfoTip term={t.label}>{GLOSSARY[t.label]}</InfoTip>
          </div>
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
