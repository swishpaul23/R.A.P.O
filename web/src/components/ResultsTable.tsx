import { ArrowDown, ArrowUp } from "lucide-react";
import { useMemo, useState } from "react";
import type { MetricRow } from "../lib/api";
import { fmtNum, fmtPct, toneOf } from "../lib/format";
import { GLOSSARY } from "../lib/glossary";
import { colorFor } from "../lib/theme";
import InfoTip from "./InfoTip";

type Key = keyof Omit<MetricRow, "name">;
type Col = { key: Key; label: string; term: string; fmt: (v: number | null | undefined) => string; tone?: boolean };
const RETURN_FULL: Col = { key: "cagr", label: "CAGR", term: "CAGR", fmt: (v) => fmtPct(v), tone: true };
const RETURN_WINDOW: Col = { key: "total_return", label: "Return", term: "Return in window", fmt: (v) => fmtPct(v), tone: true };
const REST: Col[] = [
  { key: "ann_vol", label: "Volatility", term: "Volatility", fmt: (v) => fmtPct(v) },
  { key: "sharpe", label: "Sharpe", term: "Sharpe", fmt: (v) => fmtNum(v) },
  { key: "sortino", label: "Sortino", term: "Sortino", fmt: (v) => fmtNum(v) },
  { key: "max_drawdown", label: "Max DD", term: "Max drawdown", fmt: (v) => fmtPct(v, 1), tone: true },
  { key: "calmar", label: "Calmar", term: "Calmar", fmt: (v) => fmtNum(v) },
  { key: "var_95", label: "VaR 95", term: "VaR 95% (1 day)", fmt: (v) => fmtPct(v) },
  { key: "annual_turnover", label: "Turnover", term: "Turnover / yr", fmt: (v) => (v == null ? "—" : `${fmtNum(v)}×`) },
];

interface Props {
  rows: MetricRow[];
  benchmarkLabel: string;
  focus: string;
  onFocus: (name: string) => void;
  isWindow: boolean;
}

export default function ResultsTable({ rows, benchmarkLabel, focus, onFocus, isWindow }: Props) {
  const COLS = [isWindow ? RETURN_WINDOW : RETURN_FULL, ...REST];
  const [sort, setSort] = useState<{ key: Key; desc: boolean } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a, b) => {
      const x = a[sort.key] ?? -Infinity, y = b[sort.key] ?? -Infinity;
      return sort.desc ? y - x : x - y;
    });
  }, [rows, sort]);

  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="w-full min-w-[760px] text-sm" aria-label="Strategy comparison">
        <thead className="bg-raised text-left text-xs text-fg-2">
          <tr>
            <th scope="col" className="px-3 py-3 font-medium">Strategy</th>
            {COLS.map((c) => {
              const active = sort?.key === c.key;
              return (
                <th key={c.key} scope="col" aria-sort={active ? (sort!.desc ? "descending" : "ascending") : "none"}
                  className="whitespace-nowrap px-1 py-1 text-right font-medium">
                  <button type="button"
                    onClick={() => setSort(active ? { key: c.key, desc: !sort!.desc } : { key: c.key, desc: true })}
                    className={`inline-flex min-h-11 items-center gap-1 rounded px-2 hover:text-fg ${active ? "text-fg" : ""}`}>
                    {c.label}
                    {active && (sort!.desc ? <ArrowDown className="size-3.5" aria-hidden="true" /> : <ArrowUp className="size-3.5" aria-hidden="true" />)}
                  </button>
                  <InfoTip term={c.term}>{GLOSSARY[c.term]}</InfoTip>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const isBench = r.name === benchmarkLabel;
            const selected = r.name === focus;
            return (
              <tr key={r.name}
                onClick={isBench ? undefined : () => onFocus(r.name)}
                className={`border-t border-line ${isBench ? "text-fg-2" : "cursor-pointer hover:bg-raised"} ${selected ? "bg-raised" : ""}`}>
                <th scope="row" className="px-3 py-3 text-left font-normal">
                  {isBench ? (
                    <span className="inline-flex items-center gap-3">
                      <span aria-hidden="true" className="h-0.5 w-3 border-t-2 border-dashed" style={{ borderColor: colorFor(r.name, benchmarkLabel) }} />
                      {r.name}
                    </span>
                  ) : (
                    <button type="button" aria-pressed={selected} onClick={(e) => { e.stopPropagation(); onFocus(r.name); }}
                      className="inline-flex min-h-11 items-center gap-3 text-left">
                      <span aria-hidden="true" className={`size-2.5 rounded-full ${selected ? "ring-2 ring-offset-2 ring-offset-surface" : ""}`}
                        style={{ background: colorFor(r.name, benchmarkLabel), ["--tw-ring-color" as string]: colorFor(r.name, benchmarkLabel) }} />
                      <span className={selected ? "font-medium text-fg" : "text-fg"}>{r.name}</span>
                    </button>
                  )}
                </th>
                {COLS.map((c) => {
                  const v = r[c.key];
                  const tone = c.tone ? toneOf(v) : "neutral";
                  return (
                    <td key={c.key} className={`num px-3 py-3 text-right ${tone === "pos" ? "text-gain" : tone === "neg" ? "text-loss" : ""}`}>
                      {c.fmt(v)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
