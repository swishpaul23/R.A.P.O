import { AlertCircle, ExternalLink } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import type { MetricRow } from "../../lib/api";
import { fmtNum, fmtPct } from "../../lib/format";
import { GLOSSARY } from "../../lib/glossary";
import { daysBetween, fetchRegime, type Boot, type RegimeResponse } from "../../lib/regime";
import InfoTip from "../InfoTip";
import Panel, { Segmented, Skeleton } from "../Panel";

const CycleChart = lazy(() => import("./CycleChart"));
const DOCS = "https://github.com/swishpaul23/R.A.P.O/blob/HEAD/docs/results";
const STALE_DAYS = 45;
const COLS = ["SPY", "60/40", "Equal weight", "Equal weight + regime", "HRP", "HRP + regime"];

function find(rows: MetricRow[] | undefined, name: string) {
  return rows?.find((r) => r.name === name);
}

function Th({ children, term, className = "" }: { children: string; term?: string; className?: string }) {
  return (
    <th scope="col" className={`whitespace-nowrap px-3 py-2 text-right font-medium ${className}`}>
      <span className="inline-flex items-center gap-1">{children}{term && <InfoTip term={term}>{GLOSSARY[term]}</InfoTip>}</span>
    </th>
  );
}

export default function ResearchPage({ today }: { today: string }) {
  const [data, setData] = useState<RegimeResponse | null>(null);
  const [error, setError] = useState("");
  const [period, setPeriod] = useState<"holdout" | "backward">("holdout");

  useEffect(() => {
    const ctrl = new AbortController();
    fetchRegime(ctrl.signal).then(setData).catch((e) => { if (!ctrl.signal.aborted) setError(e.message); });
    return () => ctrl.abort();
  }, []);

  const crises = useMemo(() => {
    if (!data) return [];
    const r2 = data.study.round2.holdout.stress_windows;
    const bw = data.backward?.stress_windows ?? {};
    return [...Object.entries(bw), ...Object.entries(r2)].map(([name, rows]) => ({ name, rows }));
  }, [data]);

  const boots = useMemo(() => {
    if (!data) return [] as (Boot & { period: string })[];
    return [
      ...(data.backward?.bootstrap ?? []).map((b) => ({ ...b, period: "2000–2009 (backward)" })),
      ...data.study.round2.holdout.bootstrap.map((b) => ({ ...b, period: "2020–2026 (holdout)" })),
    ];
  }, [data]);

  if (error) {
    return (
      <div role="alert" className="flex items-start gap-3 rounded-xl border border-loss/40 bg-loss/10 px-4 py-3 text-sm">
        <AlertCircle className="mt-0.5 size-4 shrink-0 text-loss" aria-hidden="true" />
        <span>{error}</span>
      </div>
    );
  }

  const study = data?.study;
  const stale = study ? daysBetween(study.as_of, today) > STALE_DAYS : false;
  const hrpBetter = crises.filter((c) => (find(c.rows, "HRP")?.max_drawdown ?? -1) > (find(c.rows, "SPY")?.max_drawdown ?? 0)).length;
  const significant = boots.filter((b) => b.significant).length;
  const chartSeries = period === "holdout" ? study?.round2.series : data?.backward?.series;

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-semibold leading-tight tracking-tight">Research</h1>
        <p className="mt-1 text-sm text-fg-2">
          Can a market-stress detector protect a portfolio? Walk-forward tests across four crises, 2000–2026.
        </p>
      </div>

      {stale && study && (
        <p role="status" className="rounded-xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm">
          Study data is from {study.as_of}. Re-run <code className="num">scripts/run_regime_study.py</code> to refresh it.
        </p>
      )}

      <section aria-label="Current reading" className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {[
          { label: "Defensive tilt", value: study ? fmtPct(study.round2.current.tilt, 1) : "—" },
          { label: "Stress probability", value: study ? fmtPct(study.current.proba) : "—" },
          { label: "Historical stress rate", value: study ? fmtPct(study.forecast_reference.base_rate, 1) : "—" },
          { label: "Data as of", value: study?.as_of ?? "—" },
        ].map((t) => (
          <div key={t.label} className="rounded-xl border border-line bg-surface px-4 py-3">
            <div className="flex items-center justify-between gap-1 text-xs font-medium text-fg-3">
              <span>{t.label}</span>
              {GLOSSARY[t.label] && <InfoTip term={t.label}>{GLOSSARY[t.label]}</InfoTip>}
            </div>
            {data ? <div className="num mt-1 text-2xl leading-tight">{t.value}</div> : <Skeleton className="mt-2 h-7 w-24" />}
          </div>
        ))}
      </section>

      {data && (
        <ul className="flex flex-col gap-1 text-sm text-fg-2" aria-label="Findings">
          <li>HRP's worst drawdown was smaller than SPY's in {hrpBetter} of {crises.length} crises.</li>
          <li>{significant} of {boots.length} overlay-versus-base differences were statistically significant.</li>
          <li>No stress forecast beat simply predicting the historical stress rate (Brier skill below 0% for every model).</li>
        </ul>
      )}

      <Panel title="Four crises" subtitle="Max drawdown inside each window; HRP return in the last column">
        {!data ? <Skeleton className="h-48 w-full" /> : (
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[820px] text-sm" aria-label="Four crises">
              <thead className="bg-raised text-xs text-fg-2">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Crisis</th>
                  {COLS.map((c) => <Th key={c}>{c}</Th>)}
                  <Th>HRP return</Th>
                </tr>
              </thead>
              <tbody>
                {crises.map(({ name, rows }) => (
                  <tr key={name} className="border-t border-line">
                    <th scope="row" className="px-3 py-3 text-left font-normal">{name}</th>
                    {COLS.map((c) => (
                      <td key={c} className="num px-3 py-3 text-right text-loss">{fmtPct(find(rows, c)?.max_drawdown, 1)}</td>
                    ))}
                    <td className={`num px-3 py-3 text-right ${(find(rows, "HRP")?.total_return ?? 0) >= 0 ? "text-gain" : "text-loss"}`}>
                      {fmtPct(find(rows, "HRP")?.total_return, 1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Does timing help?" subtitle="Regime overlay minus the same strategy without it. A positive drawdown difference means a smaller loss.">
        {!data ? <Skeleton className="h-48 w-full" /> : (
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[680px] text-sm" aria-label="Does timing help">
              <thead className="bg-raised text-xs text-fg-2">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Test period</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Base</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Metric</th>
                  <Th>Difference</Th>
                  <Th term="95% interval">95% interval</Th>
                  <Th>Significant</Th>
                </tr>
              </thead>
              <tbody>
                {boots.map((b) => {
                  const f = (v: number | null) => (b.metric === "sharpe" ? fmtNum(v) : fmtPct(v, 1));
                  return (
                    <tr key={`${b.period}${b.base}${b.metric}`} className="border-t border-line">
                      <td className="px-3 py-3">{b.period}</td>
                      <td className="px-3 py-3">{b.base}</td>
                      <td className="px-3 py-3">{b.metric === "sharpe" ? "Sharpe" : "Max drawdown"}</td>
                      <td className="num px-3 py-3 text-right">{f(b.diff)}</td>
                      <td className="num px-3 py-3 text-right text-fg-2">{f(b.lo)} to {f(b.hi)}</td>
                      <td className={`px-3 py-3 text-right ${b.significant ? "text-gain" : "text-fg-3"}`}>{b.significant ? "Yes" : "No"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="grid gap-5 lg:grid-cols-12">
        <Panel title="Locked rule in action" className="lg:col-span-7"
          subtitle={study ? `${study.round2.chosen.signal.toUpperCase()} signal, top ${Math.round((1 - study.round2.chosen.q) * 100)}% trigger. Bars = defensive tilt` : undefined}
          actions={data?.backward ? (
            <Segmented label="Test period" value={period} onChange={setPeriod}
              options={[{ value: "holdout", label: "2020–2026" }, { value: "backward", label: "2000–2009" }]} />
          ) : undefined}>
          {chartSeries ? (
            <>
              <Suspense fallback={<Skeleton className="h-[320px] w-full" />}><CycleChart series={chartSeries} /></Suspense>
              <ul className="mt-3 flex flex-wrap gap-4 text-xs text-fg-2" aria-label="Chart key">
                <li className="inline-flex items-center gap-2"><span aria-hidden="true" className="h-0.5 w-3 border-t-2 border-dashed border-[#8b949e]" />SPY</li>
                <li className="inline-flex items-center gap-2"><span aria-hidden="true" className="h-0.5 w-3 bg-[#8b85ff]" />HRP</li>
                <li className="inline-flex items-center gap-2"><span aria-hidden="true" className="h-0.5 w-3 bg-[#e3b341]" />HRP + regime</li>
              </ul>
            </>
          ) : <Skeleton className="h-[320px] w-full" />}
        </Panel>

        <Panel title="Forecast quality" subtitle="Same out-of-sample days, 2010–2026" className="lg:col-span-5">
          {!study ? <Skeleton className="h-48 w-full" /> : (
            <table className="w-full text-sm" aria-label="Forecast quality">
              <thead className="text-xs text-fg-2">
                <tr>
                  <th scope="col" className="py-2 text-left font-medium">Model</th>
                  <Th term="AUC">AUC</Th>
                  <Th term="Brier skill" className="pr-0">Brier skill</Th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-line text-fg-2">
                  <th scope="row" className="py-3 text-left font-normal">Historical rate</th>
                  <td className="num px-3 py-3 text-right">—</td><td className="num py-3 pl-3 text-right">0.0%</td>
                </tr>
                {Object.entries(study.forecast).map(([m, f]) => (
                  <tr key={m} className="border-t border-line">
                    <th scope="row" className="py-3 text-left font-normal">
                      {{ vix_rule: "VIX > 30", hmm: "HMM", logistic: "Logistic", boosting: "XGBoost" }[m] ?? m}
                    </th>
                    <td className="num px-3 py-3 text-right">{fmtNum(f.auc)}</td>
                    <td className={`num py-3 pl-3 text-right ${f.brier_skill < 0 ? "text-loss" : "text-gain"}`}>{fmtPct(f.brier_skill, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>

      <Panel title="Method">
        <ul className="flex flex-col gap-2 text-sm text-fg-2">
          <li>Every forecast is walk-forward: models only ever train on data available at the time.</li>
          <li>Round 2 chose its rule on 2010–2019, then ran once on 2020–2026 and once on 2000–2009. Both protocols were committed before the runs.</li>
          <li>Pre-2005 tests use long-history mutual funds as stand-ins for ETFs that didn't exist yet.</li>
        </ul>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          {[["phase2.md", "Round 1 results"], ["phase2-round2.md", "Round 2 results"], ["phase2-backward.md", "Backward test"]].map(([f, label]) => (
            <a key={f} href={`${DOCS}/${f}`} target="_blank" rel="noreferrer"
              className="inline-flex min-h-11 items-center gap-1 text-accent hover:underline">
              {label}<ExternalLink className="size-3.5" aria-hidden="true" />
            </a>
          ))}
        </div>
      </Panel>
    </div>
  );
}
