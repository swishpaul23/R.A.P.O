import { AlertCircle, Database, Download, Eye, EyeOff } from "lucide-react";
import { lazy, Suspense, useMemo, useRef, useState } from "react";
import AllocationBars from "./components/AllocationBars";
import CorrelationHeatmap from "./components/CorrelationHeatmap";
import FrontierChart from "./components/FrontierChart";
import KpiStrip from "./components/KpiStrip";
import Panel, { EmptyHint, Segmented, Skeleton } from "./components/Panel";
import ParamsForm from "./components/ParamsForm";
import ResultsTable from "./components/ResultsTable";
import { ApiError, runBacktest, type BacktestResponse } from "./lib/api";
import { downloadCsv } from "./lib/csv";
import { localISODate } from "./lib/date";
import { parseTickers, type ParamsInput } from "./lib/validate";
import { colorFor } from "./lib/theme";

const EquityChart = lazy(() => import("./components/EquityChart"));

const today = localISODate();
const DEFAULTS: ParamsInput = {
  tickers: "SPY, TLT, GLD, NVDA, XLP", benchmark: "SPY", start: "2015-01-01", end: today,
  lookback: "252", rebalance: "63", cost: "10",
};
const FULL = "Full period";

export default function App() {
  const [params, setParams] = useState<ParamsInput>(DEFAULTS);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error" | "ready">("idle");
  const [error, setError] = useState("");
  const [focus, setFocus] = useState("HRP");
  const [mode, setMode] = useState<"equity" | "drawdown">("equity");
  const [period, setPeriod] = useState(FULL);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const abort = useRef<AbortController | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  async function run() {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setStatus("loading");
    setError("");
    try {
      let res: BacktestResponse;
      if (new URLSearchParams(window.location.search).has("demo")) {
        res = (await import("./lib/fixture")).FIXTURE;
      } else {
        res = await runBacktest({
          tickers: parseTickers(params.tickers), benchmark: params.benchmark.trim().toUpperCase(),
          start: params.start, end: params.end, lookback: Number(params.lookback),
          rebalance_every: Number(params.rebalance), cost_bps: Number(params.cost),
        }, ctrl.signal);
      }
      setData(res);
      setStatus("ready");
      if (!res.strategies.includes(focus)) setFocus(res.strategies[0]);
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof ApiError ? e.message : "Something went wrong running the backtest. Try again.");
      setStatus("error");
    }
  }

  const loading = status === "loading";
  const rows = useMemo(() => {
    if (!data) return [];
    if (period === FULL) return data.table;
    return data.stress.find((s) => s.window === period)?.rows ?? data.table;
  }, [data, period]);
  const windowRange = useMemo(() => {
    const w = data?.stress.find((s) => s.window === period);
    return w ? { from: w.start, to: w.end } : undefined;
  }, [data, period]);
  const focusRow = rows.find((r) => r.name === focus);
  const benchRow = data ? rows.find((r) => r.name === data.benchmark_label) : undefined;
  const periods = [FULL, ...(data?.stress.map((s) => s.window) ?? [])];

  function choose(name: string) {
    setFocus(name);
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    resultsRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
  }

  function toggle(name: string) {
    setHidden((h) => {
      const next = new Set(h);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-canvas/75 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1280px] items-center gap-3 px-4 md:px-6">
          <span className="num rounded-md border border-line-strong px-2 py-0.5 text-sm font-medium tracking-wide text-accent">R.A.P.O</span>
          <span className="text-sm text-fg-2">Robo-Advisor Portfolio Optimizer</span>
          <span className="num ml-auto hidden text-xs text-fg-3 md:inline">
            {data ? `${data.meta.first_trade_date} → ${data.meta.end} · ${data.meta.tickers.length} assets` : "v2.0"}
          </span>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1280px] flex-col gap-5 px-4 py-6 md:px-6 md:py-8">
        <div className="flex flex-wrap items-start gap-3">
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold leading-tight tracking-tight">Portfolio research</h1>
            <p className="mt-1 text-sm text-fg-2">
              Walk-forward backtest of five allocation strategies against a benchmark, after trading costs.
            </p>
          </div>
          {data && <button type="button" onClick={() => downloadCsv(`rapo-${period.replace(/\s+/g, "-").toLowerCase()}.csv`, rows as never)}
            className="ml-auto inline-flex h-11 items-center gap-2 rounded-lg border border-line-strong px-4 text-sm text-fg-2 hover:bg-raised hover:text-fg">
            <Download className="size-4" aria-hidden="true" /> Export CSV
          </button>}
        </div>

        <div className="rounded-xl border border-line bg-surface p-4">
          <ParamsForm value={params} onChange={setParams} onSubmit={run} running={loading} />
        </div>

        {status === "error" && (
          <div role="alert" className="flex items-start gap-3 rounded-xl border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-fg">
            <AlertCircle className="mt-0.5 size-4 shrink-0 text-loss" aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}

        {data?.meta.data_source === "snapshot" && (
          <div role="status" className="flex items-start gap-3 rounded-xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-fg">
            <Database className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden="true" />
            <span>
              Live prices are unavailable, so these results use a saved snapshot of real historical prices.
              Recent days may be missing.
            </span>
          </div>
        )}

        {data && (
          <div className="flex flex-wrap items-center gap-3">
            <Segmented label="Period" value={period} onChange={setPeriod}
              options={periods.map((p) => ({ value: p, label: p }))} />
            <span className="text-xs text-fg-3">Focus: <span className="text-fg-2">{focus}</span> · select a row to change</span>
          </div>
        )}

        <KpiStrip row={focusRow} bench={benchRow} loading={loading} isWindow={period !== FULL} />

        <Panel title={mode === "equity" ? "Growth of $1" : "Drawdown"}
          subtitle="Out-of-sample, net of costs. Shaded bands are stress windows."
          actions={<Segmented label="Chart" value={mode} onChange={setMode}
            options={[{ value: "equity", label: "Equity" }, { value: "drawdown", label: "Drawdown" }]} />}>
          {loading ? <Skeleton className="h-[320px] w-full md:h-[380px]" /> : data ? (
            <>
              <Suspense fallback={<Skeleton className="h-[320px] w-full md:h-[380px]" />}>
                <EquityChart data={data} mode={mode} focus={focus} hidden={hidden} range={windowRange} />
              </Suspense>
              <ul className="mt-3 flex flex-wrap gap-2" aria-label="Series">
                {[...data.strategies, data.benchmark_label].map((n) => {
                  const off = hidden.has(n);
                  return (
                    <li key={n}>
                      <button type="button" onClick={() => toggle(n)} aria-pressed={!off}
                        className={`inline-flex min-h-11 items-center gap-2 rounded-md border border-line px-3 text-xs ${off ? "text-fg-3" : "text-fg-2 hover:text-fg"}`}>
                        <span aria-hidden="true" className="h-0.5 w-3" style={{ background: colorFor(n, data.benchmark_label), opacity: off ? 0.3 : 1 }} />
                        {n}
                        {off ? <EyeOff className="size-3.5" aria-hidden="true" /> : <Eye className="size-3.5" aria-hidden="true" />}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </>
          ) : (
            <div className="flex h-[320px] items-center justify-center rounded-lg border border-dashed border-line md:h-[380px]">
              <EmptyHint>Run a backtest to plot strategy growth here.</EmptyHint>
            </div>
          )}
        </Panel>

        <div className="grid gap-5 md:grid-cols-12">
          <Panel title="Efficient frontier" subtitle="3,000 random portfolios, last lookback window" className="md:col-span-7">
            {loading ? <Skeleton className="aspect-[7/4] w-full" /> : data ? <FrontierChart data={data} focus={focus} />
              : <EmptyHint>Run a backtest to see where each strategy sits on the risk/return frontier.</EmptyHint>}
          </Panel>
          <Panel title="Current allocation" subtitle={`${focus}, latest rebalance`} className="md:col-span-5">
            {loading ? <Skeleton className="h-48 w-full" /> : data ? <AllocationBars data={data} focus={focus} />
              : <EmptyHint>Target weights for the selected strategy appear here.</EmptyHint>}
          </Panel>
          <Panel title="Correlation" subtitle="Ordered by HRP clusters" className="md:col-span-12">
            {loading ? <Skeleton className="h-48 w-full" /> : data ? <CorrelationHeatmap data={data} />
              : <EmptyHint>Pairwise correlations, grouped by the HRP cluster tree.</EmptyHint>}
          </Panel>
        </div>

        <div ref={resultsRef}>
          <Panel title="Strategy comparison" subtitle={period === FULL ? "Full out-of-sample period" : `Stress window: ${period}`}>
            {loading ? <Skeleton className="h-64 w-full" /> : data ? (
              <ResultsTable rows={rows} benchmarkLabel={data.benchmark_label} focus={focus} onFocus={choose} isWindow={period !== FULL} />
            ) : (
              <div className="rounded-lg border border-line">
                <div className="border-b border-line bg-raised px-3 py-3 text-xs text-fg-2">Strategy · CAGR · Volatility · Sharpe · Max DD</div>
                <div className="px-3 py-10 text-center"><EmptyHint>Run a backtest to compare strategies.</EmptyHint></div>
              </div>
            )}
          </Panel>
        </div>

        <footer className="pb-4 text-xs text-fg-3">
          Research tool, not investment advice. Past performance does not predict future returns.
        </footer>
      </main>
    </div>
  );
}
