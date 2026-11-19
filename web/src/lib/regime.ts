import type { MetricRow } from "./api";

export interface Boot { base: string; metric: "sharpe" | "max_drawdown"; diff: number | null; lo: number | null; hi: number | null; significant: boolean }
type WindowRows = Record<string, MetricRow[]>;
interface RuleSeries { dates: string[]; tilt: number[]; spy: number[]; hrp: number[]; hrp_regime: number[]; signal?: (number | null)[] }

export interface RegimeResponse {
  study: {
    [key: string]: unknown;
    as_of: string;
    chosen_model: string;
    forecast: Record<string, { auc: number | null; brier: number; brier_skill: number; base_rate: number; n: number }>;
    forecast_reference: { auc: number | null; brier: number; base_rate: number; n: number };
    current: { date: string; proba: number; tilt: number };
    round2: {
      periods: { development_start: string; split: string; holdout_start: string; holdout_end: string };
      development: { signal: string; q: number; score: number; sharpe_gain: Record<string, number>; avg_tilt: number; whipsaws: number }[];
      chosen: { signal: string; q: number };
      holdout: { table: MetricRow[]; stress_windows: WindowRows; bootstrap: Boot[]; whipsaws: number; avg_tilt: number };
      series: RuleSeries;
      current: { date: string; tilt: number };
    };
  };
  backward: {
    rule: { signal: string; q: number };
    period: { start: string; end: string };
    signal_first_date: string;
    whipsaws: number;
    avg_tilt: number;
    table: MetricRow[];
    stress_windows: WindowRows;
    bootstrap: Boot[];
    series: RuleSeries;
  } | null;
}

export async function fetchRegime(signal?: AbortSignal): Promise<RegimeResponse> {
  if (new URLSearchParams(window.location.search).has("demo")) {
    return (await import("./regimeFixture")).REGIME_FIXTURE;
  }
  let res: Response;
  try {
    res = await fetch("/api/regime", { signal });
  } catch {
    throw new Error("Can't reach the API. Start it with `uvicorn api.main:app` and try again.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try { const b = await res.json(); if (typeof b.detail === "string") detail = b.detail; } catch { /* keep generic */ }
    throw new Error(detail);
  }
  return res.json();
}

/** Days between two ISO dates. */
export function daysBetween(a: string, b: string): number {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86_400_000);
}
