export interface MetricRow {
  name: string;
  cagr: number | null;
  ann_vol: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  calmar: number | null;
  var_95: number | null;
  annual_turnover: number | null;
}

export interface BacktestResponse {
  meta: {
    tickers: string[];
    benchmark: string;
    start: string;
    end: string;
    lookback: number;
    rebalance_every: number;
    cost_bps: number;
    first_trade_date: string;
    data_source?: "live" | "snapshot";
  };
  strategies: string[];
  benchmark_label: string;
  table: MetricRow[];
  stress: { window: string; start: string; end: string; rows: MetricRow[] }[];
  series: {
    dates: string[];
    equity: Record<string, number[]>;
    drawdown: Record<string, number[]>;
  };
  weights: Record<string, Record<string, number>>;
  eligible?: string[];
  correlation: { tickers: string[]; matrix: number[][] } | null;
  clusters: { order: string[]; linkage: number[][] } | null;
  frontier: {
    points: { vol: number; ret: number; sharpe: number }[];
    strategies: { name: string; vol: number; ret: number; sharpe: number | null }[];
  } | null;
}

export interface BacktestRequest {
  tickers: string[];
  benchmark: string;
  start: string;
  end: string;
  lookback: number;
  rebalance_every: number;
  cost_bps: number;
}

export class ApiError extends Error {}

export async function runBacktest(req: BacktestRequest, signal?: AbortSignal): Promise<BacktestResponse> {
  let res: Response;
  try {
    res = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch {
    throw new ApiError("Can't reach the API. Start it with `uvicorn api.main:app` and try again.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join(" ");
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail);
  }
  return res.json();
}
