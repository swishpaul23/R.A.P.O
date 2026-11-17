export interface ParamsInput {
  tickers: string;
  benchmark: string;
  start: string;
  end: string;
  lookback: string;
  rebalance: string;
  cost: string;
}

export type FieldErrors = Partial<Record<keyof ParamsInput, string>>;

export function parseTickers(raw: string): string[] {
  const out: string[] = [];
  for (const t of raw.split(/[\s,]+/)) {
    const s = t.trim().toUpperCase();
    if (s && !out.includes(s)) out.push(s);
  }
  return out;
}

function intIn(raw: string, lo: number, hi: number): boolean {
  const n = Number(raw);
  return raw.trim() !== "" && Number.isInteger(n) && n >= lo && n <= hi;
}

export function validateField(key: keyof ParamsInput, p: ParamsInput): string | undefined {
  switch (key) {
    case "tickers":
      return parseTickers(p.tickers).length === 0 ? "Add at least one ticker." : undefined;
    case "benchmark":
      return p.benchmark.trim() === "" ? "Add a benchmark ticker." : undefined;
    case "start":
      if (!p.start) return "Pick a start date.";
      return p.end && p.start >= p.end ? "Start must be before end." : undefined;
    case "end":
      return undefined;
    case "lookback":
      return intIn(p.lookback, 20, 1260) ? undefined : "Use 20 to 1260 days.";
    case "rebalance":
      return intIn(p.rebalance, 1, 252) ? undefined : "Use 1 to 252 days.";
    case "cost": {
      const n = Number(p.cost);
      return p.cost.trim() !== "" && n >= 0 && n <= 200 ? undefined : "Use 0 to 200 bps.";
    }
  }
}

export function validateParams(p: ParamsInput): FieldErrors {
  const errors: FieldErrors = {};
  for (const key of Object.keys(p) as (keyof ParamsInput)[]) {
    const e = validateField(key, p);
    if (e) errors[key] = e;
  }
  return errors;
}
