// Series colours. Green and red are reserved for gains/losses (see .ux-profile.md).
export const SERIES_COLORS: Record<string, string> = {
  "Equal weight": "#6cb6ff",
  "Inverse vol": "#39c5cf",
  "Min variance": "#e3b341",
  "Max Sharpe": "#f778ba",
  HRP: "#8b85ff",
};
export const BENCH_COLOR = "#8b949e";

export function colorFor(name: string, benchmarkLabel: string): string {
  return name === benchmarkLabel ? BENCH_COLOR : SERIES_COLORS[name] ?? "#c9d1d9";
}

export const STRESS_TINT = "rgba(244, 112, 103, 0.06)";
