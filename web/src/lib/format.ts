const MINUS = "−";

export type Num = number | null | undefined;

export function fmtPct(x: Num, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return "—";
  const s = (Math.abs(x) * 100).toFixed(digits);
  return (x < 0 && Number(s) !== 0 ? MINUS : "") + s + "%";
}

export function fmtNum(x: Num, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toFixed(digits);
  return x < 0 && Number(s) !== 0 ? MINUS + s : s;
}

export function toneOf(x: Num): "pos" | "neg" | "neutral" {
  if (x == null || !Number.isFinite(x) || x === 0) return "neutral";
  return x > 0 ? "pos" : "neg";
}
