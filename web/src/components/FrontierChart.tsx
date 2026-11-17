import type { BacktestResponse } from "../lib/api";
import { colorFor } from "../lib/theme";

const W = 560, H = 320, PAD = { l: 52, r: 16, t: 12, b: 36 };

function ticks(lo: number, hi: number, n = 4) {
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step);
}

export default function FrontierChart({ data, focus }: { data: BacktestResponse; focus: string }) {
  const f = data.frontier;
  if (!f) return <p className="text-sm text-fg-3">Not enough overlapping history to draw the frontier.</p>;
  const all = [...f.points, ...f.strategies];
  const vx = all.map((p) => p.vol), ry = all.map((p) => p.ret);
  const x0 = Math.min(...vx) * 0.95, x1 = Math.max(...vx) * 1.05;
  const yPad = (Math.max(...ry) - Math.min(...ry)) * 0.08 || 0.01;
  const y0 = Math.min(...ry) - yPad, y1 = Math.max(...ry) + yPad;
  const sx = (v: number) => PAD.l + ((v - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const sh = f.points.map((p) => p.sharpe);
  const s0 = Math.min(...sh), s1 = Math.max(...sh);

  return (
    <div>
    <div className="overflow-x-auto">
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[560px]" role="img"
      aria-label="Efficient frontier: random portfolios by volatility and return, with each strategy's current allocation">
      {ticks(y0, y1).map((t) => (
        <g key={`y${t}`}>
          <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} stroke="rgba(230,237,243,0.06)" />
          <text x={PAD.l - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="num" fontSize="12" fill="#8592a0">
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      {ticks(x0, x1).map((t) => (
        <text key={`x${t}`} x={sx(t)} y={H - PAD.b + 18} textAnchor="middle" className="num" fontSize="12" fill="#8592a0">
          {(t * 100).toFixed(0)}%
        </text>
      ))}
      <text x={W - PAD.r} y={H - 4} textAnchor="end" fontSize="12" fill="#8592a0">Volatility →</text>
      <text x={PAD.l} y={PAD.t + 2} fontSize="12" fill="#8592a0" dominantBaseline="hanging" dx="6">↑ Return</text>
      {f.points.map((p, i) => (
        <circle key={i} cx={sx(p.vol)} cy={sy(p.ret)} r={1.6} fill="#8b85ff"
          opacity={0.12 + 0.6 * ((p.sharpe - s0) / (s1 - s0 || 1))} />
      ))}
      {f.strategies.map((s) => {
        const x = sx(s.vol), y = sy(s.ret), c = colorFor(s.name, data.benchmark_label), r = s.name === focus ? 7 : 5;
        return (
          <g key={s.name}>
            <rect x={x - r} y={y - r} width={r * 2} height={r * 2} transform={`rotate(45 ${x} ${y})`}
              fill={c} stroke="#0a0d12" strokeWidth={1.5} />
            {s.name === focus && (
              <text x={x + 12} y={y} dominantBaseline="middle" fontSize="12" fill="#e6edf3">{s.name}</text>
            )}
          </g>
        );
      })}
    </svg>
    </div>
    <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-2" aria-label="Strategy markers">
      {f.strategies.map((s) => (
        <li key={s.name} className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="size-2 rotate-45" style={{ background: colorFor(s.name, data.benchmark_label) }} />
          {s.name}
        </li>
      ))}
    </ul>
    </div>
  );
}
