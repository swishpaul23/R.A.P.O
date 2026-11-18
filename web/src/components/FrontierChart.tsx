import { useState } from "react";
import type { BacktestResponse } from "../lib/api";
import { fmtNum, fmtPct } from "../lib/format";
import { colorFor } from "../lib/theme";

const W = 560, H = 320, PAD = { l: 52, r: 16, t: 12, b: 36 };

/** Name of the closest point to (x, y) within `radius` pixels, or null. */
export function nearest(points: { name: string; x: number; y: number }[], x: number, y: number, radius: number): string | null {
  let best: string | null = null, bestD = radius;
  for (const p of points) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d <= bestD) { best = p.name; bestD = d; }
  }
  return best;
}

function ticks(lo: number, hi: number, n = 4) {
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step);
}

export default function FrontierChart({ data, focus }: { data: BacktestResponse; focus: string }) {
  const [active, setActive] = useState<string | null>(null);
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

  // Draw the active and focused markers last so they sit on top of overlapping neighbours.
  const order = [...f.strategies].sort((a, b) =>
    Number(a.name === active) - Number(b.name === active) || Number(a.name === focus) - Number(b.name === focus));
  const tip = f.strategies.find((s) => s.name === active);

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="relative min-w-[560px]"
          onPointerMove={(e) => {
            const box = e.currentTarget.getBoundingClientRect();
            const k = box.width / W;  // svg units -> css pixels
            const pts = f.strategies.map((p) => ({ name: p.name, x: sx(p.vol) * k, y: sy(p.ret) * k }));
            setActive(nearest(pts, e.clientX - box.left, e.clientY - box.top, 22));
          }}
          onPointerLeave={() => setActive(null)}>
          <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img"
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
          </svg>

          {order.map((s) => {
            const c = colorFor(s.name, data.benchmark_label);
            const big = s.name === focus || s.name === active;
            const size = big ? 14 : 10;
            return (
              <span
                key={s.name}
                aria-hidden="true"
                className="pointer-events-none absolute flex -translate-x-1/2 -translate-y-1/2 items-center justify-center"
                style={{ left: `${(sx(s.vol) / W) * 100}%`, top: `${(sy(s.ret) / H) * 100}%` }}
              >
                <span className={`block rotate-45 outline outline-[1.5px] outline-canvas ${s.name === active ? "ring-2 ring-fg/60" : ""}`}
                  style={{ width: size, height: size, background: c }} />
              </span>
            );
          })}

          {f.strategies.filter((s) => s.name === focus && s.name !== active).map((s) => (
            <span key={s.name} aria-hidden="true" className="pointer-events-none absolute -translate-y-1/2 pl-4 text-xs text-fg"
              style={{ left: `${(sx(s.vol) / W) * 100}%`, top: `${(sy(s.ret) / H) * 100}%` }}>
              {s.name}
            </span>
          ))}

          {tip && (() => {
            const xPct = (sx(tip.vol) / W) * 100, yPct = (sy(tip.ret) / H) * 100;
            const below = yPct < 35, alignRight = xPct > 65;
            return (
              <div id="frontier-tip" role="tooltip"
                className="pointer-events-none absolute z-10 w-48 rounded-lg border border-line-strong bg-raised/90 px-3 py-2 text-xs shadow-lg backdrop-blur-md"
                style={{
                  left: `${xPct}%`, top: `${yPct}%`,
                  transform: `translate(${alignRight ? "calc(-100% - 12px)" : "12px"}, ${below ? "12px" : "calc(-100% - 12px)"})`,
                }}>
                <div className="mb-1 flex items-center gap-2 font-medium text-fg">
                  <span aria-hidden="true" className="size-2 rotate-45" style={{ background: colorFor(tip.name, data.benchmark_label) }} />
                  {tip.name}
                </div>
                <dl className="num grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-fg-2">
                  <dt>Return</dt><dd className="text-right text-fg">{fmtPct(tip.ret)}</dd>
                  <dt>Volatility</dt><dd className="text-right text-fg">{fmtPct(tip.vol)}</dd>
                  <dt>Sharpe</dt><dd className="text-right text-fg">{fmtNum(tip.sharpe)}</dd>
                </dl>
              </div>
            );
          })()}
        </div>
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-1 gap-y-1 text-xs text-fg-2" aria-label="Strategy markers">
        {f.strategies.map((s) => (
          <li key={s.name}>
            <button
              type="button"
              aria-label={`${s.name}: return ${fmtPct(s.ret)}, volatility ${fmtPct(s.vol)}, Sharpe ${fmtNum(s.sharpe)}`}
              aria-describedby={s.name === active ? "frontier-tip" : undefined}
              onMouseEnter={() => setActive(s.name)}
              onMouseLeave={() => setActive((a) => (a === s.name ? null : a))}
              onFocus={() => setActive(s.name)}
              onBlur={() => setActive((a) => (a === s.name ? null : a))}
              className={`inline-flex min-h-11 items-center gap-1.5 rounded-md px-2 ${s.name === active ? "bg-raised text-fg" : "hover:text-fg"}`}
            >
              <span aria-hidden="true" className="size-2 rotate-45" style={{ background: colorFor(s.name, data.benchmark_label) }} />
              {s.name}
            </button>
          </li>
        ))}
      </ul>
      <p className="text-xs text-fg-3">Hover a marker or a name for its values. Annualized, last lookback window.</p>
    </div>
  );
}
