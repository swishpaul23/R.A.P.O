import type { BacktestResponse } from "../lib/api";

function cellColor(v: number): string {
  // Positive: accent violet. Negative: cyan. Zero: surface.
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(139,133,255,${0.08 + 0.72 * a})` : `rgba(57,197,207,${0.08 + 0.72 * a})`;
}

export default function CorrelationHeatmap({ data }: { data: BacktestResponse }) {
  const c = data.correlation;
  if (!c) return <p className="text-sm text-fg-3">Not enough overlapping history to compute correlations.</p>;
  const order = data.clusters?.order ?? c.tickers;
  const idx = order.map((t) => c.tickers.indexOf(t));
  return (
    <div className="overflow-x-auto">
      <table className="num w-full border-separate border-spacing-1 text-sm" aria-label="Correlation matrix, ordered by HRP clusters">
        <thead>
          <tr>
            <th className="sr-only">Ticker</th>
            {order.map((t) => <th key={t} scope="col" className="px-1 pb-1 text-xs font-medium text-fg-3">{t}</th>)}
          </tr>
        </thead>
        <tbody>
          {idx.map((i, r) => (
            <tr key={order[r]}>
              <th scope="row" className="pr-2 text-right text-xs font-medium text-fg-3">{order[r]}</th>
              {idx.map((j) => {
                const v = c.matrix[i][j];
                return (
                  <td key={j} className="h-10 min-w-12 rounded text-center text-fg" style={{ background: cellColor(v) }}>
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
