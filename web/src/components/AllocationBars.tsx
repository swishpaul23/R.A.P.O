import type { BacktestResponse } from "../lib/api";
import { fmtPct } from "../lib/format";
import { colorFor } from "../lib/theme";

const HELD = 0.0005;  // below 0.05% shows as 0.0%, so treat it as not held

export default function AllocationBars({ data, focus }: { data: BacktestResponse; focus: string }) {
  const w = data.weights[focus] ?? {};
  const tickers = Object.keys(w);
  const eligible = new Set(data.eligible ?? tickers);
  const held = tickers.filter((t) => w[t] >= HELD).sort((a, b) => w[b] - w[a]);
  const notHeld = tickers.filter((t) => w[t] < HELD && eligible.has(t));
  const tooNew = tickers.filter((t) => !eligible.has(t));
  const color = colorFor(focus, data.benchmark_label);

  return (
    <div className="flex flex-col gap-4">
      <ul className="flex flex-col gap-3" aria-label={`${focus} current target weights`}>
        {held.map((t) => (
          <li key={t} data-ticker={t} className="grid grid-cols-[4rem_1fr_4.5rem] items-center gap-3 text-sm">
            <span className="num text-fg-2">{t}</span>
            <span className="h-2 rounded-full bg-raised">
              <span className="block h-2 rounded-full" style={{ width: `${w[t] * 100}%`, background: color }} />
            </span>
            <span className="num text-right text-fg">{fmtPct(w[t], 1)}</span>
          </li>
        ))}
      </ul>
      {(notHeld.length > 0 || tooNew.length > 0) && (
        <dl className="flex flex-col gap-2 border-t border-line pt-3 text-xs">
          {notHeld.length > 0 && (
            <div>
              <dt className="text-fg-3">Not held (0%)</dt>
              <dd className="num mt-1 text-fg-2">{notHeld.join(", ")}</dd>
            </div>
          )}
          {tooNew.length > 0 && (
            <div>
              <dt className="text-fg-3">Not enough history yet</dt>
              <dd className="num mt-1 text-fg-2">{tooNew.join(", ")}</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
}
