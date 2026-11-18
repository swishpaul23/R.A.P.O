import type { BacktestResponse } from "../lib/api";
import { fmtPct } from "../lib/format";
import { colorFor } from "../lib/theme";

export default function AllocationBars({ data, focus }: { data: BacktestResponse; focus: string }) {
  const w = data.weights[focus] ?? {};
  const order = data.clusters?.order ?? Object.keys(w);
  const color = colorFor(focus, data.benchmark_label);
  return (
    <ul className="flex flex-col gap-3" aria-label={`${focus} current target weights`}>
      {order.map((t) => {
        const v = w[t] ?? 0;
        return (
          <li key={t} className="grid grid-cols-[4rem_1fr_4.5rem] items-center gap-3 text-sm">
            <span className="num text-fg-2">{t}</span>
            <span className="h-2 rounded-full bg-raised">
              <span className="block h-2 rounded-full" style={{ width: `${Math.max(v * 100, 0.5)}%`, background: color }} />
            </span>
            <span className="num text-right text-fg">{fmtPct(v, 1)}</span>
          </li>
        );
      })}
    </ul>
  );
}
