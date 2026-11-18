import {
  ColorType, createChart, HistogramSeries, LineSeries, LineStyle,
  type IChartApi, type ISeriesApi, type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { BacktestResponse } from "../lib/api";
import { colorFor, STRESS_TINT } from "../lib/theme";

interface Props {
  data: BacktestResponse;
  mode: "equity" | "drawdown";
  focus: string;
  hidden: Set<string>;
}

const toTime = (d: string) => (Date.parse(d) / 1000) as UTCTimestamp;

export default function EquityChart({ data, mode, focus, hidden }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const lines = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#a8b3bf",
        fontFamily: "JetBrains Mono Variable, ui-monospace, monospace", fontSize: 12 },
      grid: { vertLines: { color: "rgba(230,237,243,0.05)" }, horzLines: { color: "rgba(230,237,243,0.05)" } },
      rightPriceScale: { borderColor: "rgba(230,237,243,0.10)" },
      timeScale: { borderColor: "rgba(230,237,243,0.10)", minBarSpacing: 0.05 },
      crosshair: { vertLine: { color: "rgba(139,133,255,0.5)" }, horzLine: { color: "rgba(139,133,255,0.5)" } },
    });
    chart.current = c;
    return () => { c.remove(); chart.current = null; lines.current.clear(); };
  }, []);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    for (const s of lines.current.values()) c.removeSeries(s);
    lines.current.clear();
    const times = data.series.dates.map(toTime);

    // Stress windows as a faint background band.
    const band = c.addSeries(HistogramSeries, { priceScaleId: "stress", color: STRESS_TINT,
      priceLineVisible: false, lastValueVisible: false });
    c.priceScale("stress").applyOptions({ scaleMargins: { top: 0, bottom: 0 }, visible: false });
    band.setData(data.series.dates.map((d, i) => ({
      time: times[i],
      value: data.stress.some((w) => d >= w.start && d <= w.end) ? 1 : 0,
    })));

    const source = mode === "equity" ? data.series.equity : data.series.drawdown;
    const names = [...data.strategies, data.benchmark_label];
    for (const name of names) {
      const isBench = name === data.benchmark_label;
      const s = c.addSeries(LineSeries, {
        color: colorFor(name, data.benchmark_label),
        lineWidth: name === focus ? 3 : 1,
        lineStyle: isBench ? LineStyle.Dashed : LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: name === focus,
        priceFormat: mode === "equity"
          ? { type: "custom", formatter: (v: number) => `$${v.toFixed(2)}` }
          : { type: "custom", formatter: (v: number) => `${(v * 100).toFixed(1)}%` },
      });
      s.setData(source[name].map((v, i) => ({ time: times[i], value: v })));
      lines.current.set(name, s);
    }
    c.timeScale().fitContent();
    const raf = requestAnimationFrame(() => c.timeScale().fitContent());
    return () => {
      cancelAnimationFrame(raf);
      // On unmount the chart effect above has already disposed the chart.
      if (chart.current !== c) return;
      for (const s of lines.current.values()) c.removeSeries(s);
      lines.current.clear();
      c.removeSeries(band);
    };
  }, [data, mode, focus]);

  useEffect(() => {
    for (const [name, s] of lines.current) s.applyOptions({ visible: !hidden.has(name) });
  }, [hidden, data, mode, focus]);

  return <div ref={el} className="h-[320px] w-full md:h-[380px]" role="img"
    aria-label={`${mode === "equity" ? "Growth of $1" : "Drawdown"} for each strategy and the benchmark`} />;
}
