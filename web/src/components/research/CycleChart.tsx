import { ColorType, createChart, HistogramSeries, LineSeries, LineStyle, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import { BENCH_COLOR, SERIES_COLORS } from "../../lib/theme";

interface Series { dates: string[]; tilt: number[]; spy: number[]; hrp: number[]; hrp_regime: number[] }

const toTime = (d: string) => (Date.parse(d) / 1000) as UTCTimestamp;

export default function CycleChart({ series }: { series: Series }) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#a8b3bf",
        fontFamily: "JetBrains Mono Variable, ui-monospace, monospace", fontSize: 12 },
      grid: { vertLines: { color: "rgba(230,237,243,0.05)" }, horzLines: { color: "rgba(230,237,243,0.05)" } },
      rightPriceScale: { borderColor: "rgba(230,237,243,0.10)" },
      leftPriceScale: { visible: true, borderColor: "rgba(230,237,243,0.10)" },
      timeScale: { borderColor: "rgba(230,237,243,0.10)", minBarSpacing: 0.05 },
    });
    const t = series.dates.map(toTime);
    const tilt = c.addSeries(HistogramSeries, { priceScaleId: "left", color: "rgba(227,179,65,0.35)",
      priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "custom", formatter: (v: number) => `${(v * 100).toFixed(0)}%` } });
    c.priceScale("left").applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });
    tilt.setData(series.tilt.map((v, i) => ({ time: t[i], value: v ?? 0 })));
    const money = { type: "custom" as const, formatter: (v: number) => `$${v.toFixed(2)}` };
    const lines: [number[], string, number, LineStyle][] = [
      [series.spy, BENCH_COLOR, 1, LineStyle.Dashed],
      [series.hrp, SERIES_COLORS.HRP, 1, LineStyle.Solid],
      [series.hrp_regime, "#e3b341", 2, LineStyle.Solid],
    ];
    for (const [vals, color, width, style] of lines) {
      const s = c.addSeries(LineSeries, { color, lineWidth: width as 1 | 2, lineStyle: style,
        priceLineVisible: false, lastValueVisible: false, priceFormat: money });
      s.setData(vals.map((v, i) => ({ time: t[i], value: v })));
    }
    c.timeScale().fitContent();
    const raf = requestAnimationFrame(() => c.timeScale().fitContent());
    return () => { cancelAnimationFrame(raf); c.remove(); };
  }, [series]);
  return <div ref={el} className="h-[320px] w-full" role="img"
    aria-label="Growth of $1 for SPY, HRP and HRP with the regime overlay, with the defensive tilt shown as bars" />;
}
