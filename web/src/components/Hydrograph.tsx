"use client";

import { useMemo, useRef, useState } from "react";
import type { Hydrograph } from "@/lib/api";
import { compact, shortDate, whole } from "@/lib/format";
import { useWidth } from "@/lib/useWidth";

const H = 210;
const M = { l: 52, r: 14, t: 10, b: 26 };
const DAY = 86_400_000;

type Point = { t: number; q: number };

function niceMax(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const f of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (f * p >= v) return f * p;
  return 10 * p;
}

export function HydrographChart({ data }: { data: Hydrograph }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const model = useMemo(() => {
    const obs: Point[] = data.observed.map((d) => ({ t: Date.parse(d.date), q: d.q }));
    const after: Point[] = data.observed_after.map((d) => ({ t: Date.parse(d.date), q: d.q }));
    const issue = Date.parse(data.issue_date);
    const fc = data.forecast.map((d) => ({ ...d, t: Date.parse(d.date) }));
    const times = [...obs.map((p) => p.t), ...fc.map((p) => p.t), ...after.map((p) => p.t), issue];
    const t0 = Math.min(...times);
    const t1 = Math.max(...times, issue + 7 * DAY);
    const values = [...obs.map((p) => p.q), ...after.map((p) => p.q), ...fc.map((p) => p.p95)];
    const dataMax = Math.max(1, ...values);
    const cap = data.gauge.design_capacity_cusecs;
    const showCap = cap != null && cap < dataMax * 1.8;
    const yMax = niceMax(Math.max(dataMax, showCap ? cap! : 0) * 1.05);
    return { obs, after, fc, issue, t0, t1, yMax, cap, showCap };
  }, [data]);

  const w = Math.max(width, 280);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const x = (t: number) => M.l + ((t - model.t0) / Math.max(model.t1 - model.t0, 1)) * iw;
  const y = (q: number) => M.t + ih - (q / model.yMax) * ih;
  const line = (pts: Point[]) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.q).toFixed(1)}`).join("");
  const band = (lo: keyof (typeof model.fc)[number], hi: keyof (typeof model.fc)[number]) => {
    if (!model.fc.length) return "";
    const anchor = model.obs.length ? model.obs[model.obs.length - 1] : null;
    const up = model.fc.map((d) => [x(d.t), y(d[hi] as number)]);
    const down = model.fc.map((d) => [x(d.t), y(d[lo] as number)]).reverse();
    const start = anchor ? [[x(anchor.t), y(anchor.q)]] : [];
    return `M${[...start, ...up, ...down, ...start].map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join("L")}Z`;
  };
  const median = model.obs.length
    ? [model.obs[model.obs.length - 1], ...model.fc.map((d) => ({ t: d.t, q: d.p50 }))]
    : model.fc.map((d) => ({ t: d.t, q: d.p50 }));

  const ticksY = [0, 0.25, 0.5, 0.75, 1].map((f) => f * model.yMax);
  const ticksX: number[] = [];
  const stepDays = Math.max(1, Math.round((model.t1 - model.t0) / DAY / Math.max(3, Math.floor(iw / 90))));
  for (let t = model.t0; t <= model.t1; t += stepDays * DAY) ticksX.push(t);

  // hover: snap to the nearest day that has any value
  const days = useMemo(() => {
    const set = new Set<number>([...model.obs, ...model.after].map((p) => p.t));
    model.fc.forEach((d) => set.add(d.t));
    return [...set].sort((a, b) => a - b);
  }, [model]);
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || !days.length) return;
    const px = e.clientX - box.left;
    let best = days[0];
    for (const d of days) if (Math.abs(x(d) - px) < Math.abs(x(best) - px)) best = d;
    setHover(best);
  };
  const at = (pts: Point[], t: number) => pts.find((p) => p.t === t)?.q;
  const fcAt = hover != null ? model.fc.find((d) => d.t === hover) : undefined;

  return (
    <div className="chart-wrap" ref={ref}>
      <svg ref={svgRef} width={w} height={H} role="img"
        aria-label={`River flow at ${data.gauge.name}: observed and forecast`}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticksY.map((q) => (
          <g key={q}>
            <line x1={M.l} x2={w - M.r} y1={y(q)} y2={y(q)} stroke="var(--grid)" strokeWidth={1} />
            <text x={M.l - 8} y={y(q)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">
              {compact(q)}
            </text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text key={t} x={x(t)} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--muted)">
            {shortDate(new Date(t).toISOString())}
          </text>
        ))}
        <line x1={M.l} x2={w - M.r} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
        {model.showCap && (
          <g>
            <line x1={M.l} x2={w - M.r} y1={y(model.cap!)} y2={y(model.cap!)} stroke="var(--muted)" strokeDasharray="4 4" />
            <text x={w - M.r} y={y(model.cap!) - 4} textAnchor="end" fontSize={11} fill="var(--ink-2)">
              design capacity {compact(model.cap!)}
            </text>
          </g>
        )}
        <path d={band("p5", "p95")} fill="var(--band-soft)" stroke="none" />
        <path d={band("p25", "p75")} fill="var(--band-strong)" stroke="none" />
        <line x1={x(model.issue)} x2={x(model.issue)} y1={M.t} y2={M.t + ih} stroke="var(--axis)" strokeWidth={1} />
        <text x={x(model.issue) + 4} y={M.t + 10} fontSize={11} fill="var(--ink-2)">forecast issued</text>
        {model.obs.length > 1 && <path d={line(model.obs)} fill="none" stroke="var(--series-1)" strokeWidth={2} />}
        {median.length > 1 && (
          <path d={line(median)} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeDasharray="5 4" />
        )}
        {model.after.length > 1 && (
          <path d={line(model.after)} fill="none" stroke="var(--ink-2)" strokeWidth={2} strokeDasharray="1.5 3.5"
            strokeLinecap="round" />
        )}
        {hover != null && (
          <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={M.t + ih} stroke="var(--ink-2)" strokeWidth={1} />
        )}
      </svg>
      {hover != null && (
        <div className="tooltip" style={{ left: Math.min(x(hover) + 10, w - 190), top: 8 }}>
          <div className="t">{new Date(hover).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })}</div>
          {at(model.obs, hover) != null && <div className="r"><span>Observed</span><span>{whole(at(model.obs, hover)!)}</span></div>}
          {fcAt && <div className="r"><span>Forecast median</span><span>{whole(fcAt.p50)}</span></div>}
          {fcAt && <div className="r"><span>5–95% range</span><span>{compact(fcAt.p5)}–{compact(fcAt.p95)}</span></div>}
          {at(model.after, hover) != null && <div className="r"><span>What happened</span><span>{whole(at(model.after, hover)!)}</span></div>}
          <div className="r"><span className="muted">cusecs</span><span /></div>
        </div>
      )}
    </div>
  );
}
