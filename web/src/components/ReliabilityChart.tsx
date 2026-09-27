"use client";

import { useState } from "react";
import { useWidth } from "@/lib/useWidth";

export type Bin = { mean_prob: number; observed: number; count: number };
export type RelSeries = { key: string; label: string; color: string; bins: Bin[] };

const M = { l: 44, r: 16, t: 12, b: 36 };

/** Forecast chance (x) against how often it actually flooded (y). On the diagonal = honest. */
export function ReliabilityChart({ series }: { series: RelSeries[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ s: RelSeries; b: Bin } | null>(null);
  const w = Math.max(width, 280);
  const size = Math.min(w - M.l - M.r, 360);
  const H = size + M.t + M.b;
  const x = (p: number) => M.l + p * size;
  const y = (p: number) => M.t + size - p * size;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chart-wrap" ref={ref}>
      <div className="chart-legend" style={{ margin: "4px 0 2px" }}>
        {series.map((s) => (
          <span className="key" key={s.key}>
            <span className="line" style={{ borderTopColor: s.color }} /> {s.label}
          </span>
        ))}
        <span className="key"><span className="line dashed" style={{ borderTopColor: "var(--muted)" }} /> perfectly honest</span>
      </div>
      <svg width={w} height={H} role="img" aria-label="Reliability: forecast chance against observed flood frequency">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">{Math.round(t * 100)}%</text>
            <text x={x(t)} y={y(0) + 16} textAnchor="middle" fontSize={11} fill="var(--muted)">{Math.round(t * 100)}%</text>
          </g>
        ))}
        <text x={x(0.5)} y={H - 4} textAnchor="middle" fontSize={11.5} fill="var(--ink-2)">forecast chance of flooding</text>
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--muted)" strokeDasharray="5 4" />
        {series.map((s) => {
          const pts = s.bins.filter((b) => b.count >= 50);
          if (!pts.length) return null;
          const last = pts[pts.length - 1];
          return (
            <g key={s.key}>
              <path d={`M${pts.map((b) => `${x(b.mean_prob).toFixed(1)},${y(b.observed).toFixed(1)}`).join("L")}`}
                fill="none" stroke={s.color} strokeWidth={2} />
              {pts.map((b, i) => (
                <circle key={i} cx={x(b.mean_prob)} cy={y(b.observed)} r={4.5} fill={s.color} stroke="var(--surface)"
                  strokeWidth={2} onMouseEnter={() => setHover({ s, b })} onMouseLeave={() => setHover(null)}
                  style={{ cursor: "default" }} />
              ))}
              <text x={Math.min(x(last.mean_prob) + 8, w - 60)} y={y(last.observed)} dy="0.32em" fontSize={11.5} fill="var(--ink)">
                {s.label}
              </text>
            </g>
          );
        })}
        <text x={M.l - 34} y={M.t - 2} fontSize={11.5} fill="var(--ink-2)">flooded</text>
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: Math.min(x(hover.b.mean_prob) + 12, w - 190), top: y(hover.b.observed) - 10 }}>
          <div className="t">{hover.s.label}</div>
          <div className="r"><span>Forecast chance</span><span>{(hover.b.mean_prob * 100).toFixed(1)}%</span></div>
          <div className="r"><span>Actually flooded</span><span>{(hover.b.observed * 100).toFixed(1)}%</span></div>
          <div className="r"><span>Pixels</span><span>{hover.b.count.toLocaleString("en-US")}</span></div>
        </div>
      )}
    </div>
  );
}
