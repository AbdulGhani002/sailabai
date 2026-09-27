"use client";

import { useState } from "react";
import { useWidth } from "@/lib/useWidth";

export type Series = { key: string; label: string; color: string; points: { lead: string; value: number | null }[] };

const H = 240;
const M = { l: 44, r: 118, t: 12, b: 28 };

/** Skill over persistence by lead time. 0 = no better than repeating the last map. */
export function SkillChart({ series, leads, yLabel }: { series: Series[]; leads: string[]; yLabel: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const w = Math.max(width, 320);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const vals = series.flatMap((s) => s.points.map((p) => p.value)).filter((v): v is number => v != null);
  const lo = Math.min(-0.1, ...vals);
  const hi = Math.max(0.1, ...vals);
  const pad = (hi - lo) * 0.1;
  const y0 = lo - pad;
  const y1 = hi + pad;
  const x = (i: number) => M.l + (leads.length > 1 ? (i / (leads.length - 1)) * iw : iw / 2);
  const y = (v: number) => M.t + ih - ((v - y0) / (y1 - y0)) * ih;
  const ticks = [y0, 0, y1].map((v) => Math.round(v * 100) / 100).filter((v, i, a) => a.indexOf(v) === i);

  return (
    <div className="chart-wrap" ref={ref}>
      <div className="chart-legend" style={{ margin: "4px 0 2px" }}>
        {series.map((s) => (
          <span className="key" key={s.key}>
            <span className="line" style={{ borderTopColor: s.color }} /> {s.label}
          </span>
        ))}
      </div>
      <svg width={w} height={H} role="img" aria-label={yLabel}
        onMouseMove={(e) => {
          const px = e.clientX - e.currentTarget.getBoundingClientRect().left;
          const i = Math.round(((px - M.l) / Math.max(iw, 1)) * (leads.length - 1));
          setHover(i >= 0 && i < leads.length ? i : null);
        }}
        onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={M.l + iw} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--axis)" : "var(--grid)"} strokeWidth={t === 0 ? 1.5 : 1} />
            <text x={M.l - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">{t.toFixed(2)}</text>
          </g>
        ))}
        <text x={M.l + iw + 6} y={y(0)} dy="0.32em" fontSize={11} fill="var(--ink-2)">persistence</text>
        {leads.map((l, i) => (
          <text key={l} x={x(i)} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--muted)">{l}</text>
        ))}
        {series.map((s) => {
          const pts = s.points.map((p, i) => (p.value == null ? null : [x(i), y(p.value)] as const)).filter(Boolean) as [number, number][];
          if (!pts.length) return null;
          const last = pts[pts.length - 1];
          return (
            <g key={s.key}>
              <path d={`M${pts.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join("L")}`} fill="none" stroke={s.color} strokeWidth={2} />
              {pts.map(([a, b], i) => (
                <circle key={i} cx={a} cy={b} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
              ))}
              <text x={last[0] + 8} y={last[1]} dy="0.32em" fontSize={11.5} fill="var(--ink)">{s.label}</text>
            </g>
          );
        })}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={M.t + ih} stroke="var(--ink-2)" />}
      </svg>
      {hover != null && (
        <div className="tooltip" style={{ left: Math.min(x(hover) + 10, w - 200), top: 10 }}>
          <div className="t">{leads[hover]} ahead</div>
          {series.map((s) => (
            <div className="r" key={s.key}>
              <span><span className="swatch" style={{ display: "inline-block", width: 8, height: 8, background: s.color, borderRadius: 2, marginRight: 6 }} />{s.label}</span>
              <span>{s.points[hover]?.value == null ? "–" : s.points[hover]!.value!.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
