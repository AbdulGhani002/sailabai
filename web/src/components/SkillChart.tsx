"use client";

import { useState } from "react";
import { useWidth } from "@/lib/useWidth";

export type Series = { key: string; label: string; color: string; points: { lead: string; value: number | null }[] };

const H = 240;
const M = { l: 44, r: 118, t: 14, b: 28 };
const FLOOR = -1; // skill below -1 (more than twice as bad as the baseline) is drawn at the edge
const LABEL_GAP = 13;

/** Skill over persistence by lead time. 0 = no better than repeating the last map. */
export function SkillChart({ series, leads, yLabel }: { series: Series[]; leads: string[]; yLabel: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const w = Math.max(width, 320);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const vals = series.flatMap((s) => s.points.map((p) => p.value)).filter((v): v is number => v != null);
  const lo = Math.max(FLOOR, Math.min(-0.1, ...vals));
  const hi = Math.min(1, Math.max(0.1, ...vals));
  const pad = (hi - lo) * 0.08;
  const y0 = lo - pad;
  const y1 = hi + pad;
  const clampV = (v: number) => Math.min(Math.max(v, lo), hi);
  const x = (i: number) => M.l + (leads.length > 1 ? (i / (leads.length - 1)) * iw : iw / 2);
  const y = (v: number) => M.t + ih - ((clampV(v) - y0) / (y1 - y0)) * ih;
  const ticks = [lo, 0, hi].map((v) => Math.round(v * 100) / 100).filter((v, i, a) => a.indexOf(v) === i);

  // direct labels at the line ends, nudged apart so they never overlap
  const ends = series
    .map((s) => {
      const lastIdx = s.points.map((p) => p.value != null).lastIndexOf(true);
      return lastIdx < 0 ? null : { key: s.key, label: s.label, x: x(lastIdx), y: y(s.points[lastIdx].value!) };
    })
    .filter((e): e is { key: string; label: string; x: number; y: number } => e != null)
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + LABEL_GAP);
  const overflow = ends.length ? ends[ends.length - 1].y - (M.t + ih) : 0;
  if (overflow > 0) ends.forEach((e) => (e.y -= overflow));

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
            <text x={M.l - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">
              {t === lo && lo === FLOOR ? "≤ -1" : t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={M.l + 4} y={y(0) - 5} fontSize={11} fill="var(--ink-2)">persistence</text>
        {leads.map((l, i) => (
          <text key={l} x={x(i)} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--muted)">{l}</text>
        ))}
        {series.map((s) => {
          const pts = s.points
            .map((p, i) => (p.value == null ? null : { cx: x(i), cy: y(p.value), off: p.value < lo }))
            .filter((p): p is { cx: number; cy: number; off: boolean } => p != null);
          if (!pts.length) return null;
          return (
            <g key={s.key}>
              <path d={`M${pts.map((p) => `${p.cx.toFixed(1)},${p.cy.toFixed(1)}`).join("L")}`} fill="none" stroke={s.color} strokeWidth={2} />
              {pts.map((p, i) =>
                p.off ? (
                  <path key={i} d={`M${p.cx - 5},${p.cy - 2}L${p.cx + 5},${p.cy - 2}L${p.cx},${p.cy + 5}Z`} fill={s.color}
                    stroke="var(--surface)" strokeWidth={1.5} />
                ) : (
                  <circle key={i} cx={p.cx} cy={p.cy} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                ),
              )}
            </g>
          );
        })}
        {ends.map((e) => (
          <text key={e.key} x={e.x + 8} y={e.y} dy="0.32em" fontSize={11.5} fill="var(--ink)">{e.label}</text>
        ))}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={M.t + ih} stroke="var(--ink-2)" />}
      </svg>
      {vals.some((v) => v < FLOOR) && (
        <p className="small ink2" style={{ margin: "2px 0 0" }}>▼ marks skill below -1 (off the scale); hover for the value.</p>
      )}
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
