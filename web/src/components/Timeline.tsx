"use client";

import { useState } from "react";
import type { RunListItem } from "@/lib/api";
import { compact, shortDate } from "@/lib/format";
import { useWidth } from "@/lib/useWidth";

const H = 56;
const PAD = 4;

/** Flooded area of every daily run; click a day to open that run. */
export function Timeline({ runs, selected, onSelect }: { runs: RunListItem[]; selected: string; onSelect: (d: string) => void }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  if (runs.length < 2) return null;
  const w = Math.max(width, 200);
  const max = Math.max(1, ...runs.map((r) => r.current_flooded_km2));
  const x = (i: number) => PAD + (i / (runs.length - 1)) * (w - 2 * PAD);
  const y = (v: number) => H - PAD - (v / max) * (H - 2 * PAD);
  const pts = runs.map((r, i) => `${x(i).toFixed(1)},${y(r.current_flooded_km2).toFixed(1)}`);
  const area = `M${x(0)},${H - PAD}L${pts.join("L")}L${x(runs.length - 1)},${H - PAD}Z`;
  const sel = runs.findIndex((r) => r.run_date === selected);
  const idx = (clientX: number, left: number) =>
    Math.max(0, Math.min(runs.length - 1, Math.round(((clientX - left - PAD) / (w - 2 * PAD)) * (runs.length - 1))));
  const shown = hover ?? sel;

  return (
    <div ref={ref} className="chart-wrap" style={{ marginTop: 6 }}>
      <svg width={w} height={H} role="img" aria-label="Flooded area over the replayed days"
        style={{ cursor: "pointer", display: "block" }}
        onMouseMove={(e) => setHover(idx(e.clientX, e.currentTarget.getBoundingClientRect().left))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onSelect(runs[idx(e.clientX, e.currentTarget.getBoundingClientRect().left)].run_date)}>
        <path d={area} fill="var(--band-soft)" />
        <path d={`M${pts.join("L")}`} fill="none" stroke="var(--series-1)" strokeWidth={2} />
        {shown >= 0 && (
          <g>
            <line x1={x(shown)} x2={x(shown)} y1={PAD} y2={H - PAD} stroke="var(--ink-2)" />
            <circle cx={x(shown)} cy={y(runs[shown].current_flooded_km2)} r={4} fill="var(--series-1)"
              stroke="var(--surface)" strokeWidth={2} />
          </g>
        )}
      </svg>
      <div className="row small ink2" style={{ justifyContent: "space-between" }}>
        <span>{shortDate(runs[0].run_date)}</span>
        {shown >= 0 && (
          <span>
            {shortDate(runs[shown].run_date)}: <strong>{compact(runs[shown].current_flooded_km2)} km²</strong> flooded
          </span>
        )}
        <span>{shortDate(runs[runs.length - 1].run_date)}</span>
      </div>
    </div>
  );
}
