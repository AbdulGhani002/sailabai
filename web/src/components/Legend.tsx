"use client";

import { useEffect, useState } from "react";
import { getJSON } from "@/lib/api";
import type { Theme } from "@/lib/theme";

type Entry = { label: string; color: string; opacity?: number };

const TITLES: Record<string, string> = {
  current: "Latest flood map",
  prob: "Chance of flooding",
  sets: "Confidence zones",
  spread: "Disagreement between runs",
};

export function Legend({ layer, theme, note }: { layer: string; theme: Theme; note?: string }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const kind = layer === "current" ? "current" : layer.split("_")[0];

  useEffect(() => {
    const ctrl = new AbortController();
    getJSON<Entry[]>(`/api/legend/${layer}?theme=${theme}`, ctrl.signal).then(setEntries).catch(() => undefined);
    return () => ctrl.abort();
  }, [layer, theme]);

  return (
    <div className="legend" aria-label="Map legend">
      <h4>{TITLES[kind] ?? "Legend"}</h4>
      {kind === "prob" && entries.length > 0 ? (
        <>
          <div className="ramp" aria-hidden>
            {entries.map((e) => (
              <span key={e.label} style={{ background: e.color }} />
            ))}
          </div>
          <div className="ticks">
            <span>5%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </>
      ) : (
        entries.map((e) => (
          <div className="item" key={e.label}>
            <span className="swatch" style={{ background: e.color, opacity: e.opacity ?? 1 }} />
            {e.label}
          </div>
        ))
      )}
      {note && <div className="small ink2" style={{ marginTop: 6, maxWidth: 220 }}>{note}</div>}
    </div>
  );
}
