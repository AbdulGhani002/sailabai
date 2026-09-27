"use client";

import { useEffect, useState } from "react";
import { api, type Meta } from "@/lib/api";

function SystemFlow() {
  // 4 inputs, 2 models, 1 dashboard (as in the team guide)
  const box = (x: number, y: number, w: number, h: number, title: string, sub: string, main = false) => (
    <g>
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={8}
        fill={main ? "var(--band-soft)" : "var(--surface)"} stroke={main ? "var(--series-1)" : "var(--axis)"}
        strokeWidth={main ? 2 : 1.25} />
      <text x={x} y={y - 3} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink)">{title}</text>
      <text x={x} y={y + 14} textAnchor="middle" fontSize={11.5} fill="var(--ink-2)">{sub}</text>
    </g>
  );
  const arrow = (d: string) => <path d={d} fill="none" stroke="var(--axis)" strokeWidth={1.25} markerEnd="url(#arrow)" />;
  return (
    <svg viewBox="0 0 760 440" role="img" aria-label="How SailabAI turns free data into flood forecasts" style={{ width: "100%", height: "auto" }}>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="var(--axis)" />
        </marker>
      </defs>
      {arrow("M104 98V160")}{arrow("M280 98V160")}{arrow("M456 98V160")}{arrow("M632 98V160")}
      {arrow("M302 188H414")}{arrow("M544 216V262")}{arrow("M544 318V364")}{arrow("M192 216V364")}
      {box(104, 72, 160, 52, "Radar images", "Sentinel-1, 3-4 days")}
      {box(280, 72, 160, 52, "Terrain", "DEM, HAND, land cover")}
      {box(456, 72, 160, 52, "Rainfall", "IMERG, ECMWF forecast")}
      {box(632, 72, 160, 52, "River flow", "GloFAS, FFD gauges")}
      {box(192, 188, 220, 56, "Model 1: flood mapping", "U-Net, TerraMind-small")}
      {box(544, 188, 260, 56, "Model 2: flood forecast", "next pass and +1 to +7 days", true)}
      {box(544, 290, 260, 56, "Confidence and exposure", "roads, buildings, people")}
      {box(380, 392, 520, 56, "Web dashboard (the digital twin)", "map, time slider, confidence, risk panel")}
      <text x={358} y={180} textAnchor="middle" fontSize={11.5} fill="var(--ink-2)">latest map</text>
      <text x={202} y={292} fontSize={11.5} fill="var(--ink-2)">current map</text>
    </svg>
  );
}

export default function AboutPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta(null));
  }, []);

  return (
    <main className="page">
      <h1>About SailabAI</h1>
      <p className="lead">
        A flood digital twin for the Chenab, Ravi and Sutlej rivers around Multan: today&apos;s flood from satellite radar
        and a forecast for the next satellite pass and 1 to 7 days ahead, with a confidence level for every area.
        Sailab means flood in Urdu.
      </p>
      <div className="card" style={{ borderColor: "var(--axis)" }}>
        <strong>Experimental research output, not an official flood warning.</strong> SailabAI is a final-year project
        for flood experts and planners (NDMA, PDMA, relief groups). Follow NDMA, PDMA and PMD for official warnings; false
        disaster warnings are an offence under the NDM Act 2010 (section 35).
        {meta?.synthetic && (
          <p style={{ marginBottom: 0 }}>
            <strong>This deployment shows the synthetic demo world</strong>: invented terrain and floods shaped like our
            study area, used to build and test the system before the real data pipeline is complete.
          </p>
        )}
      </div>

      <h2>How it works</h2>
      <p className="ink2">
        Free data comes in, Model 1 maps today&apos;s flood, Model 2 forecasts the next days, and the dashboard shows it
        all with confidence and risk. Rain and river data arrive daily, so Model 2 updates its forecast every day; when
        a new radar image arrives (every 3 to 4 days), Model 1 draws a fresh flood map and Model 2 restarts from it.
      </p>
      <div className="card"><SystemFlow /></div>

      <h2>What the percentages mean</h2>
      <p className="ink2">
        A 70% chance should flood about 70% of the time. Five copies of Model 2 trained with different seeds, each run
        on members of the GloFAS river forecast ensemble, give the spread; temperature scaling and conformal prediction,
        fitted on the 2023 Sutlej flood, make the percentages and ranges honest. &quot;Can&apos;t tell&quot; marks pixels where
        both outcomes stay plausible at 90% confidence.
      </p>

      <h2>Data and credits</h2>
      {meta?.credits?.length ? (
        <div className="card scroll-x">
          <table className="data">
            <thead><tr><th>Dataset</th><th style={{ textAlign: "left" }}>Credit</th><th style={{ textAlign: "left" }}>Licence</th></tr></thead>
            <tbody>
              {meta.credits.map((c) => (
                <tr key={c.id}>
                  <td>{c.name}</td>
                  <td style={{ textAlign: "left" }}>{c.credit}</td>
                  <td style={{ textAlign: "left" }}>{c.licence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="ink2">Credits load from the API&apos;s licence register.</p>
      )}
      <p className="small ink2">Basemap © OpenFreeMap, © OpenMapTiles, data © OpenStreetMap contributors.</p>
    </main>
  );
}
