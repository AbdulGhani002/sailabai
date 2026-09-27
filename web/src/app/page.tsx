"use client";

import { useEffect, useMemo, useState } from "react";
import { FloodMap } from "@/components/FloodMap";
import { HydrographChart } from "@/components/Hydrograph";
import { Legend } from "@/components/Legend";
import { Timeline } from "@/components/Timeline";
import { Notice } from "@/components/TopBar";
import { api, type Hydrograph, type Meta, type Range, type RoadRisk, type RunListItem, type RunSummary } from "@/lib/api";
import { compact, horizonLabel, longDate, pct, pkTime, whole } from "@/lib/format";
import { useTheme } from "@/lib/theme";

type View = "prob" | "sets" | "spread";
const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "prob", label: "Chance", hint: "Calibrated chance of flooding at each pixel" },
  { id: "sets", label: "Confidence", hint: "Likely (50%+) and possible: the possible zone caught 80% of real flooding on the validation flood" },
  { id: "spread", label: "Disagreement", hint: "How much the ensemble members disagree" },
];

function RangeText({ r, unit = "" }: { r: Range; unit?: string }) {
  return (
    <>
      <span className="range">
        range {compact(r.low)}–{compact(r.high)}
        {unit}
      </span>
      {r.likely != null && (
        <span className="range" style={{ display: "block" }}>
          {compact(r.likely)}
          {unit} where ≥50% likely
        </span>
      )}
    </>
  );
}

export default function Dashboard() {
  const [theme] = useTheme();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runDate, setRunDate] = useState<string | null>(null);
  const [run, setRun] = useState<RunSummary | null>(null);
  const [horizon, setHorizon] = useState<string>("now");
  const [view, setView] = useState<View>("prob");
  const [roads, setRoads] = useState<RoadRisk[]>([]);
  const [gauge, setGauge] = useState<string>("trimmu");
  const [hydro, setHydro] = useState<Hydrograph | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.meta()
      .then(async (m) => {
        setMeta(m);
        const list = await api.runs(m.cube);
        setRuns(list);
        if (list.length) setRunDate(list[list.length - 1].run_date);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!meta || !runDate) return;
    api.run(meta.cube, runDate).then(setRun).catch((e) => setError(String(e)));
  }, [meta, runDate]);

  const h = useMemo(() => run?.horizons.find((x) => x.name === horizon) ?? null, [run, horizon]);

  useEffect(() => {
    if (!meta || !runDate || horizon === "now") return setRoads([]);
    api.roads(meta.cube, runDate, horizon).then(setRoads).catch(() => setRoads([]));
  }, [meta, runDate, horizon]);

  useEffect(() => {
    if (!meta || !runDate) return;
    api.hydrograph(meta.cube, gauge, runDate).then(setHydro).catch(() => setHydro(null));
  }, [meta, runDate, gauge]);

  // keep the chosen horizon valid when switching runs (the next-pass horizon may not exist)
  useEffect(() => {
    if (run && horizon !== "now" && !run.horizons.some((x) => x.name === horizon)) setHorizon("d1");
  }, [run, horizon]);

  const layer = horizon === "now" ? "current" : `${h?.tiles[view] ? view : "prob"}_${horizon}`;
  const tileUrl = !run ? null : horizon === "now" ? run.tiles.current : (h?.tiles[view] ?? h?.tiles.prob ?? null);
  const themedTiles = tileUrl ? `${tileUrl}?theme=${theme}` : null;
  const runIndex = runs.findIndex((r) => r.run_date === runDate);

  if (error) {
    return (
      <main className="page">
        <h1>Can&apos;t reach the SailabAI API</h1>
        <p className="lead">{error}</p>
        <p>Start it with <code>sailab api serve</code> (and run the twin once with <code>sailab twin replay</code>).</p>
      </main>
    );
  }
  if (!meta) return <main className="page"><p className="ink2">Loading…</p></main>;

  return (
    <>
      <Notice disclaimer={meta.disclaimer} synthetic={meta.synthetic} />
      <div className="dash">
        <aside className="side" aria-label="Controls and risk summary">
          <section>
            <h2 className="section-title">Forecast run</h2>
            {runs.length === 0 ? (
              <p className="empty">No twin runs yet. Run <code>sailab twin replay</code> to create some.</p>
            ) : (
              <>
                <div className="run-picker">
                  <button className="icon-btn" type="button" aria-label="Previous day" disabled={runIndex <= 0}
                    onClick={() => setRunDate(runs[runIndex - 1].run_date)}>‹</button>
                  <div className="when">
                    <strong>{runDate ? longDate(runDate) : "–"}</strong>
                    <small>{run ? `issued ${pkTime(run.issue_time)}` : " "}</small>
                  </div>
                  <button className="icon-btn" type="button" aria-label="Next day" disabled={runIndex >= runs.length - 1}
                    onClick={() => setRunDate(runs[runIndex + 1].run_date)}>›</button>
                </div>
                <Timeline runs={runs} selected={runDate ?? ""} onSelect={setRunDate} />
              </>
            )}
          </section>

          <section>
            <h2 className="section-title">Horizon</h2>
            <div className="segmented" role="group" aria-label="Forecast horizon">
              <button type="button" aria-pressed={horizon === "now"} onClick={() => setHorizon("now")}>Now</button>
              {run?.horizons.map((x) => (
                <button key={x.name} type="button" aria-pressed={horizon === x.name} onClick={() => setHorizon(x.name)}>
                  {x.name === "next" ? "Next pass" : horizonLabel(x.name).replace(" days", "d").replace(" day", "d")}
                </button>
              ))}
            </div>
          </section>

          <section>
            <h2 className="section-title">Show</h2>
            <div className="segmented" role="group" aria-label="Map view">
              {VIEWS.map((v) => (
                <button key={v.id} type="button" aria-pressed={view === v.id} disabled={horizon === "now"}
                  onClick={() => setView(v.id)}>
                  {v.label}
                </button>
              ))}
            </div>
            <p className="small ink2" style={{ margin: "6px 0 0" }}>
              {horizon === "now" ? "Today's map from the latest radar pass." : VIEWS.find((v) => v.id === view)?.hint}
            </p>
          </section>

          {run && horizon === "now" && (
            <section>
              <h2 className="section-title">Today&apos;s flood</h2>
              <div className="tiles">
                <div className="tile wide">
                  <div className="label">Flooded area on the latest map</div>
                  <div className="value">{whole(run.current_flooded_km2)} km²</div>
                  {run.latest_pass && (
                    <div className="range">
                      Sentinel-1 pass {pkTime(run.latest_pass.time)}, mapped by{" "}
                      {run.latest_pass.mapped_by === "model1" ? "Model 1" : "GFM"}
                    </div>
                  )}
                </div>
              </div>
              {run.current_exposure && (
                <div className="tiles" style={{ marginTop: 8 }}>
                  <div className="tile">
                    <div className="label">People in flooded pixels</div>
                    <div className="value">{compact(run.current_exposure.totals.population.expected)}</div>
                  </div>
                  <div className="tile">
                    <div className="label">Buildings</div>
                    <div className="value">{compact(run.current_exposure.totals.buildings.expected)}</div>
                  </div>
                  <div className="tile wide">
                    <div className="label">Roads under water</div>
                    <div className="value">{compact(run.current_exposure.totals.road_km.expected)} km</div>
                  </div>
                </div>
              )}
              <p className="small ink2" style={{ marginTop: 8 }}>
                Pick a forecast day to see the chance of flooding, with ranges.
              </p>
            </section>
          )}

          {h && horizon !== "now" && (
            <section>
              <h2 className="section-title">
                {horizonLabel(h.name, h.lead_days)} · {pkTime(h.valid_time)}
              </h2>
              <div className="tiles">
                <div className="tile">
                  <div className="label">Flooded area</div>
                  <div className="value">{compact(h.totals.area_km2.expected)} km²</div>
                  <RangeText r={h.totals.area_km2} />
                </div>
                <div className="tile">
                  <div className="label">People</div>
                  <div className="value">{compact(h.totals.population.expected)}</div>
                  <RangeText r={h.totals.population} />
                </div>
                <div className="tile">
                  <div className="label">Buildings</div>
                  <div className="value">{compact(h.totals.buildings.expected)}</div>
                  <RangeText r={h.totals.buildings} />
                </div>
                <div className="tile">
                  <div className="label">Roads</div>
                  <div className="value">{compact(h.totals.road_km.expected)} km</div>
                  <RangeText r={h.totals.road_km} />
                </div>
              </div>
              <p className="small ink2" style={{ marginTop: 8 }}>
                Big numbers are expected values (chance times exposure); ranges cover 90% of outcomes on the
                validation flood. Possible-flood zone: {compact(h.uncertain_km2)} km².
              </p>
            </section>
          )}

          {h && horizon !== "now" && h.places.length > 0 && (
            <section>
              <h2 className="section-title">Towns: people in likely-flooded pixels (8 km)</h2>
              <ul className="list">
                {h.places.slice(0, 8).map((p) => (
                  <li key={p.name}>
                    <span className="name">{p.name}</span>
                    <span className="num">
                      {compact(p.people.expected)} ({compact(p.people.low)}–{compact(p.people.high)})
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {horizon !== "now" && (
            <section>
              <h2 className="section-title">Road sections at risk</h2>
              {roads.length === 0 ? (
                <p className="empty small">No road section above 5% chance.</p>
              ) : (
                <ul className="list">
                  {roads.slice(0, 10).map((r) => (
                    <li key={r.id}>
                      <span className="name" title={r.name}>{r.name}</span>
                      <span className="num">
                        {pct(r.max_prob)}
                        {r.max_prob_low != null && r.max_prob_high != null && ` (${pct(r.max_prob_low)}–${pct(r.max_prob_high)})`}
                      </span>
                      <span className="meter" aria-hidden><span style={{ width: `${r.max_prob * 100}%` }} /></span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {run && (
            <p className="small muted">
              Model: {run.model}. Calibration temperature {run.calibration.temperature.toFixed(2)}.
            </p>
          )}
        </aside>

        <div className="main">
          <div className="map-wrap">
            <FloodMap meta={meta} tileUrl={themedTiles} theme={theme} selectedGauge={gauge} onGaugeSelect={setGauge} />
            <Legend layer={layer} theme={theme}
              note={horizon === "now" ? "Rivers are normal water, not flood." : undefined} />
            {run?.new_passes.length ? (
              <div className="map-status">
                New radar pass today: the twin restarted from the fresh map ({run.new_passes.length} pass
                {run.new_passes.length > 1 ? "es" : ""}).
              </div>
            ) : null}
          </div>
          <section className="gauge-panel" aria-label="River gauge">
            {hydro ? (
              <>
                <header>
                  <h3>{hydro.gauge.name}</h3>
                  <span className="ink2">River {hydro.gauge.river[0].toUpperCase() + hydro.gauge.river.slice(1)} · flow in cusecs · click a gauge on the map</span>
                  <span className="spacer" />
                  <div className="chart-legend">
                    <span className="key"><span className="line" /> Observed</span>
                    <span className="key"><span className="line dashed" /> GloFAS median</span>
                    <span className="key"><span className="band" /> 5–95% of 51 members</span>
                    {hydro.observed_after.length > 0 && <span className="key"><span className="line dotted" /> What happened</span>}
                  </div>
                </header>
                {hydro.india_data_missing && (
                  <p className="small ink2">No shared data from this Indian station since April 2025; the model runs without it.</p>
                )}
                <HydrographChart data={hydro} />
              </>
            ) : (
              <p className="empty small">Select a gauge on the map to see its flow.</p>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
