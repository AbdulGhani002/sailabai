"use client";

import { useEffect, useMemo, useState } from "react";
import { ReliabilityChart } from "@/components/ReliabilityChart";
import { SkillChart, type Series } from "@/components/SkillChart";
import { api, type Reliability, type ResultRow } from "@/lib/api";

const LEADS = ["+1d", "+2d", "+3d", "+4d", "+5d", "+6d", "+7d"];
// fixed categorical order: slot 1 blue, 2 orange, 3 aqua, 4 yellow
const MODELS = [
  { key: "unet_tt", label: "UNet-TT", color: "var(--series-1)" },
  { key: "xgboost", label: "XGBoost", color: "var(--series-2)" },
  { key: "river_threshold", label: "River threshold", color: "var(--series-3)" },
  { key: "historical_frequency", label: "Hist. frequency", color: "var(--series-4)" },
];

function fmt(v: number | null | undefined, digits = 3) {
  return v == null || Number.isNaN(v) ? "–" : v.toFixed(digits);
}

function skillClass(s: number | null | undefined) {
  if (s == null) return "";
  return s > 0 ? "better" : s < 0 ? "worse" : "";
}

export default function ResultsPage() {
  const [rows, setRows] = useState<ResultRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rel, setRel] = useState<Reliability | null>(null);
  const [relLead, setRelLead] = useState("all");
  const [split, setSplit] = useState<"val" | "test">("val");
  useEffect(() => {
    setRows(null);
    api.results(split).then(setRows).catch((e) => setError(String(e)));
    api.reliability(split).then(setRel).catch(() => setRel(null));
  }, [split]);

  const forecast = useMemo(() => (rows ?? []).filter((r) => r.experiment.startsWith("model2") && !r.experiment.endsWith("-truth")), [rows]);
  const mapping = useMemo(() => (rows ?? []).filter((r) => r.experiment.startsWith("model1")), [rows]);
  const leads = LEADS.filter((l) => forecast.some((r) => r.lead === l));
  const get = (model: string, lead: string, metric: string, subset = "all") =>
    forecast.find((r) => r.model === model && r.lead === lead && r.metric === metric && r.subset === subset);

  const skillSeries = (metric: string, subset = "all"): Series[] =>
    MODELS.filter((m) => forecast.some((r) => r.model === m.key)).map((m) => ({
      ...m,
      points: leads.map((lead) => ({ lead, value: get(m.key, lead, metric, subset)?.skill ?? null })),
    }));

  const unetWins = leads.filter((l) => (get("unet_tt", l, "iou")?.skill ?? -1) > 0);

  if (error) return <main className="page"><h1>Results</h1><p className="lead">Could not load results: {error}</p></main>;
  if (!rows) return <main className="page"><p className="ink2">Loading…</p></main>;

  return (
    <main className="page">
      <h1>Results</h1>
      <p className="lead">
        Every score sits next to its baseline, on floods the models never saw in training, split by event and averaged
        per event. Normal water is removed before scoring.
      </p>
      <div className="segmented" role="group" aria-label="Which flood" style={{ marginBottom: 8 }}>
        <button type="button" aria-pressed={split === "val"} onClick={() => setSplit("val")}>
          Validation: 2023 Sutlej flood
        </button>
        <button type="button" aria-pressed={split === "test"} onClick={() => setSplit("test")}>
          Test: 2025 record flood (scored once)
        </button>
      </div>

      {forecast.length === 0 ? (
        <p className="card">No forecast results yet. Run <code>sailab evaluate forecast</code>.</p>
      ) : (
        <>
          <h2>Does Model 2 beat &quot;repeat the last map&quot;?</h2>
          <p className="ink2">
            UNet-TT has positive skill over persistence at {unetWins.length} of {leads.length} lead times
            {unetWins.length ? ` (${unetWins.join(", ")})` : ""}. The target is positive skill at 3 or more days ahead.
          </p>
          <div className="grid-2">
            <div className="card">
              <h3 style={{ margin: "0 0 4px", fontSize: 14 }}>Flood-map overlap (IoU), skill over persistence</h3>
              <SkillChart series={skillSeries("iou")} leads={leads} yLabel="IoU skill over persistence by lead time" />
            </div>
            <div className="card">
              <h3 style={{ margin: "0 0 4px", fontSize: 14 }}>Brier score, skill over persistence</h3>
              <SkillChart series={skillSeries("brier")} leads={leads} yLabel="Brier skill over persistence by lead time" />
            </div>
          </div>

          <h2>Scores by lead time (all pixels)</h2>
          <div className="card scroll-x">
            <table className="data">
              <thead>
                <tr>
                  <th>Model</th>
                  {leads.map((l) => <th key={l}>{l} IoU</th>)}
                  {leads.map((l) => <th key={`b${l}`}>{l} Brier</th>)}
                </tr>
              </thead>
              <tbody>
                {[{ key: "persistence", label: "Persistence" }, ...MODELS].filter((m) => forecast.some((r) => r.model === m.key)).map((m) => (
                  <tr key={m.key}>
                    <td>{m.label}</td>
                    {leads.map((l) => {
                      const r = get(m.key, l, "iou");
                      return <td key={l} className={skillClass(r?.skill)}>{fmt(r?.value)}</td>;
                    })}
                    {leads.map((l) => {
                      const r = get(m.key, l, "brier");
                      return <td key={`b${l}`} className={skillClass(r?.skill)}>{fmt(r?.value, 4)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small ink2">Green: better than persistence; red: worse. Lower Brier is better.</p>
          </div>

          <h2>Where things change</h2>
          <p className="ink2">Persistence can never predict a pixel that newly floods or drains, so these subsets show whether a model sees change coming.</p>
          <div className="card scroll-x">
            <table className="data">
              <thead>
                <tr><th>Model</th>{leads.map((l) => <th key={l}>{l} newly flooded F1</th>)}{leads.map((l) => <th key={`d${l}`}>{l} drained F1</th>)}</tr>
              </thead>
              <tbody>
                {MODELS.filter((m) => forecast.some((r) => r.model === m.key)).map((m) => (
                  <tr key={m.key}>
                    <td>{m.label}</td>
                    {leads.map((l) => <td key={l}>{fmt(get(m.key, l, "f1", "newly_flooded")?.value)}</td>)}
                    {leads.map((l) => <td key={`d${l}`}>{fmt(get(m.key, l, "f1", "drained")?.value)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Are the percentages honest?</h2>
          {rel && (
            <div className="card" style={{ marginBottom: 12 }}>
              <div className="row" style={{ flexWrap: "wrap", gap: 12 }}>
                <p className="ink2" style={{ margin: 0, flex: "1 1 280px" }}>
                  When a model says 30%, it should flood about 30% of the time: points on the dashed line are honest.
                  {rel.split === "test" ? "2025 test flood" : "2023 validation flood"}; bins with fewer than 50 pixels are hidden.
                </p>
                <div className="segmented" role="group" aria-label="Lead time">
                  {["all", ...LEADS.filter((l) => rel.models.unet_tt?.[l])].map((l) => (
                    <button key={l} type="button" aria-pressed={relLead === l} onClick={() => setRelLead(l)}>
                      {l === "all" ? "All leads" : l}
                    </button>
                  ))}
                </div>
              </div>
              <ReliabilityChart series={MODELS.filter((m) => rel.models[m.key]).map((m) => ({
                key: m.key, label: m.label, color: m.color, bins: rel.models[m.key][relLead] ?? [],
              }))} />
            </div>
          )}
          <div className="card scroll-x">
            <table className="data">
              <thead><tr><th>Model</th>{leads.map((l) => <th key={l}>{l} calibration error</th>)}</tr></thead>
              <tbody>
                {MODELS.filter((m) => forecast.some((r) => r.model === m.key)).map((m) => (
                  <tr key={m.key}>
                    <td>{m.label}</td>
                    {leads.map((l) => <td key={l}>{fmt(get(m.key, l, "ece")?.value)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small ink2">Expected calibration error: the average gap between the forecast chance and how often it flooded. Lower is better.</p>
          </div>
        </>
      )}

      <h2>Model 1: flood mapping</h2>
      {mapping.length === 0 ? (
        <p className="card">No mapping results yet. Run <code>sailab evaluate mapping</code>.</p>
      ) : (
        <div className="card scroll-x">
          <table className="data">
            <thead><tr><th>Model (reference)</th><th>F1</th><th>IoU</th><th>Precision</th><th>Recall</th><th>F1 skill vs Otsu</th></tr></thead>
            <tbody>
              {[...new Set(mapping.map((r) => r.model))].map((model) => {
                const m = (metric: string) => mapping.find((r) => r.model === model && r.metric === metric && r.subset === "all");
                return (
                  <tr key={model}>
                    <td>{model}</td>
                    <td>{fmt(m("f1")?.value)}</td>
                    <td>{fmt(m("iou")?.value)}</td>
                    <td>{fmt(m("precision")?.value)}</td>
                    <td>{fmt(m("recall")?.value)}</td>
                    <td className={skillClass(m("f1")?.skill)}>{fmt(m("f1")?.skill)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="small ink2">Scored against two references (GFM labels and hand-checked labels), because two expert maps of the same flood agreed only 48% in one study. Target: flood F1 around 0.8.</p>
        </div>
      )}
    </main>
  );
}
