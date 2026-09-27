// Typed client for the SailabAI FastAPI backend (reached through Next.js rewrites).

export type Feature<P = Record<string, unknown>> = {
  type: "Feature";
  properties: P;
  geometry: { type: string; coordinates: number[] | number[][] };
};
export type FeatureCollection<P = Record<string, unknown>> = { type: "FeatureCollection"; features: Feature<P>[] };

export type Credit = { id: string; name: string; credit: string; licence: string };

export type Meta = {
  name: string;
  version: string;
  disclaimer: string;
  cube: string;
  synthetic: boolean;
  synthetic_note: string | null;
  bbox: [number, number, number, number];
  center: [number, number];
  grid: { crs: string; resolution_m: number; width: number; height: number };
  places: FeatureCollection<{ name: string; kind: string }>;
  rivers: FeatureCollection<{ name: string }>;
  gauges: FeatureCollection<{ id: string; name: string; river: string; country: string; design_capacity_cusecs: number | null }>;
  breaches: FeatureCollection<{ id: string; river: string; start: string; end: string; year: number }>;
  credits: Credit[];
};

export type Range = { expected: number; low: number; high: number };

export type PlaceRisk = { name: string; lon: number; lat: number; people: Range; max_prob: number };

export type Horizon = {
  name: string; // "next" | "d1" ... "d7"
  lead_days: number;
  valid_time: string;
  flooded_km2_expected: number;
  uncertain_km2: number;
  totals: { population: Range; buildings: Range; road_km: Range; area_km2: Range };
  places: PlaceRisk[];
  layers: Record<string, string>;
  tiles: Record<string, string>;
};

export type RunSummary = {
  run_id: string;
  issue_time: string;
  cube: string;
  synthetic: boolean;
  disclaimer: string;
  model: string;
  calibration: { temperature: number; alpha: number };
  latest_pass: { scene_id: string; time: string; mapped_by: string } | null;
  new_passes: string[];
  current_flooded_km2: number;
  horizons: Horizon[];
  tiles: { current: string };
};

export type RunListItem = {
  run_date: string;
  issue_time: string;
  model: string;
  current_flooded_km2: number;
  horizons: { name: string; lead_days: number; flooded_km2_expected: number }[];
};

export type RoadRisk = {
  id: string;
  name: string;
  class: string;
  max_prob: number;
  mean_prob: number;
  max_prob_low: number | null;
  max_prob_high: number | null;
};

export type Hydrograph = {
  gauge: { id: string; name: string; river: string; country: string; design_capacity_cusecs: number | null };
  issue_date: string;
  observed: { date: string; q: number }[];
  observed_after: { date: string; q: number }[];
  forecast: { date: string; lead_day: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
  india_data_missing: boolean;
};

export type ResultRow = {
  time: string;
  commit: string;
  experiment: string;
  model: string;
  baseline: string;
  split: string;
  event: string;
  lead: string;
  subset: string;
  metric: string;
  value: number | null;
  baseline_value: number | null;
  skill: number | null;
  data_versions: string | null;
  notes: string | null;
};

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { cache: "no-store", signal });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} for ${path}`);
  }
  return (await res.json()) as T;
}

export const api = {
  meta: (cube?: string) => getJSON<Meta>(`/api/meta${cube ? `?cube=${cube}` : ""}`),
  runs: (cube: string) => getJSON<RunListItem[]>(`/api/runs?cube=${cube}`),
  run: (cube: string, date: string) => getJSON<RunSummary>(`/api/runs/${cube}/${date}`),
  roads: (cube: string, date: string, horizon: string) =>
    getJSON<RoadRisk[]>(`/api/runs/${cube}/${date}/roads?horizon=${horizon}&limit=25`),
  hydrograph: (cube: string, gauge: string, date: string) =>
    getJSON<Hydrograph>(`/api/gauges/${gauge}/hydrograph?cube=${cube}&date=${date}`),
  results: () => getJSON<ResultRow[]>(`/api/results`),
};
