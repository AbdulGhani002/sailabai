"""Synthetic river flows, rain and forecasts for the demo world.

Flood waves start at the upstream inflows (Chenab at Akhnoor, Jhelum, Ravi at Madhopur, Sutlej at
Ferozepur), travel down the network with the FFC time-lag chart delays, and add up at the
confluences. Peaks for the named events are scaled to the FFC numbers in configs/events.yaml, and
the 2025 travel times are stretched the way they were in reality (Qadirabad to Trimmu took 112
hours instead of 64). All numbers are made up apart from those anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from sailab.config import EventsConfig, GaugesConfig
from sailab.features import soil_moisture_index

# Seasonal base flow at each inflow: (1 May, mid-July to late August plateau, 31 October), cusecs
BASE_FLOW = {
    "akhnoor": (30e3, 75e3, 25e3),
    "jhelum_in": (25e3, 45e3, 15e3),
    "madhopur": (4e3, 9e3, 3e3),
    "ferozepur": (3e3, 8e3, 2e3),
}
# Flood wave size per unit storm intensity at each inflow, cusecs
STORM_SCALE = {"akhnoor": 230e3, "jhelum_in": 110e3, "madhopur": 55e3, "ferozepur": 45e3}

# Routing delays in days (official FFC lags), capacity above which floodplains store water
REACHES = {
    "akhnoor>marala": (8 / 24, None),
    "marala>khanki": (11 / 24, None),
    "khanki>qadirabad": (6 / 24, 820e3),
    "qadirabad>trimmu": (64 / 24, 640e3),
    "jhelum_in>trimmu": (1.0, None),
    "madhopur>sidhnai": (96 / 24, 140e3),
    "ferozepur>ganda_singh_wala": (6 / 24, None),
    "ganda_singh_wala>islam": (72 / 24, 300e3),
    "trimmu>panjnad": (30 / 24, 650e3),
    "sidhnai>panjnad": (36 / 24, None),
    "islam>panjnad": (48 / 24, None),
}

# Storms placed for the named events: (year, gauge whose peak is matched, storm date, inflows it
# hits hardest). The peak values themselves come from configs/events.yaml.
PEAK_ANCHORS = [
    (2014, "qadirabad", "09-03", ("akhnoor", "jhelum_in")),
    (2016, "khanki", "08-02", ("akhnoor",)),
    (2023, "ganda_singh_wala", "08-15", ("ferozepur",)),
    (2025, "qadirabad", "08-24", ("akhnoor",)),
    (2025, "panjnad", "08-24", ("madhopur", "ferozepur")),
    (2026, "marala", "08-08", ("akhnoor",)),
]

MODEL_POINTS = ["marala", "qadirabad", "trimmu", "sidhnai", "islam", "panjnad"]
INDIA_POINTS = ["akhnoor", "madhopur", "ferozepur"]
PK_GAUGES = ["marala", "khanki", "qadirabad", "trimmu", "sidhnai", "ganda_singh_wala", "islam", "panjnad"]

BREACH_SITES = [
    # name, river, lat, lon, controlling gauge, flow that opens it (cusecs)
    {"id": "rewaz_bridge", "river": "chenab", "lat": 31.00, "lon": 72.05, "gauge": "trimmu", "opens_at": 560e3},
    {"id": "sidhnai", "river": "ravi", "lat": 30.555, "lon": 72.10, "gauge": "sidhnai", "opens_at": 135e3},
]


@dataclass
class Storm:
    day: int
    intensity: float
    weights: dict[str, float]
    tp: float
    local: float


@dataclass
class SeasonFlows:
    year: int
    dates: pd.DatetimeIndex
    q: pd.DataFrame                  # true daily flow at every point, incl. internal reach points
    rain_upper: np.ndarray           # mm/day over the upper catchments
    rain_local: np.ndarray           # mm/day over the study area
    slowdown: float                  # travel-time multiplier for this season
    storms: list[Storm] = field(default_factory=list)
    breaches: list[dict] = field(default_factory=list)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _base(n_days: int, dates: pd.DatetimeIndex, spec: tuple[float, float, float], factor: float) -> np.ndarray:
    doy = dates.dayofyear.to_numpy()
    rise = _smoothstep((doy - 121) / (196 - 121))       # 1 May -> 15 Jul
    fall = _smoothstep((doy - 237) / (304 - 237))       # 25 Aug -> 31 Oct
    may, peak, octb = spec
    return factor * (may + (peak - may) * rise - (peak - octb) * fall)


def _wave(n_days: int, day: float, peak: float, tp: float, k: float = 3.0) -> np.ndarray:
    tau = np.arange(n_days) - (day - tp)
    out = np.zeros(n_days)
    pos = tau > 0
    x = tau[pos] / tp
    out[pos] = peak * x**k * np.exp(k * (1 - x))
    return out


def route(q: np.ndarray, lag_days: float, capacity: float | None = None, spill: float = 0.5,
          return_rate: float = 0.06) -> np.ndarray:
    """Delay and smooth a hydrograph; above `capacity`, part of the flow spills to floodplain storage
    and drains back slowly, which lowers and lengthens the peak downstream."""
    spread = 0.3 * lag_days + 0.25
    taps = np.arange(int(np.ceil(lag_days + 4 * spread)) + 2)
    kernel = np.exp(-0.5 * ((taps - lag_days) / spread) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.full(len(taps), q[0]), q])
    out = np.convolve(padded, kernel)[len(taps):len(taps) + len(q)]
    if capacity is None:
        return out
    shaved = np.empty_like(out)
    store = 0.0
    for i, v in enumerate(out):
        excess = max(0.0, v - capacity)
        passed = v - (1 - spill) * excess
        store += (1 - spill) * excess
        back = return_rate * store
        store -= back
        shaved[i] = passed + back
    return shaved


def _network(inflow: dict[str, np.ndarray], nullah: np.ndarray, slowdown: float) -> dict[str, np.ndarray]:
    def r(name: str, q: np.ndarray, slow: bool = True, cap_scale: float = 1.0) -> np.ndarray:
        lag, cap = REACHES[name]
        return route(q, lag * (slowdown if slow else 1.0), None if cap is None else cap * cap_scale)

    q: dict[str, np.ndarray] = dict(inflow)
    q["marala"] = r("akhnoor>marala", q["akhnoor"], slow=False) * 1.04 + nullah
    q["khanki"] = r("marala>khanki", q["marala"], slow=False) + 0.25 * nullah
    q["qadirabad"] = r("khanki>qadirabad", q["khanki"], slow=False)
    q["chenab_at_trimmu"] = r("qadirabad>trimmu", q["qadirabad"])
    q["trimmu"] = q["chenab_at_trimmu"] + r("jhelum_in>trimmu", q["jhelum_in"], slow=False)
    q["sidhnai"] = 0.8 * r("madhopur>sidhnai", q["madhopur"]) + 0.15 * nullah + 2e3
    q["ganda_singh_wala"] = r("ferozepur>ganda_singh_wala", q["ferozepur"], slow=False)
    q["islam"] = 0.8 * r("ganda_singh_wala>islam", q["ganda_singh_wala"])
    chenab = r("trimmu>panjnad", q["trimmu"])
    ravi = r("sidhnai>panjnad", q["sidhnai"])
    q["chenab_below_ravi"] = route(q["trimmu"], 0.6 * REACHES["trimmu>panjnad"][0] * slowdown) + route(q["sidhnai"], 0.5)
    q["panjnad"] = 0.95 * (chenab + ravi + r("islam>panjnad", q["islam"]))
    return q


def simulate_season(year: int, rng: np.random.Generator, targets: dict[str, float] | None = None,
                    start: str = "05-01", end: str = "10-31") -> SeasonFlows:
    """One monsoon season of true flows. `targets` maps gauge -> peak (cusecs) for named events."""
    dates = pd.date_range(f"{year}-{start}", f"{year}-{end}", freq="D")
    n = len(dates)
    doy0 = dates[0].dayofyear
    factor = rng.uniform(0.8, 1.2)
    slowdown = 1.75 if year == 2025 else float(rng.uniform(1.0, 1.2))

    storms: list[Storm] = []
    for _ in range(int(rng.poisson(3.5)) + 2):
        day = int(rng.integers(date(year, 7, 1).timetuple().tm_yday, date(year, 9, 20).timetuple().tm_yday)) - doy0
        w = rng.dirichlet([2.0, 1.2, 1.0, 0.8])
        storms.append(Storm(day, float(rng.lognormal(-0.2, 0.45)), dict(zip(BASE_FLOW, w * 2.2, strict=True)),
                            float(rng.uniform(1.4, 3.0)), float(rng.uniform(0.0, 1.0))))

    anchors = [a for a in PEAK_ANCHORS if a[0] == year and targets and a[1] in targets]
    anchor_storms: dict[str, Storm] = {}
    for _, gauge, mmdd, sources in anchors:
        day = pd.Timestamp(f"{year}-{mmdd}").dayofyear - doy0
        weights = {s: (1.0 if s in sources else 0.12) for s in BASE_FLOW}
        storm = Storm(day, 1.0, weights, 2.2, 0.8)
        storms.append(storm)
        anchor_storms[gauge] = storm

    def run() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        inflow = {s: _base(n, dates, BASE_FLOW[s], factor) for s in BASE_FLOW}
        rain_u = rng_rain_u.copy()
        rain_l = rng_rain_l.copy()
        nullah = np.zeros(n)
        for st in storms:
            for s, w in st.weights.items():
                inflow[s] = inflow[s] + _wave(n, st.day, STORM_SCALE[s] * st.intensity * w, st.tp)
            nullah += _wave(n, st.day - 0.5, 60e3 * st.intensity * st.weights["akhnoor"] / 2.2, 1.2)
            rain_u += _wave(n, st.day - 1.2, 28.0 * st.intensity, 1.0, k=2.0)
            rain_l += _wave(n, st.day - 0.8, 22.0 * st.intensity * st.local, 0.9, k=2.0)
        return _network(inflow, nullah, slowdown), rain_u, rain_l

    rng_rain_u = rng.gamma(0.35, 3.0, n) * ((dates.month >= 6) & (dates.month <= 9))
    rng_rain_l = rng.gamma(0.25, 3.5, n) * ((dates.month >= 7) & (dates.month <= 9))
    convective = rng.random(n) < 0.035
    rng_rain_l[convective] += rng.uniform(25, 85, convective.sum())

    q, rain_u, rain_l = run()
    for _ in range(6):  # scale anchor storms until the routed peaks match the FFC numbers
        changed = False
        for gauge, storm in anchor_storms.items():
            day = int(storm.day)
            window = slice(max(0, day - 2), min(n, day + 16))
            peak = float(q[gauge][window].max())
            want = targets[gauge]  # type: ignore[index]
            if abs(peak - want) / want > 0.01:
                storm.intensity *= float(np.clip(want / max(peak, 1.0), 0.3, 3.0)) ** 0.9
                changed = True
        if not changed:
            break
        q, rain_u, rain_l = run()

    frame = pd.DataFrame(q, index=dates)
    breaches = []
    for site in BREACH_SITES:
        over = np.flatnonzero(frame[site["gauge"]].to_numpy() >= site["opens_at"])
        if over.size:
            breaches.append({**site, "start": dates[over[0]].date().isoformat(),
                             "end": (dates[over[0]] + pd.Timedelta(days=12)).date().isoformat()})
    return SeasonFlows(year, dates, frame, rain_u, rain_l, slowdown, storms, breaches)


def event_targets(events: EventsConfig) -> dict[int, dict[str, float]]:
    """Peak flows per year from the events config."""
    out: dict[int, dict[str, float]] = {}
    for e in events.events:
        for peak in [p for p in [e.peak, *e.other_peaks] if p is not None]:
            out.setdefault(e.year, {})[peak.gauge] = peak.cusecs
    return out


# --------------------------------------------------------------------------- observations and forecasts


def observed_discharge(seasons: list[SeasonFlows], gauges: GaugesConfig, rng: np.random.Generator,
                       version: str) -> pd.DataFrame:
    """What we would actually receive: FFD 6 am readings with noise and gaps; India data stops in 2025."""
    cutoff = pd.Timestamp(gauges.india_data_cutoff) if gauges.india_data_cutoff else None
    rows = []
    for s in seasons:
        for point in PK_GAUGES + INDIA_POINTS:
            q = s.q[point].to_numpy() * (1 + rng.normal(0, 0.03, len(s.dates)))
            missing = rng.random(len(s.dates)) < (0.04 if point in PK_GAUGES else 0.10)
            if point in INDIA_POINTS and cutoff is not None:
                missing |= s.dates >= cutoff
            source = "india_share" if point in INDIA_POINTS else "ffd"
            rows.append(pd.DataFrame({"date": s.dates[~missing], "point_id": point,
                                      "q_cusecs": q[~missing].astype(np.float32), "source": source,
                                      "version": version}))
    return pd.concat(rows, ignore_index=True)


def weather_table(seasons: list[SeasonFlows]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        frames.append(pd.DataFrame({"date": s.dates, "rain_upper_mm": s.rain_upper.astype(np.float32),
                                    "rain_local_mm": s.rain_local.astype(np.float32),
                                    "soil_moisture": soil_moisture_index(s.rain_local).astype(np.float32)}))
    return pd.concat(frames, ignore_index=True)


def glofas_forecasts(seasons: list[SeasonFlows], rng: np.random.Generator, points: list[str] = MODEL_POINTS,
                     leads: int = 7, members: int = 51, issue_start: str = "06-01",
                     issue_end: str = "10-15") -> pd.DataFrame:
    """GloFAS-like ensembles: a fixed bias per point (the model is coarse), noise and timing errors
    that grow with lead time, and member 0 as the unperturbed control forecast."""
    bias = {p: float(rng.uniform(0.75, 1.25)) for p in points}
    lead_idx = np.arange(1, leads + 1)
    sigma = 0.08 + 0.05 * lead_idx
    frames = []
    for s in seasons:
        issues = pd.date_range(f"{s.year}-{issue_start}", f"{s.year}-{issue_end}", freq="D")
        issue_pos = np.searchsorted(s.dates, issues)
        n_i = len(issues)
        for p in points:
            truth = s.q[p].to_numpy()
            day_axis = np.arange(len(truth), dtype=np.float64)
            shift = rng.normal(0, 1, (n_i, 1, members)) * (0.15 * lead_idx)[None, :, None]
            shift[:, :, 0] = 0.0
            t = issue_pos[:, None, None] + lead_idx[None, :, None] + shift
            base = np.interp(t, day_axis, truth)
            z = np.zeros((n_i, leads, members))
            eps = rng.normal(0, 1, (n_i, leads, members))
            for j in range(leads):
                z[:, j] = (0.8 * z[:, j - 1] if j else 0.0) + np.sqrt(1 - 0.64) * eps[:, j]
            z[:, :, 0] = 0.0
            q = base * bias[p] * np.exp(sigma[None, :, None] * z - 0.5 * sigma[None, :, None] ** 2)
            ii, ll, mm = np.meshgrid(np.arange(n_i), lead_idx, np.arange(members), indexing="ij")
            frames.append(pd.DataFrame({
                "issue_date": issues[ii.ravel()], "point_id": p, "lead_day": ll.ravel().astype(np.int8),
                "member": mm.ravel().astype(np.int8), "q_cusecs": q.ravel().astype(np.float32),
            }))
    df = pd.concat(frames, ignore_index=True)
    df["point_id"] = df["point_id"].astype("category")
    return df


def rain_forecasts(seasons: list[SeasonFlows], rng: np.random.Generator, leads: int = 7,
                   issue_start: str = "06-01", issue_end: str = "10-15") -> pd.DataFrame:
    """ECMWF-like deterministic rain forecasts that lose skill with lead time."""
    frames = []
    for s in seasons:
        issues = pd.date_range(f"{s.year}-{issue_start}", f"{s.year}-{issue_end}", freq="D")
        pos = np.searchsorted(s.dates, issues)
        axis = np.arange(len(s.dates), dtype=np.float64)
        for lead in range(1, leads + 1):
            jitter = rng.normal(0, 0.25 * lead, len(issues))
            noise = rng.lognormal(0, 0.3 + 0.1 * lead, (2, len(issues)))
            frames.append(pd.DataFrame({
                "issue_date": issues, "lead_day": np.int8(lead),
                "rain_upper_mm": (np.interp(pos + lead + jitter, axis, s.rain_upper) * noise[0]).astype(np.float32),
                "rain_local_mm": (np.interp(pos + lead + jitter, axis, s.rain_local) * noise[1]).astype(np.float32),
            }))
    return pd.concat(frames, ignore_index=True)
