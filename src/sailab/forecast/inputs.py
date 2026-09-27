"""Model 2 inputs: what is known at a forecast's issue time, and nothing later.

Timing rules (issue time = 06:00 UTC on the issue day D, i.e. 11 am in Pakistan):
- river flow: FFD's 6 am reading of day D is in (01:00 UTC), so flow up to D;
- rain and soil moisture: IMERG is 4 to 15 hours late, so up to D-1;
- forecasts: GloFAS and ECMWF runs issued on D (00 UTC);
- flood map: every satellite pass up to the issue time; each pixel keeps its latest observed state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from sailab import labels as L
from sailab.cube import Datacube
from sailab.features import norm_dem, norm_flow, norm_hand, norm_rain, norm_slope

ISSUE_HOUR_UTC = 6
HISTORY_DAYS = 30
HORIZON_DAYS = 7
N_TOKENS = HISTORY_DAYS + HORIZON_DAYS
MAX_STATE_AGE_DAYS = 30.0
MAP_CHANNELS = ["flood_state", "state_known", "state_age", "normal_water", "hand", "dem", "slope", "breach",
                "map_hidden"]
WEATHER_FEATURES = ["rain_upper", "rain_local", "soil_moisture"]
STANDARD_HORIZONS = (1, 2, 3, 5, 7)


def issue_time(day: pd.Timestamp | str) -> pd.Timestamp:
    d = pd.Timestamp(day)
    d = d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC")
    return d.normalize() + pd.Timedelta(hours=ISSUE_HOUR_UTC)


# --------------------------------------------------------------------------- river, rain and forecast tokens


class SeriesBank:
    """Daily observations and forecasts as arrays, so a 37-token sequence is a few slices."""

    def __init__(self, cube: Datacube, points: list[str], members: int = 51) -> None:
        self.points = list(points)
        self.members = members
        obs = cube.discharge
        missing_cols = [p for p in self.points if p not in obs.columns]
        for p in missing_cols:
            obs[p] = np.nan
        weather = cube.weather
        gl = cube.glofas
        rf = cube.rain_forecast
        start = min(obs.index.min(), weather.index.min(), gl["issue_date"].min()) - pd.Timedelta(days=HISTORY_DAYS + 1)
        end = max(obs.index.max(), weather.index.max(), gl["issue_date"].max()) + pd.Timedelta(days=HORIZON_DAYS + 1)
        self.dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
        self.q = norm_flow(obs.reindex(self.dates)[self.points].to_numpy(dtype=np.float64))
        self.q[np.isnan(obs.reindex(self.dates)[self.points].to_numpy())] = np.nan
        w = weather.reindex(self.dates)
        self.weather = np.stack([norm_rain(w["rain_upper_mm"].to_numpy()), norm_rain(w["rain_local_mm"].to_numpy()),
                                 w["soil_moisture"].to_numpy(dtype=np.float64)], axis=1).astype(np.float32)
        self.weather[np.isnan(w[["rain_upper_mm", "rain_local_mm", "soil_moisture"]].to_numpy())] = np.nan

        # GloFAS: (day, point, lead, member) in cusecs; NaN where no forecast was issued
        self.gl = np.full((len(self.dates), len(self.points), HORIZON_DAYS, members), np.nan, dtype=np.float32)
        gl = gl[gl["point_id"].isin(self.points) & (gl["lead_day"] <= HORIZON_DAYS) & (gl["member"] < members)]
        di = self.dates.get_indexer(gl["issue_date"].dt.normalize())
        pi = pd.Index(self.points).get_indexer(gl["point_id"].astype(str))
        ok = di >= 0
        self.gl[di[ok], pi[ok], gl["lead_day"].to_numpy()[ok] - 1, gl["member"].to_numpy()[ok]] = gl["q_cusecs"].to_numpy()[ok]
        self.rf = np.full((len(self.dates), HORIZON_DAYS, 2), np.nan, dtype=np.float32)
        rf = rf[rf["lead_day"] <= HORIZON_DAYS]
        ri = self.dates.get_indexer(rf["issue_date"].dt.normalize())
        ok = ri >= 0
        self.rf[ri[ok], rf["lead_day"].to_numpy()[ok] - 1, 0] = norm_rain(rf["rain_upper_mm"].to_numpy()[ok])
        self.rf[ri[ok], rf["lead_day"].to_numpy()[ok] - 1, 1] = norm_rain(rf["rain_local_mm"].to_numpy()[ok])

    @property
    def n_features(self) -> int:
        return len(self.points) + len(WEATHER_FEATURES)

    def day_index(self, day: pd.Timestamp) -> int:
        i = self.dates.get_indexer([pd.Timestamp(day).tz_localize(None).normalize()])[0]
        if i < HISTORY_DAYS:
            raise KeyError(f"{day} is outside the series range")
        return int(i)

    def glofas_raw(self, day: pd.Timestamp) -> np.ndarray:
        """(points, leads, members) GloFAS flow in cusecs for the forecast issued on `day`."""
        return self.gl[self.day_index(day)]

    def tokens(self, day: pd.Timestamp, member: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """values (37, F), missing (37, F), is_forecast (37,). `member=None` uses the ensemble mean."""
        i = self.day_index(day)
        p = len(self.points)
        values = np.full((N_TOKENS, self.n_features), np.nan, dtype=np.float32)
        past = slice(i - HISTORY_DAYS + 1, i + 1)
        values[:HISTORY_DAYS, :p] = self.q[past]
        values[:HISTORY_DAYS, p:] = self.weather[past]
        values[HISTORY_DAYS - 1, p:] = np.nan  # today's rain is not in yet
        fc = self.gl[i]
        with np.errstate(all="ignore"):
            flow = np.nanmean(fc, axis=-1) if member is None else fc[..., member]
        values[HISTORY_DAYS:, :p] = norm_flow(flow.T)
        values[HISTORY_DAYS:, :p][np.isnan(flow.T)] = np.nan
        values[HISTORY_DAYS:, p:p + 2] = self.rf[i]
        missing = np.isnan(values).astype(np.float32)
        is_fc = np.zeros(N_TOKENS, dtype=np.float32)
        is_fc[HISTORY_DAYS:] = 1.0
        return np.nan_to_num(values, nan=0.0), missing, is_fc


# --------------------------------------------------------------------------- flood state from past passes


@dataclass
class FloodState:
    flooded: np.ndarray    # bool, latest observed state
    known: np.ndarray      # bool, observed within the look-back window
    obs_time: np.ndarray   # float64 days since epoch of each pixel's latest observation (NaN if unknown)

    def age_days(self, t: pd.Timestamp) -> np.ndarray:
        now = t.value / 86_400e9
        return np.where(self.known, now - self.obs_time, MAX_STATE_AGE_DAYS).astype(np.float32)


class StateComposer:
    """Latest known flood state per pixel from the passes before a time. By default it reads GFM
    labels; the live twin swaps in Model 1's maps via `override`."""

    def __init__(self, cube: Datacube, lookback_days: float = MAX_STATE_AGE_DAYS, truth: bool = False) -> None:
        self.cube = cube
        self.truth = truth
        self.lookback = pd.Timedelta(days=lookback_days)
        self.scenes = cube.scenes.sort_values("time").reset_index(drop=True)
        self.override: dict[str, np.ndarray] = {}

    def scenes_before(self, t: pd.Timestamp) -> pd.DataFrame:
        s = self.scenes
        return s[(s["time"] <= t) & (s["time"] > t - self.lookback)]

    def latest_index(self, t: pd.Timestamp) -> int | None:
        idx = np.flatnonzero((self.scenes["time"] <= t).to_numpy())
        return int(idx[-1]) if idx.size else None

    @lru_cache(maxsize=256)  # noqa: B019
    def _label(self, scene_id: str) -> np.ndarray:
        if scene_id in self.override:
            return self.override[scene_id]
        return self.cube.read_label(scene_id, truth=self.truth)

    @lru_cache(maxsize=64)  # noqa: B019
    def _state_at(self, last: int) -> FloodState:
        t = self.scenes.loc[last, "time"]
        shape = self.cube.grid.shape
        flooded = np.zeros(shape, dtype=bool)
        known = np.zeros(shape, dtype=bool)
        obs_time = np.full(shape, np.nan)
        for j in range(last, -1, -1):
            row = self.scenes.loc[j]
            if row["time"] <= t - self.lookback:
                break
            label = self._label(row["scene_id"])
            new = (label != L.IGNORE) & ~known
            flooded[new] = label[new] == L.FLOOD
            obs_time[new] = row["time"].value / 86_400e9
            known |= new
            if known.all():
                break
        return FloodState(flooded, known, obs_time)

    def state(self, t: pd.Timestamp) -> FloodState | None:
        last = self.latest_index(t)
        return None if last is None else self._state_at(last)

    def set_override(self, scene_id: str, label: np.ndarray) -> None:
        self.override[scene_id] = label
        self._label.cache_clear()
        self._state_at.cache_clear()


# --------------------------------------------------------------------------- static map channels


class StaticMaps:
    def __init__(self, cube: Datacube) -> None:
        dem = cube.static("dem").astype(np.float32)
        ok = np.isfinite(dem)
        self.dem_stats = (float(dem[ok].mean()), float(dem[ok].std()))
        self.normal_water = cube.static("normal_water").astype(bool)
        self.aoi = cube.static("aoi_mask").astype(bool) if cube.has_static("aoi_mask") else np.ones(cube.grid.shape, bool)
        self.channels = np.stack([
            self.normal_water.astype(np.float32),
            norm_hand(cube.static("hand")),
            norm_dem(dem, *self.dem_stats),
            norm_slope(cube.static("slope")),
        ])
        self.breaches = cube.geojson("breaches")["features"]
        self.grid = cube.grid

    def breach_channel(self, t: pd.Timestamp, radius_km: float = 12.0) -> np.ndarray:
        """1 near breaches that are open at time t (the scenario input); 0 elsewhere."""
        out = np.zeros(self.grid.shape, dtype=np.float32)
        day = t.tz_convert(None).normalize() if t.tzinfo else t.normalize()
        for f in self.breaches:
            p = f["properties"]
            if not (pd.Timestamp(p["start"]) <= day <= pd.Timestamp(p["end"])):
                continue
            lon, lat = f["geometry"]["coordinates"]
            r0, c0 = self.grid.lonlat_to_rowcol(lon, lat)
            rows, cols = np.ogrid[0:self.grid.height, 0:self.grid.width]
            d_km = np.hypot(rows - int(r0), cols - int(c0)) * self.grid.res / 1000.0
            out = np.maximum(out, (d_km <= radius_km).astype(np.float32))
        return out

    def map_stack(self, state: FloodState | None, t: pd.Timestamp, hide_map: bool = False) -> np.ndarray:
        """Model 2 map channels (9, H, W) at issue time t."""
        h, w = self.grid.shape
        if state is None or hide_map:
            flood = np.zeros((h, w), np.float32)
            known = np.zeros((h, w), np.float32)
            age = np.full((h, w), MAX_STATE_AGE_DAYS / 7.0, np.float32)
        else:
            flood = state.flooded.astype(np.float32)
            known = state.known.astype(np.float32)
            age = np.minimum(state.age_days(t), MAX_STATE_AGE_DAYS) / 7.0
        hidden = np.full((h, w), 1.0 if (hide_map or state is None) else 0.0, np.float32)
        return np.concatenate([np.stack([flood, known, age]), self.channels,
                               self.breach_channel(t)[None], hidden[None]]).astype(np.float32)


def target_arrays(label: np.ndarray, static: StaticMaps) -> tuple[np.ndarray, np.ndarray]:
    """(flood 0/1, mask of pixels that count) for a target pass, following the scoring rules."""
    mask = (label != L.IGNORE) & (label != L.NORMAL_WATER) & ~static.normal_water & static.aoi
    return (label == L.FLOOD).astype(np.float32), mask


# --------------------------------------------------------------------------- forecast pairs


def forecast_pairs(cube: Datacube, splits: list[str], max_lead_days: float = 7.5,
                   min_lead_days: float = 0.25) -> pd.DataFrame:
    """Every (issue day, target pass) pair whose target pass lies in the chosen splits' events.

    Targets (labels) only come from those events. Inputs may come from passes a few days before an
    event window opens; those passes belong to no other event, so nothing crosses a split.
    """
    scenes = cube.scenes
    scenes = scenes[scenes["split"].isin(splits)]
    rows = []
    for _, s in scenes.iterrows():
        first = issue_time(s["time"] - pd.Timedelta(days=max_lead_days))
        for k in range(0, int(np.ceil(max_lead_days)) + 2):
            t = first + pd.Timedelta(days=k)
            lead = (s["time"] - t) / pd.Timedelta(days=1)
            if min_lead_days <= lead <= max_lead_days:
                rows.append({"issue": t, "target_scene": s["scene_id"], "target_time": s["time"], "lead_days": lead,
                             "event_id": s["event_id"], "split": s["split"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["issue", "target_time"]).reset_index(drop=True)
