"""Daily flood extent in the synthetic world.

Water level at each river pixel comes from its reach's flow through a simple rating curve. A pixel
floods when that level is above its HAND. Behind the flood bunds it floods only when the bund is
overtopped or breached, and it drains slowly, which is why floods linger for weeks. Heavy local
rain ponds in old channels, a kind of flooding gauges cannot predict, as in 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from sailab.config import GaugesConfig
from sailab.synthetic.hydrology import SeasonFlows
from sailab.synthetic.terrain import River, Terrain

CELERITY_KM_PER_DAY = 65.0
FLOOD_DEPTH_M = 0.10


@dataclass
class DayState:
    flooded: np.ndarray  # bool
    depth: np.ndarray    # float16 metres


def _chainage_at(river: River, crs: str, lat: float, lon: float) -> float:
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    i = int(np.argmin(np.hypot(river.xy[:, 0] - x, river.xy[:, 1] - y)))
    return float(river.chainage_km[i])


def _intervals(terrain: Terrain, gauges: GaugesConfig) -> dict[str, list[tuple[float, float, str, float]]]:
    """For each river: (from km, to km, flow series, km where that series is measured)."""
    crs = terrain.grid.crs
    g = gauges.by_id()
    by_name = {r.name: r for r in terrain.rivers}
    out: dict[str, list[tuple[float, float, str, float]]] = {}
    if "chenab" in by_name:
        ch = by_name["chenab"]
        c_t = _chainage_at(ch, crs, g["trimmu"].lat, g["trimmu"].lon)
        c_p = _chainage_at(ch, crs, g["panjnad"].lat, g["panjnad"].lon)
        c_r = c_t + 0.6 * (c_p - c_t)
        if "ravi" in by_name:
            end = by_name["ravi"].xy[-1]
            c_r = float(ch.chainage_km[int(np.argmin(np.hypot(*(ch.xy - end).T)))])
        out["chenab"] = [(-np.inf, c_t, "chenab_at_trimmu", c_t), (c_t, c_r, "trimmu", c_t),
                         (c_r, c_p, "chenab_below_ravi", c_r), (c_p, np.inf, "panjnad", c_p)]
    if "jhelum" in by_name:
        out["jhelum"] = [(-np.inf, np.inf, "jhelum_in", 0.0)]
    if "ravi" in by_name:
        out["ravi"] = [(-np.inf, np.inf, "sidhnai", _chainage_at(by_name["ravi"], crs, g["sidhnai"].lat, g["sidhnai"].lon))]
    if "sutlej" in by_name:
        out["sutlej"] = [(-np.inf, np.inf, "islam", _chainage_at(by_name["sutlej"], crs, g["islam"].lat, g["islam"].lon))]
    return out


def river_pixel_flows(terrain: Terrain, season: SeasonFlows, gauges: GaugesConfig) -> np.ndarray:
    """(n_river_pixels, n_days) flow in cusecs, delayed by distance from the reach's gauge."""
    n = len(season.dates)
    day_axis = np.arange(n, dtype=np.float64)
    celerity = CELERITY_KM_PER_DAY / season.slowdown
    out = np.zeros((len(terrain.rp_river), n), dtype=np.float32)
    intervals = _intervals(terrain, gauges)
    for ri, river in enumerate(terrain.rivers):
        sel = np.flatnonzero(terrain.rp_river == ri)
        chain = terrain.rp_chainage[sel]
        for lo, hi, series, anchor in intervals.get(river.name, []):
            m = (chain >= lo) & (chain < hi)
            if not m.any():
                continue
            q = season.q[series].to_numpy()
            lag = (chain[m] - anchor) / celerity
            out[sel[m]] = np.interp(day_axis[None, :] - lag[:, None], day_axis, q).astype(np.float32)
    return out


def stage_from_flow(terrain: Terrain, rp_q: np.ndarray) -> np.ndarray:
    """Water level above normal (m) at each river pixel from its flow."""
    q_ref = np.array([r.q_ref for r in terrain.rivers])[terrain.rp_river][:, None]
    d_ref = np.array([r.d_ref for r in terrain.rivers])[terrain.rp_river][:, None]
    return (d_ref * ((np.maximum(rp_q, 1.0) / q_ref) ** 0.6 - 1.0)).astype(np.float32)


def breach_masks(terrain: Terrain, season: SeasonFlows, radius_km: float = 12.0) -> list[tuple[int, int, np.ndarray]]:
    """(first day, last day, pixels the breach can reach) for each breach that opened this season."""
    out = []
    names = [r.name for r in terrain.rivers]
    rows, cols = np.mgrid[0:terrain.grid.height, 0:terrain.grid.width]
    for b in season.breaches:
        r0, c0 = terrain.grid.lonlat_to_rowcol(b["lon"], b["lat"])
        d_km = np.hypot(rows - int(r0), cols - int(c0)) * terrain.grid.res / 1000.0
        same_river = terrain.rp_river[terrain.nearest] == names.index(b["river"]) if b["river"] in names else True
        mask = (d_km <= radius_km) & terrain.beyond_bund & same_river
        start = int(np.searchsorted(season.dates, np.datetime64(b["start"])))
        end = int(np.searchsorted(season.dates, np.datetime64(b["end"])))
        out.append((start, end, mask))
    return out


def simulate_inundation(terrain: Terrain, season: SeasonFlows, rp_q: np.ndarray,
                        keep_days: set[int]) -> tuple[dict[int, DayState], np.ndarray]:
    """Run the season day by day. Returns the states for `keep_days` and flooded km2 for every day."""
    stage_rp = stage_from_flow(terrain, rp_q)
    hand = terrain.hand
    drain = np.where(terrain.beyond_bund, 0.07, 0.25).astype(np.float32)
    drain[terrain.depression] = 0.04
    pond_gain = np.where(terrain.depression, 9.0, np.where(terrain.beyond_bund & (hand < 2.5), 1.5, 0.0))
    breaches = breach_masks(terrain, season)
    depth = np.zeros(terrain.grid.shape, dtype=np.float32)
    states: dict[int, DayState] = {}
    area = np.zeros(len(season.dates), dtype=np.float32)
    for d in range(len(season.dates)):
        stage = stage_rp[:, d][terrain.nearest]
        potential = stage - hand
        blocked = terrain.beyond_bund & (stage < terrain.bund_crest)
        potential = np.where(blocked, -np.inf, potential)
        for start, end, mask in breaches:
            if start <= d <= end:
                potential = np.where(mask, np.maximum(potential, stage - 1.0 - hand), potential)
        excess_m = max(0.0, float(season.rain_local[d]) - 35.0) / 1000.0
        depth = np.maximum(np.maximum(potential, depth - drain), 0.0) + excess_m * pond_gain
        flooded = (depth > FLOOD_DEPTH_M) & ~terrain.normal_water
        area[d] = flooded.sum() * terrain.grid.pixel_area_km2
        if d in keep_days:
            states[d] = DayState(flooded, depth.astype(np.float16))
    return states, area
