"""Build a complete synthetic datacube: terrain, 2014-2026 river seasons, forecasts, radar passes and labels.

    sailab synthetic generate --name demo

The result has exactly the layout the real-data pipeline writes (see sailab.cube), so training,
evaluation, the twin loop and the dashboard can all be built and tested before real data arrives.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from sailab.config import load_aoi, load_events, load_gauges
from sailab.cube import Datacube, point_features
from sailab.events import EventRegistry
from sailab.features import soil_moisture_index
from sailab.grid import GridSpec
from sailab.paths import cubes_dir
from sailab.synthetic.hydrology import (
    event_targets,
    glofas_forecasts,
    observed_discharge,
    rain_forecasts,
    simulate_season,
    weather_table,
)
from sailab.synthetic.inundation import river_pixel_flows, simulate_inundation
from sailab.synthetic.sar import gfm_like_label, pass_schedule, simulate_sar, truth_label
from sailab.synthetic.terrain import build_terrain, river_features

DATA_VERSION = "synthetic-1"
FIRST_SAR_YEAR = 2015  # the 2014 flood is before our Sentinel-1 archive: river data only


def _versions(year: int) -> dict[str, str]:
    # Mirror the real version changes so the versioning code gets exercised: GFM changed in 2025.
    return {"sentinel1": "synthetic", "gfm": "V0M2R2" if year >= 2025 else "V0M2R1", "imerg": "V07",
            "glofas": "4.5", "ecmwf": "open-0p25", "dem": "glo30-2021", "generator": DATA_VERSION}


def generate_synthetic_cube(name: str = "demo", resolution_m: float | None = None,
                            years: Iterable[int] = range(2014, 2027), seed: int = 7,
                            root: Path | None = None, log: Callable[[str], None] = print) -> Datacube:
    aoi = load_aoi()
    gauges = load_gauges()
    events = load_events()
    registry = EventRegistry(list(events.events))
    years = list(years)
    grid = GridSpec.for_aoi(aoi, resolution_m or aoi.grid.demo_resolution_m)
    rng = np.random.default_rng(seed)
    root = root or cubes_dir() / name
    log(f"grid {grid.width} x {grid.height} px at {grid.res:.0f} m, {len(years)} seasons -> {root}")

    terrain = build_terrain(grid, aoi, rng)
    cube = Datacube.create(root, grid, kind="synthetic", name=name, seed=seed, years=years,
                           data_version=DATA_VERSION,
                           note="Synthetic demo world. Not real data; do not use for any decision.")

    aoi_mask = grid.aoi_mask(aoi.effective_bbox)
    for layer, arr, kw in [
        ("dem", terrain.dem, {}), ("hand", terrain.hand, {}), ("slope", terrain.slope, {}),
        ("normal_water", terrain.normal_water.astype(np.uint8), {"dtype": "uint8"}),
        ("landcover", terrain.landcover, {"dtype": "uint8"}),
        ("population", terrain.population, {}), ("buildings", terrain.buildings, {}),
        ("road_km", terrain.road_km, {}), ("aoi_mask", aoi_mask.astype(np.uint8), {"dtype": "uint8"}),
        ("beyond_bund", terrain.beyond_bund.astype(np.uint8), {"dtype": "uint8"}),
    ]:
        cube.write_static(layer, arr, **kw)
    cube.write_geojson("roads", terrain.roads)
    cube.write_geojson("rivers", river_features(terrain))
    cube.write_geojson("places", point_features(aoi.places, "place"))
    cube.write_geojson("gauges", point_features(gauges.all(), "gauge"))
    log("terrain and static layers written")

    targets = event_targets(events)
    seasons = [simulate_season(y, rng, targets.get(y)) for y in years]
    for s in seasons:
        peaks = ", ".join(f"{g} {s.q[g].max() / 1e3:.0f}k" for g in ("marala", "qadirabad", "trimmu", "panjnad"))
        log(f"  {s.year}: {peaks}; breaches: {[b['id'] for b in s.breaches] or 'none'}")

    cube.write_table("series", "discharge", observed_discharge(seasons, gauges, rng, DATA_VERSION))
    cube.write_table("series", "weather", weather_table(seasons))
    truth_q = pd.concat([s.q.rename_axis("date").reset_index().melt(id_vars="date", var_name="point_id",
                                                                  value_name="q_cusecs") for s in seasons])
    cube.write_table("series", "truth_discharge", truth_q.astype({"q_cusecs": np.float32}))
    cube.write_table("forecasts", "glofas", glofas_forecasts(seasons, rng))
    cube.write_table("forecasts", "rain", rain_forecasts(seasons, rng))
    breach_feats = [{"type": "Feature", "properties": {k: v for k, v in b.items() if k not in ("lat", "lon")} | {"year": s.year},
                     "geometry": {"type": "Point", "coordinates": [b["lon"], b["lat"]]}}
                    for s in seasons for b in s.breaches]
    cube.write_geojson("breaches", breach_feats)
    log("river series and forecasts written")

    scene_rows = []
    area_rows = []
    for s in seasons:
        if s.year < FIRST_SAR_YEAR:
            continue
        passes = pass_schedule(s.year, grid.shape, rng)
        day_of = {id(p): int((p.time.tz_convert(None).normalize() - s.dates[0]).days) for p in passes}
        rp_q = river_pixel_flows(terrain, s, gauges)
        states, area = simulate_inundation(terrain, s, rp_q, set(day_of.values()))
        area_rows.append(pd.DataFrame({"date": s.dates, "flooded_km2": area}))
        soil = soil_moisture_index(s.rain_local)
        for orbit in sorted({p.orbit for p in passes}):
            dry_day = pd.Timestamp(f"{s.year}-06-05", tz="UTC")
            cube.write_pre_sar(f"{s.year}_{orbit}", simulate_sar(terrain, dry_day, orbit, rng, soil_moisture=0.05))
        for p in passes:
            d = day_of[id(p)]
            st = states[d]
            wind = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.25 else 0.0
            sar = simulate_sar(terrain, p.time, p.orbit, rng, st.flooded, st.depth, p.coverage,
                               soil_moisture=float(soil[d]), wind=wind)
            scene_id = f"S1_{p.time.strftime('%Y%m%dT%H%M%S')}_{p.orbit}"
            cube.write_scene_sar(scene_id, sar)
            cube.write_label(scene_id, gfm_like_label(terrain, st.flooded, sar, p.time, rng, p.coverage),
                             source="gfm-like")
            cube.write_label(scene_id, truth_label(terrain, st.flooded, p.coverage), truth=True, source="truth")
            event = registry.event_for(p.time)
            scene_rows.append({
                "scene_id": scene_id, "time": p.time, "platform": p.platform, "orbit": p.orbit,
                "pre_key": f"{s.year}_{p.orbit}", "event_id": event.id if event else None,
                "split": event.split.value if event else None,
                "coverage": round(float(p.coverage.mean()), 3), "wind": round(wind, 2),
                "flooded_km2": round(float(st.flooded.sum() * grid.pixel_area_km2), 1),
                "versions": _versions(s.year),
            })
        log(f"  {s.year}: {len(passes)} passes, peak flooded area {area.max():.0f} km2")

    cube.write_scenes(pd.DataFrame(scene_rows))
    cube.write_table("series", "truth_flood_area", pd.concat(area_rows, ignore_index=True))
    cube.update_meta(n_scenes=len(scene_rows), versions=_versions(max(years)))
    log(f"done: {len(scene_rows)} scenes")
    return cube
