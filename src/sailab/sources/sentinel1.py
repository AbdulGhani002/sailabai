"""Sentinel-1 radar images for Model 1.

Search needs no account (ASF). Terrain-corrected images come from either:
- ASF HyP3 RTC jobs (8,000 free credits a month, about 1,600 scenes; NASA Earthdata login), or
- Google Earth Engine's COPERNICUS/S1_GRD collection (already in dB; one shared team project).

Scene ids match the GFM labels: S1_<first acquisition time>_<A|D><relative orbit>, so each image
lines up with its label. Keep only the clipped study-area rasters; never keep raw scenes (~1 GB).

The HyP3 and Earth Engine paths are untested until the team's accounts exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from sailab.cube import Datacube
from sailab.sources.raster_remote import read_to_grid


def search(start: str, end: str, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    import asf_search as asf

    w, s, e, n = bbox
    wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], processingLevel=asf.PRODUCT_TYPE.GRD_HD,
                             beamMode=asf.BEAMMODE.IW, intersectsWith=wkt, start=start, end=end)
    rows = []
    for r in results:
        p = r.properties
        rows.append({"granule": p["sceneName"], "time": pd.Timestamp(p["startTime"]),
                     "orbit": f"{'A' if p['flightDirection'].upper().startswith('A') else 'D'}{int(p['pathNumber']):03d}",
                     "platform": p.get("platform", "Sentinel-1")})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def match_to_scenes(s1: pd.DataFrame, scenes: pd.DataFrame, max_minutes: float = 10.0) -> pd.DataFrame:
    """Attach each granule to the labelled pass (same orbit, within a few minutes)."""
    out = []
    for _, g in s1.iterrows():
        cand = scenes[(scenes["orbit"] == g["orbit"])]
        if cand.empty:
            continue
        dt = (cand["time"] - g["time"]).abs()
        best = dt.idxmin()
        if dt[best] <= pd.Timedelta(minutes=max_minutes):
            out.append({**g.to_dict(), "scene_id": cand.loc[best, "scene_id"]})
    return pd.DataFrame(out)


def submit_rtc(granules: list[str], name: str = "sailab"):
    """Submit HyP3 RTC jobs (20 m, gamma0, decibel). Credentials from ~/.netrc (urs.earthdata.nasa.gov)."""
    import hyp3_sdk

    hyp3 = hyp3_sdk.HyP3()
    jobs = hyp3_sdk.Batch()
    for g in granules:
        jobs += hyp3.submit_rtc_job(g, name=name, resolution=20, radiometry="gamma0", scale="decibel",
                                    include_dem=False, include_inc_map=False, dem_matching=False)
    return jobs


def write_scene_from_rtc(cube: Datacube, scene_id: str, vv_paths: list[Path], vh_paths: list[Path]) -> None:
    """Mosaic the RTC frames of one pass onto the grid and store them in the cube."""
    def mosaic(paths: list[Path]) -> np.ndarray:
        out = np.full(cube.grid.shape, np.nan, dtype=np.float32)
        for p in paths:
            part = read_to_grid(str(p), cube.grid, Resampling.average)
            take = np.isnan(out) & np.isfinite(part)
            out[take] = part[take]
        return out

    cube.write_scene_sar(scene_id, np.stack([mosaic(vv_paths), mosaic(vh_paths)]))


def earth_engine_collection(start: str, end: str, bbox: tuple[float, float, float, float], project: str):
    """Sentinel-1 IW GRD images with VV and VH over the box, in dB, from Earth Engine."""
    import ee

    ee.Initialize(project=project)
    region = ee.Geometry.Rectangle(list(bbox))
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))
