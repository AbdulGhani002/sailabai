"""Copernicus GFM flood maps as training labels (no account needed).

GFM publishes a 20 m flood map for every Sentinel-1 image since 2015, through the EODC STAC API.
One Sentinel-1 pass over our area spans several Equi7 tiles and frames; we group them into passes,
mosaic the ensemble flood extent, reference water and exclusion masks onto our grid, and write
one label per pass: land, normal water, flood, or ignore (excluded or not imaged).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from sailab import labels as L
from sailab.cube import Datacube
from sailab.events import EventRegistry
from sailab.grid import GridSpec
from sailab.sources.raster_remote import mosaic_to_grid
from sailab.versioning import gfm_version_from_filename

STAC_URL = "https://stac.eodc.eu/api/v1"
COLLECTION = "GFM"
_ORBIT = re.compile(r"_(?P<dir>[AD])(?P<orbit>\d{3})_")


@dataclass
class GfmPass:
    time: pd.Timestamp
    orbit: str
    items: list[dict]

    @property
    def scene_id(self) -> str:
        return f"S1_{self.time.strftime('%Y%m%dT%H%M%S')}_{self.orbit}"

    def hrefs(self, asset: str) -> list[str]:
        return [it["assets"][asset]["href"] for it in self.items if asset in it["assets"]]

    @property
    def version(self) -> str | None:
        for href in self.hrefs("advisory_flags"):
            v = gfm_version_from_filename(href.rsplit("/", 1)[-1])
            if v:
                return v
        return None


def search(bbox: tuple[float, float, float, float], start: str, end: str, max_items: int | None = None) -> list[dict]:
    from pystac_client import Client

    client = Client.open(STAC_URL)
    result = client.search(collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}", max_items=max_items)
    return [item.to_dict() for item in result.items()]


def orbit_of(item: dict) -> str:
    for asset in ("advisory_flags",):
        href = item["assets"].get(asset, {}).get("href", "")
        m = _ORBIT.search(href.rsplit("/", 1)[-1])
        if m:
            return f"{m.group('dir')}{m.group('orbit')}"
    return "X000"


def group_passes(items: list[dict], gap_minutes: float = 10.0) -> list[GfmPass]:
    """Items from the same orbit within a few minutes belong to one satellite pass."""
    rows = sorted(((pd.Timestamp(it["properties"]["datetime"]), orbit_of(it), it) for it in items),
                  key=lambda r: (r[1], r[0]))
    passes: list[GfmPass] = []
    for t, orbit, it in rows:
        last = passes[-1] if passes else None
        if last and last.orbit == orbit and (t - last.items[-1]["_t"]).total_seconds() <= gap_minutes * 60:
            it["_t"] = t
            last.items.append(it)
            last.time = min(last.time, t)
        else:
            it["_t"] = t
            passes.append(GfmPass(t, orbit, [it]))
    for p in passes:
        for it in p.items:
            it.pop("_t", None)
    return sorted(passes, key=lambda p: p.time)


def pass_label(p: GfmPass, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """(label, likelihood %) on our grid for one pass."""
    flood = mosaic_to_grid(p.hrefs("ensemble_flood_extent"), grid, Resampling.mode, dtype="uint8", nodata=255)
    ref = mosaic_to_grid(p.hrefs("reference_water_mask"), grid, Resampling.mode, dtype="uint8", nodata=255)
    excl = mosaic_to_grid(p.hrefs("exclusion_mask"), grid, Resampling.mode, dtype="uint8", nodata=255)
    likelihood = mosaic_to_grid(p.hrefs("ensemble_likelihood"), grid, Resampling.average, dtype="uint8", nodata=255)
    imaged = flood != 255
    normal = ref == 1
    invalid = ~imaged | (excl == 1)
    label = L.compose((flood == 1) | normal, normal, invalid=invalid & ~normal)
    return label, likelihood


def ingest(cube: Datacube, start: str, end: str, bbox: tuple[float, float, float, float] | None = None,
           registry: EventRegistry | None = None, log=print) -> pd.DataFrame:
    """Add GFM labels for every pass between start and end to a (real) datacube."""
    bbox = bbox or cube.grid.bounds_lonlat()
    registry = registry or EventRegistry.load()
    items = search(bbox, start, end)
    passes = group_passes(items)
    log(f"GFM: {len(items)} items -> {len(passes)} passes between {start} and {end}")
    rows = []
    for p in passes:
        label, likelihood = pass_label(p, cube.grid)
        known = label != L.IGNORE
        if known.mean() < 0.02:
            continue
        cube.write_label(p.scene_id, label, source="gfm", gfm_version=p.version or "unknown",
                         items=",".join(it["id"] for it in p.items))
        event = registry.event_for(p.time)
        rows.append({"scene_id": p.scene_id, "time": p.time, "platform": "S1", "orbit": p.orbit, "pre_key": None,
                     "event_id": event.id if event else None, "split": event.split.value if event else None,
                     "coverage": round(float(known.mean()), 3),
                     "flooded_km2": round(float((label == L.FLOOD).sum() * cube.grid.pixel_area_km2), 1),
                     "versions": {"gfm": p.version or "unknown"}})
        log(f"  {p.scene_id}: {rows[-1]['coverage']:.0%} imaged, {rows[-1]['flooded_km2']:.0f} km2 flood")
    new = pd.DataFrame(rows)
    old = cube.scenes
    merged = pd.concat([old[~old["scene_id"].isin(new.get("scene_id", []))], new], ignore_index=True) if len(old) else new
    if len(merged):
        cube.write_scenes(merged)
    return new

