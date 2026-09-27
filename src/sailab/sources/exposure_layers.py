"""Exposure layers: people, buildings and roads on our grid (no accounts needed).

- WorldPop Global2 R2025A 100 m population counts (CC BY 4.0), summed into our pixels.
- Microsoft Global ML Building Footprints (ODbL): building centroids counted per pixel.
- OpenStreetMap roads (ODbL) through the Overpass API, cut into ~2 km pieces so each piece gets its
  own flood chance ("risk down to single roads"). OSM stays a separate layer (team rule 5).
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod
from rasterio.enums import Resampling

from sailab.config import BBox
from sailab.cube import Datacube
from sailab.sources.http import download, http_session
from sailab.sources.raster_remote import read_to_grid

WORLDPOP_DIR = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/PAK/v1/100m/constrained/"
MS_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")


# --------------------------------------------------------------------------- population


def worldpop_url(year: int = 2025) -> str:
    base = WORLDPOP_DIR.format(year=year)
    r = http_session().get(base, timeout=60)
    r.raise_for_status()
    names = re.findall(r'href="([^"]+\.tif)"', r.text)
    if not names:
        raise FileNotFoundError(f"no GeoTIFF listed at {base}")
    return base + names[0]


def population(cube: Datacube, year: int = 2025, cache: Path | None = None) -> np.ndarray:
    """People per pixel. The ~100 m counts are summed into our pixels (not averaged)."""
    url = worldpop_url(year)
    src = str(download(url, cache / url.rsplit("/", 1)[-1])) if cache else url
    pop = read_to_grid(src, cube.grid, Resampling.sum)
    return np.nan_to_num(pop, nan=0.0).astype(np.float32)


# --------------------------------------------------------------------------- buildings


def _quadkeys(bbox: BBox, zoom: int = 9) -> set[str]:
    def tile(lon: float, lat: float) -> tuple[int, int]:
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y

    x0, y1 = tile(bbox.west, bbox.south)
    x1, y0 = tile(bbox.east, bbox.north)
    keys = set()
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            key = ""
            for z in range(zoom, 0, -1):
                digit = 0
                mask = 1 << (z - 1)
                if x & mask:
                    digit += 1
                if y & mask:
                    digit += 2
                key += str(digit)
            keys.add(key)
    return keys


def building_counts(cube: Datacube, bbox: BBox, country: str = "Pakistan", log=print) -> np.ndarray:
    """Buildings per pixel from Microsoft's footprints (each line of a file is one GeoJSON feature)."""
    s = http_session()
    links = pd.read_csv(io.StringIO(s.get(MS_LINKS, timeout=120).text))
    links = links[(links["Location"] == country) & links["QuadKey"].astype(str).str.zfill(9).isin(_quadkeys(bbox))]
    counts = np.zeros(cube.grid.shape, dtype=np.float32)
    for url in links["Url"]:
        raw = gzip.decompress(s.get(url, timeout=300).content).decode("utf-8")
        lons, lats = [], []
        for line in raw.splitlines():
            ring = json.loads(line)["geometry"]["coordinates"][0]
            xy = np.asarray(ring)
            lons.append(xy[:, 0].mean())
            lats.append(xy[:, 1].mean())
        rows, cols = cube.grid.lonlat_to_rowcol(np.asarray(lons), np.asarray(lats))
        ok = (rows >= 0) & (rows < cube.grid.height) & (cols >= 0) & (cols < cube.grid.width)
        np.add.at(counts, (rows[ok], cols[ok]), 1.0)
        log(f"  buildings: {ok.sum():,} from {url.rsplit('/', 1)[-1]}")
    return counts


# --------------------------------------------------------------------------- roads


def overpass_roads(bbox: BBox, classes: tuple[str, ...] = ROAD_CLASSES) -> list[dict[str, Any]]:
    query = (f'[out:json][timeout:300];way["highway"~"^({"|".join(classes)})$"]'
             f"({bbox.south},{bbox.west},{bbox.north},{bbox.east});out geom tags;")
    r = http_session().post(OVERPASS, data={"data": query}, timeout=400)
    r.raise_for_status()
    return r.json().get("elements", [])


def road_pieces(elements: list[dict[str, Any]], piece_km: float = 2.0) -> list[dict[str, Any]]:
    """OSM ways cut into ~2 km GeoJSON pieces with stable ids."""
    geod = Geod(ellps="WGS84")
    feats = []
    for el in elements:
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        lon = np.array([c[0] for c in coords])
        lat = np.array([c[1] for c in coords])
        seg = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2] / 1000.0
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(1, int(round(cum[-1] / piece_km)))
        cuts = np.searchsorted(cum, np.linspace(0, cum[-1], n + 1)[1:-1])
        for k, part in enumerate(np.split(np.arange(len(coords)), cuts)):
            idx = part if k == 0 else np.concatenate([[part[0] - 1], part])
            if idx.size < 2:
                continue
            feats.append({"type": "Feature", "properties": {
                "id": f"osm-{el['id']}-{k}", "name": tags.get("name") or tags.get("ref") or tags["highway"],
                "class": tags["highway"], "length_km": round(float(cum[idx[-1]] - cum[idx[0]]), 2)},
                "geometry": {"type": "LineString", "coordinates": [[round(lon[i], 6), round(lat[i], 6)] for i in idx]}})
    return feats


def road_km(cube: Datacube, features: list[dict[str, Any]]) -> np.ndarray:
    """Road length per pixel, from points sampled every quarter pixel along each piece."""
    out = np.zeros(cube.grid.shape, dtype=np.float32)
    geod = Geod(ellps="WGS84")
    for f in features:
        xy = np.asarray(f["geometry"]["coordinates"])
        for (lo0, la0), (lo1, la1) in zip(xy[:-1], xy[1:], strict=True):
            length_km = geod.inv(lo0, la0, lo1, la1)[2] / 1000.0
            n = max(1, int(length_km * 1000 / (cube.grid.res / 4)))
            t = (np.arange(n) + 0.5) / n
            rows, cols = cube.grid.lonlat_to_rowcol(lo0 + t * (lo1 - lo0), la0 + t * (la1 - la0))
            ok = (rows >= 0) & (rows < cube.grid.height) & (cols >= 0) & (cols < cube.grid.width)
            np.add.at(out, (rows[ok], cols[ok]), length_km / n)
    return out


def ingest_exposure(cube: Datacube, bbox: BBox, buildings: bool = True, log=print) -> None:
    log("population (WorldPop R2025A)...")
    cube.write_static("population", population(cube), source="WorldPop Global2 R2025A")
    if buildings:
        log("buildings (Microsoft footprints)...")
        cube.write_static("buildings", building_counts(cube, bbox, log=log), source="Microsoft Global ML Building Footprints")
    log("roads (OpenStreetMap via Overpass)...")
    feats = road_pieces(overpass_roads(bbox))
    cube.write_geojson("roads", feats)
    cube.write_static("road_km", road_km(cube, feats), source="OpenStreetMap contributors (ODbL)")
    log(f"exposure layers written ({len(feats):,} road pieces)")
