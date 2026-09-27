"""Terrain, water and land-cover layers from open cloud buckets (no account needed).

- Copernicus DEM GLO-30 and ASF GLO-30 HAND: AWS open data, 1 x 1 degree tiles.
- JRC Global Surface Water occurrence: 10 x 10 degree tiles; normal water = wet >= 75% of the time.
- ESA WorldCover 2021: 3 x 3 degree tiles.

    sailab data static --cube real-2025
"""

from __future__ import annotations

import math

import numpy as np
from rasterio.enums import Resampling

from sailab.config import BBox, load_aoi
from sailab.cube import Datacube
from sailab.sources.raster_remote import mosaic_to_grid

DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
HAND_URL = "https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_HAND.tif"
# v1.4 (1984-2021) is the latest with public per-tile downloads at this path; switch when v1.5 tiles are posted.
GSW_URL = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{lon}{ew}_{lat}{ns}v1_4_2021.tif"
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif")
NORMAL_WATER_OCCURRENCE = 75.0


def _tiles(bbox: BBox, size: int, anchor: str = "lower") -> list[tuple[int, int]]:
    """(lat, lon) corners of `size`-degree tiles covering the box. anchor 'upper' gives the top-left latitude."""
    lats = range(math.floor(bbox.south / size) * size, math.ceil(bbox.north / size) * size, size)
    lons = range(math.floor(bbox.west / size) * size, math.ceil(bbox.east / size) * size, size)
    return [((lat + size) if anchor == "upper" else lat, lon) for lat in lats for lon in lons]


def _fmt(url: str, lat: int, lon: int) -> str:
    return url.format(ns="N" if lat >= 0 else "S", ew="E" if lon >= 0 else "W", lat=abs(lat), lon=abs(lon))


def dem_urls(bbox: BBox) -> list[str]:
    return [_fmt(DEM_URL, lat, lon) for lat, lon in _tiles(bbox, 1)]


def hand_urls(bbox: BBox) -> list[str]:
    return [_fmt(HAND_URL, lat, lon) for lat, lon in _tiles(bbox, 1)]


def gsw_urls(bbox: BBox) -> list[str]:
    return [GSW_URL.format(lon=abs(lon), ew="E" if lon >= 0 else "W", lat=abs(lat), ns="N" if lat >= 0 else "S")
            for lat, lon in _tiles(bbox, 10, anchor="upper")]


def worldcover_urls(bbox: BBox) -> list[str]:
    return [_fmt(WORLDCOVER_URL, lat, lon) for lat, lon in _tiles(bbox, 3)]


def slope_degrees(dem: np.ndarray, res: float) -> np.ndarray:
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    gy, gx = np.gradient(filled, res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def ingest_static(cube: Datacube, bbox: BBox | None = None, log=print) -> None:
    bbox = bbox or load_aoi().effective_bbox
    grid = cube.grid
    avg = Resampling.average
    log("DEM (Copernicus GLO-30)...")
    dem = mosaic_to_grid(dem_urls(bbox), grid, avg, log=log)
    cube.write_static("dem", dem, source="Copernicus DEM GLO-30")
    cube.write_static("slope", slope_degrees(dem, grid.res))
    log("HAND (ASF GLO-30 HAND)...")
    cube.write_static("hand", mosaic_to_grid(hand_urls(bbox), grid, avg, log=log), source="ASF GLO-30 HAND v1")
    log("surface water occurrence (JRC GSW)...")
    occ = mosaic_to_grid(gsw_urls(bbox), grid, avg, log=log)
    cube.write_static("water_occurrence", occ, source="JRC GSW occurrence v1.4")
    cube.write_static("normal_water", (np.nan_to_num(occ) >= NORMAL_WATER_OCCURRENCE).astype(np.uint8), dtype="uint8")
    log("land cover (ESA WorldCover 2021)...")
    lc = mosaic_to_grid(worldcover_urls(bbox), grid, Resampling.mode, dtype="uint8", nodata=0, log=log)
    cube.write_static("landcover", lc, dtype="uint8", source="ESA WorldCover 2021 v200")
    cube.write_static("aoi_mask", grid.aoi_mask(bbox).astype(np.uint8), dtype="uint8")
    log("static layers written")
