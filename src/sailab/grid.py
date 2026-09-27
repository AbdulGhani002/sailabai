"""The analysis grid: one fixed raster grid that every layer is resampled onto.

Production uses 20 m pixels in UTM 42N (about 8,650 x 12,200 pixels for the study area); the
synthetic demo uses the same box at a coarser resolution so it runs in minutes on a laptop.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window

from sailab.config import AOIConfig, BBox


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: Affine
    width: int
    height: int

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_bbox(cls, bbox: BBox | tuple[float, float, float, float], crs: str, resolution: float,
                  densify: int = 64) -> GridSpec:
        """Smallest grid, snapped to whole pixels, that covers a lon/lat box in `crs`."""
        west, south, east, north = bbox.as_tuple() if isinstance(bbox, BBox) else bbox
        to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        edge = np.linspace(0.0, 1.0, densify)
        lons = np.concatenate([west + (east - west) * edge, np.full(densify, east),
                               east - (east - west) * edge, np.full(densify, west)])
        lats = np.concatenate([np.full(densify, south), south + (north - south) * edge,
                               np.full(densify, north), north - (north - south) * edge])
        xs, ys = to_crs.transform(lons, lats)
        left = math.floor(min(xs) / resolution) * resolution
        right = math.ceil(max(xs) / resolution) * resolution
        bottom = math.floor(min(ys) / resolution) * resolution
        top = math.ceil(max(ys) / resolution) * resolution
        return cls(
            crs=crs,
            transform=Affine(resolution, 0.0, left, 0.0, -resolution, top),
            width=round((right - left) / resolution),
            height=round((top - bottom) / resolution),
        )

    @classmethod
    def for_aoi(cls, aoi: AOIConfig, resolution: float | None = None) -> GridSpec:
        return cls.from_bbox(aoi.effective_bbox, aoi.grid.crs, resolution or aoi.grid.resolution_m)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        return cls(crs=d["crs"], transform=Affine(*d["transform"][:6]), width=int(d["width"]),
                   height=int(d["height"]))

    def to_dict(self) -> dict[str, Any]:
        return {"crs": self.crs, "transform": list(self.transform)[:6], "width": self.width,
                "height": self.height}

    # ------------------------------------------------------------------ geometry

    @property
    def res(self) -> float:
        return float(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return (left, top - self.height * self.res, left + self.width * self.res, top)

    @property
    def pixel_area_km2(self) -> float:
        return self.res * self.res / 1e6

    def xy(self, rows: np.ndarray | float, cols: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Map coordinates of pixel centres."""
        left, _, _, top = self.bounds
        return (left + (np.asarray(cols) + 0.5) * self.res, top - (np.asarray(rows) + 0.5) * self.res)

    def rowcol(self, xs: np.ndarray | float, ys: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        left, _, _, top = self.bounds
        cols = np.floor((np.asarray(xs) - left) / self.res).astype(int)
        rows = np.floor((top - np.asarray(ys)) / self.res).astype(int)
        return rows, cols

    def lonlat_to_rowcol(self, lons: np.ndarray | float, lats: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform(lons, lats)
        return self.rowcol(xs, ys)

    def rowcol_to_lonlat(self, rows: np.ndarray | float, cols: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = self.xy(rows, cols)
        lons, lats = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        return np.asarray(lons), np.asarray(lats)

    def lonlat_grids(self, window: Window | None = None, max_pixels: int = 20_000_000) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and latitude of every pixel centre (optionally inside a window)."""
        w = window or Window(0, 0, self.width, self.height)
        if w.width * w.height > max_pixels:
            raise MemoryError(f"window of {w.width}x{w.height} pixels is too big; read it in chunks")
        rows, cols = np.mgrid[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
        return self.rowcol_to_lonlat(rows, cols)

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        left, bottom, right, top = self.bounds
        xs = np.array([left, right, right, left])
        ys = np.array([bottom, bottom, top, top])
        lons, lats = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))

    def aoi_mask(self, bbox: BBox, window: Window | None = None) -> np.ndarray:
        """True for pixels whose centre lies inside the lon/lat study box."""
        lons, lats = self.lonlat_grids(window)
        return (lons >= bbox.west) & (lons <= bbox.east) & (lats >= bbox.south) & (lats <= bbox.north)

    def coarsen(self, factor: int) -> GridSpec:
        return GridSpec(
            crs=self.crs,
            transform=self.transform @ Affine.scale(factor),
            width=math.ceil(self.width / factor),
            height=math.ceil(self.height / factor),
        )

    # ------------------------------------------------------------------ tiling

    def windows(self, size: int, stride: int | None = None, drop_partial: bool = False) -> Iterator[Window]:
        """Square chips covering the grid, row by row. Edge chips are clipped unless dropped."""
        stride = stride or size
        for row in range(0, self.height, stride):
            for col in range(0, self.width, stride):
                h = min(size, self.height - row)
                w = min(size, self.width - col)
                if drop_partial and (h < size or w < size):
                    continue
                yield Window(col, row, w, h)

    def window_transform(self, window: Window) -> Affine:
        return self.transform @ Affine.translation(window.col_off, window.row_off)

    def profile(self, dtype: str = "float32", count: int = 1, nodata: float | int | None = None,
                window: Window | None = None) -> dict[str, Any]:
        """A rasterio profile for writing a tiled, compressed GeoTIFF on this grid."""
        width = int(window.width) if window else self.width
        height = int(window.height) if window else self.height
        transform = self.window_transform(window) if window else self.transform
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": count,
            "width": width,
            "height": height,
            "crs": self.crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 3 if dtype.startswith("float") else 2,
            "tiled": width >= 256 and height >= 256,
        }
        if profile["tiled"]:
            profile.update(blockxsize=256, blockysize=256)
        if nodata is not None:
            profile["nodata"] = nodata
        return profile
