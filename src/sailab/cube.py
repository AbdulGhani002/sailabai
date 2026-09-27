"""A datacube: every layer for one study area on one grid, as plain GeoTIFF and Parquet files.

The synthetic demo world and the real-data pipeline both write this layout, so the models, the
evaluation code and the twin never care where the data came from.

    cube.json                   grid, kind (synthetic | real), versions, notes
    static/<name>.tif           dem, hand, slope, normal_water, landcover, population, buildings, road_km
    s1/<scene_id>.tif           Sentinel-1 VV and VH in dB (int16, 0.01 dB steps)
    s1_pre/<orbit>.tif          dry-season reference image for each satellite path
    labels/<scene_id>.tif       uint8: 0 land, 1 normal water, 2 flood, 255 ignore
    truth/<scene_id>.tif        synthetic cubes only: error-free labels (stand-in for hand labels)
    scenes.parquet              one row per satellite pass
    series/<name>.parquet       daily observations (discharge, rain, soil moisture)
    forecasts/<name>.parquet    GloFAS members and rain forecasts by issue date and lead time
    vector/<name>.geojson       roads, rivers, breach sites (display and exposure)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.windows import Window

from sailab import DISCLAIMER, __version__
from sailab.grid import GridSpec
from sailab.io import decode_sar, encode_sar, read_raster, write_raster
from sailab.labels import IGNORE
from sailab.paths import cubes_dir

SCENE_COLUMNS = ["scene_id", "time", "platform", "orbit", "event_id", "split", "coverage", "versions"]


def point_features(items, layer: str) -> list[dict[str, Any]]:
    """GeoJSON points from config models that have lat/lon (places, gauges)."""
    feats = []
    for it in items:
        props = {k: v for k, v in it.model_dump().items() if k not in ("lat", "lon")}
        props["layer"] = layer
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [it.lon, it.lat]}})
    return feats


class Datacube:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        meta_path = self.root / "cube.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no datacube at {self.root} (cube.json missing)")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.grid = GridSpec.from_dict(self.meta["grid"])

    # ------------------------------------------------------------------ create / open

    @classmethod
    def create(cls, root: Path | str, grid: GridSpec, kind: str, **meta: Any) -> Datacube:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        info = {
            "kind": kind,
            "grid": grid.to_dict(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sailab_version": __version__,
            "disclaimer": DISCLAIMER,
            **meta,
        }
        (root / "cube.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        return cls(root)

    @classmethod
    def open(cls, name_or_path: str | Path) -> Datacube:
        p = Path(name_or_path)
        if not (p / "cube.json").exists():
            p = cubes_dir() / str(name_or_path)
        return cls(p)

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "real")

    @property
    def is_synthetic(self) -> bool:
        return self.kind == "synthetic"

    def update_meta(self, **meta: Any) -> None:
        self.meta.update(meta)
        (self.root / "cube.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ static layers

    def static_path(self, name: str) -> Path:
        return self.root / "static" / f"{name}.tif"

    def has_static(self, name: str) -> bool:
        return self.static_path(name).exists()

    def write_static(self, name: str, array: np.ndarray, nodata: float | int | None = None,
                     dtype: str | None = None, **tags: Any) -> Path:
        return write_raster(self.static_path(name), array, self.grid, dtype=dtype, nodata=nodata, tags=tags)

    def static(self, name: str, window: Window | None = None) -> np.ndarray:
        return read_raster(self.static_path(name), window=window)[0]

    def exclusion(self) -> np.ndarray:
        """Pixels radar flood mapping cannot judge (GFM excludes them too): bare sand, which is as
        dark as water, and built-up areas, where flood water shows as bright double bounce.
        Built from WorldCover classes 60 (bare) and 50 (built-up) the first time it is needed."""
        if not self.has_static("exclusion"):
            lc = self.static("landcover")
            self.write_static("exclusion", np.isin(lc, (50, 60)).astype(np.uint8), dtype="uint8",
                              source="WorldCover bare (60) and built-up (50)")
        return self.static("exclusion").astype(bool)

    @cached_property
    def static_names(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "static").glob("*.tif"))

    # ------------------------------------------------------------------ scenes

    def write_scene_sar(self, scene_id: str, sar_db: np.ndarray) -> Path:
        return write_raster(self.root / "s1" / f"{scene_id}.tif", encode_sar(sar_db), self.grid,
                            dtype="int16", nodata=-32768, band_names=["VV", "VH"])

    def write_pre_sar(self, orbit: str, sar_db: np.ndarray) -> Path:
        return write_raster(self.root / "s1_pre" / f"{orbit}.tif", encode_sar(sar_db), self.grid,
                            dtype="int16", nodata=-32768, band_names=["VV", "VH"])

    def write_label(self, scene_id: str, label: np.ndarray, truth: bool = False, **tags: Any) -> Path:
        folder = "truth" if truth else "labels"
        return write_raster(self.root / folder / f"{scene_id}.tif", label.astype(np.uint8), self.grid,
                            dtype="uint8", nodata=IGNORE, tags=tags)

    def read_sar(self, scene_id: str, window: Window | None = None) -> np.ndarray:
        """(2, h, w) float32 VV and VH in dB, NaN where there is no data."""
        return decode_sar(read_raster(self.root / "s1" / f"{scene_id}.tif", window=window))

    def read_pre_sar(self, orbit: str, window: Window | None = None) -> np.ndarray:
        return decode_sar(read_raster(self.root / "s1_pre" / f"{orbit}.tif", window=window))

    def has_sar(self, scene_id: str) -> bool:
        return (self.root / "s1" / f"{scene_id}.tif").exists()

    def has_truth(self, scene_id: str) -> bool:
        return (self.root / "truth" / f"{scene_id}.tif").exists()

    def read_label(self, scene_id: str, window: Window | None = None, truth: bool = False) -> np.ndarray:
        folder = "truth" if truth else "labels"
        return read_raster(self.root / folder / f"{scene_id}.tif", window=window)[0]

    def write_scenes(self, scenes: pd.DataFrame) -> None:
        df = scenes.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if "versions" in df:
            df["versions"] = df["versions"].map(lambda v: v if isinstance(v, str) else json.dumps(v or {}))
        df = df.sort_values("time").reset_index(drop=True)
        df.to_parquet(self.root / "scenes.parquet", index=False)
        self.__dict__.pop("scenes", None)

    @cached_property
    def scenes(self) -> pd.DataFrame:
        path = self.root / "scenes.parquet"
        if not path.exists():
            return pd.DataFrame(columns=SCENE_COLUMNS)
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    def scene_versions(self, scene_id: str) -> dict[str, str]:
        row = self.scenes.loc[self.scenes["scene_id"] == scene_id]
        if row.empty:
            raise KeyError(scene_id)
        raw = row.iloc[0].get("versions") or "{}"
        return json.loads(raw) if isinstance(raw, str) else dict(raw)

    # ------------------------------------------------------------------ tables

    def write_table(self, group: str, name: str, df: pd.DataFrame) -> Path:
        path = self.root / group / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def table(self, group: str, name: str) -> pd.DataFrame:
        path = self.root / group / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def has_table(self, group: str, name: str) -> bool:
        return (self.root / group / f"{name}.parquet").exists()

    @cached_property
    def discharge(self) -> pd.DataFrame:
        """Observed daily discharge (cusecs), one column per point, NaN where missing."""
        df = self.table("series", "discharge")
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot_table(index="date", columns="point_id", values="q_cusecs", aggfunc="mean").sort_index()

    @cached_property
    def weather(self) -> pd.DataFrame:
        """Daily basin rain (mm) and soil moisture index, indexed by date."""
        df = self.table("series", "weather")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    @cached_property
    def glofas(self) -> pd.DataFrame:
        """GloFAS forecasts: issue_date, point_id, lead_day, member, q_cusecs."""
        df = self.table("forecasts", "glofas")
        df["issue_date"] = pd.to_datetime(df["issue_date"])
        return df

    @cached_property
    def rain_forecast(self) -> pd.DataFrame:
        """Rain forecasts: issue_date, lead_day, rain_upper_mm, rain_local_mm."""
        df = self.table("forecasts", "rain")
        df["issue_date"] = pd.to_datetime(df["issue_date"])
        return df

    # ------------------------------------------------------------------ vectors

    def write_geojson(self, name: str, features: list[dict[str, Any]]) -> Path:
        path = self.root / "vector" / f"{name}.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        return path

    def geojson(self, name: str) -> dict[str, Any]:
        path = self.root / "vector" / f"{name}.geojson"
        if not path.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(path.read_text(encoding="utf-8"))
