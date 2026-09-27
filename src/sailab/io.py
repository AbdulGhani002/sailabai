"""Raster reading and writing on the analysis grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from sailab import DISCLAIMER
from sailab.grid import GridSpec

# Radar backscatter is stored as int16 hundredths of a dB: half the size of float32, 0.01 dB steps.
SAR_SCALE = 0.01
SAR_NODATA = -32768


def write_raster(path: Path, array: np.ndarray, grid: GridSpec, *, dtype: str | None = None,
                 nodata: float | int | None = None, tags: dict[str, Any] | None = None,
                 band_names: list[str] | None = None, cog: bool = False) -> Path:
    """Write a (bands, H, W) or (H, W) array on `grid`. `cog=True` writes a cloud-optimised GeoTIFF."""
    arr = array if array.ndim == 3 else array[None]
    dtype = dtype or str(arr.dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = grid.profile(dtype=dtype, count=arr.shape[0], nodata=nodata)
    if cog:
        profile = {k: v for k, v in profile.items() if k not in ("tiled", "blockxsize", "blockysize")}
        profile.update(driver="COG", blocksize=256, overview_resampling="average" if dtype.startswith("float") else "nearest")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype, copy=False))
        dst.update_tags(disclaimer=DISCLAIMER, **{k: str(v) for k, v in (tags or {}).items()})
        for i, name in enumerate(band_names or [], start=1):
            dst.set_band_description(i, name)
    return path


def read_raster(path: Path, window: Window | None = None, bands: list[int] | int | None = None,
                out_dtype: str | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(bands, window=window, boundless=window is not None,
                        fill_value=src.nodata if src.nodata is not None else 0)
    return data.astype(out_dtype, copy=False) if out_dtype else data


def read_tags(path: Path) -> dict[str, str]:
    with rasterio.open(path) as src:
        return dict(src.tags())


def encode_sar(db: np.ndarray) -> np.ndarray:
    out = np.round(np.nan_to_num(db, nan=SAR_NODATA * SAR_SCALE) / SAR_SCALE)
    out = np.clip(out, -32767, 32767).astype(np.int16)
    out[~np.isfinite(db)] = SAR_NODATA
    return out


def decode_sar(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float32) * SAR_SCALE
    out[raw == SAR_NODATA] = np.nan
    return out
