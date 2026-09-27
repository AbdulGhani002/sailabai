"""Read remote cloud-optimised GeoTIFFs straight onto our analysis grid.

Only the pixels over the study area are fetched (HTTP range requests), and when the target grid is
much coarser than the source, a matching overview level is read instead of full resolution.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from sailab.grid import GridSpec

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def _overview_level(src_res: float, target_res: float, n_overviews: int) -> int | None:
    factor = target_res / max(src_res, 1e-9)
    if factor < 2 or n_overviews == 0:
        return None
    level = int(math.floor(math.log2(factor))) - 1
    return max(0, min(level, n_overviews - 1))


def _src_res_meters(src) -> float:
    res = abs(src.transform.a)
    if src.crs and src.crs.is_geographic:
        return res * 111_320.0
    return res


def read_to_grid(href: str, grid: GridSpec, resampling: Resampling = Resampling.nearest, band: int = 1,
                 nodata: float | int | None = None, dtype: str = "float32") -> np.ndarray:
    """One remote raster warped onto `grid`; pixels it does not cover get `nodata` (NaN for floats)."""
    fill = np.nan if (nodata is None and dtype.startswith("float")) else (nodata if nodata is not None else 0)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as probe:
            n_ovr = len(probe.overviews(band))
            level = _overview_level(_src_res_meters(probe), grid.res, n_ovr)
        kwargs = {"overview_level": level} if level is not None else {}
        with rasterio.open(href, **kwargs) as src:
            vrt_kwargs = {"crs": grid.crs, "transform": grid.transform, "width": grid.width,
                          "height": grid.height, "resampling": resampling}
            if src.nodata is not None:
                vrt_kwargs.update(src_nodata=src.nodata, nodata=src.nodata)
            else:
                vrt_kwargs.update(add_alpha=True)  # alpha marks where the source has data
            with WarpedVRT(src, **vrt_kwargs) as vrt:
                data = vrt.read(band)
                if src.nodata is None:
                    valid = vrt.read(vrt.count) > 0
                elif isinstance(src.nodata, float) and math.isnan(src.nodata):
                    valid = ~np.isnan(data)
                else:
                    valid = data != src.nodata
    return np.where(valid, data, fill).astype(dtype)


def mosaic_to_grid(hrefs: Iterable[str], grid: GridSpec, resampling: Resampling = Resampling.nearest,
                   dtype: str = "float32", nodata: float | int | None = None, log=None) -> np.ndarray:
    """First-valid-wins mosaic of several remote tiles on `grid`."""
    fill = np.nan if (nodata is None and dtype.startswith("float")) else (nodata if nodata is not None else 0)
    out = np.full(grid.shape, fill, dtype=dtype)
    have = np.zeros(grid.shape, dtype=bool)
    for href in hrefs:
        try:
            part = read_to_grid(href, grid, resampling, dtype=dtype, nodata=nodata)
        except rasterio.errors.RasterioIOError as e:
            if log:
                log(f"  skipped {href}: {e}")
            continue
        ok = ~np.isnan(part) if dtype.startswith("float") else part != fill
        take = ok & ~have
        out[take] = part[take]
        have |= ok
    return out
