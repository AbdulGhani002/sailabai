"""GloFAS river flow (reanalysis and 51-member forecasts) from the Early Warning Data Store (EWDS).

Needs a free EWDS account and an API key in ~/.cdsapirc:

    url: https://ewds.climate.copernicus.eu/api
    key: <your-personal-access-token>

We do not train our own river model (Google needed more than 5,680 gauges for theirs); Model 2
reads GloFAS at our six points. Each point must sit on the right river cell: GloFAS is 0.05
degrees (~5 km), so a gauge's exact coordinates often fall on a neighbouring dry cell. `snap_points`
moves each point to the cell with the largest mean flow within a small window.

Untested until the team's EWDS account exists: check the request keys against the dataset's
"Show API request" button on the EWDS website, since the form changes between GloFAS versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sailab.config import Gauge

EWDS_URL = "https://ewds.climate.copernicus.eu/api"
M3S_TO_CUSECS = 35.3147
VARIABLE = "river_discharge_in_the_last_24_hours"


def _area(points: list[Gauge], pad: float = 0.3) -> list[float]:
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return [max(lats) + pad, min(lons) - pad, min(lats) - pad, max(lons) + pad]  # N, W, S, E


def client():
    import cdsapi

    return cdsapi.Client(url=EWDS_URL)


def request_reanalysis(year: int, points: list[Gauge], out: Path, version: str = "version_4_0",
                       months: tuple[int, ...] = (5, 6, 7, 8, 9, 10)) -> Path:
    client().retrieve("cems-glofas-historical", {
        "system_version": [version],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": [VARIABLE],
        "hyear": [str(year)],
        "hmonth": [f"{m:02d}" for m in months],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": _area(points),
    }, str(out))
    return out


def request_forecast(day: pd.Timestamp, points: list[Gauge], out: Path, leads: int = 7) -> Path:
    client().retrieve("cems-glofas-forecast", {
        "system_version": ["operational"],
        "hydrological_model": ["lisflood"],
        "product_type": ["control_forecast", "ensemble_perturbed_forecasts"],
        "variable": [VARIABLE],
        "year": [f"{day.year}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "leadtime_hour": [str(24 * k) for k in range(1, leads + 1)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": _area(points),
    }, str(out))
    return out


@dataclass
class SnappedPoint:
    point_id: str
    lat: float
    lon: float
    moved_km: float
    mean_flow_cusecs: float


def snap_points(mean_flow, points: list[Gauge], window: int = 2) -> list[SnappedPoint]:
    """Move each point to the highest-flow GloFAS cell within +/- `window` cells (an xarray DataArray
    of mean discharge with latitude/longitude coordinates)."""
    lat_name = "latitude" if "latitude" in mean_flow.coords else "lat"
    lon_name = "longitude" if "longitude" in mean_flow.coords else "lon"
    lats = mean_flow[lat_name].values
    lons = mean_flow[lon_name].values
    out = []
    for p in points:
        i = int(np.abs(lats - p.lat).argmin())
        j = int(np.abs(lons - p.lon).argmin())
        block = mean_flow.values[max(0, i - window):i + window + 1, max(0, j - window):j + window + 1]
        bi, bj = np.unravel_index(np.nanargmax(block), block.shape)
        si, sj = max(0, i - window) + bi, max(0, j - window) + bj
        moved = float(np.hypot((lats[si] - p.lat) * 111.32, (lons[sj] - p.lon) * 111.32 * np.cos(np.radians(p.lat))))
        out.append(SnappedPoint(p.id, float(lats[si]), float(lons[sj]), round(moved, 2),
                                float(mean_flow.values[si, sj]) * M3S_TO_CUSECS))
    return out


def forecast_table(nc_path: Path, issue_day: pd.Timestamp, snapped: list[SnappedPoint]) -> pd.DataFrame:
    """Forecast netCDF -> rows (issue_date, point_id, lead_day, member, q_cusecs) for the datacube."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    var = next(v for v in ds.data_vars if v.startswith("dis"))
    da = ds[var]
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    rows = []
    for sp in snapped:
        series = da.sel({lat_name: sp.lat, lon_name: sp.lon}, method="nearest")
        values = series.values.reshape(-1, series.shape[-1]) if series.ndim > 1 else series.values[None]
        # dims typically (number, step) for ensembles: members x leads
        for member, leads in enumerate(values):
            for k, v in enumerate(leads, start=1):
                rows.append({"issue_date": issue_day.normalize(), "point_id": sp.point_id, "lead_day": k,
                             "member": member, "q_cusecs": float(v) * M3S_TO_CUSECS})
    return pd.DataFrame(rows)
