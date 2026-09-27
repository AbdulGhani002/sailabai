"""ECMWF open-data rain forecasts (no account; CC BY 4.0). Runs stay online about 4 days, so the daily
scraper keeps every 00 UTC run's total precipitation out to 7 days.

    sailab archive run --source ecmwf

Reading the GRIB files needs cfgrib/eccodes (`pip install cfgrib eccodes`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sailab.paths import archive_dir
from sailab.sources.http import Manifest

STEPS = list(range(0, 145, 6)) + list(range(150, 169, 6))  # 0..168 h


def archive_latest(root: Path | None = None, ensemble: bool = False, log=print) -> int:
    from ecmwf.opendata import Client

    root = (root or archive_dir()) / "ecmwf"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    kind = {"type": "pf", "stream": "enfo"} if ensemble else {"type": "fc", "stream": "oper"}
    latest = client.latest(param="tp", step=168, time=0, **kind)
    target = root / f"{latest:%Y}" / f"{latest:%Y%m%d%H}_tp_{'ens' if ensemble else 'hres'}.grib2"
    manifest = Manifest(root / "manifest.csv")
    if target.name in manifest:
        log(f"ecmwf: {target.name} already archived")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve(date=latest.strftime("%Y-%m-%d"), time=0, param="tp", step=STEPS, target=str(target), **kind)
    manifest.add("ecmwf", target.name, "ecmwf-opendata", target, note=f"run {latest:%Y-%m-%d %H}Z")
    log(f"ecmwf: archived {target.name}")
    return 1


def rain_forecast_table(grib: Path, upper: tuple[float, float, float, float],
                        local: tuple[float, float, float, float]) -> pd.DataFrame:
    """Daily rain per lead day (mm) over the upper catchments and the study area."""
    import xarray as xr

    ds = xr.open_dataset(grib, engine="cfgrib")
    tp = ds["tp"] * 1000.0  # metres -> mm, accumulated from the run start
    issue = pd.Timestamp(ds["time"].values).normalize()
    rows = []
    hours = (tp["step"].values / np.timedelta64(1, "h")).astype(int)

    def daily(box: tuple[float, float, float, float], lead: int) -> float:
        w, s, e, n = box
        sub = tp.sel(latitude=slice(n, s), longitude=slice(w, e)).mean(("latitude", "longitude"))
        at = dict(zip(hours, sub.values, strict=True))
        return float(at.get(24 * lead, np.nan) - at.get(24 * (lead - 1), 0.0))

    for lead in range(1, 8):
        rows.append({"issue_date": issue, "lead_day": lead, "rain_upper_mm": max(0.0, daily(upper, lead)),
                     "rain_local_mm": max(0.0, daily(local, lead))})
    return pd.DataFrame(rows)
