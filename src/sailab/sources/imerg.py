"""NASA GPM IMERG daily rain (V07; V08 is coming) through earthaccess.

Needs a free NASA Earthdata account: `earthaccess.login()` reads EARTHDATA_USERNAME /
EARTHDATA_PASSWORD or ~/.netrc. We use the Late run (about 14 hours behind) for history and the
Early run (about 4 hours) in live mode, and save the version with every value (team rule 2).

Untested until the team's Earthdata account exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sailab.features import soil_moisture_index

SHORT_NAMES = {"early": "GPM_3IMERGDE", "late": "GPM_3IMERGDL", "final": "GPM_3IMERGDF"}
# Rain that drives our floods falls mostly on the upper catchments (above the rim stations), while
# local rain drives ponding inside the study area.
UPPER_CATCHMENTS = (73.0, 31.5, 78.0, 35.5)   # W, S, E, N
LOCAL = (70.9, 29.2, 72.7, 31.4)


def download(start: str, end: str, out_dir: Path, run: str = "late", version: str = "07") -> list[Path]:
    import earthaccess

    earthaccess.login()
    bbox = (min(UPPER_CATCHMENTS[0], LOCAL[0]), min(UPPER_CATCHMENTS[1], LOCAL[1]),
            max(UPPER_CATCHMENTS[2], LOCAL[2]), max(UPPER_CATCHMENTS[3], LOCAL[3]))
    results = earthaccess.search_data(short_name=SHORT_NAMES[run], version=version, temporal=(start, end),
                                      bounding_box=bbox)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [Path(p) for p in earthaccess.download(results, str(out_dir))]


def _box_mean(da, box: tuple[float, float, float, float]) -> float:
    w, s, e, n = box
    sub = da.sel(lon=slice(w, e), lat=slice(s, n))
    return float(np.nanmean(sub.values))


def weather_table(files: list[Path], version: str = "V07") -> pd.DataFrame:
    """Daily files -> (date, rain_upper_mm, rain_local_mm, source, version)."""
    import xarray as xr

    rows = []
    for f in sorted(files):
        ds = xr.open_dataset(f)
        var = "precipitation" if "precipitation" in ds else next(iter(ds.data_vars))
        da = ds[var].squeeze()
        if da.dims[0] == "lon":  # IMERG stores (lon, lat)
            da = da.transpose()
        day = pd.Timestamp(ds["time"].values.ravel()[0]).normalize()
        rows.append({"date": day, "rain_upper_mm": _box_mean(da, UPPER_CATCHMENTS), "rain_local_mm": _box_mean(da, LOCAL),
                     "source": "imerg", "version": version})
    df = pd.DataFrame(rows).sort_values("date")
    df["soil_moisture"] = soil_moisture_index(df["rain_local_mm"].fillna(0).to_numpy()).astype(np.float32)
    return df
