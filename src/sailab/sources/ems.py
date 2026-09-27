"""Copernicus EMS rapid-mapping products (EMSR838, the 2025 Pakistan floods) as a second test reference.

Two expert maps of the same flood agreed only 48% in one study, so mapping scores are always
reported against at least two references: GFM and these (plus our hand-labelled tiles).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from sailab.sources.http import download, http_session

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"


def activation(code: str = "EMSR838") -> dict[str, Any]:
    r = http_session().get(API, params={"code": code}, timeout=60)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise LookupError(f"no activation {code}")
    return results[0]


def download_products(code: str, out_dir: Path) -> list[Path]:
    """Download and unpack all products of an activation; returns the vector files found."""
    info = activation(code)
    zpath = download(info["productsPath"], out_dir / f"{code}_products.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir / code)
    nested = list((out_dir / code).rglob("*.zip"))
    for n in nested:
        with zipfile.ZipFile(n) as z:
            z.extractall(n.with_suffix(""))
    return [p for p in (out_dir / code).rglob("*") if p.suffix.lower() in (".shp", ".json", ".geojson", ".gpkg")]


def flood_polygons(files: list[Path]):
    """Observed flood extent polygons (EMS names these layers 'observedEvent')."""
    import geopandas as gpd
    import pandas as pd

    parts = [gpd.read_file(f) for f in files if "observedevent" in f.name.lower()]
    if not parts:
        raise FileNotFoundError("no observedEvent layers in the products")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs("EPSG:4326")
