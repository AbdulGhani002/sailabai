"""FastAPI backend for the dashboard.

    sailab api serve            # http://127.0.0.1:8000/docs

Runs, risk tables, gauges and results come from the database (PostGIS in Docker, SQLite locally);
map tiles are rendered from the twin's cloud-optimised GeoTIFFs with rio-tiler, the library
TiTiler is built on, behind an endpoint that only serves files the twin wrote.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select

from sailab import DISCLAIMER, __version__
from sailab.api.colormaps import colormap_for, legend
from sailab.config import load_aoi, load_gauges
from sailab.cube import Datacube
from sailab.db.models import Result, RoadRisk, Run
from sailab.db.session import init_db, session
from sailab.licences import LicenceRegister
from sailab.paths import twin_dir

LAYER_RE = re.compile(r"^(current|(prob|sets|spread)_(next|d[1-7]))$")
DEFAULT_CUBE = os.environ.get("SAILAB_CUBE", "demo")
DATASETS_SHOWN = ["sentinel1", "gfm", "glofas", "imerg", "ecmwf_open", "cop_dem_glo30", "asf_hand", "jrc_gsw",
                  "worldcover", "worldpop", "osm_roads", "ms_buildings"]

app = FastAPI(title="SailabAI API", version=__version__,
              description=f"Flood digital twin for the Chenab, Ravi and Sutlej around Multan. {DISCLAIMER}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SAILAB_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@lru_cache(maxsize=4)
def _cube(name: str) -> Datacube:
    try:
        return Datacube.open(name)
    except FileNotFoundError as e:
        raise HTTPException(404, f"datacube {name!r} not found") from e


def _run_or_404(cube: str, run_date: str) -> Run:
    with session() as s:
        run = s.get(Run, f"{cube}/{run_date}")
    if run is None:
        raise HTTPException(404, f"no twin run {cube}/{run_date}")
    return run


# --------------------------------------------------------------------------- metadata


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/api/meta")
def meta(cube: str = DEFAULT_CUBE) -> dict[str, Any]:
    c = _cube(cube)
    aoi = load_aoi()
    b = aoi.effective_bbox
    try:
        reg = LicenceRegister.load()
        credits = [{"id": ds, "name": reg[ds].name, "credit": reg[ds].credit, "licence": reg[ds].licence}
                   for ds in DATASETS_SHOWN if ds in reg.entries]
    except FileNotFoundError:
        credits = []
    return {
        "name": "SailabAI",
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "cube": cube,
        "synthetic": c.is_synthetic,
        "synthetic_note": c.meta.get("note") if c.is_synthetic else None,
        "bbox": [b.west, b.south, b.east, b.north],
        "center": [(b.west + b.east) / 2, (b.south + b.north) / 2],
        "grid": {"crs": c.grid.crs, "resolution_m": c.grid.res, "width": c.grid.width, "height": c.grid.height},
        "places": c.geojson("places"),
        "rivers": c.geojson("rivers"),
        "gauges": c.geojson("gauges"),
        "breaches": c.geojson("breaches"),
        "credits": credits,
    }


@app.get("/api/legend/{layer}")
def get_legend(layer: str, theme: str = Query("light", pattern="^(light|dark)$")) -> list[dict]:
    if not LAYER_RE.match(layer):
        raise HTTPException(400, "unknown layer")
    return legend(layer, theme)


# --------------------------------------------------------------------------- twin runs


@app.get("/api/runs")
def list_runs(cube: str = DEFAULT_CUBE) -> list[dict[str, Any]]:
    with session() as s:
        runs = s.scalars(select(Run).where(Run.cube == cube).order_by(Run.run_date)).all()
    return [{"run_date": r.run_date, "issue_time": r.issue_time.isoformat(), "model": r.model,
             "current_flooded_km2": r.current_flooded_km2,
             "horizons": [{"name": h["name"], "lead_days": h["lead_days"],
                           "flooded_km2_expected": h["flooded_km2_expected"]} for h in r.summary["horizons"]]}
            for r in runs]


@app.get("/api/runs/{cube}/{run_date}")
def get_run(cube: str, run_date: str) -> dict[str, Any]:
    run = _run_or_404(cube, run_date)
    summary = dict(run.summary)
    base = f"/tiles/{cube}/{run_date}"
    summary["tiles"] = {"current": f"{base}/current/{{z}}/{{x}}/{{y}}.png"}
    horizons = []
    for h in summary["horizons"]:
        h = {k: v for k, v in h.items() if k != "roads"}
        h["tiles"] = {kind: f"{base}/{kind}_{h['name']}/{{z}}/{{x}}/{{y}}.png" for kind in h["layers"]}
        horizons.append(h)
    summary["horizons"] = horizons
    return summary


@app.get("/api/runs/{cube}/{run_date}/roads")
def run_roads(cube: str, run_date: str, horizon: str = "d1", limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
    _run_or_404(cube, run_date)
    with session() as s:
        rows = s.scalars(select(RoadRisk).where(RoadRisk.run_id == f"{cube}/{run_date}", RoadRisk.horizon == horizon)
                         .order_by(RoadRisk.max_prob.desc()).limit(limit)).all()
    return [{"id": r.road_id, "name": r.name, "class": r.road_class, "max_prob": r.max_prob, "mean_prob": r.mean_prob,
             "max_prob_low": r.max_prob_low, "max_prob_high": r.max_prob_high} for r in rows]


# --------------------------------------------------------------------------- gauges


@app.get("/api/gauges/{gauge_id}/hydrograph")
def hydrograph(gauge_id: str, date: str, cube: str = DEFAULT_CUBE, history_days: int = 30) -> dict[str, Any]:
    """Observed flow before the issue date, GloFAS ensemble quantiles after it, and (in replay) what
    was observed afterwards."""
    c = _cube(cube)
    gauges = load_gauges().by_id()
    if gauge_id not in gauges:
        raise HTTPException(404, f"unknown gauge {gauge_id}")
    day = pd.Timestamp(date).normalize()
    obs = c.discharge[gauge_id] if gauge_id in c.discharge.columns else pd.Series(dtype=float)
    before = obs[(obs.index > day - pd.Timedelta(days=history_days)) & (obs.index <= day)].dropna()
    after = obs[(obs.index > day) & (obs.index <= day + pd.Timedelta(days=7))].dropna()
    gl = c.glofas
    fc = gl[(gl["issue_date"] == day) & (gl["point_id"].astype(str) == gauge_id)]
    quantiles = []
    for lead, part in fc.groupby("lead_day"):
        q = np.percentile(part["q_cusecs"].to_numpy(), [5, 25, 50, 75, 95])
        quantiles.append({"date": (day + pd.Timedelta(days=int(lead))).date().isoformat(), "lead_day": int(lead),
                          **{f"p{p}": round(float(v)) for p, v in zip((5, 25, 50, 75, 95), q, strict=True)}})
    g = gauges[gauge_id]
    return {
        "gauge": {"id": g.id, "name": g.name, "river": g.river, "country": g.country,
                  "design_capacity_cusecs": g.design_capacity_cusecs},
        "issue_date": day.date().isoformat(),
        "observed": [{"date": d.date().isoformat(), "q": round(float(v))} for d, v in before.items()],
        "observed_after": [{"date": d.date().isoformat(), "q": round(float(v))} for d, v in after.items()],
        "forecast": quantiles,
        "india_data_missing": g.country == "IN" and before.empty,
    }


# --------------------------------------------------------------------------- results


@app.get("/api/results")
def results(experiment: str | None = None) -> list[dict[str, Any]]:
    with session() as s:
        q = select(Result)
        if experiment:
            q = q.where(Result.experiment == experiment)
        rows = s.scalars(q).all()
    if not rows:
        return []
    df = pd.DataFrame([{c.name: getattr(r, c.name) for c in Result.__table__.columns} for r in rows])
    latest = df.groupby(["experiment", "model"])["time"].transform("max")
    df = df[df["time"] == latest].drop(columns=["id"])
    return df.replace({np.nan: None}).to_dict("records")


@app.get("/api/reliability")
def reliability(split: str = Query("val", pattern="^(val|test|replay|live)$"),
                experiment: str = "model2") -> dict[str, Any]:
    """Reliability curves (forecast chance vs how often it flooded) saved by `sailab evaluate forecast`."""
    import json

    from sailab.paths import results_dir

    path = results_dir() / f"reliability_{experiment}_{split}.json"
    if not path.exists():
        raise HTTPException(404, f"no reliability data for {experiment} on {split}; run sailab evaluate forecast")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- tiles


@app.get("/tiles/{cube}/{run_date}/{layer}/{z}/{x}/{y}.png")
def tile(cube: str, run_date: str, layer: str, z: int, x: int, y: int,
         theme: str = Query("light", pattern="^(light|dark)$")) -> Response:
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    if not LAYER_RE.match(layer) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_date) or not re.fullmatch(r"[\w-]+", cube):
        raise HTTPException(400, "bad tile request")
    path = twin_dir() / cube / "runs" / run_date / f"{layer}.tif"
    if not path.exists():
        raise HTTPException(404, "layer not found")
    try:
        with Reader(str(path)) as src:
            img = src.tile(x, y, z, tilesize=256, resampling_method="nearest")
    except TileOutsideBounds:
        return Response(status_code=204)
    content = img.render(img_format="PNG", colormap=colormap_for(layer, theme))
    return Response(content, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("sailab.api.app:app", host=host, port=port, reload=reload)


__all__ = ["app", "serve"]
