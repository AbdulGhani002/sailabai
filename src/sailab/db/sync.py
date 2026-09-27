"""Load twin runs, gauges, readings and the results table into the database.

    sailab db sync --cube demo

The twin writes files first (they are the source of truth and what TiTiler-style tile rendering
reads); this makes them queryable for the API and for GIS tools on PostGIS.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sailab.config import load_gauges
from sailab.cube import Datacube
from sailab.db.models import Gauge, PlaceRisk, Reading, Result, RoadRisk, Run
from sailab.db.session import engine, init_db, session
from sailab.evaluation.results import ResultsTable
from sailab.paths import twin_dir


def _point(lon: float, lat: float) -> str:
    return json.dumps({"type": "Point", "coordinates": [lon, lat]})


def sync_gauges(s: Session) -> int:
    n = 0
    for g in load_gauges().all():
        s.merge(Gauge(id=g.id, name=g.name, river=g.river, country=g.country, lat=g.lat, lon=g.lon,
                      design_capacity_cusecs=g.design_capacity_cusecs, geojson=_point(g.lon, g.lat)))
        n += 1
    return n


def sync_readings(s: Session, cube: Datacube) -> int:
    df = cube.table("series", "discharge")
    label = cube.root.name
    s.execute(delete(Reading).where(Reading.cube == label))
    rows = [Reading(gauge_id=r.point_id, day=pd.Timestamp(r.date).date(), source=r.source, q_cusecs=float(r.q_cusecs),
                    version=r.version, cube=label) for r in df.itertuples()]
    s.add_all(rows)
    return len(rows)


def sync_run(s: Session, folder: Path, cube_name: str) -> str:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    run_id = f"{cube_name}/{summary['run_id']}"
    s.execute(delete(RoadRisk).where(RoadRisk.run_id == run_id))
    s.execute(delete(PlaceRisk).where(PlaceRisk.run_id == run_id))
    s.merge(Run(id=run_id, cube=cube_name, run_date=summary["run_id"],
                issue_time=datetime.fromisoformat(summary["issue_time"]), model=summary["model"],
                synthetic=bool(summary.get("synthetic")), current_flooded_km2=summary.get("current_flooded_km2", 0.0),
                folder=str(folder), summary=summary, created=datetime.fromisoformat(summary["created"])))
    s.flush()
    try:
        roads = Datacube.open(summary["cube"]).geojson("roads")["features"]
        roads_geo = {f["properties"]["id"]: json.dumps(f["geometry"]) for f in roads}
    except FileNotFoundError:
        roads_geo = {}
    for h in summary["horizons"]:
        for r in h.get("roads", []):
            s.add(RoadRisk(run_id=run_id, horizon=h["name"], road_id=r["id"], name=r.get("name", ""),
                           road_class=r.get("class", ""), max_prob=r["max_prob"], mean_prob=r["mean_prob"],
                           max_prob_low=r.get("max_prob_low"), max_prob_high=r.get("max_prob_high"),
                           geojson=roads_geo.get(r["id"])))
        for p in h.get("places", []):
            s.add(PlaceRisk(run_id=run_id, horizon=h["name"], name=p["name"], people_expected=p["people"]["expected"],
                            people_low=p["people"]["low"], people_high=p["people"]["high"], max_prob=p["max_prob"],
                            geojson=_point(p["lon"], p["lat"])))
    return run_id


def sync_results(s: Session) -> int:
    df = ResultsTable().load()
    s.execute(delete(Result))
    for r in df.to_dict("records"):
        clean = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()
                 if k in Result.__table__.columns}
        clean["commit"] = str(clean.get("commit"))
        s.add(Result(**clean))
    return len(df)


def sync_all(cube: str = "demo", log=print) -> None:
    eng = init_db(engine())
    c = Datacube.open(cube)
    with session(eng) as s:
        log(f"gauges: {sync_gauges(s)}")
        log(f"readings: {sync_readings(s, c)}")
        runs_root = twin_dir() / c.root.name / "runs"
        n = 0
        for folder in sorted(runs_root.glob("*")) if runs_root.exists() else []:
            if (folder / "summary.json").exists():
                sync_run(s, folder, c.root.name)
                n += 1
        log(f"runs: {n}")
        log(f"results rows: {sync_results(s)}")
        s.commit()
    log(f"synced into {eng.url.render_as_string(hide_password=True)}")


def latest_run(s: Session, cube: str) -> Run | None:
    return s.scalars(select(Run).where(Run.cube == cube).order_by(Run.run_date.desc()).limit(1)).first()
