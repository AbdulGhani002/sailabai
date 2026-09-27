"""Database tables for the platform.

The same models run on SQLite (local development, tests) and PostgreSQL with PostGIS (Docker and
production). Geometries are stored as GeoJSON text so the application code is identical on both;
on PostGIS, `init_db` adds a generated `geom` column with a spatial index next to each GeoJSON
column, so QGIS and spatial SQL work on the live tables.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One daily update of the twin."""

    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)       # "<cube>/<YYYY-MM-DD>"
    cube: Mapped[str] = mapped_column(String(64), index=True)
    run_date: Mapped[str] = mapped_column(String(10), index=True)
    issue_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(200))
    synthetic: Mapped[bool] = mapped_column(default=False)
    current_flooded_km2: Mapped[float] = mapped_column(Float, default=0.0)
    folder: Mapped[str] = mapped_column(Text)
    summary: Mapped[dict] = mapped_column(JSON)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RoadRisk(Base):
    __tablename__ = "road_risk"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    horizon: Mapped[str] = mapped_column(String(16))
    road_id: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(200))
    road_class: Mapped[str] = mapped_column(String(32))
    max_prob: Mapped[float] = mapped_column(Float)
    mean_prob: Mapped[float] = mapped_column(Float)
    max_prob_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_prob_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    geojson: Mapped[str | None] = mapped_column(Text, nullable=True)


class PlaceRisk(Base):
    __tablename__ = "place_risk"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    horizon: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(120))
    people_expected: Mapped[float] = mapped_column(Float)
    people_low: Mapped[float] = mapped_column(Float)
    people_high: Mapped[float] = mapped_column(Float)
    max_prob: Mapped[float] = mapped_column(Float)
    geojson: Mapped[str | None] = mapped_column(Text, nullable=True)


class Gauge(Base):
    __tablename__ = "gauges"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    river: Mapped[str] = mapped_column(String(32))
    country: Mapped[str] = mapped_column(String(2))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    design_capacity_cusecs: Mapped[float | None] = mapped_column(Float, nullable=True)
    geojson: Mapped[str | None] = mapped_column(Text, nullable=True)


class Reading(Base):
    """Daily gauge observation (FFD 6 am reading) or shared upstream value."""

    __tablename__ = "readings"
    gauge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    q_cusecs: Mapped[float] = mapped_column(Float)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cube: Mapped[str] = mapped_column(String(64), default="real")


class Result(Base):
    """A row of the shared results table (rule 3)."""

    __tablename__ = "results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[str] = mapped_column(String(32))
    commit: Mapped[str] = mapped_column(String(40))
    experiment: Mapped[str] = mapped_column(String(120), index=True)
    model: Mapped[str] = mapped_column(String(120))
    baseline: Mapped[str] = mapped_column(String(120))
    split: Mapped[str] = mapped_column(String(16))
    event: Mapped[str] = mapped_column(String(64))
    lead: Mapped[str] = mapped_column(String(16))
    subset: Mapped[str] = mapped_column(String(32))
    metric: Mapped[str] = mapped_column(String(32))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    skill: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_versions: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# GeoJSON columns that get a PostGIS twin column on PostgreSQL
SPATIAL_COLUMNS = {"road_risk": "geojson", "place_risk": "geojson", "gauges": "geojson"}
