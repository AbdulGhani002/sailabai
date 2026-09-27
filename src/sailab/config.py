"""Typed access to the YAML files in configs/."""

from __future__ import annotations

from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from sailab.paths import config_dir


def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by file name (looked up in configs/) or by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.is_absolute() and not path.exists():
        path = config_dir() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- study area


class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _check(self) -> BBox:
        if not (self.west < self.east and self.south < self.north):
            raise ValueError(f"invalid bbox {self}")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north


class GridConfig(BaseModel):
    crs: str = "EPSG:32642"
    resolution_m: float = 20.0
    demo_resolution_m: float = 400.0


class Place(BaseModel):
    name: str
    lat: float
    lon: float
    kind: str = "town"


class AOIConfig(BaseModel):
    name: str
    description: str = ""
    bbox: BBox
    extended_east: float | None = None
    use_extension: bool = False
    grid: GridConfig = Field(default_factory=GridConfig)
    places: list[Place] = Field(default_factory=list)
    rivers: dict[str, list[tuple[float, float]]] = Field(default_factory=dict)

    @property
    def effective_bbox(self) -> BBox:
        if self.use_extension and self.extended_east:
            return BBox(west=self.bbox.west, south=self.bbox.south, east=self.extended_east, north=self.bbox.north)
        return self.bbox


# --------------------------------------------------------------------------- gauges


class Gauge(BaseModel):
    id: str
    name: str
    river: str
    country: str = "PK"
    lat: float
    lon: float
    design_capacity_cusecs: float | None = None
    next: str | None = None
    official_lag_hours_to_next: float | None = None
    in_model_inputs: bool = False


class GaugesConfig(BaseModel):
    gauges: list[Gauge]
    upstream_india: list[Gauge] = Field(default_factory=list)
    india_data_cutoff: date | None = None

    def all(self) -> list[Gauge]:
        return [*self.gauges, *self.upstream_india]

    def by_id(self) -> dict[str, Gauge]:
        return {g.id: g for g in self.all()}

    def model_input_ids(self, include_india: bool = False) -> list[str]:
        ids = [g.id for g in self.gauges if g.in_model_inputs]
        if include_india:
            ids += [g.id for g in self.upstream_india]
        return ids


# --------------------------------------------------------------------------- events


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    REPLAY = "replay"
    LIVE = "live"
    STRESS = "stress"


class Peak(BaseModel):
    gauge: str
    cusecs: float


class Event(BaseModel):
    id: str
    start: date
    end: date
    split: Split
    rivers: list[str] = Field(default_factory=list)
    peak: Peak | None = None
    other_peaks: list[Peak] = Field(default_factory=list)
    breaches: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> Event:
        if self.end < self.start:
            raise ValueError(f"event {self.id} ends before it starts")
        return self

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    @property
    def year(self) -> int:
        return self.start.year


class EventsConfig(BaseModel):
    events: list[Event]


# --------------------------------------------------------------------------- loaders


@lru_cache(maxsize=8)
def load_aoi(path: str | None = None) -> AOIConfig:
    return AOIConfig.model_validate(load_yaml(path or "aoi"))


@lru_cache(maxsize=8)
def load_gauges(path: str | None = None) -> GaugesConfig:
    return GaugesConfig.model_validate(load_yaml(path or "gauges"))


@lru_cache(maxsize=8)
def load_events(path: str | None = None) -> EventsConfig:
    return EventsConfig.model_validate(load_yaml(path or "events"))
