"""Database connection. Set SAILAB_DATABASE_URL for PostGIS, e.g.

    postgresql+psycopg://sailab:sailab@localhost:5432/sailab

Without it, a SQLite file in the data directory is used.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from sailab.db.models import SPATIAL_COLUMNS, Base
from sailab.paths import data_dir


def database_url() -> str:
    url = os.environ.get("SAILAB_DATABASE_URL")
    if url:
        return url
    path = data_dir() / "sailab.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


@lru_cache(maxsize=4)
def engine(url: str | None = None) -> Engine:
    url = url or database_url()
    eng = create_engine(url, pool_pre_ping=True, future=True)
    if eng.dialect.name == "sqlite":
        @event.listens_for(eng, "connect")
        def _fk(dbapi_conn, _):  # enforce ON DELETE CASCADE in SQLite
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return eng


def init_db(eng: Engine | None = None) -> Engine:
    """Create tables; on PostgreSQL also enable PostGIS and add indexed geometry columns."""
    eng = eng or engine()
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            for table, column in SPATIAL_COLUMNS.items():
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS geom geometry(Geometry, 4326) "
                    f"GENERATED ALWAYS AS (ST_SetSRID(ST_GeomFromGeoJSON({column}), 4326)) STORED"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {table}_geom_idx ON {table} USING GIST (geom)"))
    return eng


def session(eng: Engine | None = None) -> Session:
    return sessionmaker(bind=eng or engine(), expire_on_commit=False)()
