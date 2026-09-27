"""Where configs, data, runs and results live. Every location can be moved with an env var."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


def config_dir() -> Path:
    return _env_path("SAILAB_CONFIG_DIR", REPO_ROOT / "configs")


def data_dir() -> Path:
    """Root for downloaded and generated data. Point it at the external drive in production."""
    return _env_path("SAILAB_DATA_DIR", REPO_ROOT / "data")


def cubes_dir() -> Path:
    return data_dir() / "cubes"


def archive_dir() -> Path:
    """Daily scraper archive (FFD, IRSA, ECMWF, VIIRS). Some sources delete files within days."""
    return data_dir() / "archive"


def runs_dir() -> Path:
    """Model checkpoints and training logs."""
    return _env_path("SAILAB_RUNS_DIR", REPO_ROOT / "runs")


def results_dir() -> Path:
    """The shared results table and test lock. Committed to git."""
    return _env_path("SAILAB_RESULTS_DIR", REPO_ROOT / "results")


def twin_dir() -> Path:
    """Outputs of the live twin: flood maps, forecasts and risk layers served by the API."""
    return _env_path("SAILAB_TWIN_DIR", data_dir() / "twin")
