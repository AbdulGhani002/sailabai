"""Shared fixtures: a tiny synthetic datacube (3 km pixels, three seasons) and isolated data folders."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TINY_YEARS = [2016, 2023, 2025]  # one train, the validation and the test event


@pytest.fixture(scope="session")
def sandbox(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sailab")
    for var, sub in [("SAILAB_DATA_DIR", "data"), ("SAILAB_RUNS_DIR", "runs"), ("SAILAB_RESULTS_DIR", "results")]:
        os.environ[var] = str(root / sub)
    os.environ.pop("SAILAB_DATABASE_URL", None)
    return root


@pytest.fixture(scope="session")
def tiny_cube(sandbox):
    from sailab.synthetic.generate import generate_synthetic_cube

    return generate_synthetic_cube("tiny", resolution_m=3000, years=TINY_YEARS, seed=3, log=lambda *_: None)
