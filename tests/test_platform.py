"""The twin loop, the API (with real tile rendering) and the hand-label tooling, on tiny cubes."""

from __future__ import annotations

import numpy as np
import pytest

from sailab import labels as L
from sailab.paths import twin_dir


@pytest.fixture(scope="module")
def twin_run(tiny_cube):
    from sailab.twin.loop import Twin, TwinConfig

    return Twin(TwinConfig(cube="tiny", members=0), log=lambda *_: None).step("2025-08-28")


def test_twin_step_writes_layers_and_risk(twin_run):
    s = twin_run
    folder = twin_dir() / "tiny" / "runs" / "2025-08-28"
    assert (folder / "current.tif").exists() and (folder / "prob_d1.tif").exists()
    names = [h["name"] for h in s["horizons"]]
    assert names[-5:] == ["d1", "d2", "d3", "d5", "d7"]
    h = s["horizons"][-1]
    assert h["totals"]["population"]["low"] <= h["totals"]["population"]["expected"] <= h["totals"]["population"]["high"]
    assert s["current_exposure"]["totals"]["area_km2"]["expected"] >= 0
    assert "not an official flood warning" in s["disclaimer"]


def test_api_serves_runs_tiles_and_hydrographs(twin_run):
    from fastapi.testclient import TestClient

    from sailab.api.app import app

    with TestClient(app) as client:
        meta = client.get("/api/meta", params={"cube": "tiny"}).json()
        assert meta["synthetic"] and meta["gauges"]["features"]
        runs = client.get("/api/runs", params={"cube": "tiny"}).json()
        assert runs[-1]["run_date"] == "2025-08-28"
        run = client.get("/api/runs/tiny/2025-08-28").json()
        assert run["tiles"]["current"].startswith("/tiles/tiny/2025-08-28/current/")
        # z8 tile over Multan
        tile = client.get("/tiles/tiny/2025-08-28/current/8/178/105.png", params={"theme": "dark"})
        assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
        assert client.get("/tiles/tiny/2025-08-28/../../etc/1/1/1.png").status_code in (400, 404)
        hydro = client.get("/api/gauges/trimmu/hydrograph", params={"cube": "tiny", "date": "2025-08-28"}).json()
        assert len(hydro["observed"]) > 20 and len(hydro["forecast"]) == 7
        assert client.get("/api/legend/prob_d1", params={"theme": "light"}).json()[0]["label"] == "5-15%"


def test_hand_label_roundtrip(sandbox, tmp_path):
    from sailab.labeling import export_tiles, import_tiles, pick_tiles
    from sailab.synthetic.generate import generate_synthetic_cube

    cube = generate_synthetic_cube("labels", resolution_m=4000, years=[2025], seed=5, log=lambda *_: None)
    tiles = pick_tiles(cube, "test", n=8, size=8)
    assert len(tiles) and set(tiles["stratum"]) <= {"flood_rich", "flood_edge", "dry_near_river", "random"}
    out = export_tiles(cube, tiles, tmp_path / "hl")
    first = tiles.iloc[0]
    from sailab.io import read_raster, write_raster

    path = out / f"{first['tile_id']}_label.tif"
    edited = read_raster(path)[0]
    edited[:] = L.FLOOD  # the labeller decides the whole tile is flooded
    from rasterio.windows import Window

    from sailab.grid import GridSpec

    sub = GridSpec(cube.grid.crs, cube.grid.window_transform(Window(first["col"], first["row"], 8, 8)), 8, 8)
    write_raster(path, edited, sub, dtype="uint8", nodata=L.IGNORE)
    assert import_tiles(cube, out) == len(tiles)
    truth = cube.read_label(first["scene_id"], truth=True)
    r, c = int(first["row"]), int(first["col"])
    assert (truth[r:r + 8, c:c + 8] == L.FLOOD).all()
    assert np.isin(np.unique(truth), [L.LAND, L.NORMAL_WATER, L.FLOOD, L.IGNORE]).all()
