from __future__ import annotations

import numpy as np
import pytest

from sailab.config import Split, load_aoi, load_events, load_gauges
from sailab.events import EventRegistry, LeakageError, TestAlreadyUsedError, TestLock, assert_no_leakage
from sailab.grid import GridSpec
from sailab.licences import LicenceError, LicenceRegister
from sailab.versioning import DataVersions, gfm_version_from_filename, mixed_versions


def test_configs_load():
    aoi = load_aoi()
    assert aoi.bbox.as_tuple() == (70.9, 29.2, 72.7, 31.4)
    gauges = load_gauges()
    ids = gauges.model_input_ids()
    assert ids == ["marala", "qadirabad", "trimmu", "sidhnai", "islam", "panjnad"]
    assert "akhnoor" in gauges.model_input_ids(include_india=True)
    assert gauges.by_id()["qadirabad"].design_capacity_cusecs == 807000


def test_production_grid_matches_doc_size():
    grid = GridSpec.for_aoi(load_aoi())
    # the team guide quotes about 8,650 x 12,200 pixels at 20 m
    assert grid.res == 20
    assert 8_500 < grid.width < 9_000
    assert 12_000 < grid.height < 12_500


def test_grid_roundtrip_and_windows():
    grid = GridSpec.for_aoi(load_aoi(), 2000)
    rows, cols = grid.lonlat_to_rowcol(71.5249, 30.1575)  # Multan
    lon, lat = grid.rowcol_to_lonlat(rows, cols)
    assert abs(lon - 71.5249) < 0.03 and abs(lat - 30.1575) < 0.03
    wins = list(grid.windows(16))
    assert sum(int(w.width * w.height) for w in wins) == grid.width * grid.height
    assert GridSpec.from_dict(grid.to_dict()) == grid
    mask = grid.aoi_mask(load_aoi().bbox)
    assert 0.8 < mask.mean() < 1.0


def test_event_registry_and_splits():
    reg = EventRegistry.load()
    assert reg.split_for("2025-08-28") == Split.TEST
    assert reg.split_for("2023-08-15") == Split.VAL
    assert reg.split_for("2019-08-01") == Split.TRAIN
    assert reg.split_for("2019-12-01") is None
    assert reg.ids_in("val") == ["sutlej-2023"]
    assert {e.split for e in load_events().events} >= {Split.TRAIN, Split.VAL, Split.TEST, Split.REPLAY}


def test_leakage_guard():
    assert_no_leakage(["monsoon-2016"], ["sutlej-2023"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["monsoon-2016", "sutlej-2023"], ["sutlej-2023"])


def test_test_lock_is_one_shot_per_cube(tmp_path):
    lock = TestLock(tmp_path / "lock.json")
    lock.acquire("flood-2025", "exp-a", cube="synthetic:demo")
    with pytest.raises(TestAlreadyUsedError):
        lock.acquire("flood-2025", "exp-a", cube="synthetic:demo")  # same model, second look
    rec = lock.acquire("flood-2025", "exp-b", cube="synthetic:demo")  # another model gets its own shot
    assert rec["attempt"] == 1 and rec["touches_of_event"] == 2
    with pytest.raises(TestAlreadyUsedError):
        lock.acquire("flood-2025", "exp-a", cube="synthetic:demo", force=True)  # no reason given
    rec = lock.acquire("flood-2025", "exp-a", cube="synthetic:demo", force=True, reason="bug in scoring fixed")
    assert rec["attempt"] == 2
    lock.acquire("flood-2025", "exp-a", cube="real:real")  # a different cube has its own lock
    assert len(lock.runs("flood-2025")) == 4


def test_versions():
    v = DataVersions(gfm="V0M2R2", imerg="V07", extra={"generator": "x"})
    assert DataVersions.from_dict(v.to_dict()) == v
    assert v.key() == "generator=x;gfm=V0M2R2;imerg=V07"
    assert gfm_version_from_filename("ADVFLAG_20250904T133651__VV_A042_E018N030T3_EQUI7_AS020M_V0M2R2_S1.tif") == "V0M2R2"
    mixed = mixed_versions([DataVersions(gfm="V0M2R1"), DataVersions(gfm="V0M2R2"), DataVersions(gfm="V0M2R2")])
    assert mixed["gfm"]["V0M2R2"] == 2


def test_licence_register():
    reg = LicenceRegister.load()
    assert reg.problems() == []
    reg.check_use(["sentinel1", "gfm", "osm_roads"], "product")
    with pytest.raises(LicenceError):
        reg.check_use(["fabdem"], "product")       # rule 4: tests only
    with pytest.raises(LicenceError):
        reg.check_use(["ffd_bulletins"], "product")  # not yet cleared with FFD
    assert "osm_roads" in reg.share_alike_layers()   # rule 5: separate layer


def test_licence_mark_downloaded(tmp_path):
    src = LicenceRegister.load().path.read_text(encoding="utf-8")
    p = tmp_path / "licences.yaml"
    p.write_text(src, encoding="utf-8")
    reg = LicenceRegister.load(p)
    reg.mark_downloaded("gfm", "V0M2R2")
    again = LicenceRegister.load(p)
    assert again["gfm"].version == "V0M2R2" and again["gfm"].download_date is not None
    assert p.read_text(encoding="utf-8").startswith("# Licence register")
    assert np.isclose(len(again.entries), len(reg.entries))
