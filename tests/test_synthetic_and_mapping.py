from __future__ import annotations

import numpy as np
import pytest

from sailab import labels as L
from sailab.evaluation.metrics import Confusion
from sailab.mapping.otsu import otsu_flood_map, split_based_threshold


def test_tiny_cube_layout(tiny_cube):
    c = tiny_cube
    assert c.is_synthetic
    for layer in ("dem", "hand", "slope", "normal_water", "landcover", "population", "buildings", "road_km", "aoi_mask"):
        assert c.has_static(layer), layer
    s = c.scenes
    assert set(s["split"].dropna()) == {"train", "val", "test"}
    assert s["time"].dt.tz is not None
    sid = s.iloc[0]["scene_id"]
    sar = c.read_sar(sid)
    assert sar.shape == (2, *c.grid.shape)
    label = c.read_label(sid)
    assert set(np.unique(label)) <= {L.LAND, L.NORMAL_WATER, L.FLOOD, L.IGNORE}
    assert c.has_truth(sid)
    assert set(c.discharge.columns) >= {"marala", "qadirabad", "trimmu", "sidhnai", "islam", "panjnad"}
    gl = c.glofas
    assert gl["member"].nunique() == 51 and gl["lead_day"].max() == 7


def test_named_peaks_follow_the_ffc_numbers(tiny_cube):
    truth = tiny_cube.table("series", "truth_discharge")
    truth["year"] = truth["date"].dt.year
    peak = truth.groupby(["year", "point_id"])["q_cusecs"].max()
    assert peak[(2025, "qadirabad")] == pytest.approx(1_077_951, rel=0.03)
    assert peak[(2023, "ganda_singh_wala")] == pytest.approx(278_297, rel=0.03)
    assert peak[(2016, "khanki")] == pytest.approx(418_736, rel=0.03)


def test_india_data_stops_in_2025(tiny_cube):
    d = tiny_cube.discharge
    assert d.loc["2025-06-01":, "akhnoor"].isna().all()
    assert d.loc["2016-07-01":"2016-09-30", "akhnoor"].notna().mean() > 0.8


def test_the_2025_flood_is_the_biggest(tiny_cube):
    s = tiny_cube.scenes
    by_event = s.groupby("event_id")["flooded_km2"].max()
    assert by_event["flood-2025"] > by_event["monsoon-2016"]


def test_otsu_finds_water_in_a_bimodal_image():
    rng = np.random.default_rng(0)
    vv = rng.normal(-9, 1.2, (128, 128))
    vv[40:90, 30:100] = rng.normal(-21, 1.2, (50, 70))
    t, used = split_based_threshold(vv, tile=32)
    assert -19 < t < -11 and used > 0


def test_otsu_on_a_synthetic_scene(tiny_cube):
    c = tiny_cube
    s = c.scenes.sort_values("flooded_km2").iloc[-1]
    sar = c.read_sar(s["scene_id"])
    pre = c.read_pre_sar(s["pre_key"])
    res = otsu_flood_map(sar, c.static("normal_water").astype(bool), c.static("hand"), pre, tile=8)
    truth = c.read_label(s["scene_id"], truth=True)
    valid = (truth != L.IGNORE) & (truth != L.NORMAL_WATER)
    conf = Confusion.from_arrays(res.label == L.FLOOD, truth == L.FLOOD, valid)
    assert conf.recall > 0.3  # the coarse 3 km test grid is harsh; the demo grid does far better


def test_unet_matches_doc_size():
    torch = pytest.importorskip("torch")
    from sailab.nn.unet import ResNetUNet, count_parameters, pad_to_multiple

    m = ResNetUNet(7, 3, encoder="resnet34")
    assert 23e6 < count_parameters(m) < 26e6  # "24 million parameters"
    x, (h, w) = pad_to_multiple(torch.zeros(1, 7, 100, 70))
    assert m(x).shape == (1, 3, 128, 96) and (h, w) == (100, 70)
