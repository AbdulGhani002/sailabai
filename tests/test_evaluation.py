from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sailab import labels as L
from sailab.evaluation.metrics import (
    Confusion,
    ProbAccumulator,
    brier,
    expected_calibration_error,
    skill_score,
)
from sailab.evaluation.protocol import EventScorer, compare_to_baseline, lead_bucket
from sailab.evaluation.results import ResultsTable


def test_confusion_scores():
    pred = np.array([1, 1, 0, 0, 1], bool)
    truth = np.array([1, 0, 1, 0, 1], bool)
    c = Confusion.from_arrays(pred, truth)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert c.iou == pytest.approx(0.5)
    assert c.f1 == pytest.approx(2 / 3)
    valid = np.array([1, 1, 1, 1, 0], bool)
    assert Confusion.from_arrays(pred, truth, valid).tp == 1


def test_brier_and_skill():
    truth = np.array([1, 0, 1, 0])
    assert brier(np.array([1.0, 0.0, 1.0, 0.0]), truth) == 0
    assert brier(np.full(4, 0.5), truth) == pytest.approx(0.25)
    assert skill_score(0.1, 0.2, perfect=0.0) == pytest.approx(0.5)   # Brier skill
    assert skill_score(0.6, 0.5, perfect=1.0) == pytest.approx(0.2)   # IoU skill over persistence


def test_perfectly_calibrated_forecast_has_small_ece():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = rng.random(200_000) < p
    assert expected_calibration_error(p, y) < 0.01
    # an overconfident forecast is badly calibrated
    assert expected_calibration_error(np.clip(p * 2 - 0.5, 0, 1), y) > 0.1


def test_prob_accumulator_merges():
    rng = np.random.default_rng(1)
    p, y = rng.random(1000), rng.random(1000) < 0.3
    a, b = ProbAccumulator(), ProbAccumulator()
    a.update(p[:400], y[:400])
    b.update(p[400:], y[400:])
    whole = ProbAccumulator()
    whole.update(p, y)
    assert a.merge(b).brier == pytest.approx(whole.brier)


def test_scoring_rules_remove_normal_water_and_split_changes():
    # 1 x 6 strip: normal water, stays flooded, newly flooded, drains, stays dry, unknown
    label = np.array([[L.NORMAL_WATER, L.FLOOD, L.FLOOD, L.LAND, L.LAND, L.IGNORE]], np.uint8)
    prev = np.array([[0, 1, 0, 1, 0, 0]], bool)
    normal = np.array([[1, 0, 0, 0, 0, 0]], bool)
    persistence = np.where(prev, 1.0, 0.0)
    sc = EventScorer()
    sc.add("e1", persistence, label, prev_flood=prev, normal_water=normal, lead_days=2.2)
    per = sc.per_event().set_index(["subset", "metric"])["value"]
    assert per[("all", "iou")] == pytest.approx(1 / 3)          # 1 hit, 1 miss, 1 false alarm
    assert per[("newly_flooded", "recall")] == pytest.approx(0)  # persistence never predicts change
    assert per[("drained", "recall")] == pytest.approx(0)
    assert lead_bucket(2.2) == "+2d" and lead_bucket(None) == "map"


def test_macro_average_weights_events_equally():
    sc = EventScorer()
    big = np.ones((100, 100), np.uint8) * L.FLOOD
    sc.add("big", np.ones((100, 100)), big)                        # perfect on a big event
    small = np.array([[L.FLOOD, L.LAND]], np.uint8)
    sc.add("small", np.array([[0.0, 1.0]]), small)                 # all wrong on a small one
    m = sc.macro().set_index(["subset", "metric"])["value"]
    assert m[("all", "iou")] == pytest.approx(0.5)                 # not ~1.0 as pooling would say


def test_results_table_requires_baseline(tmp_path):
    table = ResultsTable(tmp_path / "results.csv")
    model = pd.DataFrame({"lead": ["+1d"], "subset": ["all"], "metric": ["iou"], "value": [0.6], "n_events": [2], "n_pixels": [10]})
    base = model.assign(value=[0.5])
    with pytest.raises(ValueError):
        table.append(model, None, experiment="x", model="m", baseline="persistence", split="val")
    rows = table.append(model, base, experiment="x", model="m", baseline="persistence", split="val")
    assert rows.iloc[0]["skill"] == pytest.approx(0.2)
    text = table.render_markdown()
    assert "x: m vs persistence" in text
    assert "0.600 / 0.500 / 0.200" in text
    merged = compare_to_baseline(model, base)
    assert merged["skill"].iloc[0] == pytest.approx(0.2)
