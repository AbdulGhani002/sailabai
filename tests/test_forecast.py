from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sailab.forecast.baselines import LevelBaselines, persistence
from sailab.forecast.calibration import Calibration, CalibrationFitter, conformal_quantile, fit_temperature
from sailab.forecast.inputs import (
    HISTORY_DAYS,
    MAP_CHANNELS,
    N_TOKENS,
    SeriesBank,
    StateComposer,
    StaticMaps,
    forecast_pairs,
    issue_time,
)
from sailab.risk.exposure import exposure_summary, total_with_range

POINTS = ["marala", "qadirabad", "trimmu", "sidhnai", "islam", "panjnad"]


def test_series_tokens_respect_timing(tiny_cube):
    bank = SeriesBank(tiny_cube, POINTS)
    values, missing, is_fc = bank.tokens(pd.Timestamp("2025-08-20"))
    assert values.shape == (N_TOKENS, len(POINTS) + 3) and is_fc.sum() == 7
    p = len(POINTS)
    assert missing[HISTORY_DAYS - 1, p:].all()          # today's rain is not in yet
    assert not missing[HISTORY_DAYS:, :p].all()         # GloFAS forecasts present
    member = bank.tokens(pd.Timestamp("2025-08-20"), member=7)[0]
    assert not np.allclose(member[HISTORY_DAYS:, :p], values[HISTORY_DAYS:, :p])


def test_state_composer_only_uses_the_past(tiny_cube):
    comp = StateComposer(tiny_cube)
    t = issue_time("2025-08-28")
    before = comp.scenes_before(t)
    assert (before["time"] <= t).all()
    state = comp.state(t)
    assert state is not None and state.known.any()
    age = state.age_days(t)
    assert (age[state.known] >= 0).all() and (age[state.known] <= 30).all()


def test_map_stack_and_pairs(tiny_cube):
    static = StaticMaps(tiny_cube)
    comp = StateComposer(tiny_cube)
    t = issue_time("2025-08-28")
    maps = static.map_stack(comp.state(t), t)
    assert maps.shape == (len(MAP_CHANNELS), *tiny_cube.grid.shape)
    hidden = static.map_stack(comp.state(t), t, hide_map=True)
    assert hidden[0].sum() == 0 and hidden[-1].min() == 1
    pairs = forecast_pairs(tiny_cube, ["val"])
    assert len(pairs) and pairs["lead_days"].between(0.25, 7.5).all()
    assert set(pairs["event_id"]) == {"sutlej-2023"}


def test_persistence():
    assert persistence(None, (2, 2)).max() < 0.05


def test_level_baselines_fit_and_predict(tiny_cube, tmp_path):
    lb = LevelBaselines.fit(tiny_cube, POINTS)
    bank = SeriesBank(tiny_cube, POINTS)
    flows = lb.flows_at(bank, pd.Timestamp("2025-08-28"), 2)
    for which in ("historical_frequency", "river_threshold"):
        p = lb.predict(which, flows)
        assert p.shape == tiny_cube.grid.shape and p.min() >= 0 and p.max() <= 1
    lb.save(tmp_path / "lb.npz")
    again = LevelBaselines.load(tmp_path / "lb.npz")
    assert np.allclose(again.predict("river_threshold", flows), lb.predict("river_threshold", flows))


def test_temperature_scaling_undoes_overconfidence():
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 2, 100_000)
    y = rng.random(100_000) < 1 / (1 + np.exp(-logit))
    overconfident = 1 / (1 + np.exp(-3 * logit))
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(3.0, rel=0.1)


def test_conformal_sets_cover_the_truth():
    rng = np.random.default_rng(1)
    p = rng.random(50_000)
    y = rng.random(50_000) < p
    fitter = CalibrationFitter(alpha=0.1)
    fitter.add(p, y.astype(float), np.ones_like(p, bool), lead_days=1.0)
    cal = fitter.fit(["val"])
    sets = cal.pixel_sets(p)
    covered = np.where(sets == 2, True, np.where(sets == 1, y, ~y))
    assert covered.mean() >= 0.88
    assert conformal_quantile(np.arange(10.0), 0.1) == 9.0


def test_calibration_roundtrip(tmp_path):
    cal = Calibration(temperatures={"+1d": 1.3}, pixel_qhat=0.7, total_quantiles={"buildings": (-0.3, 0.5)})
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json")
    assert again.temperature(1.2) == 1.3 and again.total_range("buildings", 100) == (70.0, 150.0)


def test_exposure_ranges(tiny_cube):
    prob = np.full(tiny_cube.grid.shape, 0.5, np.float32)
    members = np.stack([prob * 0.5, prob, prob * 1.5])
    pop = tiny_cube.static("population")
    stats = total_with_range(prob, pop, members, Calibration(), "population")
    assert stats["low"] <= stats["expected"] <= stats["high"]
    summary = exposure_summary(prob, members, {"population": pop, "buildings": tiny_cube.static("buildings")},
                               tiny_cube.grid, Calibration(), tiny_cube.geojson("roads"), tiny_cube.geojson("places"))
    assert summary["roads"] and summary["places"]
    assert {"population", "buildings", "area_km2"} <= set(summary["totals"])


def test_unet_tt_shapes():
    torch = pytest.importorskip("torch")
    from sailab.nn.unet_tt import UNetTT, time_encoder_parameters

    m = UNetTT(len(MAP_CHANNELS), 9, N_TOKENS, encoder="resnet18")
    assert 0.6e6 < time_encoder_parameters(m) < 1.0e6  # "about 0.8 million parameters"
    out = m(torch.zeros(2, len(MAP_CHANNELS), 64, 96), torch.zeros(2, N_TOKENS, 9), torch.zeros(2, N_TOKENS, 9),
            torch.zeros(2, N_TOKENS), torch.tensor([1.0, 3.5]))
    assert out.shape == (2, 64, 96)
