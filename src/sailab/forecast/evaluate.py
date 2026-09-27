"""Score Model 2 and the four baselines on an event split, the way the team agreed.

    sailab evaluate forecast --cube demo --split val
    sailab evaluate forecast --cube demo --split test   # once, guarded by the test lock

Every model is scored on the same (issue day, target pass) pairs; results go to the shared
results table next to persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sailab.cube import Datacube
from sailab.evaluation.protocol import EventScorer
from sailab.evaluation.results import ResultsTable
from sailab.events import TestLock
from sailab.forecast.baselines import LevelBaselines, persistence
from sailab.forecast.calibration import CalibrationFitter
from sailab.forecast.inputs import SeriesBank, StateComposer, StaticMaps, forecast_pairs, target_arrays
from sailab.forecast.model import ForecastEnsemble
from sailab.forecast.xgb import predict_xgb
from sailab.versioning import DataVersions

BASELINE = "persistence"


@dataclass
class ForecastEvaluation:
    scorers: dict[str, EventScorer]
    pairs: pd.DataFrame

    def macro(self, model: str, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
        return self.scorers[model].macro(thresholds)

    def tuned_thresholds(self) -> dict[str, dict[str, float]]:
        """Per model and lead, the F1-best threshold on this split (use only on validation)."""
        return {name: sc.best_thresholds() for name, sc in self.scorers.items()}

    def reliability(self, bins: int = 10) -> dict[str, dict[str, list[dict[str, float]]]]:
        """Pooled reliability curves per model: 'all' leads and each lead bucket."""
        out: dict[str, dict[str, list[dict[str, float]]]] = {}
        for name, sc in self.scorers.items():
            leads = sorted({lead for (_, lead, _) in sc._cells})
            curves = {}
            for lead in [None, *leads]:
                rel = sc.pooled(lead).reliability(bins)
                curves["all" if lead is None else lead] = [
                    {"mean_prob": round(float(m), 4), "observed": round(float(o), 4), "count": int(n)}
                    for m, o, n in zip(rel.mean_prob, rel.observed_freq, rel.counts, strict=True) if n > 0]
            out[name] = curves
        return out

    def summary(self, thresholds: dict[str, dict[str, float]] | None = None,
                metrics: tuple[str, ...] = ("brier", "iou", "f1", "ece")) -> pd.DataFrame:
        rows = []
        for name, sc in self.scorers.items():
            m = sc.macro((thresholds or {}).get(name))
            m = m[(m["subset"] == "all") & m["metric"].isin(metrics)]
            for _, r in m.iterrows():
                rows.append({"model": name, "lead": r["lead"], "metric": r["metric"], "value": r["value"]})
        return pd.DataFrame(rows).pivot_table(index=["lead", "model"], columns="metric", values="value")


def evaluate_forecasts(cube: Datacube, split: str, ensemble: ForecastEnsemble | None,
                       level: LevelBaselines | None = None, xgb_model=None, max_pairs: int | None = None,
                       calibrated: bool = True, truth: bool = False,
                       calibration_fitter: CalibrationFitter | None = None, log=print) -> ForecastEvaluation:
    """Score every available model on the split. `truth=True` scores against the error-free labels
    (hand labels on real data) instead of GFM labels."""
    points = ensemble.points if ensemble else (level.points if level else [])
    series = SeriesBank(cube, points)
    composer = StateComposer(cube)
    static = StaticMaps(cube, points)
    pairs = forecast_pairs(cube, [split])
    pairs = pairs[[composer.latest_index(t) is not None for t in pairs["issue"]]].reset_index(drop=True)
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs.iloc[np.linspace(0, len(pairs) - 1, max_pairs).astype(int)].reset_index(drop=True)
    names = [BASELINE] + (["historical_frequency", "river_threshold"] if level else []) + \
            (["xgboost"] if xgb_model is not None else []) + (["unet_tt"] if ensemble else [])
    scorers = {n: EventScorer() for n in names}
    buildings = cube.static("buildings") if cube.has_static("buildings") else None
    population = cube.static("population") if cube.has_static("population") else None
    log(f"scoring {len(pairs)} (issue, pass) pairs on split '{split}' for {', '.join(names)}")

    for i, pr in pairs.iterrows():
        t, lead = pr["issue"], float(pr["lead_days"])
        state = composer.state(t)
        label = cube.read_label(pr["target_scene"], truth=truth)
        maps = static.map_stack(state, t, flows=series.flow_features(t, lead) if points else None)
        breach = maps[7] > 0
        common = {"prev_flood": state.flooded if state else None, "prev_known": state.known if state else None,
                  "normal_water": static.normal_water, "extra_valid": static.aoi, "lead_days": lead,
                  "extra_subsets": {"near_breach": breach} if breach.any() else None}
        preds: dict[str, np.ndarray] = {BASELINE: persistence(state, label.shape)}
        if level is not None:
            flows = level.flows_at(series, t, lead)
            preds["historical_frequency"] = level.predict("historical_frequency", flows)
            preds["river_threshold"] = level.predict("river_threshold", flows)
        if xgb_model is not None and level is not None:
            preds["xgboost"] = predict_xgb(xgb_model, maps, series, t, lead, level)
        if ensemble is not None:
            values, missing, is_fc = series.tokens(t)
            out = ensemble.predict(maps, values, missing, is_fc, [lead], calibrated=calibrated)
            preds["unet_tt"] = out["prob"][0]
            if calibration_fitter is not None:
                target, mask = target_arrays(label, static)
                weights = {k: v for k, v in (("buildings", buildings), ("population", population)) if v is not None}
                calibration_fitter.add(preds["unet_tt"], target, mask, lead, weights)
        for name, prob in preds.items():
            scorers[name].add(pr["event_id"], prob, label, **common)
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(pairs)} pairs")
    return ForecastEvaluation(scorers, pairs)


def record_results(ev: ForecastEvaluation, experiment: str, split: str, cube: Datacube, notes: str = "",
                   thresholds: dict[str, dict[str, float]] | None = None, table: ResultsTable | None = None) -> None:
    """Write every model's macro scores to the results table next to persistence."""
    table = table or ResultsTable()
    thresholds = thresholds or {}
    base = ev.scorers[BASELINE].macro()
    versions = DataVersions.from_dict(cube.meta.get("versions"))
    for name, sc in ev.scorers.items():
        if name == BASELINE:
            table.append(sc.macro(), None, experiment=experiment, model=name, baseline="none", split=split,
                         data_versions=versions, notes=f"reference forecast (repeat the last map). {notes}".strip())
            continue
        table.append(sc.macro(thresholds.get(name)), base, experiment=experiment, model=name, baseline=BASELINE,
                     split=split, data_versions=versions, notes=notes)
    table.render_markdown()


def save_thresholds(path: Path, thresholds: dict[str, dict[str, float]], split: str) -> None:
    path.write_text(json.dumps({"tuned_on": split, "metric": "f1", "thresholds": thresholds}, indent=2),
                    encoding="utf-8")


def load_thresholds(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["thresholds"]


def guard_test(split: str, cube: Datacube, experiment: str, force: bool = False, reason: str = "") -> None:
    """The final test event is scored once per datacube; this records (or refuses) the attempt."""
    if split != "test":
        return
    cube_name = f"{cube.kind}:{cube.meta.get('name', cube.root.name)}"
    for event in sorted(cube.scenes.loc[cube.scenes["split"] == "test", "event_id"].dropna().unique()):
        TestLock().acquire(event, experiment, cube=cube_name, force=force, reason=reason)


def load_level_baselines(folder: Path) -> LevelBaselines | None:
    path = folder / "level_baselines.npz"
    return LevelBaselines.load(path) if path.exists() else None


__all__ = ["ForecastEvaluation", "evaluate_forecasts", "guard_test", "load_level_baselines", "load_thresholds",
           "record_results", "save_thresholds"]
