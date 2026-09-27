"""Baseline 4: XGBoost on the same inputs as Model 2, one pixel at a time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sailab.cube import Datacube
from sailab.forecast.baselines import XGB_FEATURES, LevelBaselines, xgb_pixel_features
from sailab.forecast.inputs import SeriesBank, StateComposer, StaticMaps, forecast_pairs, target_arrays


def training_table(cube: Datacube, series: SeriesBank, level: LevelBaselines, pairs: pd.DataFrame,
                   pixels_per_pair: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sampled pixels from training pairs; half of them from pixels that changed state."""
    rng = np.random.default_rng(seed)
    composer = StateComposer(cube)
    static = StaticMaps(cube)
    xs, ys = [], []
    for _, pr in pairs.iterrows():
        t = pr["issue"]
        state = composer.state(t)
        if state is None:
            continue
        maps = static.map_stack(state, t)
        target, mask = target_arrays(cube.read_label(pr["target_scene"]), static)
        feats = xgb_pixel_features(maps, series, t, float(pr["lead_days"]), level.ref, level.coarsen)
        m = mask.ravel()
        changed = m & ((target.ravel() > 0) != state.flooded.ravel())
        pool_change = np.flatnonzero(changed)
        pool_other = np.flatnonzero(m & ~changed)
        n_change = min(pool_change.size, pixels_per_pair // 2)
        pick = np.concatenate([
            rng.choice(pool_change, n_change, replace=False) if n_change else np.array([], dtype=int),
            rng.choice(pool_other, min(pool_other.size, pixels_per_pair - n_change), replace=False),
        ])
        xs.append(feats[pick])
        ys.append(target.ravel()[pick])
    return np.concatenate(xs), np.concatenate(ys)


def train_xgb(cube: Datacube, series: SeriesBank, level: LevelBaselines, out: Path, max_pairs: int = 600,
              seed: int = 0, log=print):
    import xgboost as xgb

    pairs = forecast_pairs(cube, ["train"])
    pairs = pairs.iloc[:: max(1, len(pairs) // max_pairs)]
    x, y = training_table(cube, series, level, pairs, seed=seed)
    log(f"XGBoost: {len(y):,} pixels from {len(pairs)} training pairs, {y.mean():.1%} flooded")
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=seed)
    model.fit(x, y)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out)
    return model


def load_xgb(path: Path):
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def predict_xgb(model, maps: np.ndarray, series: SeriesBank, issue: pd.Timestamp, lead_days: float,
                level: LevelBaselines) -> np.ndarray:
    feats = xgb_pixel_features(maps, series, issue, lead_days, level.ref, level.coarsen)
    return model.predict_proba(feats)[:, 1].reshape(maps.shape[1:]).astype(np.float32)


__all__ = ["XGB_FEATURES", "load_xgb", "predict_xgb", "train_xgb", "training_table"]
