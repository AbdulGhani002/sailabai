"""Input normalisation shared by Model 1, Model 2 and the XGBoost baseline.

Keeping it in one place means a model trained today and the twin running next monsoon see
identical inputs.
"""

from __future__ import annotations

import numpy as np

from sailab import labels as L


def norm_sar(db: np.ndarray) -> np.ndarray:
    """Backscatter in dB to roughly [-1, 1]; missing pixels become 0."""
    out = (db + 15.0) / 7.0
    return np.nan_to_num(np.clip(out, -3, 3), nan=0.0).astype(np.float32)


def norm_hand(hand: np.ndarray) -> np.ndarray:
    return (np.log1p(np.clip(np.nan_to_num(hand, nan=50.0), 0, 500)) / 2.0).astype(np.float32)


def norm_dem(dem: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.nan_to_num((dem - mean) / max(std, 1e-3), nan=0.0).astype(np.float32)


def norm_slope(slope_deg: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(slope_deg, nan=0.0) / 2.0, 0, 5).astype(np.float32)


def norm_flow(q_cusecs: np.ndarray) -> np.ndarray:
    """Log flow centred near 60,000 cusecs."""
    return ((np.log(np.maximum(q_cusecs, 100.0)) - 11.0) / 1.5).astype(np.float32)


def norm_rain(mm: np.ndarray) -> np.ndarray:
    return (np.log1p(np.maximum(mm, 0.0)) / 3.0).astype(np.float32)


def flood_state(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flooded, known) from a label map."""
    return (label == L.FLOOD), (label != L.IGNORE)


MAPPING_CHANNELS = ["post_vv", "post_vh", "pre_vv", "pre_vh", "dem", "hand", "slope"]


def mapping_inputs(post_db: np.ndarray, pre_db: np.ndarray | None, dem: np.ndarray, hand: np.ndarray,
                   slope: np.ndarray, dem_stats: tuple[float, float]) -> np.ndarray:
    """Model 1 input stack (7, H, W): radar after and before the flood, then terrain."""
    pre = pre_db if pre_db is not None else np.full_like(post_db, np.nan)
    return np.concatenate([
        norm_sar(post_db), norm_sar(pre),
        norm_dem(dem, *dem_stats)[None], norm_hand(hand)[None], norm_slope(slope)[None],
    ]).astype(np.float32)


def soil_moisture_index(rain_mm: np.ndarray, k: float = 0.92) -> np.ndarray:
    """Antecedent precipitation index scaled to 0-1."""
    api = np.zeros_like(rain_mm, dtype=np.float64)
    for i, r in enumerate(rain_mm):
        api[i] = (api[i - 1] * k if i else 0.0) + r
    return np.clip(api / 150.0, 0, 1)
