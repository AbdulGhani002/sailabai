"""Model 1, step 1: an Otsu threshold. No training.

Water is dark in radar, so a cut-off on VV backscatter already finds much of it. A single cut-off
over a whole scene fails when water is a small share of the pixels (the histogram is not
bimodal), so we use the split-based variant: pick tiles whose histograms are clearly bimodal,
threshold each, and average. Terrain rules then remove impossible water (high above the river), and
comparing with the dry-season image removes things that are always dark, like desert sand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu

from sailab import labels as L


@dataclass
class OtsuResult:
    label: np.ndarray        # uint8 label map
    flood_prob: np.ndarray   # soft score in [0, 1], for scoring alongside the models
    threshold_db: float
    tiles_used: int


def ashman_d(values: np.ndarray, threshold: float) -> float:
    """Separation of the two classes either side of a threshold; > 2 means clearly bimodal."""
    lo, hi = values[values < threshold], values[values >= threshold]
    if lo.size < 20 or hi.size < 20:
        return 0.0
    return float(np.sqrt(2) * abs(lo.mean() - hi.mean()) / np.sqrt(lo.var() + hi.var() + 1e-9))


def split_based_threshold(vv_db: np.ndarray, tile: int = 32, min_valid: float = 0.8, min_d: float = 2.0,
                          first_guess_db: float = -15.0, min_gap_db: float = 3.0,
                          fallback: float = -16.0) -> tuple[float, int]:
    """Median Otsu threshold over tiles that hold both water and land (split-based thresholding).

    A tile qualifies when 10-90% of its pixels are darker than a first-guess water threshold and the
    two classes Otsu finds are clearly apart. The Ashman D test alone is not enough: splitting a
    single-mode tile of pure land at its mean already gives D of about 2.7.
    """
    h, w = vv_db.shape
    thresholds = []
    for r in range(0, h - tile + 1, tile):
        for c in range(0, w - tile + 1, tile):
            block = vv_db[r:r + tile, c:c + tile]
            v = block[np.isfinite(block)]
            if v.size < min_valid * tile * tile:
                continue
            dark_share = float((v < first_guess_db).mean())
            if not 0.1 <= dark_share <= 0.9:
                continue
            t = float(threshold_otsu(v))
            lo, hi = v[v < t], v[v >= t]
            if lo.size < 20 or hi.size < 20 or hi.mean() - lo.mean() < min_gap_db:
                continue
            if ashman_d(v, t) >= min_d:
                thresholds.append(t)
    if len(thresholds) < 3:
        v = vv_db[np.isfinite(vv_db)]
        return (float(threshold_otsu(v)) if v.size > 100 else fallback), 0
    return float(np.median(thresholds)), len(thresholds)


def otsu_flood_map(post_db: np.ndarray, normal_water: np.ndarray, hand: np.ndarray | None = None,
                   pre_db: np.ndarray | None = None, max_hand_m: float = 15.0, change_db: float = -3.0,
                   tile: int = 32) -> OtsuResult:
    """Label map (land / normal water / flood / ignore) from one radar image."""
    vv = post_db[0]
    valid = np.isfinite(vv)
    threshold, used = split_based_threshold(vv, tile=tile)
    water = valid & (vv < threshold)
    if hand is not None:
        water &= hand <= max_hand_m
    if pre_db is not None and np.isfinite(pre_db[0]).any():
        pre_vv = pre_db[0]
        was_dark = np.isfinite(pre_vv) & (pre_vv < threshold)
        # always-dark surfaces (sand, tarmac) are not new water unless they got darker still
        water &= ~was_dark | ((vv - pre_vv) < change_db)
    normal = normal_water.astype(bool)
    label = L.compose(water | normal, normal, invalid=~valid)
    score = 1.0 / (1.0 + np.exp((np.nan_to_num(vv, nan=0.0) - threshold) / 1.5))
    if hand is not None:
        score = np.where(hand > max_hand_m, 0.0, score)
    score = np.where(normal | ~valid, 0.0, score).astype(np.float32)
    return OtsuResult(label, score, threshold, used)
