"""Label encoding shared by the data pipeline, both models and the evaluation code."""

from __future__ import annotations

import numpy as np

LAND = 0           # dry land
NORMAL_WATER = 1   # rivers, lakes and canals that are wet in normal conditions
FLOOD = 2          # water that is not normally there
IGNORE = 255       # unsure pixels: GFM exclusion mask, outside the radar swath, no data

CLASS_NAMES = {LAND: "land", NORMAL_WATER: "normal water", FLOOD: "flood"}
NUM_CLASSES = 3


def flood_mask(label: np.ndarray) -> np.ndarray:
    return label == FLOOD


def water_mask(label: np.ndarray) -> np.ndarray:
    return (label == FLOOD) | (label == NORMAL_WATER)


def known_mask(label: np.ndarray) -> np.ndarray:
    return label != IGNORE


def compose(water: np.ndarray, normal_water: np.ndarray, invalid: np.ndarray | None = None) -> np.ndarray:
    """Build a label map from an observed-water mask and a normal-water mask."""
    out = np.full(water.shape, LAND, dtype=np.uint8)
    out[water & ~normal_water] = FLOOD
    out[normal_water] = NORMAL_WATER
    if invalid is not None:
        out[invalid] = IGNORE
    return out
