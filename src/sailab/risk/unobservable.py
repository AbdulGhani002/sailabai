"""Forecasts where radar cannot see.

GFM (and so every label we train on) leaves out towns, where flood water shows as bright double
bounce, and bare sand, which is as dark as water. Model 2 therefore never learns anything there and
its raw output in those pixels is noise. Towns matter most for exposure, so instead of dropping them
we give every such pixel the forecast of the nearest pixel radar can judge: towns up to 8 km away (a
town beside a flooding floodplain carries its risk), sand up to 2 km (sand bars in the river bed
flood with the river, desert interiors do not). Anything further gets the base rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sailab.cube import Datacube

BUILT, BARE = 50, 60
DEFAULT_REACH_KM = {BUILT: 8.0, BARE: 2.0}


@dataclass
class UnobservableFill:
    excluded: np.ndarray        # bool: pixels radar cannot judge
    source_rows: np.ndarray     # nearest observable pixel for every pixel
    source_cols: np.ndarray
    use_neighbour: np.ndarray   # bool: excluded and close enough to borrow a neighbour's value
    base_rate: float = 0.001

    @classmethod
    def from_cube(cls, cube: Datacube, reach_km: dict[int, float] | None = None,
                  default_km: float = 2.0, base_rate: float = 0.001) -> UnobservableFill:
        excluded = cube.exclusion()
        landcover = cube.static("landcover")
        dist, (rows, cols) = ndimage.distance_transform_edt(excluded, return_indices=True)
        dist_km = dist * cube.grid.res / 1000.0
        limit = np.full(excluded.shape, default_km, dtype=np.float32)
        for code, km in (reach_km or DEFAULT_REACH_KM).items():
            limit[landcover == code] = km
        return cls(excluded, rows, cols, excluded & (dist_km <= limit), base_rate)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Fill excluded pixels of a (..., H, W) array of flood chances."""
        out = np.array(arr, dtype=np.float32, copy=True)
        borrowed = out[..., self.source_rows, self.source_cols]
        out = np.where(self.use_neighbour, borrowed, out)
        return np.where(self.excluded & ~self.use_neighbour, self.base_rate, out).astype(np.float32)
