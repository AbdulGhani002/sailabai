"""Training chips for Model 2, with modality dropout.

Each input group is hidden 20 to 30% of the time and flagged as missing, so the model learns to
cope when the satellite map is old, a gauge report is late, or upstream data from India is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sailab.cube import Datacube
from sailab.forecast.inputs import (
    HISTORY_DAYS,
    SeriesBank,
    StateComposer,
    StaticMaps,
    forecast_pairs,
    target_arrays,
)


@dataclass
class DropoutConfig:
    map: float = 0.25          # hide the whole flood map (as if no recent pass)
    point: float = 0.2         # hide one river point's observed flow
    india: float = 0.5         # hide the Indian upstream points (besides the real 2025 cut-off)
    rain_obs: float = 0.2      # hide observed rain and soil moisture
    forecast: float = 0.25     # hide all forecasts (GloFAS and ECMWF)
    random_member: float = 0.5  # use one random GloFAS member instead of the ensemble mean


class ForecastChips(Dataset):
    def __init__(self, cube: Datacube, splits: list[str], points: list[str], india_points: tuple[str, ...] = (),
                 chip: int = 128, samples_per_epoch: int = 3000, dropout: DropoutConfig | None = None,
                 augment: bool = True, seed: int = 0, change_share: float = 0.6) -> None:
        self.cube = cube
        self.pairs = forecast_pairs(cube, splits)
        if self.pairs.empty:
            raise ValueError(f"no forecast pairs for splits {splits}")
        self.series = SeriesBank(cube, points)
        self.points = points
        self.india_idx = [points.index(p) for p in india_points if p in points]
        self.composer = StateComposer(cube)
        self.static = StaticMaps(cube)
        self.chip = chip
        self.n = samples_per_epoch
        self.dropout = dropout or DropoutConfig()
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.change_share = change_share
        keep = [self.composer.latest_index(t) is not None for t in self.pairs["issue"]]
        self.pairs = self.pairs[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return self.n

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @lru_cache(maxsize=128)  # noqa: B019
    def _target(self, scene_id: str) -> tuple[np.ndarray, np.ndarray]:
        return target_arrays(self.cube.read_label(scene_id), self.static)

    def _crop_origin(self, rng: np.random.Generator, interest: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
        h, w = mask.shape
        c = self.chip
        if rng.random() < self.change_share:
            ys, xs = np.nonzero(interest)
            if ys.size:
                k = int(rng.integers(ys.size))
                r = int(np.clip(ys[k] - rng.integers(c // 4, 3 * c // 4), 0, h - c))
                q = int(np.clip(xs[k] - rng.integers(c // 4, 3 * c // 4), 0, w - c))
                return r, q
        for _ in range(20):
            r, q = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
            if mask[r:r + c, q:q + c].mean() > 0.3:
                return r, q
        return int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        pair = self.pairs.iloc[int(rng.integers(len(self.pairs)))]
        t = pair["issue"]
        state = self.composer.state(t)
        hide_map = rng.random() < self.dropout.map
        member = int(rng.integers(0, self.series.members)) if rng.random() < self.dropout.random_member else None
        values, missing, is_fc = self.series.tokens(t, member)

        p = len(self.points)
        drop = np.zeros_like(missing)
        for j in range(p):
            if rng.random() < self.dropout.point:
                drop[:HISTORY_DAYS, j] = 1
        if self.india_idx and rng.random() < self.dropout.india:
            drop[:HISTORY_DAYS, self.india_idx] = 1
        if rng.random() < self.dropout.rain_obs:
            drop[:HISTORY_DAYS, p:] = 1
        if rng.random() < self.dropout.forecast:
            drop[HISTORY_DAYS:, :] = 1
        missing = np.maximum(missing, drop)
        values = values * (1 - missing)

        target, mask = self._target(pair["target_scene"])
        prev = state.flooded if state is not None else np.zeros_like(mask)
        interest = mask & ((target > 0) | prev)
        r, c = self._crop_origin(rng, interest, mask)
        s = np.s_[r:r + self.chip, c:c + self.chip]
        maps_c = self.static.map_stack(state, t, hide_map=hide_map, window=s)
        target_c, mask_c = target[s], mask[s]
        prev_c = prev[s].astype(np.float32)
        if self.augment:
            k = int(rng.integers(4))
            maps_c = np.rot90(maps_c, k, axes=(1, 2))
            target_c, mask_c, prev_c = (np.rot90(a, k) for a in (target_c, mask_c, prev_c))
            if rng.random() < 0.5:
                maps_c, target_c, mask_c, prev_c = maps_c[:, :, ::-1], target_c[:, ::-1], mask_c[:, ::-1], prev_c[:, ::-1]
        return {
            "maps": torch.from_numpy(np.ascontiguousarray(maps_c)),
            "values": torch.from_numpy(values),
            "missing": torch.from_numpy(missing),
            "is_forecast": torch.from_numpy(is_fc),
            "lead": torch.tensor(float(pair["lead_days"]), dtype=torch.float32),
            "target": torch.from_numpy(np.ascontiguousarray(target_c)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_c)),
            "prev": torch.from_numpy(np.ascontiguousarray(prev_c)),
        }


def pairs_for(cube: Datacube, splits: list[str]) -> pd.DataFrame:
    return forecast_pairs(cube, splits)
