"""Training chips for Model 1: radar after and before the flood plus terrain, with GFM labels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from sailab import labels as L
from sailab.cube import Datacube
from sailab.features import mapping_inputs


@dataclass
class TerrainStack:
    dem: np.ndarray
    hand: np.ndarray
    slope: np.ndarray
    dem_stats: tuple[float, float]

    @classmethod
    def from_cube(cls, cube: Datacube) -> TerrainStack:
        dem = cube.static("dem").astype(np.float32)
        valid = np.isfinite(dem)
        return cls(dem, cube.static("hand").astype(np.float32), cube.static("slope").astype(np.float32),
                   (float(dem[valid].mean()), float(dem[valid].std())))

    def window(self, w: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, c, h, wd = int(w.row_off), int(w.col_off), int(w.height), int(w.width)
        return self.dem[r:r + h, c:c + wd], self.hand[r:r + h, c:c + wd], self.slope[r:r + h, c:c + wd]


def scene_inputs(cube: Datacube, scene_id: str, terrain: TerrainStack, window: Window | None = None) -> np.ndarray:
    """Model 1 input stack for a scene (or a window of it)."""
    row = cube.scenes.loc[cube.scenes["scene_id"] == scene_id].iloc[0]
    post = cube.read_sar(scene_id, window)
    pre_key = row.get("pre_key")
    pre = cube.read_pre_sar(pre_key, window) if isinstance(pre_key, str) and (cube.root / "s1_pre" / f"{pre_key}.tif").exists() else None
    w = window or Window(0, 0, cube.grid.width, cube.grid.height)
    dem, hand, slope = terrain.window(w)
    return mapping_inputs(post, pre, dem, hand, slope, terrain.dem_stats)


class MappingChips(Dataset):
    """Random chips from a list of scenes. Half the chips are drawn from places that flooded,
    because floods cover a few percent of pixels and the model would otherwise rarely see one."""

    def __init__(self, cube: Datacube, scene_ids: list[str], chip: int = 128, samples_per_epoch: int = 2000,
                 flood_share: float = 0.5, min_known: float = 0.5, augment: bool = True, seed: int = 0,
                 truth: bool = False) -> None:
        if not scene_ids:
            raise ValueError("no scenes for this split")
        self.cube = cube
        self.scene_ids = list(scene_ids)
        self.chip = chip
        self.n = samples_per_epoch
        self.flood_share = flood_share
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.truth = truth
        self.terrain = TerrainStack.from_cube(cube)
        stride = chip // 2
        self.candidates: list[tuple[int, int, int, bool]] = []  # scene index, row, col, has flood
        for si, sid in enumerate(self.scene_ids):
            label = cube.read_label(sid, truth=truth)
            for w in cube.grid.windows(chip, stride, drop_partial=True):
                r, c = int(w.row_off), int(w.col_off)
                block = label[r:r + chip, c:c + chip]
                if (block != L.IGNORE).mean() < min_known:
                    continue
                self.candidates.append((si, r, c, bool((block == L.FLOOD).mean() > 0.005)))
        self.flood_idx = [i for i, cnd in enumerate(self.candidates) if cnd[3]]
        self.dry_idx = [i for i, cnd in enumerate(self.candidates) if not cnd[3]]
        if not self.candidates:
            raise ValueError("no usable chips: labels are mostly unknown")
        self._by_scene: dict[int, list[int]] = {}
        for i, cnd in enumerate(self.candidates):
            self._by_scene.setdefault(cnd[0], []).append(i)
        # half the groups start from a scene with flood in it, so floods are not rare in training
        self._flood_scenes = sorted({self.candidates[i][0] for i in self.flood_idx})

    def __len__(self) -> int:
        return self.n

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @lru_cache(maxsize=64)  # noqa: B019 - one dataset per training run
    def _scene(self, si: int) -> tuple[np.ndarray, np.ndarray]:
        sid = self.scene_ids[si]
        return scene_inputs(self.cube, sid, self.terrain), self.cube.read_label(sid, truth=self.truth)

    def _chip(self, si: int, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cube.grid.width * self.cube.grid.height <= 4_000_000:  # demo grids: cache whole scenes
            x_full, y_full = self._scene(si)
            return x_full[:, r:r + self.chip, c:c + self.chip], y_full[r:r + self.chip, c:c + self.chip]
        w = Window(c, r, self.chip, self.chip)  # 20 m grids: read just the window
        sid = self.scene_ids[si]
        return scene_inputs(self.cube, sid, self.terrain, w), self.cube.read_label(sid, w, truth=self.truth)

    GROUP = 8  # consecutive samples share a scene, so a whole-scene read serves 8 chips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        group = np.random.default_rng((self.seed, self.epoch, idx // self.GROUP, 1))
        if self._flood_scenes and group.random() < self.flood_share:
            scene = self._flood_scenes[int(group.integers(len(self._flood_scenes)))]
        else:
            scene = self.candidates[int(group.integers(len(self.candidates)))][0]
        same = self._by_scene[scene]
        flood_here = [i for i in same if self.candidates[i][3]]
        pool = flood_here if (flood_here and rng.random() < self.flood_share) else same
        si, r, c, _ = self.candidates[pool[int(rng.integers(len(pool)))]]
        x, y = self._chip(si, r, c)
        y = y.astype(np.int64)
        if self.augment:
            k = int(rng.integers(4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
