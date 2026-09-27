"""Run Model 1 on whole scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sailab import labels as L
from sailab.cube import Datacube
from sailab.mapping.dataset import TerrainStack, scene_inputs
from sailab.nn.unet import ResNetUNet
from sailab.torchutils import autocast, load_checkpoint, sliding_window


def load_mapping_model(path: Path, dev: torch.device) -> tuple[ResNetUNet, dict[str, Any]]:
    ckpt = load_checkpoint(path, dev)
    meta = ckpt["meta"]
    model = ResNetUNet(meta["in_channels"], meta["num_classes"], encoder=meta["encoder"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dev).eval(), meta


@torch.no_grad()
def predict_probs(model: ResNetUNet, x: np.ndarray, dev: torch.device, amp: bool = True, tile: int = 512) -> np.ndarray:
    """Class probabilities (3, H, W) for one input stack (7, H, W)."""
    xt = torch.from_numpy(x).to(dev)

    def run(batch: torch.Tensor) -> torch.Tensor:
        with autocast(dev, amp):
            return torch.softmax(model(batch).float(), dim=1)

    return sliding_window(run, xt, tile=tile).cpu().numpy().astype(np.float32)


def probs_to_label(probs: np.ndarray, normal_water: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Most likely class, with the reference water mask taking precedence for normal water."""
    cls = probs.argmax(axis=0).astype(np.uint8)
    cls[(cls == L.NORMAL_WATER) & ~normal_water] = L.FLOOD  # water outside normal channels is flood
    cls[(cls == L.FLOOD) & normal_water] = L.NORMAL_WATER
    cls[normal_water & (cls == L.LAND)] = L.NORMAL_WATER
    cls[~valid] = L.IGNORE
    return cls


def map_scene(model: ResNetUNet, cube: Datacube, scene_id: str, dev: torch.device,
              terrain: TerrainStack | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(label map, flood probability) for a scene."""
    terrain = terrain or TerrainStack.from_cube(cube)
    x = scene_inputs(cube, scene_id, terrain)
    probs = predict_probs(model, x, dev)
    sar = cube.read_sar(scene_id)
    valid = np.isfinite(sar[0])
    normal = cube.static("normal_water").astype(bool)
    flood_prob = np.where(valid & ~normal, probs[L.FLOOD], 0.0).astype(np.float32)
    return probs_to_label(probs, normal, valid), flood_prob
