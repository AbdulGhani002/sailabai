"""Small PyTorch helpers: device choice, seeding, mixed precision, checkpoints, tiled inference."""

from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sailab.paths import REPO_ROOT


def device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed legacy global users (torch, third-party code)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(dev: torch.device, enabled: bool = True):
    """bfloat16 autocast on GPUs that support it (RTX 40xx do), float16 otherwise, nothing on CPU."""
    if not enabled or dev.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {**meta, "git_commit": git_commit()}}, path)
    (path.with_suffix(".json")).write_text(json.dumps({**meta, "git_commit": git_commit()}, indent=2, default=str),
                                          encoding="utf-8")
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def sliding_window(predict: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, tile: int = 512,
                   overlap: int = 64, multiple: int = 32) -> torch.Tensor:
    """Run `predict` on overlapping tiles of a (C, H, W) tensor and blend the results.

    `predict` maps (1, C, h, w) with h, w divisible by `multiple` to (1, K, h, w). Tiles are
    blended with a window that down-weights their edges, where a U-Net is least reliable.
    """
    c, h, w = x.shape
    if h <= tile and w <= tile:
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        xp = torch.nn.functional.pad(x[None], (0, pw, 0, ph))
        return predict(xp)[0, :, :h, :w]
    step = tile - overlap
    ramp = torch.ones(tile, device=x.device)
    edge = torch.linspace(0.1, 1.0, overlap, device=x.device)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge.flip(0)
    weight_tile = ramp[:, None] * ramp[None, :]
    out = None
    weight = torch.zeros(h, w, device=x.device)
    rows = list(range(0, max(h - tile, 0) + 1, step))
    cols = list(range(0, max(w - tile, 0) + 1, step))
    if rows[-1] + tile < h:
        rows.append(h - tile)
    if cols[-1] + tile < w:
        cols.append(w - tile)
    for r in rows:
        for c0 in cols:
            patch = x[:, r:r + tile, c0:c0 + tile]
            ph, pw = tile - patch.shape[1], tile - patch.shape[2]
            if ph or pw:
                patch = torch.nn.functional.pad(patch, (0, pw, 0, ph))
            pred = predict(patch[None])[0, :, :tile - ph, :tile - pw]
            wt = weight_tile[:tile - ph, :tile - pw]
            if out is None:
                out = torch.zeros(pred.shape[0], h, w, device=x.device)
            out[:, r:r + pred.shape[1], c0:c0 + pred.shape[2]] += pred * wt
            weight[r:r + pred.shape[1], c0:c0 + pred.shape[2]] += wt
    return out / weight.clamp_min(1e-6)
