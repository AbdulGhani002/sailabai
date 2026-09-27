"""Loading and running Model 2: one checkpoint or a deep ensemble of several seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sailab.forecast.calibration import Calibration
from sailab.nn.unet import pad_to_multiple
from sailab.nn.unet_clstm import UNetConvLSTM
from sailab.nn.unet_tt import UNetTT
from sailab.torchutils import autocast, load_checkpoint

ARCHS = {"unet_tt": UNetTT, "unet_clstm": UNetConvLSTM}


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Model 2 network from its saved config (\"arch\" picks UNet-TT or the ConvLSTM comparison)."""
    cfg = dict(config)
    arch = cfg.pop("arch", "unet_tt")
    if arch not in ARCHS:
        raise ValueError(f"unknown Model 2 architecture {arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[arch](**cfg)


class ForecastEnsemble:
    """Several copies of Model 2 trained with different seeds. Their disagreement is the model
    uncertainty; running them on each GloFAS member adds the weather uncertainty."""

    def __init__(self, models: list[torch.nn.Module], metas: list[dict[str, Any]], dev: torch.device,
                 calibration: Calibration | None = None) -> None:
        if not models:
            raise ValueError("no models")
        self.models = models
        self.metas = metas
        self.dev = dev
        self.calibration = calibration or Calibration()

    @property
    def points(self) -> list[str]:
        return list(self.metas[0]["points"])

    @classmethod
    def load(cls, paths: list[Path], dev: torch.device, calibration: Calibration | None = None) -> ForecastEnsemble:
        models, metas = [], []
        for p in paths:
            ckpt = load_checkpoint(p, dev)
            meta = ckpt["meta"]
            model = build_model(meta["model_config"])
            model.load_state_dict(ckpt["state_dict"])
            models.append(model.to(dev).eval())
            metas.append(meta)
        points = {tuple(m["points"]) for m in metas}
        if len(points) != 1:
            raise ValueError("ensemble members were trained on different river points")
        return cls(models, metas, dev, calibration)

    @classmethod
    def from_dir(cls, folder: Path, dev: torch.device) -> ForecastEnsemble:
        paths = sorted(folder.glob("*_seed*.pt"))
        if not paths:
            raise FileNotFoundError(f"no Model 2 checkpoints (<arch>_seed<n>.pt) in {folder}")
        cal_path = folder / "calibration.json"
        return cls.load(paths, dev, Calibration.load(cal_path) if cal_path.exists() else None)

    @torch.no_grad()
    def logits(self, maps: np.ndarray, values: np.ndarray, missing: np.ndarray, is_forecast: np.ndarray,
               leads: list[float], amp: bool = True, batch: int = 8,
               map_index: list[int] | None = None) -> np.ndarray:
        """Raw logits (models, items, H, W) for one issue time.

        `maps` is (C, H, W), shared by every item, or (M, C, H, W) with `map_index` saying which map
        each item uses (maps differ by lead time through the river-flow channels). `values`/`missing`
        are (T, F) for one token set or (items, T, F), e.g. one per GloFAS member and lead."""
        n = len(leads)
        m = torch.from_numpy(maps if maps.ndim == 4 else maps[None]).to(self.dev)
        m, (h, w) = pad_to_multiple(m)
        idx = torch.as_tensor(map_index if map_index is not None else
                              ([0] * n if maps.ndim == 3 else list(range(n))), device=self.dev)
        v = torch.from_numpy(values).to(self.dev)
        mi = torch.from_numpy(missing).to(self.dev)
        if v.dim() == 2:
            v, mi = v.expand(n, -1, -1), mi.expand(n, -1, -1)
        f = torch.from_numpy(is_forecast).to(self.dev).expand(n, -1)
        ld = torch.tensor(leads, dtype=torch.float32, device=self.dev)
        out = np.empty((len(self.models), n, h, w), dtype=np.float32)
        for k, model in enumerate(self.models):
            for s in range(0, n, batch):
                e = min(n, s + batch)
                with autocast(self.dev, amp):
                    lg = model(m[idx[s:e]], v[s:e], mi[s:e], f[s:e], ld[s:e])
                out[k, s:e] = lg.float()[:, :h, :w].cpu().numpy()
        return out

    def predict(self, maps: np.ndarray, values: np.ndarray, missing: np.ndarray, is_forecast: np.ndarray,
                leads: list[float], calibrated: bool = True) -> dict[str, np.ndarray]:
        """Flood chance per lead: calibrated mean over the ensemble, plus the spread between models.
        `maps` is (C, H, W) or one map per lead (L, C, H, W)."""
        lg = self.logits(maps, values, missing, is_forecast, leads)
        probs = 1.0 / (1.0 + np.exp(-lg))
        mean = probs.mean(axis=0)
        if calibrated:
            mean = np.stack([self.calibration.apply(mean[i], leads[i]) for i in range(len(leads))])
        return {"prob": mean.astype(np.float32), "spread": probs.std(axis=0).astype(np.float32),
                "members": probs.astype(np.float32)}
