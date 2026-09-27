"""Train Model 2 (UNet-TT). Run it with several seeds for the deep ensemble.

    sailab train forecast --cube demo --seeds 0 1 2 3 4

The loss only counts pixels where the target pass has a label, normal water is excluded, and
model selection uses the Brier score on the validation event (a proper score, so it rewards
honest probabilities rather than sharp ones).
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sailab.config import load_gauges
from sailab.cube import Datacube
from sailab.evaluation.metrics import Confusion, ProbAccumulator
from sailab.events import assert_no_leakage
from sailab.forecast.baselines import persistence
from sailab.forecast.dataset import DropoutConfig, ForecastChips
from sailab.forecast.inputs import MAP_CHANNELS, N_TOKENS, forecast_pairs, target_arrays
from sailab.forecast.model import ForecastEnsemble
from sailab.nn.unet import count_parameters
from sailab.nn.unet_tt import UNetTT, time_encoder_parameters
from sailab.paths import runs_dir
from sailab.torchutils import autocast, device, save_checkpoint, seed_everything


@dataclass
class ForecastConfig:
    cube: str = "demo"
    encoder: str = "resnet18"
    include_india: bool = False
    chip: int = 128
    batch_size: int = 12
    epochs: int = 14
    samples_per_epoch: int = 3000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 0
    num_workers: int = 0
    max_val_pairs: int = 120
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    out_dir: str = field(default_factory=lambda: str(runs_dir() / "model2"))


def model_points(include_india: bool) -> list[str]:
    return load_gauges().model_input_ids(include_india=include_india)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def validate(ensemble: ForecastEnsemble, ds_val: ForecastChips, pairs, max_pairs: int) -> dict[str, float]:
    """Brier and IoU of the model vs persistence on validation pairs (full scenes)."""
    pairs = pairs.iloc[:: max(1, len(pairs) // max_pairs)]
    acc_m, acc_p = ProbAccumulator(), ProbAccumulator()
    conf_m, conf_p = Confusion(), Confusion()
    for _, pr in pairs.iterrows():
        t = pr["issue"]
        state = ds_val.composer.state(t)
        maps = ds_val.static.map_stack(state, t)
        values, missing, is_fc = ds_val.series.tokens(t)
        prob = ensemble.predict(maps, values, missing, is_fc, [float(pr["lead_days"])], calibrated=False)["prob"][0]
        target, mask = target_arrays(ds_val.cube.read_label(pr["target_scene"]), ds_val.static)
        base = persistence(state, mask.shape)
        acc_m.update(prob, target, mask)
        acc_p.update(base, target, mask)
        conf_m = conf_m + Confusion.from_arrays(prob >= 0.5, target > 0, mask)
        conf_p = conf_p + Confusion.from_arrays(base >= 0.5, target > 0, mask)
    return {"brier": acc_m.brier, "brier_persistence": acc_p.brier, "iou": conf_m.iou, "iou_persistence": conf_p.iou}


def train_forecast(cfg: ForecastConfig, log=print) -> Path:
    seed_everything(cfg.seed)
    dev = device()
    cube = Datacube.open(cfg.cube)
    points = model_points(cfg.include_india)
    india = tuple(g.id for g in load_gauges().upstream_india) if cfg.include_india else ()
    ds = ForecastChips(cube, ["train"], points, india, chip=cfg.chip, samples_per_epoch=cfg.samples_per_epoch,
                       dropout=cfg.dropout, seed=cfg.seed)
    val_pairs = forecast_pairs(cube, ["val"])
    assert_no_leakage(ds.pairs["event_id"].tolist(), val_pairs["event_id"].tolist())
    ds_val = ForecastChips(cube, ["val"], points, india, chip=cfg.chip, samples_per_epoch=1, augment=False)
    val_pairs = ds_val.pairs
    log(f"{len(ds.pairs)} training pairs from {ds.pairs['event_id'].nunique()} events, {len(val_pairs)} validation "
        f"pairs; points {points}; device {dev}")

    model_config = {"map_channels": len(MAP_CHANNELS), "series_features": ds.series.n_features,
                    "n_tokens": N_TOKENS, "encoder": cfg.encoder}
    model = UNetTT(**model_config).to(dev)
    log(f"UNet-TT {cfg.encoder}: {count_parameters(model) / 1e6:.1f}M parameters "
        f"(time transformer {time_encoder_parameters(model) / 1e6:.2f}M)")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader), pct_start=0.1)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"unet_tt_seed{cfg.seed}.pt"
    best = np.inf
    with open(out / f"unet_tt_seed{cfg.seed}_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_brier", "val_brier_persistence", "val_iou", "val_iou_persistence", "seconds"])
        for epoch in range(cfg.epochs):
            ds.set_epoch(epoch)
            model.train()
            t0 = time.time()
            losses = []
            for batch in loader:
                b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
                with autocast(dev, cfg.amp):
                    logits = model(b["maps"], b["values"], b["missing"], b["is_forecast"], b["lead"])
                loss = masked_bce(logits, b["target"], b["mask"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(float(loss))
            model.eval()
            meta = {"kind": "model2-unet-tt", "model_config": model_config, "points": points, "channels": MAP_CHANNELS,
                    "config": asdict(cfg), "epoch": epoch, "cube": cube.meta.get("name", cfg.cube),
                    "train_events": sorted(ds.pairs["event_id"].unique()),
                    "val_events": sorted(val_pairs["event_id"].unique())}
            scores = validate(ForecastEnsemble([model], [meta], dev), ds_val, val_pairs, cfg.max_val_pairs)
            secs = time.time() - t0
            writer.writerow([epoch, round(float(np.mean(losses)), 5), *(round(scores[k], 5) for k in
                             ("brier", "brier_persistence", "iou", "iou_persistence")), round(secs, 1)])
            fh.flush()
            log(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val Brier {scores['brier']:.4f} "
                f"(persistence {scores['brier_persistence']:.4f})  IoU {scores['iou']:.3f} "
                f"(persistence {scores['iou_persistence']:.3f})  {secs:.0f}s")
            if scores["brier"] < best:
                best = scores["brier"]
                save_checkpoint(path, model, {**meta, "val": scores})
    log(f"best val Brier {best:.4f} -> {path}")
    return path
