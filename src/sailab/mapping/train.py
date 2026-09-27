"""Train Model 1 (flood mapping U-Net) on event-split data.

    sailab train mapping --cube demo --epochs 12

Training uses the train-split events, model selection uses the validation event (2023), and the
test event is never touched here.
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

from sailab import labels as L
from sailab.cube import Datacube
from sailab.evaluation.metrics import Confusion
from sailab.events import assert_no_leakage
from sailab.features import MAPPING_CHANNELS
from sailab.mapping.dataset import MappingChips, TerrainStack, scene_inputs
from sailab.mapping.predict import predict_probs
from sailab.nn.unet import ResNetUNet, count_parameters
from sailab.paths import runs_dir
from sailab.torchutils import autocast, device, save_checkpoint, seed_everything


@dataclass
class MappingConfig:
    cube: str = "demo"
    encoder: str = "resnet34"
    pretrained: bool = False
    chip: int = 128
    batch_size: int = 16
    epochs: int = 12
    samples_per_epoch: int = 2400
    lr: float = 3e-4
    weight_decay: float = 1e-4
    class_weights: tuple[float, float, float] = (1.0, 1.0, 3.0)
    dice_weight: float = 0.5
    amp: bool = True
    seed: int = 0
    num_workers: int = 0
    max_val_scenes: int = 40
    out_dir: str = field(default_factory=lambda: str(runs_dir() / "model1"))


def soft_dice(logits: torch.Tensor, target: torch.Tensor, cls: int, ignore: int = L.IGNORE) -> torch.Tensor:
    """1 - Dice on one class, counting known pixels only."""
    prob = torch.softmax(logits.float(), dim=1)[:, cls]
    known = (target != ignore).float()
    truth = (target == cls).float() * known
    p = prob * known
    inter = (p * truth).sum()
    return 1.0 - (2 * inter + 1.0) / (p.sum() + truth.sum() + 1.0)


def evaluate_scenes(model: ResNetUNet, cube: Datacube, scene_ids: list[str], terrain: TerrainStack,
                    dev: torch.device, amp: bool) -> dict[str, float]:
    """Flood-class scores against labels, normal water removed (the team's scoring rule)."""
    conf = Confusion()
    normal = cube.static("normal_water").astype(bool)
    for sid in scene_ids:
        probs = predict_probs(model, scene_inputs(cube, sid, terrain), dev, amp)
        label = cube.read_label(sid)
        valid = (label != L.IGNORE) & (label != L.NORMAL_WATER) & ~normal
        conf = conf + Confusion.from_arrays(probs.argmax(0) == L.FLOOD, label == L.FLOOD, valid)
    return {"f1": conf.f1, "iou": conf.iou, "precision": conf.precision, "recall": conf.recall}


def train_mapping(cfg: MappingConfig, log=print) -> Path:
    seed_everything(cfg.seed)
    dev = device()
    cube = Datacube.open(cfg.cube)
    scenes = cube.scenes
    train = scenes[scenes["split"] == "train"]
    val = scenes[scenes["split"] == "val"]
    assert_no_leakage(train["event_id"].tolist(), val["event_id"].tolist())
    val_ids = val["scene_id"].tolist()
    if len(val_ids) > cfg.max_val_scenes:
        val_ids = val_ids[:: int(np.ceil(len(val_ids) / cfg.max_val_scenes))]
    log(f"train on {len(train)} scenes from {train['event_id'].nunique()} events, validate on {len(val_ids)} scenes "
        f"({', '.join(sorted(val['event_id'].unique()))}); device {dev}")

    ds = MappingChips(cube, train["scene_id"].tolist(), chip=cfg.chip, samples_per_epoch=cfg.samples_per_epoch,
                      seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)
    model = ResNetUNet(len(MAPPING_CHANNELS), L.NUM_CLASSES, encoder=cfg.encoder, pretrained=cfg.pretrained).to(dev)
    log(f"U-Net {cfg.encoder}: {count_parameters(model) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    weights = torch.tensor(cfg.class_weights, device=dev)
    terrain = TerrainStack.from_cube(cube)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / f"unet_{cfg.encoder}_seed{cfg.seed}.pt"
    log_path = out / f"unet_{cfg.encoder}_seed{cfg.seed}_log.csv"
    best = -1.0
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_f1", "val_iou", "val_precision", "val_recall", "seconds"])
        for epoch in range(cfg.epochs):
            ds.set_epoch(epoch)
            model.train()
            t0 = time.time()
            losses = []
            for x, y in loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with autocast(dev, cfg.amp):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y, weight=weights, ignore_index=L.IGNORE)
                loss = loss + cfg.dice_weight * soft_dice(logits, y, L.FLOOD)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            model.eval()
            scores = evaluate_scenes(model, cube, val_ids, terrain, dev, cfg.amp)
            secs = time.time() - t0
            writer.writerow([epoch, np.mean(losses), *(round(scores[k], 4) for k in ("f1", "iou", "precision", "recall")),
                             round(secs, 1)])
            fh.flush()
            log(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val F1 {scores['f1']:.3f}  IoU {scores['iou']:.3f}  "
                f"({secs:.0f}s)")
            if scores["f1"] > best:
                best = scores["f1"]
                save_checkpoint(best_path, model, {
                    "kind": "model1-unet", "encoder": cfg.encoder, "in_channels": len(MAPPING_CHANNELS),
                    "num_classes": L.NUM_CLASSES, "channels": MAPPING_CHANNELS, "dem_stats": terrain.dem_stats,
                    "epoch": epoch, "val": scores, "config": asdict(cfg), "cube": cube.meta.get("name", cfg.cube),
                    "train_events": sorted(train["event_id"].unique()), "val_events": sorted(val["event_id"].unique()),
                })
    log(f"best val F1 {best:.3f} -> {best_path}")
    return best_path
