"""Model 1, step 3: TerraMind-small (IBM and ESA, Apache-2.0) fine-tuned with TerraTorch.

TerraMind was pre-trained on large amounts of Earth data and reads radar and elevation directly.
We export 224 x 224 chips in the folder layout TerraTorch's GenericMultiModalDataModule reads, then
fine-tune with the CLI (on Kaggle's T4s or locally):

    sailab export terratorch --cube demo --out data/terratorch/demo
    terratorch fit -c configs/terramind_small.yaml

Layout written:
    <out>/S1GRD/<chip>.tif    VV, VH backscatter in dB (float32, NaN outside the swath -> 0)
    <out>/DEM/<chip>.tif      elevation in metres
    <out>/mask/<chip>.tif     0 land, 1 normal water, 2 flood, -1 ignore
    <out>/splits/{train,val,test}.txt   chip ids, split by flood event (team rule 1)
    <out>/stats.json          per-band means and stds for the config
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sailab import labels as L
from sailab.cube import Datacube
from sailab.io import write_raster

CHIP = 224  # TerraMind's ViT patch size is 16, so 224 gives a 14 x 14 token grid


def export_terratorch(cube: Datacube, out: Path, chip: int = CHIP, min_known: float = 0.5,
                      splits: tuple[str, ...] = ("train", "val", "test"), log=print) -> dict[str, int]:
    dem = cube.static("dem").astype(np.float32)
    counts: dict[str, int] = {}
    lists: dict[str, list[str]] = {s: [] for s in splits}
    sums = {"vv": [0.0, 0.0, 0], "vh": [0.0, 0.0, 0], "dem": [0.0, 0.0, 0]}
    for split in splits:
        scenes = cube.scenes[cube.scenes["split"] == split]
        for sid in scenes["scene_id"]:
            sar = cube.read_sar(sid)
            label = cube.read_label(sid)
            for w in cube.grid.windows(chip, chip, drop_partial=True):
                r, c = int(w.row_off), int(w.col_off)
                lab = label[r:r + chip, c:c + chip]
                if (lab != L.IGNORE).mean() < min_known:
                    continue
                name = f"{sid}_r{r:05d}_c{c:05d}"
                s1 = np.nan_to_num(sar[:, r:r + chip, c:c + chip], nan=0.0)
                grid = cube.grid
                sub = type(grid)(grid.crs, grid.window_transform(w), chip, chip)
                write_raster(out / "S1GRD" / f"{name}.tif", s1, sub, dtype="float32")
                write_raster(out / "DEM" / f"{name}.tif", dem[r:r + chip, c:c + chip], sub, dtype="float32")
                mask = np.where(lab == L.IGNORE, -1, lab).astype(np.int16)
                write_raster(out / "mask" / f"{name}.tif", mask, sub, dtype="int16", nodata=-1)
                lists[split].append(name)
                if split == "train":
                    for key, arr in (("vv", s1[0]), ("vh", s1[1]), ("dem", dem[r:r + chip, c:c + chip])):
                        v = arr[np.isfinite(arr) & (arr != 0)]
                        sums[key][0] += float(v.sum())
                        sums[key][1] += float((v.astype(np.float64) ** 2).sum())
                        sums[key][2] += int(v.size)
        counts[split] = len(lists[split])
        log(f"{split}: {counts[split]} chips")
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for split, names in lists.items():
        (out / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    stats = {}
    for key, (s, sq, n) in sums.items():
        if n:
            mean = s / n
            stats[key] = {"mean": round(mean, 4), "std": round(float(np.sqrt(max(sq / n - mean * mean, 1e-9))), 4)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"band statistics for configs/terramind_small.yaml: {stats}")
    return counts


def build_model(num_classes: int = L.NUM_CLASSES, pretrained: bool = True):
    """TerraMind-small encoder with a U-Net decoder through TerraTorch's EncoderDecoderFactory.

    Mirrors configs/terramind_small.yaml, for use from Python or notebooks. Needs the `terramind`
    extra (terratorch); check the argument names against the installed TerraTorch version.
    """
    from terratorch.models import EncoderDecoderFactory

    return EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone="terramind_v1_small",
        backbone_pretrained=pretrained,
        backbone_modalities=["S1GRD", "DEM"],
        necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
               {"name": "ReshapeTokensToImage", "remove_cls_token": False},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        head_dropout=0.1,
        num_classes=num_classes,
    )
