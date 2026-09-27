"""Hand-checked test tiles: the second mapping reference (30 to 60 tiles, test and validation events).

GFM labels are good but not perfect (rice fields and sand look like water to radar), and two expert
maps of one flood agreed only 48% in one study. So we hand-check a small, well-chosen set of tiles.

    sailab labels export --cube real --split test --n 40     # tiles to data/handlabels/<cube>/
    (correct each *_label.tif in QGIS: 0 land, 1 normal water, 2 flood, 255 unsure)
    sailab labels import --cube real                         # back into truth/<scene>.tif

Tiles are stratified so the check covers what matters: flood-rich tiles, flood edges, dry land
right next to rivers (where false alarms hide), and random tiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.windows import Window

from sailab import labels as L
from sailab.cube import Datacube
from sailab.grid import GridSpec
from sailab.io import read_raster, write_raster

STRATA = ("flood_rich", "flood_edge", "dry_near_river", "random")

GUIDE = """# Hand-labelling guide

Each tile has three files: *_sar.tif (VV, VH and the dry-season VV, in dB), *_label.tif (the GFM
label to correct) and a row in tiles.csv.

1. Open *_sar.tif in QGIS (VV band, stretch -25 to 0 dB; compare with the dry-season band).
2. Edit *_label.tif with the Serval plugin: 0 land, 1 normal water (river, canal, lake),
   2 flood, 255 unsure. Paint 255 wherever you cannot tell; never guess.
3. Rice paddies in July and desert sand are dark but are not flood. Flooded villages look bright.
4. Save in place, fill `labeller` and `minutes` in tiles.csv, then run `sailab labels import`.
"""


def pick_tiles(cube: Datacube, split: str, n: int = 40, size: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hand = cube.static("hand")
    scenes = cube.scenes[cube.scenes["split"] == split]
    if scenes.empty:
        raise ValueError(f"no scenes in split {split!r}")
    candidates = []
    for _, s in scenes.iterrows():
        label = cube.read_label(s["scene_id"])
        for w in cube.grid.windows(size, size, drop_partial=True):
            r, c = int(w.row_off), int(w.col_off)
            block = label[r:r + size, c:c + size]
            known = block != L.IGNORE
            if known.mean() < 0.8:
                continue
            flood = float((block == L.FLOOD)[known].mean())
            near_river = float((hand[r:r + size, c:c + size] < 5).mean())
            if flood > 0.10:
                stratum = "flood_rich"
            elif flood > 0.01:
                stratum = "flood_edge"
            elif near_river > 0.3:
                stratum = "dry_near_river"
            else:
                stratum = "random"
            candidates.append((s["scene_id"], s["event_id"], r, c, stratum, round(flood, 3)))
    df = pd.DataFrame(candidates, columns=["scene_id", "event_id", "row", "col", "stratum", "gfm_flood_share"])
    per = max(1, n // len(STRATA))
    picks = []
    for stratum in STRATA:
        part = df[df["stratum"] == stratum]
        if part.empty:
            continue
        # spread picks over different passes: at most one tile per scene per stratum where possible
        part = part.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).drop_duplicates("scene_id")
        picks.append(part.head(per))
    out = pd.concat(picks, ignore_index=True) if picks else df.head(0)
    out.insert(0, "tile_id", [f"t{i:03d}" for i in range(len(out))])
    out["size"] = size
    out["labeller"] = ""
    out["minutes"] = ""
    return out


def export_tiles(cube: Datacube, tiles: pd.DataFrame, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for _, t in tiles.iterrows():
        w = Window(t["col"], t["row"], t["size"], t["size"])
        sub = GridSpec(cube.grid.crs, cube.grid.window_transform(w), int(t["size"]), int(t["size"]))
        sar = cube.read_sar(t["scene_id"], w)
        row = cube.scenes.loc[cube.scenes["scene_id"] == t["scene_id"]].iloc[0]
        pre_key = row.get("pre_key")
        pre = (cube.read_pre_sar(pre_key, w)[:1] if isinstance(pre_key, str) and
               (cube.root / "s1_pre" / f"{pre_key}.tif").exists() else np.full((1, *sar.shape[1:]), np.nan, np.float32))
        write_raster(out / f"{t['tile_id']}_sar.tif", np.concatenate([sar, pre]), sub, dtype="float32",
                     band_names=["VV", "VH", "VV_dry_season"])
        write_raster(out / f"{t['tile_id']}_label.tif", cube.read_label(t["scene_id"], w), sub, dtype="uint8",
                     nodata=L.IGNORE, tags={"scene_id": t["scene_id"], "codes": "0 land, 1 normal water, 2 flood, 255 unsure"})
    tiles.to_csv(out / "tiles.csv", index=False)
    (out / "LABELLING.md").write_text(GUIDE, encoding="utf-8")
    return out


def import_tiles(cube: Datacube, folder: Path) -> int:
    """Write corrected tiles into truth/<scene>.tif (unknown everywhere else)."""
    tiles = pd.read_csv(folder / "tiles.csv")
    by_scene: dict[str, np.ndarray] = {}
    for _, t in tiles.iterrows():
        sid = t["scene_id"]
        if sid not in by_scene:
            by_scene[sid] = cube.read_label(sid, truth=True) if cube.has_truth(sid) else \
                np.full(cube.grid.shape, L.IGNORE, dtype=np.uint8)
        edited = read_raster(folder / f"{t['tile_id']}_label.tif")[0]
        r, c, n = int(t["row"]), int(t["col"]), int(t["size"])
        by_scene[sid][r:r + n, c:c + n] = edited
    for sid, arr in by_scene.items():
        cube.write_label(sid, arr, truth=True, source="hand-labelled tiles")
    return len(tiles)
