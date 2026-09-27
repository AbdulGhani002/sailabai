"""Score Model 1 (U-Net) and the Otsu threshold on an event split, against two references.

GFM labels are one reference; the second is our hand-checked tiles (the error-free "truth" labels
in a synthetic cube). Two expert maps of the same flood agreed only 48% in one study, so a mapping
score against a single reference is never reported alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sailab.cube import Datacube
from sailab.evaluation.protocol import EventScorer
from sailab.evaluation.results import ResultsTable
from sailab.mapping.otsu import otsu_flood_map
from sailab.versioning import DataVersions


def evaluate_mapping(cube: Datacube, split: str, model_path: Path | None, max_scenes: int | None = None,
                     log=print) -> dict[str, dict[str, EventScorer]]:
    """{reference: {model: scorer}} for references 'gfm' and (when available) 'truth'."""
    scenes = cube.scenes[cube.scenes["split"] == split]
    if max_scenes and len(scenes) > max_scenes:
        scenes = scenes.iloc[np.linspace(0, len(scenes) - 1, max_scenes).astype(int)]
    normal = cube.static("normal_water").astype(bool)
    hand = cube.static("hand")
    aoi = cube.static("aoi_mask").astype(bool) if cube.has_static("aoi_mask") else np.ones(cube.grid.shape, bool)
    excluded = cube.exclusion()
    scored = aoi & ~excluded  # radar cannot judge sand and towns; GFM leaves them out too
    refs = ["gfm"] + (["truth"] if cube.has_truth(scenes.iloc[0]["scene_id"]) else [])
    scorers = {r: {"otsu": EventScorer()} for r in refs}
    unet = None
    if model_path is not None:
        from sailab.mapping.dataset import TerrainStack
        from sailab.mapping.predict import load_mapping_model, map_scene
        from sailab.torchutils import device

        dev = device()
        unet, _ = load_mapping_model(model_path, dev)
        terrain = TerrainStack.from_cube(cube)
        for r in refs:
            scorers[r]["unet"] = EventScorer()
    log(f"scoring {len(scenes)} scenes of split '{split}' against {', '.join(refs)}")
    for _, s in scenes.iterrows():
        sid = s["scene_id"]
        sar = cube.read_sar(sid)
        pre_key = s.get("pre_key")
        pre = cube.read_pre_sar(pre_key) if isinstance(pre_key, str) and (cube.root / "s1_pre" / f"{pre_key}.tif").exists() else None
        otsu = otsu_flood_map(sar, normal, hand, pre)
        preds = {"otsu": np.where(excluded, 0.0, otsu.flood_prob)}
        if unet is not None:
            _, prob = map_scene(unet, cube, sid, dev, terrain)
            preds["unet"] = prob
        for ref in refs:
            label = cube.read_label(sid, truth=(ref == "truth"))
            for name, prob in preds.items():
                scorers[ref][name].add(s["event_id"], prob, label, normal_water=normal, extra_valid=scored)
    return scorers


def record_mapping_results(scorers: dict[str, dict[str, EventScorer]], cube: Datacube, split: str,
                           experiment: str, table: ResultsTable | None = None) -> None:
    table = table or ResultsTable()
    versions = DataVersions.from_dict(cube.meta.get("versions"))
    for ref, models in scorers.items():
        base = models["otsu"].macro()
        table.append(base, None, experiment=experiment, model=f"otsu (vs {ref})", baseline="none", split=split,
                     data_versions=versions, notes="step 1, no training: reference for the U-Net")
        if "unet" in models:
            table.append(models["unet"].macro(), base, experiment=experiment, model=f"unet (vs {ref})",
                         baseline="otsu", split=split, data_versions=versions)
    table.render_markdown()
