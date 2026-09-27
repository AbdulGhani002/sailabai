"""The digital twin loop.

Rain and river data arrive every day, so Model 2 updates its forecast daily. When a new radar pass
arrives (every 3 to 4 days), Model 1 draws a fresh flood map and Model 2 restarts from it, so
errors do not pile up.

    sailab twin replay --cube demo --start 2025-08-15 --end 2025-09-15

Each daily run writes cloud-optimised GeoTIFFs (served as map tiles by the API) and a JSON summary
under data/twin/<cube>/runs/<YYYY-MM-DD>/, and is listed in data/twin/<cube>/index.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sailab import DISCLAIMER
from sailab import labels as L
from sailab.cube import Datacube
from sailab.forecast.baselines import LevelBaselines, persistence
from sailab.forecast.calibration import Calibration
from sailab.forecast.inputs import STANDARD_HORIZONS, SeriesBank, StateComposer, StaticMaps, issue_time
from sailab.io import write_raster
from sailab.paths import runs_dir, twin_dir
from sailab.risk.exposure import exposure_summary


@dataclass
class TwinConfig:
    cube: str = "demo"
    model1: str | None = None          # Model 1 checkpoint; None uses the GFM label of each pass
    model2_dir: str | None = None      # folder with unet_tt_seed*.pt and calibration.json
    members: int = 11                  # GloFAS members to run (0 = ensemble mean only)
    horizons: tuple[int, ...] = STANDARD_HORIZONS
    out_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _percent(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.round(prob * 100), 0, 100).astype(np.uint8)


class Twin:
    def __init__(self, cfg: TwinConfig, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self.cube = Datacube.open(cfg.cube)
        self.composer = StateComposer(self.cube)
        self.out = Path(cfg.out_dir) if cfg.out_dir else twin_dir() / self.cube.root.name
        self.calibration = Calibration()
        self.ensemble = None
        self.mapper = None
        self.level: LevelBaselines | None = None
        self._load_models()
        points = self.ensemble.points if self.ensemble else (self.level.points if self.level else [])
        self.series = SeriesBank(self.cube, points) if points else None
        self.static = StaticMaps(self.cube, points or None)
        self.mapped: dict[str, str] = {}

    # ------------------------------------------------------------------ models

    def _load_models(self) -> None:
        m2 = Path(self.cfg.model2_dir) if self.cfg.model2_dir else runs_dir() / "model2"
        try:
            import torch  # noqa: F401

            from sailab.forecast.model import ForecastEnsemble
            from sailab.torchutils import device

            dev = device()
            if any(m2.glob("*_seed*.pt")):
                self.ensemble = ForecastEnsemble.from_dir(m2, dev)
                self.calibration = self.ensemble.calibration
                self.log(f"Model 2: {len(self.ensemble.models)}-member ensemble from {m2}")
            m1 = Path(self.cfg.model1) if self.cfg.model1 else next(iter(sorted((runs_dir() / "model1").glob("unet_*.pt"))), None)
            if m1 and Path(m1).exists():
                from sailab.mapping.dataset import TerrainStack
                from sailab.mapping.predict import load_mapping_model

                model, meta = load_mapping_model(Path(m1), dev)
                self.mapper = (model, meta, TerrainStack.from_cube(self.cube), dev)
                self.log(f"Model 1: {m1}")
        except ImportError:
            self.log("PyTorch not installed: using GFM labels and baselines only")
        lb = m2 / "level_baselines.npz"
        if lb.exists():
            self.level = LevelBaselines.load(lb)
        if self.ensemble is None:
            self.log("no Model 2 checkpoints: forecasts fall back to the river-threshold baseline" if self.level
                     else "no Model 2 or baselines: forecasts fall back to persistence")

    # ------------------------------------------------------------------ one pass through the loop

    def _map_new_passes(self, t: pd.Timestamp) -> list[str]:
        """Model 1 maps every pass up to t that has not been mapped yet; Model 2 then starts from those maps."""
        new = []
        scenes = self.composer.scenes_before(t)
        for sid in scenes["scene_id"]:
            if sid in self.mapped:
                continue
            if self.mapper and self.cube.has_sar(sid):
                from sailab.mapping.predict import map_scene

                model, _, terrain, dev = self.mapper
                label, _ = map_scene(model, self.cube, sid, dev, terrain)
                self.composer.set_override(sid, label)
                self.mapped[sid] = "model1"
            else:
                self.mapped[sid] = "gfm"
            new.append(sid)
        return new

    def _next_pass(self, t: pd.Timestamp) -> pd.Series | None:
        # Sentinel-1 acquisition plans are published ahead, so the next pass time is known at issue.
        later = self.cube.scenes[self.cube.scenes["time"] > t]
        return later.iloc[0] if not later.empty else None

    def step(self, day: str | pd.Timestamp) -> dict[str, Any]:
        t = issue_time(day)
        new_passes = self._map_new_passes(t)
        state = self.composer.state(t)
        leads: list[tuple[str, float]] = [(f"d{h}", float(h)) for h in self.cfg.horizons]
        nxt = self._next_pass(t)
        if nxt is not None:
            lead = (nxt["time"] - t) / pd.Timedelta(days=1)
            if lead <= max(self.cfg.horizons) + 0.5:
                leads.insert(0, ("next", float(lead)))

        folder = self.out / "runs" / t.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        grid = self.cube.grid
        write_raster(folder / "current.tif", self._current_label(state), grid, dtype="uint8", nodata=L.IGNORE, cog=True,
                     tags={"layer": "current flood map", "issue_time": t.isoformat()})

        probs, members, model_name = self._forecast(state, t, [lv for _, lv in leads])
        layers = {"population": self.cube.static("population"), "buildings": self.cube.static("buildings"),
                  "road_km": self.cube.static("road_km")}
        roads, places = self.cube.geojson("roads"), self.cube.geojson("places")
        horizon_rows = []
        for i, (name, lead) in enumerate(leads):
            prob = np.where(self.static.normal_water, 0.0, probs[i]).astype(np.float32)
            write_raster(folder / f"prob_{name}.tif", _percent(prob), grid, dtype="uint8", nodata=255, cog=True,
                         tags={"layer": "flood chance (%)", "lead_days": f"{lead:.2f}", "model": model_name})
            sets = self.calibration.zones(prob)
            write_raster(folder / f"sets_{name}.tif", sets, grid, dtype="uint8", nodata=255, cog=True,
                         tags={"layer": "0 unlikely, 1 likely (>= 50%), 2 possible",
                               "coverage": self.calibration.coverage,
                               "possible_threshold": self.calibration.possible_threshold})
            mem = members[:, i] if members is not None else None
            if mem is not None:
                write_raster(folder / f"spread_{name}.tif", _percent(mem.std(axis=0)), grid, dtype="uint8", nodata=255,
                             cog=True, tags={"layer": "spread between ensemble members (percentage points)"})
            exposure = exposure_summary(prob, mem, layers, grid, self.calibration, roads, places, self.static.aoi)
            horizon_rows.append({
                "name": name, "lead_days": round(lead, 2), "valid_time": (t + pd.Timedelta(days=lead)).isoformat(),
                "flooded_km2_expected": exposure["totals"]["area_km2"]["expected"],
                "totals": exposure["totals"], "roads": exposure["roads"][:200], "places": exposure["places"],
                "uncertain_km2": round(float((sets == 2).sum() * grid.pixel_area_km2), 1),
                "likely_km2": round(float((sets == 1).sum() * grid.pixel_area_km2), 1),
                "layers": {"prob": f"prob_{name}.tif", "sets": f"sets_{name}.tif",
                           **({"spread": f"spread_{name}.tif"} if mem is not None else {})},
            })

        now_map = (state.flooded & state.known).astype(np.float32) if state is not None else np.zeros(grid.shape, np.float32)
        # today's map is an observation, not a forecast: counts without a forecast range
        exact = Calibration(total_quantiles={m: (0.0, 0.0) for m in (*layers, "area_km2")})
        now_exposure = exposure_summary(now_map, None, layers, grid, exact, roads, places, self.static.aoi)
        latest = self.composer.scenes_before(t)
        summary = {
            "run_id": t.strftime("%Y-%m-%d"),
            "issue_time": t.isoformat(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cube": self.cube.root.name,
            "synthetic": self.cube.is_synthetic,
            "disclaimer": DISCLAIMER,
            "model": model_name,
            "calibration": {"temperature": self.calibration.default_temperature, "alpha": self.calibration.alpha,
                            "coverage": self.calibration.coverage,
                            "possible_threshold": self.calibration.possible_threshold},
            "latest_pass": None if latest.empty else {
                "scene_id": latest.iloc[-1]["scene_id"], "time": latest.iloc[-1]["time"].isoformat(),
                "mapped_by": self.mapped.get(latest.iloc[-1]["scene_id"], "gfm")},
            "new_passes": new_passes,
            "current_flooded_km2": round(float(state.flooded[self.static.aoi].sum() * grid.pixel_area_km2), 1) if state else 0.0,
            "current_exposure": {"totals": now_exposure["totals"], "places": now_exposure["places"][:10]},
            "horizons": horizon_rows,
            "layers": {"current": "current.tif"},
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        self._update_index(summary)
        self._sync_db(folder)
        return summary

    def _sync_db(self, folder: Path) -> None:
        """Make the run visible to the API (PostGIS or the local SQLite file)."""
        try:
            from sailab.db.session import engine, init_db, session
            from sailab.db.sync import sync_run
        except ImportError:  # the `api` extra is not installed: files only
            return
        if not getattr(self, "_db_ready", False):
            init_db(engine())
            self._db_ready = True
        with session() as s:
            sync_run(s, folder, self.cube.root.name)
            s.commit()

    def _current_label(self, state) -> np.ndarray:
        out = np.full(self.cube.grid.shape, L.IGNORE, dtype=np.uint8)
        if state is not None:
            out[state.known] = L.LAND
            out[state.known & state.flooded] = L.FLOOD
        out[self.static.normal_water] = L.NORMAL_WATER
        return out

    def _forecast(self, state, t: pd.Timestamp, leads: list[float]) -> tuple[np.ndarray, np.ndarray | None, str]:
        """(probs per lead, member probs (K, leads, H, W) or None, model name)."""
        if self.ensemble is not None and self.series is not None:
            # one map stack per lead: the river-flow channels hold the forecast flow for that lead
            maps = np.stack([self.static.map_stack(state, t, flows=self.series.flow_features(t, lv)) for lv in leads])
            n_members = self.cfg.members
            member_ids = [None] if n_members <= 0 else list(np.linspace(0, 50, n_members).round().astype(int))
            vals, miss, lead_list, map_index = [], [], [], []
            for m in member_ids:
                v, mi, is_fc = self.series.tokens(t, None if m is None else int(m))
                for li, lv in enumerate(leads):
                    vals.append(v)
                    miss.append(mi)
                    lead_list.append(lv)
                    map_index.append(li)
            lg = self.ensemble.logits(maps, np.stack(vals), np.stack(miss), is_fc, lead_list, map_index=map_index)
            p = 1.0 / (1.0 + np.exp(-lg))                            # (models, members*leads, H, W)
            p = p.reshape(len(self.ensemble.models), len(member_ids), len(leads), *p.shape[-2:])
            members = p.reshape(-1, len(leads), *p.shape[-2:])       # every (model, member) as a sample
            mean = p.mean(axis=(0, 1))
            calibrated = np.stack([self.calibration.apply(mean[i], leads[i]) for i in range(len(leads))])
            return calibrated, members, f"unet-tt x{len(self.ensemble.models)} seeds x{len(member_ids)} glofas members"
        if self.level is not None and self.series is not None:
            out = np.stack([self.level.predict("river_threshold", self.level.flows_at(self.series, t, lv)) for lv in leads])
            return out, None, "river-threshold baseline"
        return np.stack([persistence(state, self.cube.grid.shape) for _ in leads]), None, "persistence baseline"

    def _update_index(self, summary: dict[str, Any]) -> None:
        path = self.out / "index.json"
        index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
        runs = [r for r in index["runs"] if r["run_id"] != summary["run_id"]]
        runs.append({"run_id": summary["run_id"], "issue_time": summary["issue_time"], "model": summary["model"],
                     "current_flooded_km2": summary["current_flooded_km2"],
                     "horizons": [{"name": h["name"], "lead_days": h["lead_days"],
                                   "flooded_km2_expected": h["flooded_km2_expected"]} for h in summary["horizons"]]})
        index["runs"] = sorted(runs, key=lambda r: r["run_id"])
        index.update({"cube": self.cube.root.name, "synthetic": self.cube.is_synthetic, "disclaimer": DISCLAIMER})
        path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    def replay(self, start: str, end: str) -> list[dict[str, Any]]:
        """Run the loop day by day as if live (the doc's 2026 replay mode)."""
        out = []
        for day in pd.date_range(start, end, freq="D"):
            s = self.step(day)
            h1 = next((h for h in s["horizons"] if h["name"] == "d1"), None)
            self.log(f"{s['run_id']}: now {s['current_flooded_km2']:.0f} km2 flooded; +1d expected "
                     f"{h1['flooded_km2_expected'] if h1 else float('nan'):.0f} km2; new passes {len(s['new_passes'])}")
            out.append(s)
        return out
