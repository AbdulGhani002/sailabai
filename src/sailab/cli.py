"""The `sailab` command line. Run `sailab --help` for the full list."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sailab import DISCLAIMER, __version__

app = typer.Typer(help="SailabAI: flood digital twin for the Chenab, Ravi and Sutlej around Multan.",
                  no_args_is_help=True, add_completion=False)
console = Console()


def log(msg: str) -> None:
    console.print(msg, highlight=False)


def _years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


@app.command()
def info() -> None:
    """Show versions, paths, the study area and the GPU."""
    from sailab.config import load_aoi, load_events
    from sailab.grid import GridSpec
    from sailab.paths import config_dir, cubes_dir, results_dir, runs_dir, twin_dir

    aoi = load_aoi()
    grid = GridSpec.for_aoi(aoi)
    log(f"[bold]SailabAI {__version__}[/bold]  ({DISCLAIMER})")
    log(f"study area {aoi.name}: {aoi.effective_bbox.as_tuple()}, grid {grid.width} x {grid.height} px at "
        f"{grid.res:.0f} m ({grid.width * grid.height * grid.pixel_area_km2:,.0f} km2 incl. margins)")
    t = Table("event", "window", "split")
    for e in load_events().events:
        t.add_row(e.id, f"{e.start} to {e.end}", e.split.value)
    console.print(t)
    for name, p in [("configs", config_dir()), ("cubes", cubes_dir()), ("runs", runs_dir()),
                    ("results", results_dir()), ("twin", twin_dir())]:
        log(f"{name:8s} {p}")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
        log(f"torch {torch.__version__}, GPU: {gpu}")
    except ImportError:
        log("torch not installed (install the `ml` extra)")


# --------------------------------------------------------------------------- licences

licences_app = typer.Typer(help="The licence register (team rules 4 and 5).", no_args_is_help=True)
app.add_typer(licences_app, name="licences")


@licences_app.command("check")
def licences_check() -> None:
    """Validate configs/licences.yaml."""
    from sailab.licences import LicenceRegister

    reg = LicenceRegister.load()
    problems = reg.problems()
    t = Table("dataset", "licence", "commercial-safe", "share-alike", "uses", "downloaded")
    for e in reg.entries.values():
        t.add_row(e.id, e.licence, str(e.commercial_safe), "yes" if e.share_alike else "", ", ".join(e.uses),
                  str(e.download_date or ""))
    console.print(t)
    if problems:
        for p in problems:
            log(f"[red]problem[/red] {p}")
        raise typer.Exit(1)
    log("[green]register OK[/green]")


@licences_app.command("mark")
def licences_mark(dataset: str, version: str = typer.Option(None, help="version downloaded")) -> None:
    """Record the first download date (and version) of a dataset."""
    from sailab.licences import LicenceRegister

    reg = LicenceRegister.load()
    reg.mark_downloaded(dataset, version)
    log(f"recorded {dataset} {version or ''}")


# --------------------------------------------------------------------------- synthetic world

synthetic_app = typer.Typer(help="The synthetic demo world.", no_args_is_help=True)
app.add_typer(synthetic_app, name="synthetic")


@synthetic_app.command("generate")
def synthetic_generate(name: str = "demo", resolution: float = typer.Option(None, help="metres per pixel"),
                       years: str = "2014-2026", seed: int = 7) -> None:
    """Build a synthetic datacube with the same layout as the real one."""
    from sailab.synthetic.generate import generate_synthetic_cube

    generate_synthetic_cube(name, resolution, _years(years), seed, log=log)


@app.command("cube")
def cube_info(name: str = "demo") -> None:
    """Summarise a datacube: scenes per event, versions, layers."""
    from sailab.cube import Datacube

    cube = Datacube.open(name)
    log(f"{cube.root} ({cube.kind}), grid {cube.grid.width} x {cube.grid.height} at {cube.grid.res:.0f} m")
    log(f"static layers: {', '.join(cube.static_names)}")
    s = cube.scenes
    if not s.empty:
        t = Table("event", "split", "passes", "first", "last")
        for (event, split), part in s.groupby(["event_id", "split"]):
            t.add_row(event, split, str(len(part)), str(part["time"].min().date()), str(part["time"].max().date()))
        console.print(t)


# --------------------------------------------------------------------------- training

train_app = typer.Typer(help="Train Model 1, Model 2 and the baselines.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("mapping")
def train_mapping_cmd(cube: str = "demo", encoder: str = "resnet34", epochs: int = 12, batch_size: int = 16,
                      chip: int = 128, samples: int = 2400, lr: float = 3e-4, seed: int = 0,
                      pretrained: bool = False) -> None:
    """Model 1: U-Net flood mapping."""
    from sailab.mapping.train import MappingConfig, train_mapping

    train_mapping(MappingConfig(cube=cube, encoder=encoder, epochs=epochs, batch_size=batch_size, chip=chip,
                                samples_per_epoch=samples, lr=lr, seed=seed, pretrained=pretrained), log=log)


@train_app.command("forecast")
def train_forecast_cmd(cube: str = "demo", seeds: list[int] = typer.Option([0], help="one model per seed"),
                       encoder: str = "resnet18", epochs: int = 14, batch_size: int = 12, chip: int = 128,
                       samples: int = 3000, lr: float = 3e-4, include_india: bool = False,
                       map_dropout: float = 0.25) -> None:
    """Model 2: UNet-TT. Pass several --seeds for the deep ensemble."""
    from sailab.forecast.dataset import DropoutConfig
    from sailab.forecast.train import ForecastConfig, train_forecast

    for seed in seeds:
        log(f"[bold]seed {seed}[/bold]")
        train_forecast(ForecastConfig(cube=cube, encoder=encoder, epochs=epochs, batch_size=batch_size, chip=chip,
                                      samples_per_epoch=samples, lr=lr, seed=seed, include_india=include_india,
                                      dropout=DropoutConfig(map=map_dropout)), log=log)


@train_app.command("baselines")
def train_baselines_cmd(cube: str = "demo", xgboost: bool = True, include_india: bool = False) -> None:
    """Fit the historical-frequency, river-threshold and XGBoost baselines on the training events."""
    from sailab.cube import Datacube
    from sailab.forecast.baselines import LevelBaselines
    from sailab.forecast.inputs import SeriesBank
    from sailab.forecast.train import model_points
    from sailab.paths import runs_dir

    c = Datacube.open(cube)
    points = model_points(include_india)
    out = runs_dir() / "model2"
    level = LevelBaselines.fit(c, points)
    level.save(out / "level_baselines.npz")
    log(f"level baselines fitted on training events -> {out / 'level_baselines.npz'}")
    if xgboost:
        from sailab.forecast.xgb import train_xgb

        train_xgb(c, SeriesBank(c, points), level, out / "xgboost.json", log=log)
        log(f"XGBoost -> {out / 'xgboost.json'}")


@app.command()
def calibrate(cube: str = "demo", model_dir: Path = typer.Option(None), alpha: float = 0.1,
              max_pairs: int = 200) -> None:
    """Fit temperature scaling and conformal ranges for Model 2 on the validation event."""
    from sailab.cube import Datacube
    from sailab.forecast.calibration import CalibrationFitter
    from sailab.forecast.evaluate import evaluate_forecasts
    from sailab.forecast.model import ForecastEnsemble
    from sailab.paths import runs_dir
    from sailab.torchutils import device

    folder = model_dir or runs_dir() / "model2"
    c = Datacube.open(cube)
    ens = ForecastEnsemble.from_dir(folder, device())
    fitter = CalibrationFitter(alpha=alpha)
    evaluate_forecasts(c, "val", ens, max_pairs=max_pairs, calibrated=False, calibration_fitter=fitter, log=log)
    cal = fitter.fit(sorted(c.scenes.loc[c.scenes["split"] == "val", "event_id"].dropna().unique()))
    cal.save(folder / "calibration.json")
    log(f"temperatures {cal.temperatures} (default {cal.default_temperature}); pixel q-hat {cal.pixel_qhat}; "
        f"totals {cal.total_quantiles} -> {folder / 'calibration.json'}")


# --------------------------------------------------------------------------- evaluation

evaluate_app = typer.Typer(help="Score models on an event split and add them to the results table.",
                           no_args_is_help=True)
app.add_typer(evaluate_app, name="evaluate")


@evaluate_app.command("mapping")
def evaluate_mapping_cmd(cube: str = "demo", split: str = "val", model: Path = typer.Option(None),
                         max_scenes: int = typer.Option(None), experiment: str = "model1",
                         force: bool = False, reason: str = "") -> None:
    """Model 1 (U-Net) vs the Otsu threshold, against GFM labels and hand labels."""
    from sailab.cube import Datacube
    from sailab.forecast.evaluate import guard_test
    from sailab.mapping.evaluate import evaluate_mapping, record_mapping_results
    from sailab.paths import runs_dir

    c = Datacube.open(cube)
    guard_test(split, c, experiment, force, reason)
    path = model or next(iter(sorted((runs_dir() / "model1").glob("unet_*.pt"))), None)
    scorers = evaluate_mapping(c, split, path, max_scenes, log=log)
    record_mapping_results(scorers, c, split, experiment)
    for ref, models in scorers.items():
        for name, sc in models.items():
            m = sc.macro()
            row = m[(m["subset"] == "all")].set_index("metric")["value"]
            log(f"vs {ref:5s} {name:6s} F1 {row.get('f1', float('nan')):.3f}  IoU {row.get('iou', float('nan')):.3f}  "
                f"precision {row.get('precision', float('nan')):.3f}  recall {row.get('recall', float('nan')):.3f}")


@evaluate_app.command("forecast")
def evaluate_forecast_cmd(cube: str = "demo", split: str = "val", model_dir: Path = typer.Option(None),
                          max_pairs: int = typer.Option(None), experiment: str = "model2",
                          truth: bool = False, force: bool = False, reason: str = "") -> None:
    """Model 2 and the four baselines, per lead time, next to persistence."""
    from sailab.cube import Datacube
    from sailab.forecast.evaluate import evaluate_forecasts, guard_test, load_level_baselines, record_results
    from sailab.paths import runs_dir

    folder = model_dir or runs_dir() / "model2"
    c = Datacube.open(cube)
    guard_test(split, c, experiment, force, reason)
    level = load_level_baselines(folder)
    xgb_model = None
    if (folder / "xgboost.json").exists():
        from sailab.forecast.xgb import load_xgb

        xgb_model = load_xgb(folder / "xgboost.json")
    ensemble = None
    if any(folder.glob("unet_tt_seed*.pt")):
        from sailab.forecast.model import ForecastEnsemble
        from sailab.torchutils import device

        ensemble = ForecastEnsemble.from_dir(folder, device())
    ev = evaluate_forecasts(c, split, ensemble, level, xgb_model, max_pairs=max_pairs, truth=truth, log=log)
    record_results(ev, experiment + ("-truth" if truth else ""), split, c)
    console.print(ev.summary().round(4).to_string())


results_app = typer.Typer(help="The shared results table.", no_args_is_help=True)
app.add_typer(results_app, name="results")


@results_app.command("render")
def results_render() -> None:
    """Regenerate results/RESULTS.md from results/results_table.csv."""
    from sailab.evaluation.results import ResultsTable

    ResultsTable().render_markdown()
    log("results/RESULTS.md updated")


# --------------------------------------------------------------------------- twin

twin_app = typer.Typer(help="The digital twin loop.", no_args_is_help=True)
app.add_typer(twin_app, name="twin")


@twin_app.command("run")
def twin_run(day: str, cube: str = "demo", members: int = 11) -> None:
    """One daily update: map new passes, forecast, write layers and risk."""
    from sailab.twin.loop import Twin, TwinConfig

    s = Twin(TwinConfig(cube=cube, members=members), log=log).step(day)
    log(f"{s['run_id']}: {s['current_flooded_km2']} km2 flooded now; model {s['model']}")


@twin_app.command("replay")
def twin_replay(start: str, end: str, cube: str = "demo", members: int = 11) -> None:
    """Replay a period day by day as if live."""
    from sailab.twin.loop import Twin, TwinConfig

    Twin(TwinConfig(cube=cube, members=members), log=log).replay(start, end)


# --------------------------------------------------------------------------- real data

data_app = typer.Typer(help="Real-data connectors (see docs/data.md for accounts).", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("init")
def data_init(name: str = "real", resolution: float = typer.Option(None, help="metres; default from aoi.yaml")) -> None:
    """Create an empty real-data datacube on the study-area grid."""
    from sailab.config import load_aoi, load_gauges
    from sailab.cube import Datacube, point_features
    from sailab.grid import GridSpec
    from sailab.paths import cubes_dir

    aoi = load_aoi()
    grid = GridSpec.for_aoi(aoi, resolution or aoi.grid.resolution_m)
    cube = Datacube.create(cubes_dir() / name, grid, kind="real", name=name)
    cube.write_geojson("places", point_features(aoi.places, "place"))
    cube.write_geojson("gauges", point_features(load_gauges().all(), "gauge"))
    log(f"created {cube.root}: {grid.width} x {grid.height} px at {grid.res:.0f} m")


@data_app.command("static")
def data_static(cube: str = "real") -> None:
    """DEM, HAND, slope, JRC water, WorldCover onto the cube's grid (no account)."""
    from sailab.cube import Datacube
    from sailab.sources.static_layers import ingest_static

    ingest_static(Datacube.open(cube), log=log)


@data_app.command("gfm")
def data_gfm(start: str, end: str, cube: str = "real") -> None:
    """GFM flood labels for every Sentinel-1 pass between two dates (no account)."""
    from sailab.cube import Datacube
    from sailab.sources.gfm import ingest

    ingest(Datacube.open(cube), start, end, log=log)


@data_app.command("exposure")
def data_exposure(cube: str = "real", buildings: bool = True) -> None:
    """WorldPop population, Microsoft buildings, OSM roads (no account)."""
    from sailab.config import load_aoi
    from sailab.cube import Datacube
    from sailab.sources.exposure_layers import ingest_exposure

    ingest_exposure(Datacube.open(cube), load_aoi().effective_bbox, buildings=buildings, log=log)


@data_app.command("ffd-parse")
def data_ffd_parse(cube: str = "real", folder: Path = typer.Option(None, help="archived bulletins")) -> None:
    """Turn archived FFD bulletins into the cube's discharge table."""
    import pandas as pd

    from sailab.cube import Datacube
    from sailab.paths import archive_dir
    from sailab.sources.ffd import bulletins_to_discharge

    c = Datacube.open(cube)
    df = bulletins_to_discharge(folder or archive_dir() / "ffd")
    if c.has_table("series", "discharge"):
        old = c.table("series", "discharge")
        df = pd.concat([old[old["source"] != "ffd"], df], ignore_index=True)
    c.write_table("series", "discharge", df)
    log(f"{len(df):,} gauge readings from {df['date'].nunique() if len(df) else 0} bulletin days")


archive_app = typer.Typer(help="Daily scrapers for sources that delete files within days.", no_args_is_help=True)
app.add_typer(archive_app, name="archive")


@archive_app.command("run")
def archive_run(source: list[str] = typer.Option(None, help="ffd, irsa, ecmwf, viirs (default: all)")) -> None:
    """Fetch anything new from the daily sources into data/archive/."""
    from sailab.sources.scrapers import run_all

    results = run_all(source or None, log=log)
    if any(v < 0 for v in results.values()):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- platform

db_app = typer.Typer(help="PostGIS / SQLite database.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@db_app.command("sync")
def db_sync(cube: str = "demo") -> None:
    """Load gauges, readings, twin runs and results into the database."""
    from sailab.db.sync import sync_all

    sync_all(cube, log=log)


api_app = typer.Typer(help="The FastAPI backend.", no_args_is_help=True)
app.add_typer(api_app, name="api")


@api_app.command("serve")
def api_serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, cube: str = "demo") -> None:
    """Serve the API (docs at /docs)."""
    import os

    from sailab.api.app import serve

    os.environ.setdefault("SAILAB_CUBE", cube)
    serve(host, port, reload)


# --------------------------------------------------------------------------- tools


@app.command("gpu-memtest")
def gpu_memtest(chip: int = 256, batch_size: int = 8) -> None:
    """Peak GPU memory for Model 1 and Model 2 training steps (first-two-weeks checklist)."""
    from sailab.tools.gpu_memtest import run_memtest

    run_memtest(chip, batch_size, log=log)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
