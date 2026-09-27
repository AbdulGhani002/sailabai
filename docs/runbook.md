# Runbook

## First two weeks (from the team guide), as commands

Some data disappears if we wait; start the scrapers first.

- [ ] **Everyone**: Kaggle account (one per person), NASA Earthdata, Copernicus Data Space.
- [ ] **Everyone**: install and clone:
  ```powershell
  git clone git@github.com:AbdulGhani002/sailabai.git; cd sailabai
  python -m pip install uv; python -m uv venv .venv --python 3.12
  python -m uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -m uv pip install --python .venv\Scripts\python.exe -e ".[ml,geo,data,api,dev]"
  .venv\Scripts\activate; pytest -q
  ```
- [ ] **Abdul**: one shared Earth Engine noncommercial project (Community tier), add both teammates;
  register on EWDS (GloFAS) and ASF.
- [ ] **Abdul**: start the daily scrapers (FFD bulletins, IRSA PDFs kept ~10 days, ECMWF open
  forecasts kept ~4 days, VIIRS flood files kept ~7 days):
  ```powershell
  sailab archive run
  ```
  Schedule it daily. Windows Task Scheduler (runs at 12:00 PKT, after FFD's midday bulletin):
  ```powershell
  schtasks /Create /SC DAILY /ST 12:00 /TN "SailabAI archive" /TR "C:\CC\Code\sailabai\.venv\Scripts\sailab.exe archive run"
  ```
  On the server, the `worker` service in `docker-compose.yml` does the same.
- [ ] **Abdul**: email FFD Lahore for old gauge data and permission to use their data feed (the
  bulletins are "all rights reserved"); ask about student internships too.
- [ ] **Member 2**: download Sen1Floods11 and run IBM's TerraMind-small flood notebook on Kaggle;
  then `sailab export terratorch` and `terratorch fit -c configs/terramind_small.yaml`.
- [ ] **Member 3**: pull GloFAS flow for Marala, Qadirabad, Trimmu, Sidhnai, Islam and Panjnad and
  check each sits on the right river cell (`sources/glofas.py: snap_points`).
- [ ] **One laptop**: GPU memory test (`sailab gpu-memtest`), after setting the NVIDIA "CUDA
  Sysmem Fallback Policy" to "Prefer No Sysmem Fallback". Results so far are in docs/models.md.
- [ ] **Everyone**: keep the licence register current (`configs/licences.yaml`,
  `sailab licences mark <dataset> --version ...`, `sailab licences check`).
- [ ] **Supervisor**: apply with us for the Earth Engine Partner tier.

## The demo, end to end (about an hour on the laptop)

```powershell
sailab synthetic generate --name demo
sailab train mapping --epochs 12
sailab train baselines
sailab train forecast --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4
sailab calibrate
sailab evaluate mapping --split val
sailab evaluate forecast --split val              # also tunes decision thresholds
sailab evaluate forecast --split test             # once: recorded in results/test_lock.json
sailab twin replay 2025-08-10 2025-09-20
sailab api serve                                   # then `npm run dev` in web/
```

## Daily operations (live mode, 2027 monsoon)

1. 12:00 PKT: `sailab archive run` (bulletins, IRSA, ECMWF, VIIRS).
2. After the day's GFM passes are published (06:00 and 18:30 PKT passes, a few hours later):
   `sailab data gfm <yesterday> <today> --cube real`, then the radar images for Model 1.
3. After GloFAS and IMERG update: refresh `forecasts/glofas.parquet` and `series/weather.parquet`.
4. `sailab twin run <today> --cube real` (writes layers and risk, syncs PostGIS).
5. Score each new satellite pass against yesterday's forecasts for the live scorecard
   (`sailab evaluate forecast --split live`).

## When something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `LeakageError` | a training script saw a validation or test event | fix the split, never the guard |
| `TestAlreadyUsedError` | the test event was already scored | tune on `val`; rerun test only with `--force --reason` |
| `LicenceError` | a non-commercial or unchecked dataset in product use | use a commercial-safe source or keep it in tests |
| Dashboard says it can't reach the API | `sailab api serve` not running, or no twin runs | start it; run `sailab twin replay` |
| GPU training crawls on Windows | driver spilling into system RAM | set "Prefer No Sysmem Fallback" |
| FFD parser misses a station | bulletin layout changed | adjust `sources/ffd.py` and add a snippet test |
