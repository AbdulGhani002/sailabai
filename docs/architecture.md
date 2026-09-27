# Architecture

SailabAI is a live computer copy of the river area that updates itself when new satellite, rain or
river data arrives. Four inputs, two models, one dashboard.

![system flow](img/system-flow.svg)

## The twin loop

`sailab twin run <day>` (daily, or `twin replay` over a period as if live):

1. **New passes.** Every Sentinel-1 pass since the last run is mapped by Model 1 (U-Net) when its
   radar image is available, otherwise its GFM label is used. Model 2 always restarts from the
   freshest map, so errors do not pile up between passes (every 3 to 4 days).
2. **State.** Each pixel keeps its latest observed state (flooded or not) and how old that
   observation is: passes often cover only part of the area.
3. **Forecast.** At 06:00 UTC (11 am PKT) Model 2 forecasts the next pass (Sentinel-1 acquisition
   plans are public) and +1, +2, +3, +5 and +7 days. It runs every ensemble seed on a spread of
   GloFAS members; the mean is calibrated; spreads are kept.
4. **Confidence and risk.** Conformal sets mark pixels where the forecast can't tell; people,
   buildings and road kilometres get ranges; each road piece and town gets its own chance.
5. **Outputs.** Cloud-optimised GeoTIFFs and a JSON summary per run in
   `data/twin/<cube>/runs/<date>/`, synced into the database for the API.

Timing rules for inputs (what is really known at issue time) live in
`src/sailab/forecast/inputs.py`: gauge flow up to the 6 am reading of the day, rain up to the
previous day (IMERG is 4 to 15 hours late), forecasts issued that day.

## The datacube

Every layer for one study area on one grid, as plain files (`src/sailab/cube.py`). The synthetic
demo world and the real-data pipeline write the same layout, so nothing downstream cares where the
data came from.

```
cube.json                  grid (UTM 42N), kind (synthetic | real), versions
static/*.tif               dem, hand, slope, normal_water, landcover, population, buildings, road_km, aoi_mask
s1/<scene>.tif             Sentinel-1 VV, VH in dB (int16, 0.01 dB)
s1_pre/<key>.tif           dry-season reference image per satellite path
labels/<scene>.tif         GFM labels: 0 land, 1 normal water, 2 flood, 255 ignore
truth/<scene>.tif          hand-checked labels (whole scenes in the synthetic world)
scenes.parquet             one row per pass: time, orbit, event, split, coverage, versions
series/*.parquet           discharge (FFD 6 am readings), weather (rain, soil moisture)
forecasts/*.parquet        GloFAS members and rain forecasts by issue date and lead day
vector/*.geojson           roads (2 km pieces), rivers, places, gauges, breach sites
```

Production grid: 20 m, about 8,650 x 12,200 pixels. Demo grid: 400 m. Scene ids follow the GFM
pass: `S1_<first frame time>_<A|D><relative orbit>`.

## Storage and API

- **Files are the source of truth** (datacube, twin outputs, checkpoints) and live on the external
  drive (`SAILAB_DATA_DIR`).
- **PostGIS** (Docker, Neon in production) holds runs, road and place risk, gauges, readings and the
  results table for the API and for GIS tools. The same SQLAlchemy models run on SQLite locally;
  on PostgreSQL, `init_db` adds generated `geom` columns with GIST indexes.
- **FastAPI** (`src/sailab/api/app.py`): `/api/meta`, `/api/runs`, `/api/runs/{cube}/{date}`,
  `/api/runs/{cube}/{date}/roads`, `/api/gauges/{id}/hydrograph`, `/api/results`,
  `/api/legend/{layer}`, and `/tiles/{cube}/{date}/{layer}/{z}/{x}/{y}.png`. Tiles are rendered by
  rio-tiler (TiTiler's engine) behind an endpoint that only serves files the twin wrote.
- **Dashboard** (`web/`): Next.js + MapLibre GL, OpenFreeMap basemaps, talks only to its own origin
  (`/api` and `/tiles` are forwarded to FastAPI).

## Uncertainty, end to end

| Source | How it is shown |
|---|---|
| Model (what the network doesn't know) | spread between the 5 ensemble seeds |
| Weather and river forecast | running Model 2 on GloFAS members |
| Calibration (is 70% really 70%?) | temperature scaling per lead time, fitted on 2023 |
| Per pixel | conformal sets: flood / dry / can't tell at 90% |
| Totals (people, buildings, roads) | expected value with a range covering both the member spread and the conformal range |
