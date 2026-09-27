# Data

All data is free; most sources need only a free account. Credits and licences are in
`configs/licences.yaml` (`sailab licences check`).

| Data | Source | Detail | Connector | Account |
|---|---|---|---|---|
| Radar images | Sentinel-1 (Copernicus) | 10-20 m, every 3-4 days | `sources/sentinel1.py`: ASF search, HyP3 RTC jobs, Earth Engine | Earthdata / Earth Engine |
| Flood labels | Copernicus GFM | 20 m map for every Sentinel-1 image since 2015 | `sources/gfm.py`: EODC STAC `stac.eodc.eu/api/v1`, collection `GFM` | none |
| Rainfall | NASA GPM IMERG V07 | ~10 km, 4-15 h late | `sources/imerg.py` (earthaccess) | Earthdata |
| Rain forecast | ECMWF open data | 0.25°, to 7 days | `sources/ecmwf_open.py` (daily archive) | none |
| River flow | GloFAS v4.x | daily, 51-member forecasts | `sources/glofas.py` (EWDS cdsapi) | EWDS |
| Gauge readings | PMD FFD Lahore | daily 6 am | `sources/scrapers.py` + `sources/ffd.py` parser | none |
| Terrain | Copernicus DEM GLO-30, ASF HAND | 30 m | `sources/static_layers.py` | none |
| Normal water | JRC Global Surface Water | 30 m | `sources/static_layers.py` (occurrence >= 75%) | none |
| Land cover | ESA WorldCover 2021 | 10 m | `sources/static_layers.py` | none |
| Buildings | Microsoft footprints | outlines -> counts per pixel | `sources/exposure_layers.py` | none |
| Roads | OpenStreetMap | 2 km pieces | `sources/exposure_layers.py` (Overpass) | none |
| Population | WorldPop Global2 R2025A | 100 m | `sources/exposure_layers.py` | none |
| Test maps | Copernicus EMS EMSR838 | 2025 flood | `sources/ems.py` | none |

## Building a real datacube

```powershell
sailab data init --name real --resolution 100      # 20 m for production; 100 m to start fast
sailab data static --cube real                     # DEM, HAND, slope, JRC water, WorldCover
sailab data exposure --cube real                   # WorldPop, buildings, OSM roads
sailab data gfm 2025-07-15 2025-10-10 --cube real  # GFM labels for every pass (repeat per monsoon)
sailab data ffd-parse --cube real                  # gauge readings from archived bulletins
```

Then, once accounts exist: Sentinel-1 images (HyP3 or Earth Engine) into `s1/`, GloFAS forecasts
and reanalysis into `forecasts/glofas.parquet` and `series/discharge.parquet`, IMERG into
`series/weather.parquet`. Every table records its source and version.

Verified on 2026-09-27: the GFM STAC search returns passes over our box (e.g. 4 Sep 2025, orbits
D034 at 06:00 PKT and A042 at 18:36 PKT, product version V0M2R2); DEM, HAND, JRC and WorldCover
tiles, the FFD archive (bulletins June 2024 to October 2025), IRSA's daily page (last 8 days),
ECMWF open data and the EMS EMSR838 activation all respond.

## Things that disappear

| Source | Kept online | What we do |
|---|---|---|
| IRSA daily PDFs | about 8-10 days | `sailab archive run` daily from week one |
| ECMWF open forecasts | about 4 days | archived daily (00 UTC run, total precipitation to 168 h) |
| VIIRS flood files | about 7 days | archived daily once `SAILAB_VIIRS_URL` is set |
| FFD bulletins | archive page, but gaps happen | archived daily with a SHA-256 manifest |

FFD bulletins say "All rights reserved; no reproduction without prior written permission": keep them
for internal research, never redistribute the PDFs, and get written permission for the data feed
(email FFD Lahore; ask about internships too).

## Versions change mid-project

NASA is moving IMERG from V07 to V08, GloFAS v5 is coming, and GFM changed product version in 2025
(the version is in its file names, e.g. `V0M2R2`). Every scene row and results row records the
versions it used; `versioning.mixed_versions()` flags datasets that mix them.

## Upstream data from India

Routine sharing stopped when the Indus Waters Treaty was put in abeyance (April 2025). Indian
points (Akhnoor, Madhopur, Ferozepur) are optional Model 2 inputs (`--include-india`); training
hides them often, and they are missing entirely from the cut-off date in `configs/gauges.yaml`.

## Accounts checklist

- Everyone: Kaggle (one per person), NASA Earthdata, Copernicus Data Space.
- Abdul: one shared Google Earth Engine noncommercial project (Community tier), teammates added;
  EWDS and ASF registrations.
- Supervisor: apply with us for the Earth Engine Partner tier (100,000 compute-hours a month).
- Credentials stay in each tool's own file (`~/.cdsapirc`, `~/.netrc`, `earthengine authenticate`),
  never in the repo.
