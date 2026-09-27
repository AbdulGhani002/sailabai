# The synthetic demo world

`sailab synthetic generate` builds a complete datacube for the study area with invented data, so
every part of SailabAI can be built, tested and demonstrated before the real data pipeline is
complete. **Nothing in it describes real floods.** The dashboard shows a "Synthetic demo data"
badge whenever it serves a synthetic cube, and every file carries the disclaimer.

## What is modelled

- **Terrain.** The real river courses (approximate centre lines in `configs/aoi.yaml`) with
  meanders, floodplains 2-9 km wide between flood bunds, height above the river (HAND) rising behind
  the bunds, old river channels that pond rain, desert sand in the Thal (north-west) and Cholistan
  (south-east), towns sized by population, rice paddies, and roads linking the towns (with the
  Multan-Sukkur motorway line).
- **Rivers.** Flood waves start at the upstream inflows and travel with the FFC time-lag chart delays
  (Marala to Panjnad about 111 hours; in 2025 Qadirabad to Trimmu 112 hours instead of 64). The
  named events match the FFC peaks in `configs/events.yaml` (Qadirabad 1,077,951 cusecs in 2025,
  Panjnad 703,698, Ganda Singh Wala 278,297 in 2023, Khanki 418,736 in 2016, Marala 275,600 in 2026,
  Qadirabad 904,285 in 2014). In 2025 the hill torrents between Marala and Khanki and the planned
  breaches at Qadirabad, Rewaz bridge and Sidhnai shape the flood, as they did in reality.
- **Floods.** Water level from a rating curve on each reach; pixels flood when the level is above
  their HAND, behind the bunds only through overtopping or breaches, and drain slowly. Heavy local
  rain ponds in low ground, which gauges cannot predict.
- **Radar passes.** Every 3 days until 2021, 5-6 days in 2022-2024 (Sentinel-1B was lost), 3-4
  days from 2025 (Sentinel-1C, then only Sentinel-1D from late June 2026); mornings (06:00 PKT,
  descending) and evenings (18:36 PKT, ascending); a third of passes cover only part of the area.
- **Radar images.** Water is dark, but so are desert sand and freshly planted rice; flooded towns
  and trees are bright (double bounce); wind roughens water; speckle as in a 20 m product.
- **Labels.** GFM-style labels carry typical errors (missed floods under trees and in towns, rice
  false alarms, ragged edges, sand excluded); they agree with the error-free truth labels at IoU
  0.53-0.73, which stand in for hand-checked tiles.
- **Inputs for Model 2.** FFD-style readings with noise and gaps; Indian upstream data stops on the
  cut-off date in `configs/gauges.yaml`; GloFAS-style 51-member forecasts with a fixed bias per point
  and errors growing with lead time; ECMWF-style rain forecasts.

## What it is good for, and what it is not

Good for: building and testing code paths, the evaluation protocol, the twin loop, the dashboard,
demos, and checking that a model can learn flood dynamics at all.

Not good for: any claim about real floods or real model skill. Scores on the demo world say the
pipeline works; only scores on real data (GFM labels, hand-labelled tiles, the 2025 test, the 2027
live season) say the model works.
