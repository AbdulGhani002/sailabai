# Models

Two models, both small enough for the RTX 4050 laptop (6 GB). Numbers below were measured on
2026-09-27 with PyTorch 2.11 (CUDA 12.8), bf16 autocast, batch 8 of 256 x 256 chips
(`sailab gpu-memtest`).

| Model | Parameters | Peak GPU memory (train step) | Guide's estimate |
|---|---|---|---|
| Model 1 U-Net, ResNet-34 encoder | 24.4 M | 0.89 GB | 24 M, 1-2 GB |
| Model 2 UNet-TT, ResNet-18 encoder | 15.9 M (time transformer 0.82 M) | 0.73 GB | 7-25 M, 2-4 GB |
| Model 2 UNet-TT, ResNet-34 encoder | 26.0 M | 0.95 GB | |

Before measuring on Windows, set NVIDIA Control Panel, "CUDA - Sysmem Fallback Policy" to "Prefer
No Sysmem Fallback", or the driver silently spills into system RAM.

## Model 1: flood mapping (`src/sailab/mapping/`)

Job: mark every pixel of a radar image as land, normal water or flood.

Inputs (7 channels, `features.mapping_inputs`): post-flood VV and VH (dB), pre-flood VV and VH from
the same satellite path, DEM, HAND and slope.

1. **Otsu threshold, no training** (`otsu.py`). Split-based: only tiles that hold both water and
   land (10-90% of pixels darker than -15 dB, classes at least 3 dB apart) vote; their median
   threshold is applied to the scene. Water higher than 15 m above the river (HAND) is removed, and
   surfaces that were already dark in the dry-season image (sand, tarmac) only count if they got
   darker by 3 dB. It is the baseline every trained mapper must beat.
2. **U-Net with a ResNet-34 encoder** (`nn/unet.py`, `train.py`). Cross-entropy (flood weighted x3)
   plus soft Dice on the flood class, pixels labelled "ignore" skipped. Chips are drawn 8 per scene
   with half the groups from scenes that flooded. Selected on validation F1 (2023 Sutlej flood).
3. **TerraMind-small** (`terramind.py`, `configs/terramind_small.yaml`). `sailab export terratorch`
   writes 224 x 224 chips (S1GRD + DEM, split by event); fine-tune with `terratorch fit` on Kaggle.
   Target: flood F1 around 0.8. On Sen1Floods11 a plain U-Net scored 91.4 mIoU against 90.8 for
   TerraMind-L, so expect TerraMind's advantage mainly on new, unseen floods. Prithvi is skipped:
   it reads only optical images, which the monsoon clouds block.

## Model 2: flood forecasting, UNet-TT (`src/sailab/forecast/`, `nn/unet_tt.py`)

Job: the chance of flooding (0-100%) at every pixel for the next satellite pass and +1, +2, +3, +5
and +7 days.

- **Map part.** A ResNet U-Net reads 9 channels: the latest flood state per pixel, whether it was
  observed, how old that observation is, normal water, HAND, DEM, slope, a breach-scenario channel
  (1 within 12 km of an open breach), and a "map hidden" flag.
- **Time part.** A 4-layer transformer (d = 128, 4 heads, 0.82 M parameters) reads 37 day tokens:
  30 days of observed flow at the six points (Marala, Qadirabad, Trimmu, Sidhnai, Islam, Panjnad),
  upper-catchment rain, local rain and soil moisture, then 7 days of GloFAS and ECMWF forecasts.
  Every value arrives with a "missing" flag.
- **Joining.** The bottleneck features attend to the day tokens (cross-attention), and a lead-time
  token scales them (FiLM). One network answers any lead time, so every pair of passes up to 7.5
  days apart becomes a training example, and the next pass (at a fractional lead) is just another
  query.
- **Missing data.** Training hides the flood map 25% of the time, each river point 20%, the Indian
  points 50% (they are gone since April 2025 anyway), observed rain 20% and all forecasts 25%.
- **Labels.** GFM flood maps at the real target pass; the loss counts only labelled, non-normal-water
  pixels. Model selection uses the validation Brier score.
- **Not our job:** our own river-flow model. GloFAS forecasts are the river inputs.

## Baselines Model 2 must beat (`forecast/baselines.py`, `forecast/xgb.py`)

1. **Persistence**: repeat the last map (98% where flooded, 1% elsewhere).
2. **Historical frequency**: how often each pixel flooded at this river level (8 flow bins at the
   pixel's reference point, shrunk toward the area-wide rate).
3. **River threshold** (Google's method): each pixel gets the flow above which it floods, fitted
   by minimising errors on the training passes; probabilities from the rates above and below.
4. **XGBoost** on the same inputs, one pixel at a time (state, age, terrain, flows now and at the
   target, rain, lead time).

Each pixel's reference point is the gauge whose flow best explains its flooding in training. At
forecast time the level baselines use the GloFAS ensemble-mean flow for the target day, so every
model sees the same information. If Model 2 cannot beat these, that is a documented result.

## Confidence (`forecast/calibration.py`, `forecast/model.py`)

- **Deep ensemble**: 5 copies of Model 2 with different seeds; their spread is model uncertainty.
- **GloFAS members**: every copy runs on a spread of the 51 members (11 by default in the twin,
  all 51 on request) to show weather and river uncertainty separately.
- **Temperature scaling** per lead time on the validation event, so 70% means about 70%.
- **Conformal prediction** at 90%: per pixel a set (flood, dry, or "can't tell"), and for totals
  (people, buildings, road km) a range that covered the true total 90% of the time on validation.

## Stretch goal

An hourly animation between daily forecasts from a Fourier neural operator trained on the
FloodCastBench 2022 simulation, clearly labelled as a simulation. Not started.
