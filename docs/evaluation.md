# How we test

We only call it a success if a model beats simple baselines on floods it never saw in training.

## Splits (by event, never by random tiles)

| Event | Split | Use |
|---|---|---|
| 2015-2022 and 2024 monsoons | train | training |
| Aug 2023, Sutlej | val | model selection, calibration, decision thresholds |
| Aug-Sep 2025, Chenab, Ravi, Sutlej | test | final test, **scored once** |
| 2026 monsoon | replay | run the twin as if live |
| 2027 monsoon | live | live test |
| Sep 2014, Chenab and Jhelum | stress | river data only (before our Sentinel-1 archive) |

Random tile splits made scores look up to 28% better than they really were in one study, because
neighbouring tiles of one flood land on both sides. `assert_no_leakage` stops a training run that
mixes events. The test event is guarded by `results/test_lock.json`: the first scoring run records
the commit, user and time; a second needs `--force --reason "..."` and is logged, so the report can
state how often the test set was touched. The lock is per datacube: scoring the synthetic demo
world never uses up the real 2025 test.

## Scoring rules (`src/sailab/evaluation/protocol.py`)

1. Remove normal water (rivers, lakes) before scoring.
2. Ignore unsure pixels (GFM exclusion mask, outside the swath, no data).
3. Score newly flooded and drained pixels separately. "Repeat the last map" gets every unchanged
   pixel right, so the change subsets show whether a model sees change coming.
4. Score each flood event on its own, then average across events (macro average).
5. Report forecasts per lead time (+1 to +7 days) next to persistence; pixels near an open breach
   are also scored as their own subset.
6. Mapping is scored against at least two references: GFM labels and hand-checked tiles (the
   error-free labels in the synthetic world; 30-60 hand-labelled tiles in reality). Two expert maps
   of the same flood agreed only 48% in one study.

## Thresholds

The Brier score, log loss and calibration error use the probabilities directly. IoU, F1, precision
and recall need a cut-off. A calibrated forecast rarely says above 50% where only a quarter of
flooded pixels stay flooded until the next pass, so a fixed 0.5 would penalise honesty. Each model
is therefore scored at its F1-best threshold per lead time, **tuned on the validation event** and
applied unchanged to the test event (`runs/model2/thresholds.json`). Scores keep 1% probability
histograms, so any threshold can be read off without re-running the models.

## Metrics and targets

| Metric | What it tells us | Target |
|---|---|---|
| IoU and F1 (flood class) | how well the flood area matches | mapping F1 about 0.8 |
| Precision and recall | false alarms versus missed floods | report both |
| Skill over persistence | better than "repeat the last map"? | positive at 3+ days ahead |
| Brier score, calibration error, reliability | is our % honest? | low calibration error on 2025 |

Skill = (score - persistence) / (perfect - persistence): 1 is perfect, 0 is no better, negative is
worse.

## The results table

`results/results_table.csv` is append-only and committed; every row holds the model's score, the
baseline's score and the skill, with the commit, split and data versions. `results/RESULTS.md` is
regenerated for the weekly meeting (`sailab results render`), and the dashboard's Results page
reads it through the API.
