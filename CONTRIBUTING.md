# Working on SailabAI

One GitHub repo, pull requests, a 30-minute weekly meeting, and one shared results table where every
new score sits next to its baseline.

## Workflow

1. Branch from `main` (`feature/<short-name>`), commit small steps with clear messages.
2. Open a pull request; CI must pass (ruff, pytest, PostGIS integration, dashboard build).
3. One teammate reviews. The owner of the area (see README) merges.
4. Any change that produces a score appends it to `results/results_table.csv` through
   `sailab evaluate ...` and commits the updated `results/RESULTS.md` in the same PR.

## Before you push

```powershell
ruff check src tests
pytest -q
cd web; npm run typecheck
```

## The rules (from the team guide)

1. **Split data by flood event, never by random tiles.** Use `EventRegistry` and `forecast_pairs`;
   never sample training chips from validation or test events. `assert_no_leakage` guards the
   training scripts.
2. **Save the data version** (IMERG, GloFAS, GFM) with every sample. Connectors write versions into
   `scenes.parquet`; bump `configs/licences.yaml` when a version changes.
3. **Every new score goes in the results table, next to the baseline.** No screenshots of numbers
   in chat.
4. **Only commercial-safe data in the product.** FABDEM and WorldFloods are tests only;
   `sailab licences check` must pass.
5. **OpenStreetMap roads stay a separate layer** (share-alike).
6. **Never publish public warnings.** Everything we show says "Experimental research output, not an
   official flood warning."

The final test (the 2025 flood) is scored **once**. `sailab evaluate ... --split test` records the
attempt in `results/test_lock.json`; a second attempt needs `--force --reason "..."` and is logged.
Develop and tune on the 2023 validation event and the 2026 replay.

## Code style

- Python 3.10+, type hints, `ruff` (line length 110). Plain names, short docstrings that say *why*.
- New data source: a module in `src/sailab/sources/`, an entry in `configs/licences.yaml`, a parser
  test on a small made-up snippet (never commit third-party data or PDFs).
- New model or baseline: score it with `EventScorer` and append it to the results table.
- Data never goes in git: it lives under `SAILAB_DATA_DIR` (the external drive).
