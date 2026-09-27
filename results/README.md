# Shared results table

Team rule 3: every new score sits next to its baseline.

- `results_table.csv`: append-only; written by `sailab evaluate mapping|forecast`. One row per
  (model, lead time, pixel subset, metric) with the baseline's value, the skill, the commit, the
  split and the data versions.
- `RESULTS.md`: regenerated from the table (`sailab results render`) for the weekly meeting.
- `test_lock.json`: every scoring run on a test event, per datacube. The real 2025 test is scored once.

Commit these files with the code change that produced them.
