"""The scoring rules the whole team uses, so every number in the results table means the same thing.

1. Remove normal water (rivers, lakes) before scoring: it is easy to get right and inflates scores.
2. Ignore unsure pixels (GFM exclusion mask, outside the swath, no data).
3. Score newly flooded and drained pixels separately from the full map. "Repeat the last map"
   gets every unchanged pixel right, so change pixels are where a forecast has to earn its keep.
4. Score each flood event on its own, then average across events (macro average), so one huge
   flood does not drown out the others.
5. Forecasts are reported per lead time, next to persistence.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from sailab import labels as L
from sailab.evaluation.metrics import ProbAccumulator, skill_score

SUBSETS = ("all", "newly_flooded", "drained")


def scoring_mask(target_label: np.ndarray, normal_water: np.ndarray | None = None,
                 extra_valid: np.ndarray | None = None) -> np.ndarray:
    """Pixels that count: known label, not normal water, inside the study area."""
    valid = (target_label != L.IGNORE) & (target_label != L.NORMAL_WATER)
    if normal_water is not None:
        valid &= ~normal_water.astype(bool)
    if extra_valid is not None:
        valid &= extra_valid.astype(bool)
    return valid


def lead_bucket(lead_days: float | None) -> str:
    """Report forecasts by whole-day lead time: +1d ... +7d. Mapping scores use 'map'."""
    if lead_days is None:
        return "map"
    return f"+{int(min(7, max(1, round(lead_days))))}d"


Thresholds = float | dict[str, float] | None


def _threshold_for(thresholds: Thresholds, lead: str, default: float) -> float:
    if thresholds is None:
        return default
    if isinstance(thresholds, dict):
        return float(thresholds.get(lead, default))
    return float(thresholds)


class EventScorer:
    """Accumulates scores for one model, grouped by (event, lead, subset).

    Probabilities are kept as 1% histograms, so threshold scores (IoU, F1, ...) can be reported at
    0.5 or at thresholds tuned on the validation event, without a second pass over the data.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._cells: dict[tuple[str, str, str], ProbAccumulator] = defaultdict(ProbAccumulator)

    def add(self, event_id: str, prob: np.ndarray, target_label: np.ndarray, *,
            prev_flood: np.ndarray | None = None, prev_known: np.ndarray | None = None,
            normal_water: np.ndarray | None = None, extra_valid: np.ndarray | None = None,
            lead_days: float | None = None, extra_subsets: dict[str, np.ndarray] | None = None) -> None:
        """Add one predicted flood-chance map and the label it is scored against.

        prev_flood is the flood map the forecast started from (for the change subsets);
        prev_known marks where that map was actually observed. extra_subsets adds named pixel
        groups scored on their own, e.g. pixels near a dam breach.
        """
        valid = scoring_mask(target_label, normal_water, extra_valid)
        truth = target_label == L.FLOOD
        lead = lead_bucket(lead_days)
        self._cells[(event_id, lead, "all")].update(prob, truth, valid)
        for name, subset in (extra_subsets or {}).items():
            sub = valid & subset.astype(bool)
            if sub.any():
                self._cells[(event_id, lead, name)].update(prob, truth, sub)
        if prev_flood is not None:
            prev = prev_flood.astype(bool)
            base = valid if prev_known is None else valid & prev_known.astype(bool)
            self._cells[(event_id, lead, "newly_flooded")].update(prob, truth, base & ~prev)
            # drained: the event is "dry again", so score the chance of drying on pixels that were wet
            self._cells[(event_id, lead, "drained")].update(1.0 - prob, ~truth, base & prev)

    def per_event(self, thresholds: Thresholds = None) -> pd.DataFrame:
        rows = []
        for (event, lead, subset), acc in sorted(self._cells.items()):
            t = _threshold_for(thresholds, lead, self.threshold)
            if subset == "drained":
                t = 1.0 - t  # the drained subset scores the chance of drying
            conf = acc.confusion_at(t)
            metrics = {
                "iou": conf.iou, "f1": conf.f1, "precision": conf.precision, "recall": conf.recall,
                "brier": acc.brier, "ece": acc.ece, "log_loss": acc.log_loss,
            }
            for metric, value in metrics.items():
                rows.append({"event": event, "lead": lead, "subset": subset, "metric": metric,
                             "value": value, "n_pixels": acc.n, "threshold": round(t, 2)})
        return pd.DataFrame(rows, columns=["event", "lead", "subset", "metric", "value", "n_pixels", "threshold"])

    def macro(self, thresholds: Thresholds = None) -> pd.DataFrame:
        """Average of per-event scores (NaN events skipped), with the number of events used."""
        per = self.per_event(thresholds)
        if per.empty:
            return per.assign(n_events=[])
        grouped = per.groupby(["lead", "subset", "metric"], sort=True)
        out = grouped["value"].agg(lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"))
        n_events = grouped["value"].agg(lambda v: int(np.isfinite(v).sum()))
        n_pixels = grouped["n_pixels"].sum()
        return pd.DataFrame({"value": out, "n_events": n_events, "n_pixels": n_pixels}).reset_index()

    def pooled(self, lead: str | None = None, subset: str = "all") -> ProbAccumulator:
        """All events pooled for one lead and subset (for reliability plots and threshold tuning)."""
        acc = ProbAccumulator()
        for (_, cell_lead, cell_subset), cell in self._cells.items():
            if cell_subset == subset and (lead is None or cell_lead == lead):
                acc = acc.merge(cell)
        return acc

    reliability = pooled

    def best_thresholds(self, metric: str = "f1") -> dict[str, float]:
        """Per lead time, the threshold that maximises `metric` over all pixels (pooled events).
        Tune this on the validation event only; apply it unchanged to the test event."""
        leads = sorted({lead for (_, lead, _) in self._cells})
        return {lead: self.pooled(lead).best_threshold(metric) for lead in leads}


def compare_to_baseline(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Join two macro tables and add skill over the baseline for each metric.

    Skill uses the metric's perfect value: 1 for iou/f1/precision/recall, 0 for brier/ece/log_loss.
    """
    perfect = {"iou": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "brier": 0.0, "ece": 0.0, "log_loss": 0.0}
    merged = model.merge(baseline, on=["lead", "subset", "metric"], how="left", suffixes=("", "_baseline"))
    merged["skill"] = [
        skill_score(v, b, perfect.get(m, 1.0)) for v, b, m in
        zip(merged["value"], merged["value_baseline"], merged["metric"], strict=True)
    ]
    return merged
