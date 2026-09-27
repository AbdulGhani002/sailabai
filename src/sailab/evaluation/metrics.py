"""Scores for flood maps and flood-chance forecasts.

Everything works on flat arrays plus an optional `valid` mask, and every score has a streaming
accumulator so a whole monsoon can be scored chip by chip without holding it in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def _flat(*arrays: np.ndarray | None, valid: np.ndarray | None = None) -> list[np.ndarray]:
    mask = None if valid is None else np.asarray(valid, dtype=bool).ravel()
    out = []
    for a in arrays:
        a = np.asarray(a).ravel()
        out.append(a if mask is None else a[mask])
    return out


# --------------------------------------------------------------------------- binary maps


@dataclass
class Confusion:
    """Confusion counts for the flood class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @classmethod
    def from_arrays(cls, pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> Confusion:
        p, t = _flat(pred, truth, valid=valid)
        p = p.astype(bool)
        t = t.astype(bool)
        tp = int(np.count_nonzero(p & t))
        fp = int(np.count_nonzero(p & ~t))
        fn = int(np.count_nonzero(~p & t))
        return cls(tp, fp, fn, int(p.size - tp - fp - fn))

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else float("nan")

    @property
    def iou(self) -> float:
        """Intersection over union of the flood class; the same number as CSI."""
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")

    csi = iou

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"iou": self.iou, "f1": self.f1, "precision": self.precision, "recall": self.recall,
                "accuracy": self.accuracy, "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


# --------------------------------------------------------------------------- probabilities


def brier(prob: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> float:
    p, t = _flat(prob, truth, valid=valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p.astype(np.float64) - t.astype(np.float64)) ** 2))


def log_loss(prob: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> float:
    p, t = _flat(prob, truth, valid=valid)
    if p.size == 0:
        return float("nan")
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    t = t.astype(np.float64)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def skill_score(score: float, reference: float, perfect: float) -> float:
    """Generic skill: 1 is perfect, 0 is no better than the reference, negative is worse.

    For Brier (perfect 0) this is the Brier skill score; for IoU/CSI (perfect 1) it is the skill
    over persistence we report at each lead time.
    """
    denom = perfect - reference
    if abs(denom) < EPS or np.isnan(score) or np.isnan(reference):
        return float("nan")
    return float((score - reference) / denom)


@dataclass
class ReliabilityCurve:
    bin_edges: np.ndarray
    mean_prob: np.ndarray
    observed_freq: np.ndarray
    counts: np.ndarray

    @property
    def ece(self) -> float:
        """Expected calibration error: count-weighted gap between forecast % and observed %."""
        total = self.counts.sum()
        if total == 0:
            return float("nan")
        filled = self.counts > 0
        gaps = np.abs(self.mean_prob[filled] - self.observed_freq[filled])
        return float(np.sum(gaps * self.counts[filled]) / total)

    @property
    def mce(self) -> float:
        """Maximum calibration error over non-empty bins."""
        filled = self.counts > 0
        if not filled.any():
            return float("nan")
        return float(np.max(np.abs(self.mean_prob[filled] - self.observed_freq[filled])))


@dataclass
class ProbAccumulator:
    """Streaming Brier score, log loss, reliability and threshold scores for flood-chance maps.

    Probabilities are counted in 1% bins, so IoU, F1, precision and recall can be read off at any
    threshold afterwards (e.g. one tuned on the validation event), in a single pass over the data.
    """

    n_bins: int = 100
    n: int = 0
    sq_err: float = 0.0
    nll: float = 0.0
    positives: int = 0
    bin_count: np.ndarray = field(default=None)  # type: ignore[assignment]
    bin_prob: np.ndarray = field(default=None)  # type: ignore[assignment]
    bin_pos: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bin_count is None:
            self.bin_count = np.zeros(self.n_bins, dtype=np.int64)
            self.bin_prob = np.zeros(self.n_bins, dtype=np.float64)
            self.bin_pos = np.zeros(self.n_bins, dtype=np.int64)

    def update(self, prob: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> None:
        p, t = _flat(prob, truth, valid=valid)
        if p.size == 0:
            return
        p = np.clip(p.astype(np.float64), 0.0, 1.0)
        t = t.astype(bool)
        self.n += p.size
        self.positives += int(t.sum())
        self.sq_err += float(np.sum((p - t) ** 2))
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        self.nll += float(-np.sum(np.where(t, np.log(pc), np.log(1 - pc))))
        idx = np.minimum((p * self.n_bins + 1e-9).astype(int), self.n_bins - 1)
        self.bin_count += np.bincount(idx, minlength=self.n_bins)
        self.bin_prob += np.bincount(idx, weights=p, minlength=self.n_bins)
        self.bin_pos += np.bincount(idx, weights=t, minlength=self.n_bins).astype(np.int64)

    def merge(self, other: ProbAccumulator) -> ProbAccumulator:
        if other.n_bins != self.n_bins:
            raise ValueError("bin counts differ")
        return ProbAccumulator(self.n_bins, self.n + other.n, self.sq_err + other.sq_err, self.nll + other.nll,
                               self.positives + other.positives, self.bin_count + other.bin_count,
                               self.bin_prob + other.bin_prob, self.bin_pos + other.bin_pos)

    @property
    def brier(self) -> float:
        return self.sq_err / self.n if self.n else float("nan")

    @property
    def climatology_brier(self) -> float:
        """Brier score of always forecasting the observed base rate (the 'no skill' reference)."""
        if not self.n:
            return float("nan")
        base = self.positives / self.n
        return base * (1 - base)

    @property
    def log_loss(self) -> float:
        return self.nll / self.n if self.n else float("nan")

    def confusion_at(self, threshold: float = 0.5) -> Confusion:
        """Flood-class confusion counts when pixels with chance >= threshold are called flooded."""
        k = int(round(threshold * self.n_bins))
        tp = int(self.bin_pos[k:].sum())
        fp = int(self.bin_count[k:].sum()) - tp
        fn = self.positives - tp
        return Confusion(tp, fp, fn, self.n - tp - fp - fn)

    def best_threshold(self, metric: str = "f1", lo: float = 0.02, hi: float = 0.98) -> float:
        """Threshold (on the 1% grid) that maximises `metric` on the data seen so far."""
        grid = np.arange(int(round(lo * self.n_bins)), int(round(hi * self.n_bins)) + 1) / self.n_bins
        scores = [getattr(self.confusion_at(t), metric) for t in grid]
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1.0)
        return float(grid[int(np.argmax(scores))])

    def reliability(self, display_bins: int = 10) -> ReliabilityCurve:
        group = max(1, self.n_bins // display_bins)
        count = self.bin_count.reshape(-1, group).sum(1)
        psum = self.bin_prob.reshape(-1, group).sum(1)
        pos = self.bin_pos.reshape(-1, group).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_prob = np.where(count > 0, psum / np.maximum(count, 1), np.nan)
            observed = np.where(count > 0, pos / np.maximum(count, 1), np.nan)
        return ReliabilityCurve(np.linspace(0, 1, len(count) + 1), mean_prob, observed, count)

    @property
    def ece(self) -> float:
        return self.reliability().ece


def reliability_curve(prob: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None,
                      n_bins: int = 10) -> ReliabilityCurve:
    acc = ProbAccumulator()
    acc.update(prob, truth, valid)
    return acc.reliability(n_bins)


def expected_calibration_error(prob: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None,
                               n_bins: int = 10) -> float:
    return reliability_curve(prob, truth, valid, n_bins).ece
