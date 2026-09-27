"""Making "70% chance" mean it floods about 70% of the time.

Temperature scaling rescales the ensemble's logits with one number per lead time, fitted on the
validation event (2023). Split conformal prediction then gives two guarantees on data like the
validation event: per pixel, a set of plausible outcomes (flood, dry, or "can't tell"), and for
area totals such as buildings flooded, a range that contains the true value 90% of the time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(prob: np.ndarray, truth: np.ndarray, bounds: tuple[float, float] = (0.25, 8.0)) -> float:
    """Temperature T minimising log loss of sigmoid(logit(p) / T) on held-out pixels."""
    z = _logit(prob.astype(np.float64))
    y = truth.astype(np.float64)

    def nll(log_t: float) -> float:
        q = np.clip(_sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of nonconformity scores."""
    n = scores.size
    if n == 0:
        return 1.0
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.partition(scores, k - 1)[k - 1])


@dataclass
class Calibration:
    temperatures: dict[str, float] = field(default_factory=dict)  # lead bucket ("+1d".."+7d") -> T
    default_temperature: float = 1.0
    alpha: float = 0.1
    pixel_qhat: float | None = None                               # conformal threshold for pixel sets
    total_quantiles: dict[str, tuple[float, float]] = field(default_factory=dict)  # measure -> (lo, hi) relative errors
    fitted_on: list[str] = field(default_factory=list)

    @staticmethod
    def bucket(lead_days: float) -> str:
        return f"+{int(min(7, max(1, round(lead_days))))}d"

    def temperature(self, lead_days: float) -> float:
        return self.temperatures.get(self.bucket(lead_days), self.default_temperature)

    def apply(self, prob: np.ndarray, lead_days: float) -> np.ndarray:
        t = self.temperature(lead_days)
        if abs(t - 1.0) < 1e-6:
            return prob
        return _sigmoid(_logit(prob) / t).astype(np.float32)

    def pixel_sets(self, prob: np.ndarray) -> np.ndarray:
        """0 = dry, 1 = flood, 2 = can't tell (both outcomes plausible at level 1 - alpha)."""
        if self.pixel_qhat is None:
            return (prob >= 0.5).astype(np.uint8)
        flood_ok = (1 - prob) <= self.pixel_qhat
        dry_ok = prob <= self.pixel_qhat
        out = np.where(flood_ok & dry_ok, 2, np.where(flood_ok, 1, 0))
        return out.astype(np.uint8)

    def total_range(self, measure: str, expected: float) -> tuple[float, float]:
        """Conformal range for an area total (e.g. buildings in flooded pixels)."""
        lo, hi = self.total_quantiles.get(measure, (-0.5, 1.0))
        return max(0.0, expected * (1 + lo)), max(0.0, expected * (1 + hi))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["total_quantiles"] = {k: tuple(v) for k, v in d.get("total_quantiles", {}).items()}
        return cls(**d)


class CalibrationFitter:
    """Collects validation predictions, then fits temperatures and conformal quantiles."""

    def __init__(self, alpha: float = 0.1, max_pixels_per_bucket: int = 400_000, seed: int = 0) -> None:
        self.alpha = alpha
        self.max_pixels = max_pixels_per_bucket
        self.rng = np.random.default_rng(seed)
        self.probs: dict[str, list[np.ndarray]] = {}
        self.truths: dict[str, list[np.ndarray]] = {}
        self.totals: dict[str, list[tuple[float, float]]] = {}

    def add(self, prob: np.ndarray, truth: np.ndarray, valid: np.ndarray, lead_days: float,
            weights: dict[str, np.ndarray] | None = None) -> None:
        b = Calibration.bucket(lead_days)
        p, y = prob[valid], truth[valid].astype(bool)
        take = min(p.size, 40_000)
        if take == 0:
            return
        idx = self.rng.choice(p.size, take, replace=False)
        self.probs.setdefault(b, []).append(p[idx].astype(np.float32))
        self.truths.setdefault(b, []).append(y[idx])
        for name, w in (weights or {}).items():
            self.totals.setdefault(name, []).append((float((prob * w)[valid].sum()), float((truth * w)[valid].sum())))

    def fit(self, fitted_on: list[str]) -> Calibration:
        cal = Calibration(alpha=self.alpha, fitted_on=fitted_on)
        all_p, all_y = [], []
        for b in sorted(self.probs):
            p = np.concatenate(self.probs[b])[: self.max_pixels]
            y = np.concatenate(self.truths[b])[: self.max_pixels]
            if y.any() and (~y).any():
                cal.temperatures[b] = round(fit_temperature(p, y), 4)
            all_p.append(p)
            all_y.append(y)
        if all_p:
            p = np.concatenate(all_p)
            y = np.concatenate(all_y)
            cal.default_temperature = round(fit_temperature(p, y), 4) if (y.any() and (~y).any()) else 1.0
            # conformal pixel sets on calibrated probabilities (buckets pooled)
            pc = np.concatenate([cal.apply(np.concatenate(self.probs[b])[: self.max_pixels],
                                           float(b.strip("+d"))) for b in sorted(self.probs)])
            scores = np.where(y, 1 - pc, pc)
            cal.pixel_qhat = round(conformal_quantile(scores, self.alpha), 4)
        for name, pairs in self.totals.items():
            arr = np.asarray(pairs)
            rel = (arr[:, 1] - arr[:, 0]) / np.maximum(arr[:, 0], 1.0)
            lo = -conformal_quantile(-rel, self.alpha / 2)
            hi = conformal_quantile(rel, self.alpha / 2)
            cal.total_quantiles[name] = (round(float(lo), 4), round(float(hi), 4))
        return cal
