"""The four baselines Model 2 must beat.

1. Persistence: repeat the last flood map.
2. Historical frequency: how often each pixel flooded at this river level.
3. River threshold: a pixel floods when flow passes its own learned level (Google's method).
4. XGBoost on the same inputs, pixel by pixel.

Baselines 2 and 3 need a river level for every pixel. We give each pixel the forecast point whose
flow best explains that pixel's flooding in the training events, then use the GloFAS ensemble mean
at the forecast's lead time (the flow we would actually know when issuing the forecast).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sailab import labels as L
from sailab.cube import Datacube
from sailab.features import norm_flow
from sailab.forecast.inputs import FloodState, SeriesBank

EPS = 1e-6


def persistence(state: FloodState | None, shape: tuple[int, int], p_flood: float = 0.98,
                p_dry: float = 0.01) -> np.ndarray:
    """Repeat the last map, as a probability (never exactly 0 or 1, so log loss stays finite)."""
    if state is None:
        return np.full(shape, p_dry, dtype=np.float32)
    return np.where(state.flooded, p_flood, p_dry).astype(np.float32)


# --------------------------------------------------------------------------- shared: training observations


@dataclass
class PixelHistory:
    """Label observations of every pixel on every training pass, with the flow at each point."""

    flooded: np.ndarray   # (S, H, W) uint8: 1 flood, 0 dry, 255 unknown
    flows: np.ndarray     # (S, P) observed cusecs at each point on the pass day
    points: list[str]


def pixel_history(cube: Datacube, splits: list[str], points: list[str], coarsen: int = 1) -> PixelHistory:
    scenes = cube.scenes[cube.scenes["split"].isin(splits)]
    disc = cube.discharge
    stack, flows = [], []
    for _, s in scenes.iterrows():
        label = cube.read_label(s["scene_id"])
        if coarsen > 1:
            label = label[::coarsen, ::coarsen]
        obs = np.where(label == L.IGNORE, 255, np.where(label == L.FLOOD, 1, 0)).astype(np.uint8)
        obs[label == L.NORMAL_WATER] = 255
        day = s["time"].tz_convert(None).normalize()
        row = disc.reindex([day])[points].to_numpy()[0] if day in disc.index else np.full(len(points), np.nan)
        stack.append(obs)
        flows.append(row)
    return PixelHistory(np.stack(stack), np.asarray(flows, dtype=np.float64), list(points))


def assign_reference_points(history: PixelHistory, min_obs: int = 8) -> np.ndarray:
    """Index of the point whose (log) flow correlates best with each pixel's flooding."""
    s, h, w = history.flooded.shape
    known = history.flooded != 255
    y = np.where(known, history.flooded, 0).astype(np.float32)
    best = np.zeros((h, w), dtype=np.int16)
    best_r = np.full((h, w), -np.inf, dtype=np.float32)
    for j in range(len(history.points)):
        q = np.log(np.maximum(history.flows[:, j], 1.0))
        ok = np.isfinite(history.flows[:, j])
        if ok.sum() < min_obs:
            continue
        qj = np.where(ok, q, 0.0)[:, None, None].astype(np.float32)
        kj = known & ok[:, None, None]
        nj = kj.sum(0).astype(np.float32)
        mq = (qj * kj).sum(0) / np.maximum(nj, 1)
        my = (y * kj).sum(0) / np.maximum(nj, 1)
        cov = ((qj - mq) * (y - my) * kj).sum(0)
        vq = (((qj - mq) ** 2) * kj).sum(0)
        vy = (((y - my) ** 2) * kj).sum(0)
        r = cov / np.sqrt(np.maximum(vq * vy, EPS))
        r = np.where(nj >= min_obs, r, -np.inf)
        better = r > best_r
        best[better] = j
        best_r[better] = r[better]
    return best


def _upsample(a: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    if factor == 1:
        return a
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)[:shape[0], :shape[1]]


# --------------------------------------------------------------------------- 2. historical frequency


class HistoricalFrequency:
    """P(flood | pixel, flow bin), shrunk towards the area-wide rate for rarely observed bins."""

    def __init__(self, n_bins: int = 8, prior_strength: float = 2.0) -> None:
        self.n_bins = n_bins
        self.prior_strength = prior_strength

    def fit(self, history: PixelHistory, ref: np.ndarray) -> HistoricalFrequency:
        s, h, w = history.flooded.shape
        logq = np.log(np.maximum(history.flows, 1.0))
        self.edges = np.nanquantile(logq, np.linspace(0, 1, self.n_bins + 1)[1:-1], axis=0)  # (bins-1, P)
        bins = np.stack([np.searchsorted(self.edges[:, j], logq[:, j]) for j in range(logq.shape[1])], axis=1)
        pix_bins = bins[:, ref.ravel()].reshape(s, h, w)  # flow bin of each pixel's reference point, per pass
        known = history.flooded != 255
        flooded = (history.flooded == 1) & known
        counts = np.zeros((self.n_bins, h, w), dtype=np.float32)
        hits = np.zeros((self.n_bins, h, w), dtype=np.float32)
        for b in range(self.n_bins):
            in_bin = (pix_bins == b) & known
            counts[b] = in_bin.sum(0)
            hits[b] = (in_bin & flooded).sum(0)
        area_rate = hits.sum((1, 2)) / np.maximum(counts.sum((1, 2)), 1)
        k = self.prior_strength
        self.table = ((hits + k * area_rate[:, None, None]) / (counts + k)).astype(np.float32)
        self.ref = ref
        return self

    def predict(self, flows_now: np.ndarray) -> np.ndarray:
        """flows_now: (P,) cusecs at the target time for each point."""
        logq = np.log(np.maximum(flows_now, 1.0))
        bins = np.array([np.searchsorted(self.edges[:, j], logq[j]) if np.isfinite(logq[j]) else self.n_bins // 2
                         for j in range(len(logq))])
        pix_bin = bins[self.ref]
        return np.take_along_axis(self.table, pix_bin[None].astype(np.int64), axis=0)[0]


# --------------------------------------------------------------------------- 3. river threshold


class RiverThreshold:
    """Per-pixel flow threshold that best separates flooded from dry passes (Nevo et al. 2022 style)."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing

    def fit(self, history: PixelHistory, ref: np.ndarray) -> RiverThreshold:
        s, h, w = history.flooded.shape
        known = history.flooded != 255
        flooded = (history.flooded == 1) & known
        self.threshold = np.full((h, w), np.inf, dtype=np.float32)
        self.p_above = np.zeros((h, w), dtype=np.float32)
        self.p_below = np.zeros((h, w), dtype=np.float32)
        for j in range(len(history.points)):
            sel = ref == j
            if not sel.any():
                continue
            q = history.flows[:, j]
            ok = np.isfinite(q)
            order = np.argsort(np.where(ok, q, np.inf))
            order = order[ok[order]]
            if order.size < 4:
                continue
            qs = q[order]
            f = flooded[order][:, sel].astype(np.float32)   # (S, n)
            k = known[order][:, sel].astype(np.float32)
            d = k - f                                          # dry and known
            # threshold between sorted flows i-1 and i: flood if q >= qs[i]
            fb = np.vstack([np.zeros((1, f.shape[1])), np.cumsum(f, 0)])   # floods at or below cut
            db = np.vstack([np.zeros((1, d.shape[1])), np.cumsum(d, 0)])   # dry at or below cut
            errors = fb + (db[-1] - db)                                     # missed floods + false alarms
            cut = errors.argmin(0)
            thr = np.where(cut < len(qs), qs[np.minimum(cut, len(qs) - 1)], np.inf)
            n_below = np.take_along_axis(np.vstack([np.zeros((1, k.shape[1])), np.cumsum(k, 0)]), cut[None], 0)[0]
            f_below = np.take_along_axis(fb, cut[None], 0)[0]
            n_above = k.sum(0) - n_below
            f_above = f.sum(0) - f_below
            a = self.smoothing
            self.threshold[sel] = thr
            self.p_above[sel] = (f_above + a * 0.5) / (n_above + a)
            self.p_below[sel] = (f_below + a * 0.02) / (n_below + a)
        self.ref = ref
        return self

    def predict(self, flows_now: np.ndarray) -> np.ndarray:
        q = flows_now[self.ref]
        return np.where(np.isfinite(q) & (q >= self.threshold), self.p_above, self.p_below).astype(np.float32)


# --------------------------------------------------------------------------- the level-based baselines on a cube


@dataclass
class LevelBaselines:
    points: list[str]
    ref: np.ndarray
    freq: HistoricalFrequency
    thresh: RiverThreshold
    coarsen: int
    shape: tuple[int, int]

    @classmethod
    def fit(cls, cube: Datacube, points: list[str], splits: tuple[str, ...] = ("train",)) -> LevelBaselines:
        coarsen = max(1, int(np.ceil(np.sqrt(cube.grid.width * cube.grid.height / 2_000_000))))
        hist = pixel_history(cube, list(splits), points, coarsen)
        ref = assign_reference_points(hist)
        return cls(points, ref, HistoricalFrequency().fit(hist, ref), RiverThreshold().fit(hist, ref), coarsen,
                   cube.grid.shape)

    def flows_at(self, series: SeriesBank, issue_day: pd.Timestamp, lead_days: float) -> np.ndarray:
        """GloFAS ensemble-mean flow at each point for the target time (what we know at issue)."""
        raw = series.glofas_raw(issue_day)  # (P, leads, members)
        lead = int(np.clip(round(lead_days), 1, raw.shape[1]))
        with np.errstate(all="ignore"):
            q = np.nanmean(raw[:, lead - 1, :], axis=-1)
        idx = [series.points.index(p) for p in self.points]
        return q[idx]

    def predict(self, which: str, flows: np.ndarray) -> np.ndarray:
        model = self.freq if which == "historical_frequency" else self.thresh
        return _upsample(model.predict(flows), self.coarsen, self.shape)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ref=self.ref, table=self.freq.table, edges=self.freq.edges,
                            threshold=self.thresh.threshold, p_above=self.thresh.p_above, p_below=self.thresh.p_below,
                            meta=json.dumps({"points": self.points, "coarsen": self.coarsen, "shape": list(self.shape),
                                             "n_bins": self.freq.n_bins}))

    @classmethod
    def load(cls, path: Path) -> LevelBaselines:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        freq = HistoricalFrequency(meta["n_bins"])
        freq.table, freq.edges, freq.ref = z["table"], z["edges"], z["ref"]
        thresh = RiverThreshold()
        thresh.threshold, thresh.p_above, thresh.p_below, thresh.ref = z["threshold"], z["p_above"], z["p_below"], z["ref"]
        return cls(meta["points"], z["ref"], freq, thresh, meta["coarsen"], tuple(meta["shape"]))


# --------------------------------------------------------------------------- 4. XGBoost


XGB_FEATURES = ["state", "state_known", "state_age", "hand", "dem", "slope", "normal_water", "breach",
                "q_ref_now", "q_ref_target", "q_ref_max", "rain_obs_3d", "rain_fc_to_target", "soil_moisture",
                "lead_days"]


def xgb_pixel_features(maps: np.ndarray, series: SeriesBank, issue_day: pd.Timestamp, lead_days: float,
                       ref: np.ndarray, coarsen: int) -> np.ndarray:
    """(H*W, F) features from Model 2's own inputs, so the comparison is on the same information."""
    h, w = maps.shape[1:]
    i = series.day_index(issue_day)
    p = len(series.points)
    ref_full = _upsample(ref, coarsen, (h, w)).astype(np.int64)
    q_now = np.nan_to_num(series.q[i, :p], nan=0.0)
    raw = series.gl[i]
    with np.errstate(all="ignore"):
        mean_fc = np.nanmean(raw, axis=-1)  # (P, leads)
    lead = int(np.clip(round(lead_days), 1, mean_fc.shape[1]))
    q_target = np.nan_to_num(norm_flow(mean_fc[:, lead - 1]), nan=0.0)
    q_max = np.nan_to_num(norm_flow(np.nanmax(mean_fc[:, :lead], axis=1)), nan=0.0)
    rain_obs = np.nansum(series.weather[i - 3:i, 0])
    rain_fc = np.nansum(series.rf[i, :lead, 0])
    soil = np.nan_to_num(series.weather[i - 1, 2])
    cols = [maps[0], maps[1], maps[2], maps[4], maps[5], maps[6], maps[3], maps[7],
            q_now[ref_full], q_target[ref_full], q_max[ref_full],
            np.full((h, w), rain_obs), np.full((h, w), rain_fc), np.full((h, w), soil), np.full((h, w), lead_days)]
    return np.stack([np.asarray(c, dtype=np.float32).ravel() for c in cols], axis=1)
