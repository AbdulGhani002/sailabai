"""Risk down to single roads and buildings, shown as ranges.

A single number ("12,400 buildings will flood") claims more than we know. Every total here comes
with a range built from two sources of doubt: the spread between ensemble members (model and
weather uncertainty) and the conformal range fitted on the validation event (how wrong totals like
this have actually been). The reported range covers both.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sailab.forecast.calibration import Calibration
from sailab.grid import GridSpec


def total_with_range(prob: np.ndarray, weight: np.ndarray, members: np.ndarray | None, calibration: Calibration,
                     measure: str, valid: np.ndarray | None = None) -> dict[str, float]:
    w = np.where(valid, weight, 0.0) if valid is not None else weight
    expected = float(np.nansum(prob * w))
    likely = float(np.nansum(np.where(prob >= 0.5, w, 0.0)))  # in pixels more likely flooded than not
    lo, hi = calibration.total_range(measure, expected)
    if members is not None and len(members) > 1:
        per_member = np.nansum(members * w[None], axis=(1, 2))
        lo = min(lo, float(np.percentile(per_member, 5)))
        hi = max(hi, float(np.percentile(per_member, 95)))
    return {"expected": round(expected, 1), "low": round(max(0.0, lo), 1), "high": round(hi, 1),
            "likely": round(likely, 1)}


def _line_pixels(coords: list[list[float]], grid: GridSpec, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    lon = np.array([c[0] for c in coords])
    lat = np.array([c[1] for c in coords])
    rows, cols = grid.lonlat_to_rowcol(lon, lat)
    # densify in pixel space so short pixels along the line are not skipped
    pts_r, pts_c = [rows[:1]], [cols[:1]]
    for r0, c0, r1, c1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:], strict=True):
        n = int(max(abs(r1 - r0), abs(c1 - c0), 1) * max(1.0, grid.res / step_m))
        t = np.linspace(0, 1, n + 1)[1:]
        pts_r.append(np.round(r0 + t * (r1 - r0)).astype(int))
        pts_c.append(np.round(c0 + t * (c1 - c0)).astype(int))
    r = np.concatenate(pts_r)
    c = np.concatenate(pts_c)
    ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    return r[ok], c[ok]


def road_risk(prob: np.ndarray, members: np.ndarray | None, roads: dict[str, Any], grid: GridSpec,
              min_prob: float = 0.05) -> list[dict[str, Any]]:
    """Flood chance along each road piece: the worst pixel on it, with the member range."""
    out = []
    for f in roads.get("features", []):
        r, c = _line_pixels(f["geometry"]["coordinates"], grid, step_m=grid.res / 2)
        if r.size == 0:
            continue
        p = prob[r, c]
        worst = float(p.max())
        if worst < min_prob:
            continue
        item = {**f["properties"], "max_prob": round(worst, 3), "mean_prob": round(float(p.mean()), 3),
                "share_over_50": round(float((p >= 0.5).mean()), 3)}
        if members is not None and len(members) > 1:
            m = members[:, r, c].max(axis=1)
            item["max_prob_low"] = round(float(np.percentile(m, 5)), 3)
            item["max_prob_high"] = round(float(np.percentile(m, 95)), 3)
        out.append(item)
    return sorted(out, key=lambda d: -d["max_prob"])


def place_risk(prob: np.ndarray, members: np.ndarray | None, population: np.ndarray, places: dict[str, Any],
               grid: GridSpec, calibration: Calibration, radius_km: float = 8.0) -> list[dict[str, Any]]:
    """People in likely-flooded pixels within `radius_km` of each town, with a range."""
    rows, cols = np.ogrid[0:grid.height, 0:grid.width]
    out = []
    for f in places.get("features", []):
        lon, lat = f["geometry"]["coordinates"]
        r0, c0 = grid.lonlat_to_rowcol(lon, lat)
        near = np.hypot(rows - int(r0), cols - int(c0)) * grid.res / 1000.0 <= radius_km
        stats = total_with_range(prob, population, members, calibration, "population", near)
        out.append({"name": f["properties"].get("name"), "lon": lon, "lat": lat, "people": stats,
                    "max_prob": round(float(prob[near].max()) if near.any() else 0.0, 3)})
    return sorted(out, key=lambda d: -d["people"]["expected"])


def exposure_summary(prob: np.ndarray, members: np.ndarray | None, layers: dict[str, np.ndarray], grid: GridSpec,
                     calibration: Calibration, roads: dict[str, Any], places: dict[str, Any],
                     valid: np.ndarray | None = None) -> dict[str, Any]:
    totals = {}
    for measure, weight in layers.items():
        totals[measure] = total_with_range(prob, weight, members, calibration, measure, valid)
    area = np.full(prob.shape, grid.pixel_area_km2, dtype=np.float32)
    totals["area_km2"] = total_with_range(prob, area, members, calibration, "area_km2", valid)
    return {"totals": totals, "roads": road_risk(prob, members, roads, grid),
            "places": place_risk(prob, members, layers.get("population", area), places, grid, calibration)}
