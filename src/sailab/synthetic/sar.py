"""Synthetic Sentinel-1 passes, radar images and GFM-style labels.

The radar model is deliberately imperfect in the ways real radar is: open water is dark, but so
are desert sand and freshly planted rice paddies; flooded towns and trees look bright (double
bounce); wind roughens water. GFM-style labels inherit those errors, while the separate "truth"
labels do not, which stands in for our hand-checked tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from sailab import labels as L
from sailab.synthetic.terrain import BARE, BUILT, CROP, GRASS, SHRUB, TREE, WATER, WETLAND, Terrain

ENL = 5.0  # equivalent number of looks: speckle strength of a 20 m GRD pixel

# Dry-state backscatter (VV, VH) in dB per WorldCover class
BACKSCATTER = {
    TREE: (-7.5, -13.0), SHRUB: (-10.0, -17.0), GRASS: (-11.0, -18.5), CROP: (-10.5, -17.5),
    BUILT: (-2.0, -9.0), BARE: (-19.0, -27.0), WATER: (-21.0, -28.0), WETLAND: (-13.5, -21.0),
}
ORBITS = ["D034", "A042", "D107", "A115"]


@dataclass
class Pass:
    time: pd.Timestamp
    platform: str
    orbit: str
    coverage: np.ndarray  # bool mask of imaged pixels


def _platforms(t: pd.Timestamp, k: int) -> str:
    if t < pd.Timestamp("2022-01-01", tz="UTC"):
        return "S1A" if k % 2 == 0 else "S1B"
    if t < pd.Timestamp("2025-01-01", tz="UTC"):
        return "S1A"
    if t < pd.Timestamp("2026-06-25", tz="UTC"):
        return "S1A" if k % 2 == 0 else "S1C"
    return "S1D"  # since late June 2026 only Sentinel-1D covers Pakistan


def revisit_days(year: int) -> tuple[int, ...]:
    if year <= 2021:
        return (3,)
    if year <= 2024:
        return (5, 6)  # Sentinel-1B failed in December 2021
    return (3, 4)


def pass_schedule(year: int, shape: tuple[int, int], rng: np.random.Generator,
                  start: str = "06-01", end: str = "10-15") -> list[Pass]:
    """Passes every 3 to 6 days, alternating morning (descending, ~6 am PKT) and evening (ascending,
    ~6 pm PKT). About a third of passes only cover part of the study area."""
    gaps = revisit_days(year)
    t = pd.Timestamp(f"{year}-{start}", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, gaps[0])))
    stop = pd.Timestamp(f"{year}-{end}", tz="UTC")
    passes: list[Pass] = []
    k = int(rng.integers(0, len(ORBITS)))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    yy = yy / shape[0] - 0.5
    xx = xx / shape[1] - 0.5
    while t <= stop:
        orbit = ORBITS[k % len(ORBITS)]
        when = t + (pd.Timedelta(hours=1, minutes=0, seconds=50) if orbit.startswith("D")
                    else pd.Timedelta(hours=13, minutes=36, seconds=26))
        if rng.random() < 0.68:
            coverage = np.ones(shape, dtype=bool)
        else:
            theta = rng.uniform(0, 2 * np.pi)
            proj = np.cos(theta) * xx + np.sin(theta) * yy
            coverage = proj > np.quantile(proj, 1 - rng.uniform(0.55, 0.9))
        passes.append(Pass(when, _platforms(when, len(passes)), orbit, coverage))
        t += pd.Timedelta(days=int(rng.choice(gaps)))
        k += 1
    return passes


def simulate_sar(terrain: Terrain, when: pd.Timestamp, orbit: str, rng: np.random.Generator,
                 flooded: np.ndarray | None = None, depth: np.ndarray | None = None,
                 coverage: np.ndarray | None = None, soil_moisture: float = 0.2,
                 wind: float = 0.0) -> np.ndarray:
    """(2, H, W) VV and VH backscatter in dB, NaN outside the swath."""
    shape = terrain.grid.shape
    lc = terrain.landcover
    vv = np.full(shape, -10.5, dtype=np.float32)
    vh = np.full(shape, -17.5, dtype=np.float32)
    for code, (a, b) in BACKSCATTER.items():
        sel = lc == code
        vv[sel] = a
        vh[sel] = b

    doy = when.dayofyear
    growth = float(np.clip((doy - 170) / 90.0, 0, 1))  # crops grow from late June to mid-September
    crop = lc == CROP
    vv[crop] += 1.5 * growth
    vh[crop] += 3.5 * growth
    if doy < 213:  # July: freshly transplanted rice fields hold standing water
        vv[terrain.paddy] = -16.5
        vh[terrain.paddy] = -24.0
    else:
        vv[terrain.paddy] += 1.0 + 1.5 * growth
    wet = (lc == BARE) | crop | (lc == GRASS)
    vv[wet] += 1.5 * soil_moisture
    vh[wet] += 1.0 * soil_moisture

    water = terrain.normal_water.copy()
    if flooded is not None:
        d = depth.astype(np.float32) if depth is not None else np.ones(shape, np.float32)
        open_flood = flooded & ~np.isin(lc, [BUILT, TREE])
        partial = open_flood & crop & (d < 0.4) & (growth > 0.6)  # sugarcane sticking out of shallow water
        water |= open_flood & ~partial
        vv[partial] = -8.5
        vh[partial] = -15.0
        vv[flooded & (lc == BUILT)] = 2.0     # double bounce between water and walls
        vh[flooded & (lc == BUILT)] = -7.0
        vv[flooded & (lc == TREE)] = -4.5
        vh[flooded & (lc == TREE)] = -11.0
    vv[water] = -21.0
    vh[water] = -28.0
    if wind > 0:
        gust = ndimage.gaussian_filter(rng.standard_normal(shape), 8)
        gust = np.clip(gust / (gust.std() + 1e-9), 0, None) * wind
        vv[water] += 5.5 * gust[water]
        vh[water] += 3.0 * gust[water]

    ascending = orbit.startswith("A")
    across = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)[None, :] * (1 if ascending else -1)
    vv += 1.2 * across + terrain.sar_texture[0]
    vh += 1.0 * across + terrain.sar_texture[1]
    field = ndimage.gaussian_filter(rng.standard_normal(shape), 3)
    field = (0.7 * field / (field.std() + 1e-9)).astype(np.float32)
    vv += field
    vh += field

    out = np.empty((2,) + shape, dtype=np.float32)
    for i, band in enumerate((vv, vh)):
        linear = 10 ** (band / 10.0) * rng.gamma(ENL, 1.0 / ENL, shape)
        out[i] = 10.0 * np.log10(np.maximum(linear, 1e-6))
    if coverage is not None:
        out[:, ~coverage] = np.nan
    return out


def gfm_like_label(terrain: Terrain, flooded: np.ndarray, sar: np.ndarray, when: pd.Timestamp,
                   rng: np.random.Generator, coverage: np.ndarray) -> np.ndarray:
    """Labels as an automatic radar product would draw them, including its typical mistakes."""
    lc = terrain.landcover
    vv = np.nan_to_num(sar[0], nan=0.0)
    observed = flooded.copy()
    observed &= ~((lc == BUILT) | (lc == TREE))       # double bounce hides flooded towns and trees
    observed &= ~(flooded & (vv > -14.0))             # wind-roughened or partly submerged water
    observed |= terrain.paddy & (vv < -15.0) & (rng.random(lc.shape) < 0.6)  # dark rice paddies
    edge = flooded ^ ndimage.binary_erosion(flooded)
    observed ^= edge & (rng.random(lc.shape) < 0.15)  # ragged flood edges
    exclusion = (lc == BARE) | (lc == BUILT) | ~coverage
    return L.compose(observed | terrain.normal_water, terrain.normal_water, invalid=exclusion)


def truth_label(terrain: Terrain, flooded: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """What a careful analyst would draw with field knowledge: our stand-in for hand labels."""
    return L.compose(flooded | terrain.normal_water, terrain.normal_water, invalid=~coverage)


__all__ = ["ORBITS", "Pass", "gfm_like_label", "pass_schedule", "revisit_days", "simulate_sar", "truth_label"]
