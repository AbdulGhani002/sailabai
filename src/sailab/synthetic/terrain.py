"""Synthetic terrain for the demo world: rivers, floodplains, bunds, land cover and exposure.

Nothing here is real data. The layout follows the real rivers roughly (from configs/aoi.yaml) so
the demo looks like our study area, and the physics is simple on purpose: HAND-based flooding
behind flood embankments, with desert sand and rice paddies to fool the radar like they do in
reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from sailab.config import AOIConfig
from sailab.grid import GridSpec

# ESA WorldCover codes
TREE, SHRUB, GRASS, CROP, BUILT, BARE, WATER, WETLAND = 10, 20, 30, 40, 50, 60, 80, 90

RIVER_PARAMS: dict[str, dict[str, Any]] = {
    # fp: floodplain half-width range (km) between the flood bunds; ch: channel half-width (km)
    # q_ref: flow (cusecs) that just fills the channel; d_ref: stage scale (m) of the rating curve
    "chenab": {"fp": (4.0, 9.0), "ch": 0.55, "q_ref": 150_000, "d_ref": 3.0},
    "jhelum": {"fp": (2.5, 4.5), "ch": 0.40, "q_ref": 60_000, "d_ref": 2.5},
    "ravi": {"fp": (1.5, 3.0), "ch": 0.30, "q_ref": 25_000, "d_ref": 2.2},
    "sutlej": {"fp": (3.0, 6.0), "ch": 0.40, "q_ref": 45_000, "d_ref": 2.5},
}
DEFAULT_RIVER = {"fp": (2.0, 4.0), "ch": 0.3, "q_ref": 40_000, "d_ref": 2.5}

PLACE_SIZE_KM = {"city": 3.5, "town": 1.4}
BIG_CITY = {"Multan": 7.0, "Bahawalpur": 4.5}


@dataclass
class River:
    name: str
    xy: np.ndarray            # (n, 2) projected coordinates, upstream to downstream
    chainage_km: np.ndarray   # (n,) distance from the upstream end
    q_ref: float
    d_ref: float
    channel_km: float


@dataclass
class Terrain:
    grid: GridSpec
    rivers: list[River]
    nearest: np.ndarray           # (H, W) index of the nearest river pixel
    rp_river: np.ndarray          # (N,) river index of each river pixel
    rp_chainage: np.ndarray       # (N,) km along its river
    rp_rc: np.ndarray             # (N, 2) row, col of each river pixel
    dist_km: np.ndarray           # distance to the nearest river channel
    fp_halfwidth_km: np.ndarray   # floodplain half-width at the nearest river pixel
    hand: np.ndarray              # height above nearest drainage (m)
    dem: np.ndarray               # elevation (m)
    slope: np.ndarray             # degrees
    normal_water: np.ndarray      # bool, rivers that are wet in normal conditions
    beyond_bund: np.ndarray       # bool, outside the flood embankments
    bund_crest: np.ndarray        # stage (m above normal water) that overtops the nearest bund
    depression: np.ndarray        # bool, old channels where rain water ponds
    landcover: np.ndarray         # uint8 WorldCover codes
    paddy: np.ndarray             # bool, rice fields (dark in July radar)
    population: np.ndarray        # people per pixel
    buildings: np.ndarray         # buildings per pixel
    road_km: np.ndarray           # road length (km) per pixel
    roads: list[dict[str, Any]]   # GeoJSON features, one per road piece
    sar_texture: np.ndarray       # (2, H, W) persistent per-pixel backscatter offsets (dB)
    lon: np.ndarray
    lat: np.ndarray


# --------------------------------------------------------------------------- helpers


def fbm(shape: tuple[int, int], rng: np.random.Generator, scales: tuple[float, ...] = (1.5, 6, 24),
        weights: tuple[float, ...] = (0.3, 0.6, 1.0)) -> np.ndarray:
    """Smooth multi-scale noise with zero mean and unit variance."""
    out = np.zeros(shape, dtype=np.float64)
    for s, w in zip(scales, weights, strict=True):
        n = ndimage.gaussian_filter(rng.standard_normal(shape), s, mode="reflect")
        out += w * n / (n.std() + 1e-9)
    return (out / np.sqrt(sum(w * w for w in weights))).astype(np.float32)


def _densify(line: np.ndarray, step: float) -> np.ndarray:
    seg = np.diff(line, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    pts = [line[0]]
    for p0, d, n in zip(line[:-1], seg, np.maximum(1, np.ceil(seg_len / step)).astype(int), strict=True):
        t = np.arange(1, n + 1)[:, None] / n
        pts.extend(p0 + t * d)
    return np.asarray(pts)


def _meander(line: np.ndarray, rng: np.random.Generator, amp_m: float, half_wave_m: float, step: float) -> np.ndarray:
    """Sinuous version of a polyline that still passes through every original vertex."""
    out = [line[:1]]
    for p0, p1 in zip(line[:-1], line[1:], strict=True):
        d = p1 - p0
        length = float(np.hypot(*d))
        n = max(2, int(np.ceil(length / step)))
        u = np.linspace(0, 1, n + 1)[1:]
        k = max(1, round(length / half_wave_m))
        normal = np.array([-d[1], d[0]]) / (length + 1e-9)
        amp = amp_m * rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
        offset = amp * np.sin(np.pi * k * u)
        out.append(p0 + u[:, None] * d + offset[:, None] * normal)
    return np.vstack(out)


def _smooth_profile(chainage: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    phase1, phase2 = rng.uniform(0, 2 * np.pi, 2)
    wave = 0.65 * np.sin(2 * np.pi * chainage / 60.0 + phase1) + 0.35 * np.sin(2 * np.pi * chainage / 23.0 + phase2)
    return (lo + hi) / 2 + (hi - lo) / 2 * wave


def _burn_line(grid: GridSpec, xy: np.ndarray, target: np.ndarray, value: float, step: float) -> None:
    """Accumulate `value` per sample point along a densified line into a raster (e.g. road km)."""
    dense = _densify(xy, step)
    rows, cols = grid.rowcol(dense[:, 0], dense[:, 1])
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    np.add.at(target, (rows[ok], cols[ok]), value)


# --------------------------------------------------------------------------- main


def build_terrain(grid: GridSpec, aoi: AOIConfig, rng: np.random.Generator) -> Terrain:
    shape = grid.shape
    res = grid.res
    to_xy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    lon, lat = grid.lonlat_grids()
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)

    # ---- rivers: meandering centrelines burnt into a river-pixel table (Chenab first so it owns confluences)
    order = sorted(aoi.rivers, key=lambda n: 0 if n == "chenab" else 1)
    rivers: list[River] = []
    owner = np.full(shape, -1, dtype=np.int32)
    rp_rows, rp_cols, rp_river, rp_chain = [], [], [], []
    for ri, name in enumerate(order):
        pts = np.asarray(aoi.rivers[name], dtype=np.float64)
        xs, ys = to_xy.transform(pts[:, 1], pts[:, 0])
        line = np.column_stack([xs, ys])
        params = RIVER_PARAMS.get(name, DEFAULT_RIVER)
        wiggly = _meander(line, rng, amp_m=2.2 * params["ch"] * 1000 + 600, half_wave_m=9000, step=res / 3)
        chain = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(wiggly, axis=0).T))]) / 1000.0
        rivers.append(River(name, wiggly, chain, params["q_ref"], params["d_ref"], params["ch"]))
        rows, cols = grid.rowcol(wiggly[:, 0], wiggly[:, 1])
        seen: set[tuple[int, int]] = set()
        for r, c, ch in zip(rows, cols, chain, strict=True):
            if not (0 <= r < grid.height and 0 <= c < grid.width) or (r, c) in seen or owner[r, c] >= 0:
                continue
            seen.add((r, c))
            owner[r, c] = len(rp_rows)
            rp_rows.append(r)
            rp_cols.append(c)
            rp_river.append(ri)
            rp_chain.append(ch)
    rp_rc = np.column_stack([rp_rows, rp_cols]).astype(np.int32)
    rp_river_arr = np.asarray(rp_river, dtype=np.int16)
    rp_chain_arr = np.asarray(rp_chain, dtype=np.float32)

    dist_px, (ir, ic) = ndimage.distance_transform_edt(owner < 0, return_indices=True)
    nearest = owner[ir, ic]
    dist_km = (dist_px * res / 1000.0).astype(np.float32)
    rp_river_arr[nearest]

    # ---- floodplain width, channel and bund crest vary along each river
    fp_rp = np.empty(len(rp_rows), dtype=np.float32)
    ch_rp = np.empty(len(rp_rows), dtype=np.float32)
    bund_rp = np.empty(len(rp_rows), dtype=np.float32)
    for ri, river in enumerate(rivers):
        sel = rp_river_arr == ri
        lo, hi = RIVER_PARAMS.get(river.name, DEFAULT_RIVER)["fp"]
        fp_rp[sel] = _smooth_profile(rp_chain_arr[sel], lo, hi, rng)
        ch_rp[sel] = river.channel_km
        weak = rng.random(sel.sum()) < 0.03  # a few weak sections
        bund_rp[sel] = np.where(weak, 2.0, 3.0) + 0.3 * np.sin(rp_chain_arr[sel] / 7.0)
    bund_rp = ndimage.uniform_filter1d(bund_rp, 5)
    fp_km = fp_rp[nearest]
    channel = dist_km <= np.maximum(ch_rp[nearest], res / 2000.0)

    # ---- HAND: low floodplain inside the bunds, a step up behind them, then a gentle rise
    noise = fbm(shape, rng)
    inside = dist_km <= fp_km
    edge_hand = 0.4 + 0.42 * fp_km
    hand_in = 0.4 + 0.42 * dist_km + 0.35 * noise
    hand_out = edge_hand + 1.1 + 0.16 * (dist_km - fp_km) + 0.0025 * (dist_km - fp_km) ** 2 + 0.8 * noise
    hand = np.where(inside, hand_in, hand_out)

    # old river channels far from today's rivers: they pond water after heavy local rain
    depression = np.zeros(shape, dtype=bool)
    for _ in range(max(6, int(shape[0] * shape[1] / 30_000))):
        start = np.array([rng.uniform(0, grid.width), rng.uniform(0, grid.height)])
        heading = rng.uniform(0, 2 * np.pi)
        pts = [start]
        for _ in range(rng.integers(25, 70)):
            heading += rng.normal(0, 0.35)
            pts.append(pts[-1] + 1.2 * np.array([np.cos(heading), np.sin(heading)]))
        pts_arr = np.asarray(pts)
        r = np.clip(pts_arr[:, 1].astype(int), 0, grid.height - 1)
        c = np.clip(pts_arr[:, 0].astype(int), 0, grid.width - 1)
        depression[r, c] = True
    depression = ndimage.binary_dilation(depression, iterations=1) & ~inside
    hand = np.where(depression, np.maximum(0.25, hand - 2.2), hand)
    hand = np.clip(hand, 0.05, 45.0).astype(np.float32)
    hand[channel] = 0.0

    bund_crest = (edge_hand + bund_rp[nearest]).astype(np.float32)
    beyond_bund = ~inside

    # ---- elevation: a plain sloping gently down to the south-west, rivers cut into it
    plane = 100.0 + 28.0 * (lat - 29.2) + 10.0 * (lon - 70.9)
    bed = plane[rp_rc[:, 0], rp_rc[:, 1]] - 4.0
    dem = (bed[nearest] + hand + 0.3 * fbm(shape, rng, scales=(1.0,), weights=(1.0,))).astype(np.float32)
    gy, gx = np.gradient(dem, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    # ---- land cover
    n1 = fbm(shape, rng)
    n2 = fbm(shape, rng, scales=(1.0, 4.0), weights=(0.6, 1.0))
    u1 = ndimage.uniform_filter(rng.random(shape), 3)

    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    thal = sig((lat - 30.75) / 0.10) * sig((71.62 - lon) / 0.10)
    cholistan = sig((29.55 - lat) / 0.07) * sig((lon - 71.95) / 0.10)
    sand = ((thal + cholistan) * (0.65 + 0.35 * sig(n1)) > 0.5) & beyond_bund
    sand |= inside & (dist_km < 1.6) & (n2 > 1.6) & ~channel  # sand bars in the river bed

    landcover = np.full(shape, CROP, dtype=np.uint8)
    landcover[(n2 > 1.3) & ~sand] = TREE
    landcover[inside & (hand < 1.0) & (n1 > 0.4)] = WETLAND
    landcover[inside & (dist_km < fp_km * 0.8) & (n2 > 1.0)] = TREE
    landcover[sand] = BARE
    landcover[beyond_bund & (u1 > 0.62) & ~sand & (n1 < -1.2)] = GRASS

    population_density = np.full(shape, 380.0, dtype=np.float32)
    population_density[inside] = 160.0
    population_density[sand] = 15.0
    population_density *= (0.7 + 0.6 * sig(n1)).astype(np.float32)
    built = np.zeros(shape, dtype=bool)
    to_rc = grid.lonlat_to_rowcol
    rows_idx, cols_idx = np.mgrid[0:grid.height, 0:grid.width]
    for place in aoi.places:
        radius = BIG_CITY.get(place.name, PLACE_SIZE_KM.get(place.kind, 1.4))
        r0, c0 = to_rc(place.lon, place.lat)
        d_km = np.hypot(rows_idx - int(r0), cols_idx - int(c0)) * res / 1000.0
        built |= d_km < radius * (0.85 + 0.3 * sig(n2))
        peak = 22_000.0 if place.name == "Multan" else (12_000.0 if place.kind == "city" else 6_000.0)
        population_density += (peak * np.exp(-((d_km / (0.6 * radius)) ** 2))).astype(np.float32)
    landcover[built & ~channel] = BUILT
    landcover[channel] = WATER
    paddy = (landcover == CROP) & (((lat > 30.35) & (n2 > 1.0)) | (n2 > 1.8))

    population_density[channel] = 0.0
    population = (population_density * grid.pixel_area_km2).astype(np.float32)
    buildings = np.round(population / np.where(built, 9.0, 6.5)).astype(np.float32)

    # ---- roads: each place linked to its two nearest neighbours, plus the Multan-Sukkur motorway line
    names = [p.name for p in aoi.places]
    pxy = np.array(to_xy.transform([p.lon for p in aoi.places], [p.lat for p in aoi.places])).T
    edges: set[tuple[int, int]] = set()
    for i in range(len(names)):
        d = np.hypot(*(pxy - pxy[i]).T)
        for j in np.argsort(d)[1:3]:
            edges.add((min(i, int(j)), max(i, int(j))))
    motorway = [n for n in ("Multan", "Shujabad", "Jalalpur Pirwala", "Uch Sharif") if n in names]
    motorway_edges = {(min(names.index(a), names.index(b)), max(names.index(a), names.index(b)))
                      for a, b in zip(motorway, motorway[1:], strict=False)}
    edges |= motorway_edges
    road_km = np.zeros(shape, dtype=np.float32)
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    roads: list[dict[str, Any]] = []
    step = res / 4
    for i, j in sorted(edges):
        a, b = pxy[i], pxy[j]
        mid = (a + b) / 2 + rng.normal(0, 0.04) * np.array([-(b - a)[1], (b - a)[0]])
        line = _densify(np.vstack([a, mid, b]), step)
        _burn_line(grid, line, road_km, step / 1000.0, step=step)
        klass = "motorway" if (i, j) in motorway_edges else "highway"
        length_km = float(np.sum(np.hypot(*np.diff(line, axis=0).T)) / 1000.0)
        n_pieces = max(1, int(round(length_km / 3.0)))
        for k, piece in enumerate(np.array_split(line, n_pieces)):
            if len(piece) < 2:
                continue
            lons, lats = to_ll.transform(piece[:, 0], piece[:, 1])
            roads.append({
                "type": "Feature",
                "properties": {"id": f"road-{names[i]}-{names[j]}-{k}".replace(" ", "_").lower(),
                               "name": f"{names[i]} - {names[j]}", "class": klass,
                               "length_km": round(length_km / n_pieces, 2)},
                "geometry": {"type": "LineString",
                             "coordinates": [[round(x, 5), round(y, 5)] for x, y in zip(lons[::3], lats[::3], strict=True)]},
            })

    texture = np.stack([fbm(shape, rng, scales=(0.8, 3.0), weights=(1.0, 0.6)) for _ in range(2)])
    texture[1] = 0.7 * texture[0] + 0.3 * texture[1]

    return Terrain(
        grid=grid, rivers=rivers, nearest=nearest.astype(np.int32), rp_river=rp_river_arr,
        rp_chainage=rp_chain_arr, rp_rc=rp_rc, dist_km=dist_km, fp_halfwidth_km=fp_km.astype(np.float32),
        hand=hand, dem=dem, slope=slope, normal_water=channel, beyond_bund=beyond_bund,
        bund_crest=bund_crest, depression=depression, landcover=landcover, paddy=paddy,
        population=population, buildings=buildings, road_km=road_km, roads=roads,
        sar_texture=texture.astype(np.float32), lon=lon, lat=lat,
    )


def river_features(terrain: Terrain) -> list[dict[str, Any]]:
    """River centrelines as GeoJSON for the dashboard overlay."""
    to_ll = Transformer.from_crs(terrain.grid.crs, "EPSG:4326", always_xy=True)
    feats = []
    for river in terrain.rivers:
        xy = river.xy[:: max(1, len(river.xy) // 400)]
        lons, lats = to_ll.transform(xy[:, 0], xy[:, 1])
        feats.append({"type": "Feature", "properties": {"name": river.name.title()},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(x, 5), round(y, 5)] for x, y in zip(lons, lats, strict=True)]}})
    return feats
