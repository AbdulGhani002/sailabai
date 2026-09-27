"""Colour ramps for map tiles and legends.

Flood chance is a magnitude, so it uses one hue (blue), light to dark, with the lightest step
meaning "near zero" and receding into the basemap; in dark mode the anchor flips. Everything below
5% is left transparent so the basemap stays readable. "Can't tell" pixels use the second
categorical colour (orange), and normal water is recessive grey so rivers never read as flood.
Hex values come from the data-viz reference palette.
"""

from __future__ import annotations

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FLOOD = "#2a78d6"
FLOOD_DARK = "#3987e5"
UNSURE = "#eb6834"
UNSURE_DARK = "#d95926"
NORMAL_WATER = "#898781"

# (upper bound of bin in %, ramp index) for the flood-chance legend: 10 bins over 5..100%
PROB_BINS = [(15, 1), (25, 2), (35, 3), (45, 4), (55, 5), (65, 6), (75, 7), (85, 9), (95, 10), (101, 12)]


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(round(alpha * 255)))


def prob_colormap(theme: str = "light") -> dict[int, tuple[int, int, int, int]]:
    """uint8 percent (0-100) -> RGBA; 255 (no data) and < 5% transparent."""
    cmap = {v: (0, 0, 0, 0) for v in range(256)}
    lo = 5
    for i, (hi, idx) in enumerate(PROB_BINS):
        ramp_idx = idx if theme == "light" else len(BLUE_RAMP) - 1 - idx
        alpha = 0.55 + 0.35 * i / (len(PROB_BINS) - 1)
        for v in range(lo, min(hi, 101)):
            cmap[v] = _rgba(BLUE_RAMP[ramp_idx], alpha)
        lo = hi
    return cmap


def label_colormap(theme: str = "light") -> dict[int, tuple[int, int, int, int]]:
    """Current flood map: 0 land (clear), 1 normal water (grey), 2 flood (blue), 255 unknown (clear)."""
    cmap = {v: (0, 0, 0, 0) for v in range(256)}
    cmap[1] = _rgba(NORMAL_WATER, 0.55)
    cmap[2] = _rgba(FLOOD if theme == "light" else FLOOD_DARK, 0.85)
    return cmap


def sets_colormap(theme: str = "light") -> dict[int, tuple[int, int, int, int]]:
    """Conformal sets: 0 dry (clear), 1 flood (blue), 2 can't tell (orange)."""
    cmap = {v: (0, 0, 0, 0) for v in range(256)}
    cmap[1] = _rgba(FLOOD if theme == "light" else FLOOD_DARK, 0.8)
    cmap[2] = _rgba(UNSURE if theme == "light" else UNSURE_DARK, 0.75)
    return cmap


def spread_colormap(theme: str = "light") -> dict[int, tuple[int, int, int, int]]:
    """Ensemble spread in percentage points: one hue (orange) whose opacity grows with the spread."""
    base = UNSURE if theme == "light" else UNSURE_DARK
    cmap = {v: (0, 0, 0, 0) for v in range(256)}
    for v in range(3, 101):
        cmap[v] = _rgba(base, min(0.9, 0.15 + 0.75 * min(v, 30) / 30))
    return cmap


def colormap_for(layer: str, theme: str) -> dict[int, tuple[int, int, int, int]]:
    if layer == "current":
        return label_colormap(theme)
    if layer.startswith("sets"):
        return sets_colormap(theme)
    if layer.startswith("spread"):
        return spread_colormap(theme)
    return prob_colormap(theme)


def legend(layer: str, theme: str = "light") -> list[dict]:
    """Legend entries (label, colour) for the dashboard."""
    if layer == "current":
        return [{"label": "Flooded", "color": FLOOD if theme == "light" else FLOOD_DARK},
                {"label": "Normal water (rivers)", "color": NORMAL_WATER}]
    if layer.startswith("sets"):
        return [{"label": "Flood (confident)", "color": FLOOD if theme == "light" else FLOOD_DARK},
                {"label": "Can't tell", "color": UNSURE if theme == "light" else UNSURE_DARK}]
    if layer.startswith("spread"):
        return [{"label": "Low disagreement", "color": UNSURE, "opacity": 0.25},
                {"label": "High disagreement (30+ pts)", "color": UNSURE, "opacity": 0.9}]
    out, lo = [], 5
    for hi, idx in PROB_BINS:
        ramp_idx = idx if theme == "light" else len(BLUE_RAMP) - 1 - idx
        out.append({"label": f"{lo}-{min(hi, 100)}%", "color": BLUE_RAMP[ramp_idx]})
        lo = hi
    return out
