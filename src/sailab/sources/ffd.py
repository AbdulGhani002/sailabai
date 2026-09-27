"""Read gauge flows out of FFD daily flood bulletins.

Page 2 of each bulletin has "QUANTITATIVE FLOOD FORECAST OF GAUGING STATIONS (IN THOUSANDS OF
CUSECS)": for every station its design capacity, the 6 am inflow and outflow, a 24-hour forecast
and the season's peak. We keep the observed inflow (and outflow) at our gauges.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# bulletin station name (as printed) -> our gauge id
STATIONS = {
    "Marala": "marala",
    "Khanki": "khanki",
    "Qadirabad": "qadirabad",
    "Trimmu": "trimmu",
    "Punjnad": "panjnad",
    "Panjnad": "panjnad",
    "Sidhnai": "sidhnai",
    "G.S. Wala": "ganda_singh_wala",
    "G.S.Wala": "ganda_singh_wala",
    "Islam": "islam",
    "Balloki": "balloki",
    "Shahdara": "shahdara",
    "Jassar": "jassar",
    "Sulemanki": "sulemanki",
    "Mangla": "mangla",
    "Rasul": "rasul",
    "Taunsa": "taunsa",
    "Chashma": "chashma",
    "Tarbela": "tarbela",
    "Guddu": "guddu",
}

_NUM = r"(\d+(?:\.\d+)?)"
_DATE = re.compile(r"Dated:\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[-\s,]*(\d{4})")


def _row_pattern(name: str) -> re.Pattern[str]:
    # name, design capacity (number or dashes), inflow, outflow
    return re.compile(re.escape(name) + r"\s+(?:\d+(?:\.\d+)?|-+)\s+" + _NUM + r"\s+" + _NUM)


def bulletin_date(text: str) -> date | None:
    m = _DATE.search(text)
    if not m:
        return None
    day, month, year = m.groups()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{day} {month} {year}", fmt).date()
        except ValueError:
            continue
    return None


def parse_bulletin_text(text: str) -> pd.DataFrame:
    """Rows (date, gauge_id, inflow_cusecs, outflow_cusecs) from a bulletin's extracted text."""
    day = bulletin_date(text)
    start = text.find("QUANTITATIVE FLOOD FORECAST")
    table = text[start:] if start >= 0 else text
    rows = []
    for name, gauge in STATIONS.items():
        m = _row_pattern(name).search(table)
        if m:
            rows.append({"date": day, "gauge_id": gauge, "inflow_cusecs": float(m.group(1)) * 1000,
                         "outflow_cusecs": float(m.group(2)) * 1000})
    df = pd.DataFrame(rows, columns=["date", "gauge_id", "inflow_cusecs", "outflow_cusecs"])
    return df.drop_duplicates("gauge_id")


def parse_bulletin_pdf(path: Path) -> pd.DataFrame:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    df = parse_bulletin_text(text)
    df["file"] = path.name
    return df


def bulletins_to_discharge(folder: Path, version: str = "ffd-bulletin") -> pd.DataFrame:
    """All archived bulletins -> the datacube's discharge table (inflow at 6 am, cusecs)."""
    frames = []
    for pdf in sorted(folder.rglob("*.pdf")):
        try:
            frames.append(parse_bulletin_pdf(pdf))
        except Exception:  # a malformed PDF must not stop the batch
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "point_id", "q_cusecs", "source", "version"])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    return pd.DataFrame({"date": pd.to_datetime(df["date"]), "point_id": df["gauge_id"],
                         "q_cusecs": df["inflow_cusecs"].astype("float32"), "source": "ffd", "version": version})
