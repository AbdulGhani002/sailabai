"""Parsers for the scraped sources, tested on small made-up snippets shaped like the real pages."""

from __future__ import annotations

from datetime import date

from sailab.config import BBox
from sailab.sources.ffd import bulletin_date, parse_bulletin_text
from sailab.sources.gfm import group_passes, orbit_of
from sailab.sources.scrapers import parse_ffd_archive_page, parse_irsa_page
from sailab.sources.static_layers import dem_urls, gsw_urls, worldcover_urls

FFD_ROW = """
<tr><td class="td-date">
    27 Aug 2025
</td><td class="td-type"><span class="badge-bulletin">Bulletin</span></td>
<td class="td-title">Flood Bulletin 27 Aug 2025</td>
<td class="td-file d-none d-lg-table-cell"><i class="bi bi-file-earmark-pdf-fill text-danger mr-1"></i>27082025.pdf</td>
<td class="td-act"><button type="button" class="btn" data-view-url="https://ffd.pmd.gov.pk/bulletins/archive/238/view"
 data-download-url="https://ffd.pmd.gov.pk/bulletins/archive/238/download" data-title="x">View</button></td></tr>
"""

# made-up numbers in the bulletin's text layout (name, design capacity, inflow, outflow, ...)
BULLETIN_TEXT = """BULLETIN No. B-001/25 Dated: 3rd September-2025
II: QUANTITATIVE FLOOD FORECAST OF GAUGING STATIONS (IN THOUSANDS OF CUSECS)
CHENAB
Marala
1100
123.4
120.0
100-120
Low
Qadirabad
807
456.7
450.1
400 R 500
High
Punjnad
700
88.8
80.2
SUTLEJ
G.S. Wala
--
99.5
99.5
"""


def test_parse_ffd_archive_row():
    items = parse_ffd_archive_page(FFD_ROW)
    assert len(items) == 1
    it = items[0]
    assert it.date == "2025-08-27" and it.key == "238" and it.url.endswith("/238/download")
    assert it.filename == "2025-08-27_27082025.pdf"


def test_parse_irsa_links():
    html = '<a href="Doc/Data26-09-2026.pdf">x</a><a href="Doc/Data25-09-2026.pdf">y</a>'
    items = parse_irsa_page(html)
    assert [i.date for i in items] == ["2026-09-26", "2026-09-25"]
    assert items[0].url == "http://pakirsa.gov.pk/Doc/Data26-09-2026.pdf"


def test_parse_bulletin_text():
    assert bulletin_date(BULLETIN_TEXT) == date(2025, 9, 3)
    df = parse_bulletin_text(BULLETIN_TEXT).set_index("gauge_id")
    assert df.loc["marala", "inflow_cusecs"] == 123_400
    assert df.loc["qadirabad", "outflow_cusecs"] == 450_100
    assert df.loc["panjnad", "inflow_cusecs"] == 88_800
    assert df.loc["ganda_singh_wala", "inflow_cusecs"] == 99_500


def test_gfm_pass_grouping():
    def item(t: str, tile: str, orbit: str) -> dict:
        return {"id": f"ENSEMBLE_FLOOD_{t}_{tile}", "properties": {"datetime": t},
                "assets": {"advisory_flags": {"href": f"https://x/ADVFLAG_{t}__VV_{orbit}_{tile}_EQUI7_AS020M_V0M2R2_S1.tif"}}}

    items = [item("2025-09-04T13:36:51Z", "E018N030T3", "A042"), item("2025-09-04T13:36:26Z", "E021N030T3", "A042"),
             item("2025-09-04T01:00:50Z", "E021N030T3", "D034")]
    assert orbit_of(items[0]) == "A042"
    passes = group_passes(items)
    assert [p.orbit for p in passes] == ["D034", "A042"]
    assert len(passes[1].items) == 2 and passes[1].scene_id == "S1_20250904T133626_A042"
    assert passes[1].version == "V0M2R2"


def test_static_tile_urls_cover_the_study_area():
    box = BBox(west=70.9, south=29.2, east=72.7, north=31.4)
    assert len(dem_urls(box)) == 9
    assert any("N30_00_E071_00" in u for u in dem_urls(box))
    assert sorted(u.split("/")[-1] for u in gsw_urls(box)) == ["occurrence_70E_30Nv1_4_2021.tif", "occurrence_70E_40Nv1_4_2021.tif"]
    assert len(worldcover_urls(box)) == 4
