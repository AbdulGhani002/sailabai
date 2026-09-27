"""Daily scrapers for sources that delete their files within days (start these in week one).

    sailab archive run            # all scrapers; schedule it daily (cron, Task Scheduler, or the API's scheduler)

- FFD bulletins (PMD Flood Forecasting Division, Lahore): daily 6 am gauge readings and forecasts.
- IRSA daily water data PDFs: kept online about 10 days.
- ECMWF open-data rain forecasts: kept about 4 days.
- VIIRS flood files: kept about 7 days.

Everything lands in data/archive/<source>/ with a manifest (URL, time, SHA-256). The PDFs are for
internal research: FFD bulletins are "all rights reserved", so never redistribute them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from sailab.paths import archive_dir
from sailab.sources.http import Manifest, download, http_session

FFD_ARCHIVE = "https://ffd.pmd.gov.pk/bulletins/archive"
IRSA_DAILY = "http://pakirsa.gov.pk/DailyData.aspx"


@dataclass
class ArchiveItem:
    key: str
    url: str
    filename: str
    date: str | None = None
    title: str | None = None


# --------------------------------------------------------------------------- FFD bulletins

_FFD_ROW = re.compile(
    r'<td class="td-date">\s*(?P<date>.*?)\s*</td>.*?<td class="td-title">(?P<title>.*?)</td>'
    r'.*?</i>(?P<file>[^<]*?\.pdf).*?data-download-url="(?P<url>[^"]+)"', re.S)


def parse_ffd_archive_page(html: str) -> list[ArchiveItem]:
    items = []
    for m in _FFD_ROW.finditer(html):
        date = datetime.strptime(m.group("date").strip(), "%d %b %Y").date().isoformat()
        url = m.group("url")
        items.append(ArchiveItem(key=url.rstrip("/").split("/")[-2], url=url, filename=f"{date}_{m.group('file').strip()}",
                                 date=date, title=m.group("title").strip()))
    return items


def list_ffd_bulletins(max_pages: int = 20, kind: str = "bulletin") -> list[ArchiveItem]:
    s = http_session()
    out: list[ArchiveItem] = []
    for page in range(1, max_pages + 1):
        r = s.get(FFD_ARCHIVE, params={"type": kind, "page": page}, timeout=60)
        r.raise_for_status()
        items = parse_ffd_archive_page(r.text)
        if not items:
            break
        out.extend(items)
        if f"page={page + 1}" not in r.text:
            break
    return out


# --------------------------------------------------------------------------- IRSA

_IRSA_LINK = re.compile(r'href="(?P<href>Doc/Data(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})\.pdf)"')


def parse_irsa_page(html: str, base: str = IRSA_DAILY) -> list[ArchiveItem]:
    items = []
    for m in _IRSA_LINK.finditer(html):
        date = f"{m.group('y')}-{m.group('m')}-{m.group('d')}"
        items.append(ArchiveItem(key=date, url=urljoin(base, m.group("href")), filename=f"irsa_{date}.pdf", date=date))
    return items


def list_irsa() -> list[ArchiveItem]:
    r = http_session().get(IRSA_DAILY, timeout=60)
    r.raise_for_status()
    return parse_irsa_page(r.text)


# --------------------------------------------------------------------------- runner


def archive_items(source: str, items: list[ArchiveItem], root: Path | None = None, log=print) -> int:
    root = (root or archive_dir()) / source
    manifest = Manifest(root / "manifest.csv")
    have = manifest.keys()
    s = http_session()
    n = 0
    for it in items:
        if it.key in have:
            continue
        dest = root / (it.date[:4] if it.date else "undated") / it.filename
        try:
            download(it.url, dest, s)
        except Exception as e:  # keep going: one broken link must not stop the daily run
            log(f"  {source}: failed {it.url}: {e}")
            continue
        manifest.add(source, it.key, it.url, dest, note=it.title or "")
        n += 1
    log(f"{source}: {n} new file(s), {len(have) + n} archived")
    return n


SCRAPERS: dict[str, Callable[[], list[ArchiveItem]]] = {
    "ffd": list_ffd_bulletins,
    "irsa": list_irsa,
}


def run_all(sources: list[str] | None = None, log=print) -> dict[str, int]:
    """Run each scraper; also pulls today's ECMWF open-data rain forecast and VIIRS files if configured."""
    results = {}
    for name in sources or [*SCRAPERS, "ecmwf", "viirs"]:
        try:
            if name in SCRAPERS:
                results[name] = archive_items(name, SCRAPERS[name](), log=log)
            elif name == "ecmwf":
                from sailab.sources.ecmwf_open import archive_latest

                results[name] = archive_latest(log=log)
            elif name == "viirs":
                from sailab.sources.viirs import archive_viirs

                results[name] = archive_viirs(log=log)
        except Exception as e:
            log(f"{name}: scraper failed: {e}")
            results[name] = -1
    return results
