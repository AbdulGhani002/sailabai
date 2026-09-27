"""VIIRS flood products (NOAA/GMU JPSS flood maps), kept online about 7 days.

The GMU server publishes daily and 5-day composite flood GeoTIFFs by area tile. Set
SAILAB_VIIRS_URL to the directory that holds our area's files (ftp:// or https://) and
SAILAB_VIIRS_MATCH to a substring of our tiles' file names; the daily scraper then keeps every new
file. The exact directory layout is not documented publicly, so confirm it once by browsing the
server (https://jpssflood.gmu.edu/ links to its FTP root).
"""

from __future__ import annotations

import ftplib
import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from sailab.paths import archive_dir
from sailab.sources.http import Manifest, download, http_session


def _list_ftp(url: str) -> list[str]:
    u = urlparse(url)
    with ftplib.FTP(u.hostname, timeout=60) as ftp:
        ftp.login()
        ftp.cwd(u.path or "/")
        return [f"ftp://{u.hostname}{(u.path.rstrip('/') + '/' + n)}" for n in ftp.nlst()]


def _list_http(url: str) -> list[str]:
    r = http_session().get(url, timeout=60)
    r.raise_for_status()
    return [urljoin(url, h) for h in re.findall(r'href="([^"]+\.tif[f]?)"', r.text, re.I)]


def _fetch_ftp(url: str, dest: Path) -> None:
    u = urlparse(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ftplib.FTP(u.hostname, timeout=120) as ftp, open(dest, "wb") as f:
        ftp.login()
        ftp.retrbinary(f"RETR {u.path}", f.write)


def archive_viirs(root: Path | None = None, log=print) -> int:
    base = os.environ.get("SAILAB_VIIRS_URL")
    match = os.environ.get("SAILAB_VIIRS_MATCH", "")
    if not base:
        log("viirs: SAILAB_VIIRS_URL not set, skipping (see sailab/sources/viirs.py)")
        return 0
    root = (root or archive_dir()) / "viirs"
    manifest = Manifest(root / "manifest.csv")
    have = manifest.keys()
    files = _list_ftp(base) if base.startswith("ftp://") else _list_http(base)
    n = 0
    for url in files:
        name = url.rsplit("/", 1)[-1]
        if match and match not in name or name in have:
            continue
        dest = root / name
        if url.startswith("ftp://"):
            _fetch_ftp(url, dest)
        else:
            download(url, dest)
        manifest.add("viirs", name, url, dest)
        n += 1
    log(f"viirs: {n} new file(s)")
    return n
