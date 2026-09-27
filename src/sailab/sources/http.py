"""HTTP helpers for the data connectors: retries, polite headers, atomic downloads and a manifest."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sailab import __version__

USER_AGENT = f"SailabAI/{__version__} (flood research FYP; contact via GitHub repo)"


def http_session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD", "POST"))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    return s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, session: requests.Session | None = None, timeout: float = 120.0,
             overwrite: bool = False) -> Path:
    """Stream a URL to `dest` via a temporary file, so a failed download never leaves a partial file."""
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or http_session()
    with s.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return dest


class Manifest:
    """Append-only CSV log of everything a scraper fetched: what, when, from where, its hash."""

    FIELDS = ["fetched_at", "source", "key", "url", "path", "bytes", "sha256", "note"]

    def __init__(self, path: Path) -> None:
        self.path = path

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        with open(self.path, newline="", encoding="utf-8") as f:
            return {row["key"] for row in csv.DictReader(f)}

    def add(self, source: str, key: str, url: str, path: Path, note: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
                        "key": key, "url": url, "path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256_of(path), "note": note})
