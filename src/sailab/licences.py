"""The licence register (team rules 4 and 5).

Only commercial-safe data may reach the product (FABDEM and WorldFloods are tests only), and
share-alike data such as OpenStreetMap stays a separate layer. The register lives in
configs/licences.yaml; `sailab licences check` validates it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from sailab.paths import config_dir

Use = Literal["train", "test", "product"]


class LicenceError(RuntimeError):
    """Raised when a dataset is used in a way its licence (or our register) does not allow."""


class LicenceEntry(BaseModel):
    id: str
    name: str
    licence: str
    licence_url: str | None = None
    credit: str
    commercial_safe: bool | Literal["unknown"] = "unknown"
    share_alike: bool = False
    uses: list[Use] = Field(default_factory=list)
    version: str | None = None
    download_date: date | None = None
    notes: str = ""


class LicenceRegister:
    def __init__(self, entries: list[LicenceEntry], path: Path | None = None) -> None:
        self.entries = {e.id: e for e in entries}
        self.path = path

    @classmethod
    def load(cls, path: Path | None = None) -> LicenceRegister:
        path = path or config_dir() / "licences.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([LicenceEntry.model_validate(d) for d in raw.get("datasets", [])], path)

    def __getitem__(self, dataset_id: str) -> LicenceEntry:
        try:
            return self.entries[dataset_id]
        except KeyError:
            raise LicenceError(f"{dataset_id!r} is not in the licence register; add it before using it") from None

    def problems(self) -> list[str]:
        """Inconsistencies in the register itself."""
        out = []
        for e in self.entries.values():
            if "product" in e.uses and e.commercial_safe is not True:
                out.append(f"{e.id}: listed for product use but commercial_safe={e.commercial_safe}")
            if not e.credit.strip():
                out.append(f"{e.id}: missing credit text")
            if not e.uses:
                out.append(f"{e.id}: no allowed uses listed")
        return out

    def check_use(self, dataset_ids: list[str], use: Use) -> None:
        """Raise LicenceError if any dataset may not be used for `use`."""
        errors = []
        for ds in dataset_ids:
            e = self[ds]
            if use not in e.uses:
                errors.append(f"{ds} may not be used for {use} (allowed: {', '.join(e.uses)})")
            elif use == "product" and e.commercial_safe is not True:
                errors.append(f"{ds} is not confirmed commercial-safe ({e.licence})")
        if errors:
            raise LicenceError("; ".join(errors))

    def credits(self, dataset_ids: list[str]) -> list[str]:
        return [self[ds].credit for ds in dataset_ids]

    def share_alike_layers(self) -> list[str]:
        return [e.id for e in self.entries.values() if e.share_alike]

    def mark_downloaded(self, dataset_id: str, version: str | None, when: date | None = None) -> None:
        """Record the first download date and the version. Rewrites licences.yaml keeping its header."""
        if self.path is None:
            raise LicenceError("register was not loaded from a file")
        entry = self[dataset_id]
        if entry.download_date is None:
            entry.download_date = when or date.today()
        if version:
            entry.version = version
        header = ""
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("#"):
                break
            header += line + "\n"
        body = {"datasets": [e.model_dump(mode="json") for e in self.entries.values()]}
        self.path.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100),
                             encoding="utf-8")
