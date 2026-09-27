"""Data versions saved with every sample (team rule 2).

IMERG moves from V07 to V08, GloFAS from v4 to v5, and GFM changed product version in 2025. A
model trained on one version and scored on another can look better or worse for reasons that have
nothing to do with the model, so every sample records the versions it was built from.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

TRACKED = ("sentinel1", "gfm", "imerg", "glofas", "ecmwf", "dem")


@dataclass(frozen=True)
class DataVersions:
    sentinel1: str | None = None
    gfm: str | None = None
    imerg: str | None = None
    glofas: str | None = None
    ecmwf: str | None = None
    dem: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "extra" and v is not None}
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DataVersions:
        d = dict(d or {})
        known = {k: (str(d.pop(k)) if d.get(k) is not None else None) for k in TRACKED if k in d}
        return cls(**known, extra={k: str(v) for k, v in d.items()})

    def key(self) -> str:
        """Stable one-line form, e.g. 'gfm=V0M2R2;glofas=4.5;imerg=V07'."""
        return ";".join(f"{k}={v}" for k, v in sorted(self.to_dict().items()))

    def merge(self, other: DataVersions) -> DataVersions:
        merged = self.to_dict()
        merged.update(other.to_dict())
        return DataVersions.from_dict(merged)


_GFM_VERSION = re.compile(r"_(V\d+M\d+R\d+)_")


def gfm_version_from_filename(name: str) -> str | None:
    """GFM encodes its product version in file names, e.g. ..._EQUI7_AS020M_V0M2R2_S1.tif."""
    m = _GFM_VERSION.search(name)
    return m.group(1) if m else None


def mixed_versions(versions: Iterable[DataVersions]) -> dict[str, Counter]:
    """Sources that appear with more than one version in a dataset, with counts per version."""
    counts: dict[str, Counter] = {}
    for v in versions:
        for k, val in v.to_dict().items():
            counts.setdefault(k, Counter())[val] += 1
    return {k: c for k, c in counts.items() if len(c) > 1}
