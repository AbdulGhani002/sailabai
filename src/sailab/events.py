"""Flood events, event-based splits and the one-shot test lock.

Team rule 1: split data by flood event, never by random tiles. Random tile splits made scores look
up to 28% better than they really were in one study, because neighbouring tiles of the same flood
end up on both sides of the split.
"""

from __future__ import annotations

import getpass
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from sailab.config import Event, Split, load_events
from sailab.paths import REPO_ROOT, results_dir


class LeakageError(RuntimeError):
    """Raised when the same flood event appears on both sides of a split."""


class TestAlreadyUsedError(RuntimeError):
    """Raised when someone tries to score the final test event a second time."""

    __test__ = False  # not a pytest test class


def _as_date(t: date | datetime | str) -> date:
    if isinstance(t, datetime):
        return t.date()
    if isinstance(t, date):
        return t
    return datetime.fromisoformat(str(t).replace("Z", "+00:00")).date()


@dataclass
class EventRegistry:
    events: list[Event]

    def __post_init__(self) -> None:
        ids = [e.id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event ids in events config")
        ordered = sorted(self.events, key=lambda e: e.start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start <= a.end:
                raise ValueError(f"events {a.id} and {b.id} overlap; each day must belong to one event")

    @classmethod
    def load(cls, path: str | None = None) -> EventRegistry:
        return cls(list(load_events(path).events))

    def __getitem__(self, event_id: str) -> Event:
        for e in self.events:
            if e.id == event_id:
                return e
        raise KeyError(event_id)

    def event_for(self, t: date | datetime | str) -> Event | None:
        day = _as_date(t)
        for e in self.events:
            if e.contains(day):
                return e
        return None

    def split_for(self, t: date | datetime | str) -> Split | None:
        e = self.event_for(t)
        return e.split if e else None

    def events_in(self, *splits: Split | str) -> list[Event]:
        wanted = {Split(s) for s in splits}
        return [e for e in self.events if e.split in wanted]

    def ids_in(self, *splits: Split | str) -> list[str]:
        return [e.id for e in self.events_in(*splits)]

    def assign(self, times: Iterable[date | datetime | str]) -> list[str | None]:
        """Event id for each timestamp (None when it falls outside every event window)."""
        return [(e.id if (e := self.event_for(t)) else None) for t in times]


def assert_no_leakage(train_events: Sequence[str | None], eval_events: Sequence[str | None]) -> None:
    """Refuse any split where one flood event feeds both training and evaluation."""
    shared = {e for e in train_events if e} & {e for e in eval_events if e}
    if shared:
        raise LeakageError(f"events used for both training and evaluation: {sorted(shared)}")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TestLock:
    """The 2025 flood is the final test: each model is scored on it once.

    The first scoring run of an experiment (e.g. "model1", "model2") on a test event writes
    results/test_lock.json (committed to git). A second run of the same experiment on the same event
    is refused unless `force=True` with a written reason, and is logged; other experiments may take
    their own single shot, and every attempt of every experiment stays in the log, so the report can
    say honestly how often the test set was touched.
    """

    __test__ = False  # not a pytest test class

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or results_dir() / "test_lock.json"

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"runs": []}

    def runs(self, event_id: str | None = None, cube: str | None = None, experiment: str | None = None) -> list[dict]:
        runs = self._load()["runs"]
        return [r for r in runs if (event_id is None or r["event"] == event_id)
                and (cube is None or r.get("cube", "real") == cube)
                and (experiment is None or r["experiment"] == experiment)]

    def acquire(self, event_id: str, experiment: str, cube: str = "real", force: bool = False,
                reason: str = "") -> dict:
        """Record a scoring run on a test event. The lock is per datacube and experiment, so scoring
        the synthetic demo cube never uses up the real 2025 test, and Model 1 scoring its test does
        not block Model 2 from scoring its own."""
        previous = self.runs(event_id, cube, experiment)
        if previous and not force:
            first = previous[0]
            raise TestAlreadyUsedError(
                f"test event {event_id} was already scored by {experiment} on {first['time']} "
                f"(commit {first['commit']}). Re-scoring needs force=True and a written reason."
            )
        if previous and not reason.strip():
            raise TestAlreadyUsedError("re-scoring a test event needs a written reason")
        record = {
            "event": event_id,
            "cube": cube,
            "experiment": experiment,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "user": getpass.getuser(),
            "attempt": len(previous) + 1,
            "touches_of_event": len(self.runs(event_id, cube)) + 1,
            "reason": reason,
        }
        data = self._load()
        data["runs"].append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return record
