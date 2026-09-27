"""The shared results table (team rule 3): every new score sits next to its baseline.

results/results_table.csv is append-only and committed to git; results/RESULTS.md is regenerated
from it for the weekly meeting.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sailab.evaluation.protocol import compare_to_baseline
from sailab.paths import REPO_ROOT, results_dir
from sailab.versioning import DataVersions

COLUMNS = [
    "time", "commit", "experiment", "model", "baseline", "split", "event", "lead", "subset", "metric",
    "value", "baseline_value", "skill", "n_events", "n_pixels", "data_versions", "notes",
]

HEADLINE_METRICS = ("iou", "f1", "precision", "recall", "brier", "ece")


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class ResultsTable:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or results_dir() / "results_table.csv"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path)

    def append(self, model_macro: pd.DataFrame, baseline_macro: pd.DataFrame | None, *, experiment: str,
               model: str, baseline: str, split: str, event: str = "macro",
               data_versions: DataVersions | None = None, notes: str = "",
               metrics: tuple[str, ...] = HEADLINE_METRICS) -> pd.DataFrame:
        """Append macro-averaged scores for one model, each row joined with the baseline's score."""
        if baseline_macro is None and baseline != "none":
            raise ValueError("pass the baseline scores, or baseline='none' with a note explaining why")
        if baseline_macro is None:
            merged = model_macro.assign(value_baseline=np.nan, skill=np.nan)
        else:
            merged = compare_to_baseline(model_macro, baseline_macro)
        merged = merged[merged["metric"].isin(metrics)]
        rows = pd.DataFrame({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _commit(),
            "experiment": experiment,
            "model": model,
            "baseline": baseline,
            "split": split,
            "event": event,
            "lead": merged["lead"].to_numpy(),
            "subset": merged["subset"].to_numpy(),
            "metric": merged["metric"].to_numpy(),
            "value": merged["value"].round(4).to_numpy(),
            "baseline_value": merged["value_baseline"].round(4).to_numpy(),
            "skill": merged["skill"].round(4).to_numpy(),
            "n_events": merged.get("n_events", pd.Series([np.nan] * len(merged))).to_numpy(),
            "n_pixels": merged.get("n_pixels", pd.Series([np.nan] * len(merged))).to_numpy(),
            "data_versions": (data_versions or DataVersions()).key(),
            "notes": notes,
        }, columns=COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        return rows

    def render_markdown(self, out: Path | None = None) -> str:
        """A readable summary: latest run of each (experiment, model), headline metrics on all pixels."""
        df = self.load()
        lines = ["# Results", "",
                 "Generated from `results/results_table.csv` by `sailab results render`. Each score sits "
                 "next to its baseline; skill > 0 means better than the baseline.", ""]
        if df.empty:
            lines.append("No results yet.")
        else:
            latest = df.sort_values("time").groupby(["experiment", "model"], sort=False)["time"].transform("max")
            df = df[df["time"] == latest]
            for (experiment, model), part in df.groupby(["experiment", "model"], sort=False):
                first = part.iloc[0]
                lines += [f"## {experiment}: {model} vs {first['baseline']}", "",
                          f"Split `{first['split']}`, commit `{first['commit']}`, {first['time']}, "
                          f"data `{first['data_versions']}`.", ""]
                view = part[part["subset"].isin(["all", "newly_flooded", "drained"])]
                table = view.pivot_table(index=["lead", "subset"], columns="metric",
                                         values=["value", "baseline_value", "skill"], aggfunc="first")
                keep = [m for m in HEADLINE_METRICS if ("value", m) in table.columns]
                lines.append("| lead | subset | " + " | ".join(f"{m} (model / base / skill)" for m in keep) + " |")
                lines.append("|---|---|" + "---|" * len(keep))
                for (lead, subset), row in table.iterrows():
                    cells = []
                    for m in keep:
                        v, b, s = row.get(("value", m)), row.get(("baseline_value", m)), row.get(("skill", m))
                        cells.append(" / ".join("-" if pd.isna(x) else f"{x:.3f}" for x in (v, b, s)))
                    lines.append(f"| {lead} | {subset} | " + " | ".join(cells) + " |")
                if isinstance(first["notes"], str) and first["notes"]:
                    lines += ["", f"Notes: {first['notes']}"]
                lines.append("")
        text = "\n".join(lines) + "\n"
        target = out or self.path.with_name("RESULTS.md")
        target.write_text(text, encoding="utf-8")
        return text
