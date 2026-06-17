"""Multi-task probe benchmark over the user-level downstream tasks.

Embeds users once with a trained model, then probes each task (a gradient-boosting
head on the frozen embedding vs the same head on raw counts), with eval-point
truncation. Reports ROC-AUC and PR-AUC, renders a provenance-stamped markdown table,
and writes it into the README ``<!-- MULTITASK_PROBE_RESULTS -->`` marker.

Scope: the user-level binary tasks with one row per user and an eval point —
``default_12m`` (credit), ``churn_6m``, ``ltv_positive``. Event-level ``fraud``
and ``recurring`` are transaction/series-level (multiple rows per user), and
``comm_uplift`` is a treatment-effect task — those are evaluated by their own
paths (transaction scoring / ``api.uplift``), not this user-embedding probe.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_LEVEL_TASKS: tuple[str, ...] = ("default_12m", "churn_6m", "ltv_positive")

MARKER = "<!-- MULTITASK_PROBE_RESULTS -->"


@dataclass
class MultiTaskRow:
    """One task's probe vs raw-count-baseline ROC-AUC and PR-AUC."""

    task: str
    probe_auc: float
    baseline_auc: float
    probe_pr_auc: float
    baseline_pr_auc: float
    prevalence: float
    n_test: int


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_multitask_probe(
    shard_dir: str | Path,
    run_dir: str | Path,
    labels_dir: str | Path,
    tasks: tuple[str, ...] = USER_LEVEL_TASKS,
    seed: int = 0,
    device: str = "auto",
) -> list[MultiTaskRow]:
    """Probe each task's label table against the raw-count baseline."""
    from pragmatiq import api

    labels_dir = Path(labels_dir)
    rows: list[MultiTaskRow] = []
    for task in tasks:
        lp = labels_dir / f"{task}.parquet"
        if not lp.exists():
            continue
        res = api.probe(shard_dir, run_dir, lp, device=device, seed=seed)
        rows.append(MultiTaskRow(task=task, probe_auc=res["probe_auc"],
                                 baseline_auc=res["baseline_auc"],
                                 probe_pr_auc=res["probe_pr_auc"],
                                 baseline_pr_auc=res["baseline_pr_auc"],
                                 prevalence=res["prevalence"], n_test=res["n_test"]))
    return rows


def multitask_results_markdown(rows: list[MultiTaskRow], scale: dict[str, Any]) -> str:
    """Render probe rows as a provenance-stamped markdown table."""
    lines = ["| task | probe ROC-AUC | baseline ROC-AUC | probe PR-AUC | baseline PR-AUC | prevalence |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.task} | {r.probe_auc:.3f} | {r.baseline_auc:.3f} | "
                     f"{r.probe_pr_auc:.3f} | {r.baseline_pr_auc:.3f} | {r.prevalence:.2f} |")
    lines.append("")
    lines.append(
        f"<sub>provenance: n_users={scale.get('n_users', '?')}, model={scale.get('model', '?')}, "
        f"steps={scale.get('steps', '?')}, seed={scale.get('seed', 0)}, commit={_git_commit()}</sub>"
    )
    return "\n".join(lines)


def write_multitask_report(
    rows: list[MultiTaskRow], scale: dict[str, Any], readme_path: str | Path = "README.md",
) -> None:
    """Write the table into the README marker; refuse to shrink the reported scale."""
    md = multitask_results_markdown(rows, scale)
    path = Path(readme_path)
    if not path.exists():
        return
    text = path.read_text()
    if MARKER not in text:
        return
    existing = re.search(rf"{re.escape(MARKER)}.*?provenance: n_users=(\d+)", text, flags=re.S)
    if existing and existing.group(1).isdigit() and int(existing.group(1)) > int(scale.get("n_users", 0)):
        import logging

        logging.getLogger(__name__).warning(
            "existing multi-task table is from a larger run (n_users=%s > %s); not overwriting",
            existing.group(1), scale.get("n_users"))
        return
    text = re.sub(re.escape(MARKER) + r".*?(?=\n## |\Z)",
                  MARKER + "\n\n" + md + "\n", text, count=1, flags=re.S)
    path.write_text(text)
