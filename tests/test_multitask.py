"""Multi-task probe benchmark: markdown rendering + README writeback."""

from __future__ import annotations

from pathlib import Path

from pragmatiq.inference.multitask import (
    MARKER,
    USER_LEVEL_TASKS,
    MultiTaskRow,
    multitask_results_markdown,
    write_multitask_report,
)

# MultiTaskRow: task, probe ROC-AUC, baseline ROC-AUC, probe PR-AUC, baseline PR-AUC, prevalence, n_test
_ROWS = [
    MultiTaskRow("default_12m", 0.6195766, 0.5144699, 0.0834, 0.0312, 0.02, 974),
    MultiTaskRow("churn_6m", 0.6847288, 0.6171318, 0.2451, 0.1602, 0.11, 935),
    MultiTaskRow("ltv_positive", 0.7227447, 0.6097792, 0.8866, 0.8203, 0.81, 935),
]
_SCALE = {"n_users": 4000, "model": "nano", "steps": 1200, "seed": 0}


def test_markdown_has_table_and_provenance() -> None:
    md = multitask_results_markdown(_ROWS, _SCALE)
    assert "probe ROC-AUC" in md and "baseline ROC-AUC" in md
    assert "probe PR-AUC" in md and "baseline PR-AUC" in md
    assert "| default_12m | 0.620 | 0.514 | 0.083 | 0.031 | 0.02 |" in md
    assert "provenance: n_users=4000" in md and "commit=" in md


def test_writeback_replaces_marker(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n## Probes\n\n{MARKER}\n\nold table\n\n## Next\n")
    write_multitask_report(_ROWS, _SCALE, readme_path=readme)
    out = readme.read_text()
    assert "0.620" in out and "## Next" in out and "old table" not in out
    assert out.count(MARKER) == 1


def test_writeback_refuses_scale_downgrade(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{MARKER}\n\n"
                      f"<sub>provenance: n_users=100000, model=small, steps=2000, seed=0, "
                      f"commit=abc</sub>\n\n## Next\n")
    write_multitask_report(_ROWS, _SCALE, readme_path=readme)  # 4000 < 100000
    out = readme.read_text()
    assert "n_users=100000" in out  # larger-scale table preserved
    assert "0.620" not in out


def test_tasks_scope_is_user_level_only() -> None:
    assert USER_LEVEL_TASKS == ("default_12m", "churn_6m", "ltv_positive")
    for excluded in ("fraud", "recurring", "comm_uplift", "aml"):
        assert excluded not in USER_LEVEL_TASKS
