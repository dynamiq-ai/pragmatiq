"""Unit tests for pragmatiq.experiments.tracking.MetricLogger.

Covers JSONL logging (always on), optional TensorBoard/wandb backends,
and the graceful-degradation contract: missing optional backends must warn,
never raise.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from pragmatiq.experiments.tracking import MetricLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path, **kwargs) -> MetricLogger:
    return MetricLogger(run_dir=tmp_path, **kwargs)


# ---------------------------------------------------------------------------
# JSONL always-on
# ---------------------------------------------------------------------------


def test_jsonl_file_created(tmp_path: Path) -> None:
    """MetricLogger always creates metrics.jsonl in run_dir."""
    with _make_logger(tmp_path) as logger:
        logger.log(0, {"loss": 1.0})
    assert (tmp_path / "metrics.jsonl").exists()


def test_jsonl_rows_written(tmp_path: Path) -> None:
    """Each log() call appends a JSON row with step and metrics."""
    with _make_logger(tmp_path) as logger:
        logger.log(0, {"loss": 2.0, "lr": 1e-3})
        logger.log(1, {"loss": 1.5})

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line]
    assert rows[0] == {"step": 0, "loss": 2.0, "lr": 1e-3}
    assert rows[1] == {"step": 1, "loss": 1.5}


def test_truncate_after_drops_later_steps(tmp_path: Path) -> None:
    """truncate_after(n) removes rows with step > n from the JSONL file."""
    logger = _make_logger(tmp_path)
    for i in range(5):
        logger.log(i, {"loss": float(i)})
    logger.truncate_after(2)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line]
    assert all(r["step"] <= 2 for r in rows)
    assert len(rows) == 3
    logger.close()


# ---------------------------------------------------------------------------
# wandb graceful degradation — Bugbot finding #1
# ---------------------------------------------------------------------------


def test_wandb_missing_warns_not_raises(tmp_path: Path, monkeypatch) -> None:
    """wandb=True with wandb absent must degrade gracefully and NOT raise.

    This is the Bugbot-reported regression: the old code raised MissingExtraError;
    the fixed code warns (see test_wandb_missing_warning_message) and degrades to
    JSONL-only. Here we assert only the no-raise + JSONL-still-works contract.
    """
    # Block wandb so the import fails, regardless of whether it is installed.
    monkeypatch.setitem(sys.modules, "wandb", None)

    # Must not raise — must degrade gracefully.
    logger = MetricLogger(run_dir=tmp_path, wandb=True)
    logger.log(0, {"loss": 0.5})
    logger.close()

    jsonl = (tmp_path / "metrics.jsonl").read_text().strip()
    assert jsonl, "JSONL logging must still work when wandb is absent"


def test_wandb_missing_warning_message(tmp_path: Path, monkeypatch, caplog) -> None:
    """The warning emitted when wandb is absent must mention wandb and installation."""
    monkeypatch.setitem(sys.modules, "wandb", None)

    with caplog.at_level(logging.WARNING, logger="pragmatiq.experiments.tracking"):
        MetricLogger(run_dir=tmp_path, wandb=True).close()

    assert any(
        "wandb" in r.message.lower() and "install" in r.message.lower()
        for r in caplog.records
    ), f"Expected a warning about wandb installation, got: {[r.message for r in caplog.records]}"


def test_wandb_false_no_warning(tmp_path: Path, caplog) -> None:
    """No wandb warning when wandb=False (the default)."""
    with caplog.at_level(logging.WARNING, logger="pragmatiq.experiments.tracking"):
        MetricLogger(run_dir=tmp_path, wandb=False).close()

    assert not any("wandb" in r.message.lower() for r in caplog.records)


def test_jsonl_works_when_wandb_absent(tmp_path: Path, monkeypatch) -> None:
    """JSONL logging is fully functional even when wandb is missing."""
    monkeypatch.setitem(sys.modules, "wandb", None)

    with MetricLogger(run_dir=tmp_path, wandb=True) as logger:
        logger.log(0, {"loss": 9.9})
        logger.log(1, {"loss": 8.8})

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["loss"] == pytest.approx(9.9)


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------


def test_context_manager_closes_file(tmp_path: Path) -> None:
    """MetricLogger.__exit__ must flush and close the JSONL file handle."""
    with _make_logger(tmp_path) as logger:
        logger.log(0, {"x": 1})
    assert logger._fh.closed
