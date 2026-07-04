"""Actionable-error and warning guards on the public API / ingest paths."""

from __future__ import annotations

import datetime as dt
import logging

import pytest

import pragmatiq.core.schema as schema
from pragmatiq import api
from pragmatiq.core.schema import UserRecord


def test_load_yaml_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n")  # a top-level sequence, not a mapping
    with pytest.raises(ValueError, match="top-level mapping"):
        api._load_yaml(p)


def test_naive_datetime_warns_once(caplog) -> None:
    schema._warned_naive_ts = False  # reset the module-level one-time guard
    with caplog.at_level(logging.WARNING):
        UserRecord.from_dict({
            "user_id": "u",
            "events": [{"ts": dt.datetime(2024, 1, 1, 9, 0), "source": "transaction", "fields": {}}],
        })
        UserRecord.from_dict({
            "user_id": "v",
            "events": [{"ts": dt.datetime(2024, 1, 2, 9, 0), "source": "transaction", "fields": {}}],
        })
    naive_warnings = [r for r in caplog.records if "naive datetime" in r.getMessage()]
    assert len(naive_warnings) == 1  # warned, and only once


# ---------------------------------------------------------------------------
# synthesize(write_report=None) auto mode — a slim install must synthesize
# out of the box (Bugbot PR #10 finding 3449267053 / 3449249152)
# ---------------------------------------------------------------------------


def test_synthesize_default_skips_report_without_matplotlib(tmp_path, monkeypatch, caplog) -> None:
    """Default write_report=None skips the report (with a warning) when matplotlib is absent."""
    import importlib.util as ilu

    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec",
        lambda name, *a, **k: None if name == "matplotlib" else real_find_spec(name, *a, **k),
    )
    with caplog.at_level(logging.WARNING):
        api.synthesize({"n_users": 20, "seed": 1}, out=tmp_path / "raw")
    assert (tmp_path / "raw" / "manifest.json").exists()
    assert not (tmp_path / "raw" / "realism_report.html").exists()
    assert any("realism_report" in r.getMessage() for r in caplog.records)


def test_synthesize_explicit_report_raises_without_matplotlib(tmp_path, monkeypatch) -> None:
    """An explicit write_report=True still demands the [data] extra with a clear error."""
    import sys

    from pragmatiq.core.errors import MissingExtraError

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(MissingExtraError, match=r"pragmatiq\[data\]"):
        api.synthesize({"n_users": 20, "seed": 1}, out=tmp_path / "raw", write_report=True)


def test_synthesize_default_writes_report_when_matplotlib_present(tmp_path) -> None:
    """With matplotlib installed (the dev env), the default still writes the report."""
    pytest.importorskip("matplotlib")
    api.synthesize({"n_users": 20, "seed": 1}, out=tmp_path / "raw")
    assert (tmp_path / "raw" / "realism_report.html").exists()
