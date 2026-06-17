"""Actionable-error and warning guards on the public API / ingest paths."""

from __future__ import annotations

import datetime as dt
import logging

import pytest

import pragmatiq.data.schema as schema
from pragmatiq import api
from pragmatiq.data.schema import UserRecord


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
