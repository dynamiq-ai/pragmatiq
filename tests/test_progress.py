"""Tests for pragmatiq.progress — display-only iteration wrappers."""

from __future__ import annotations

import logging

import pragmatiq.progress as P


class TestProgress:
    def test_yields_all_items_in_order(self, monkeypatch) -> None:
        monkeypatch.setattr(P, "_interactive", lambda: False)
        items = list(range(100))
        assert list(P.progress(iter(items), total=100, desc="t")) == items

    def test_disabled_passthrough(self) -> None:
        items = ["a", "b"]
        assert list(P.progress(iter(items), enabled=False)) == items

    def test_lazy_no_prefetch(self, monkeypatch) -> None:
        monkeypatch.setattr(P, "_interactive", lambda: False)
        seen: list[int] = []

        def gen():
            for i in range(10):
                seen.append(i)
                yield i

        it = P.progress(gen(), total=10)
        assert next(it) == 0
        assert seen == [0], "wrapper must not pull ahead of the consumer"

    def test_non_tty_fallback_logs(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(P, "LOG_INTERVAL_S", 0.0)
        monkeypatch.setattr(P, "_interactive", lambda: False)
        with caplog.at_level(logging.INFO, logger="pragmatiq.progress"):
            list(P.progress(range(5), total=5, desc="phase", unit="user"))
        assert any("phase" in r.message for r in caplog.records)

    def test_quiet_when_fast(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(P, "_interactive", lambda: False)
        with caplog.at_level(logging.INFO, logger="pragmatiq.progress"):
            list(P.progress(range(5), total=5, desc="phase"))
        assert not caplog.records, "short phases must stay silent in CI logs"


class TestTokenizeManifestBestEffort:
    """A foreign/corrupt manifest.json must never break tokenize (display-only)."""

    def _dataset(self, tmp_path):
        from pragmatiq import api

        out = tmp_path / "ds"
        api.synthesize({"n_users": 12, "seed": 3}, out=out, write_report=False)
        return out

    def test_corrupt_manifest_ignored(self, tmp_path) -> None:
        from pragmatiq import api

        ds = self._dataset(tmp_path)
        (ds / "manifest.json").write_text("{not json")
        manifest = api.tokenize(ds, tmp_path / "tok")
        assert manifest["n_users"] == 12

    def test_non_numeric_n_users_with_max_users(self, tmp_path) -> None:
        from pragmatiq import api

        ds = self._dataset(tmp_path)
        (ds / "manifest.json").write_text('{"n_users": "twelve"}')
        manifest = api.tokenize(ds, tmp_path / "tok", max_users=5)
        assert manifest["n_users"] == 5


class TestParallelTokenize:
    """n_workers must not change a single output byte (rule 2)."""

    def test_parallel_matches_inline_byte_identical(self, tmp_path) -> None:
        import hashlib

        from pragmatiq import api

        ds = tmp_path / "ds"
        api.synthesize({"n_users": 80, "seed": 4}, out=ds, write_report=False)
        m0 = api.tokenize(ds, tmp_path / "tok0", n_workers=0)
        m2 = api.tokenize(ds, tmp_path / "tok2", n_workers=2)
        assert m0 == m2

        def shard_hashes(d):
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((d / "shards").glob("*.parquet"))}

        h0, h2 = shard_hashes(tmp_path / "tok0"), shard_hashes(tmp_path / "tok2")
        assert h0 == h2, "parallel tokenize must produce byte-identical shards"
