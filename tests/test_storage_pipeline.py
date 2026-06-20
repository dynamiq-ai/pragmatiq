"""End-to-end pipeline equivalence test: local vs remote (memory://) staging.

Runs synthesize → tokenize → pretrain (nano, 2 steps) → embed(out=parquet)
with both local and memory:// paths, then asserts the embeddings are identical.
"""
from __future__ import annotations

import io

import numpy as np
import pyarrow.parquet as pq
import pytest

import pragmatiq.api as api
import pragmatiq.storage as storage


@pytest.fixture(autouse=True)
def _clean_memory_fs():
    import fsspec

    mem = fsspec.filesystem("memory")
    mem.store.clear()
    yield
    mem.store.clear()


SYNTH_CFG = dict(
    n_users=30,
    seed=42,
    months=16,
    n_merchants=200,
    mule_ring_count=1,
    eval_month_credit=4,
    eval_month_short=6,
)
TRAIN_CFG = dict(max_steps=2, token_budget=512, warmup_steps=1, seed=0)


def _read_embeddings(parquet_path):
    """Read parquet from local path and return {user_id: np.array}."""
    t = pq.read_table(str(parquet_path))
    df = t.to_pandas()
    return {row["user_id"]: np.array(row["embedding"]) for _, row in df.iterrows()}


# ------------------------------------------------------------------ #
# staging unit tests
# ------------------------------------------------------------------ #


class TestStagingLocalPassthrough:
    """Local paths pass through unchanged — no temp dir, no upload."""

    def test_input_local_unchanged(self, tmp_path):
        from pragmatiq.storage.staging import staging

        p = tmp_path / "data"
        p.mkdir()
        with staging() as stage:
            result = stage.input(str(p))
        assert str(result) == str(p)

    def test_output_local_unchanged(self, tmp_path):
        from pragmatiq.storage.staging import staging

        p = tmp_path / "out"
        with staging() as stage:
            result = stage.output(str(p), is_dir=True)
        assert str(result) == str(p)

    def test_input_none_unchanged(self):
        from pragmatiq.storage.staging import staging

        with staging() as stage:
            result = stage.input(None)
        assert result is None

    def test_output_none_unchanged(self):
        from pragmatiq.storage.staging import staging

        with staging() as stage:
            result = stage.output(None, is_dir=False)
        assert result is None


class TestStagingRemoteInput:
    """Remote inputs are materialized into a local temp slot."""

    def test_dir_materialized(self, tmp_path):
        from pathlib import Path

        import fsspec

        from pragmatiq.storage.staging import staging

        # Write a small dir tree to memory://
        mem = fsspec.filesystem("memory")
        mem.makedirs("/testinput", exist_ok=True)
        with mem.open("/testinput/a.txt", "wb") as f:
            f.write(b"hello")
        with mem.open("/testinput/sub/b.txt", "wb") as f:
            f.write(b"world")

        # Re-run and check inside context
        mem.makedirs("/testinput2", exist_ok=True)
        with mem.open("/testinput2/c.txt", "wb") as f:
            f.write(b"check")

        with staging() as stage2:
            local2 = stage2.input("memory:///testinput2")
            assert (Path(local2) / "c.txt").exists()

    def test_file_materialized(self):
        from pathlib import Path

        import fsspec

        from pragmatiq.storage.staging import staging

        mem = fsspec.filesystem("memory")
        mem.makedirs("/testfile", exist_ok=True)
        with mem.open("/testfile/payload.bin", "wb") as f:
            f.write(b"\xde\xad\xbe\xef")

        with staging() as stage:
            local = stage.input("memory:///testfile/payload.bin")
            assert Path(local).exists()
            assert Path(local).read_bytes() == b"\xde\xad\xbe\xef"


class TestStagingRemoteOutput:
    """Remote outputs are uploaded on clean exit; NOT on exception."""

    def test_dir_uploaded_on_clean_exit(self, tmp_path):
        from pathlib import Path

        import fsspec

        from pragmatiq.storage.staging import staging

        with staging() as stage:
            local_out = Path(stage.output("memory:///outtest/mydir", is_dir=True))
            (local_out / "result.txt").write_text("done")

        # After clean exit, check memory://
        mem = fsspec.filesystem("memory")
        assert mem.exists("/outtest/mydir/result.txt")
        with mem.open("/outtest/mydir/result.txt", "rb") as f:
            assert f.read() == b"done"

    def test_file_uploaded_on_clean_exit(self):
        from pathlib import Path

        from pragmatiq.storage.staging import staging

        with staging() as stage:
            local_out = Path(stage.output("memory:///outtest/file/result.bin", is_dir=False))
            local_out.write_bytes(b"\xca\xfe\xba\xbe")

        assert storage.read_bytes("memory:///outtest/file/result.bin") == b"\xca\xfe\xba\xbe"

    def test_no_upload_on_exception(self):
        from pathlib import Path

        from pragmatiq.storage.staging import staging

        with pytest.raises(RuntimeError, match="intentional"):
            with staging() as stage:
                local_out = Path(stage.output("memory:///noupload/dir", is_dir=True))
                (local_out / "bad.txt").write_text("never uploaded")
                raise RuntimeError("intentional")

        assert not storage.exists("memory:///noupload/dir/bad.txt")


# ------------------------------------------------------------------ #
# put_dir tests
# ------------------------------------------------------------------ #


class TestPutDir:
    def test_local_to_local(self, tmp_path):
        from pragmatiq.storage.cache import put_dir

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("aaa")
        (src / "sub").mkdir()
        (src / "sub" / "b.txt").write_text("bbb")

        dst = tmp_path / "dst"
        put_dir(src, str(dst))
        assert (dst / "a.txt").read_text() == "aaa"
        assert (dst / "sub" / "b.txt").read_text() == "bbb"

    def test_local_to_memory(self, tmp_path):
        import fsspec

        from pragmatiq.storage.cache import put_dir  # noqa: PLC0415

        src = tmp_path / "src"
        src.mkdir()
        (src / "x.txt").write_text("xxx")
        (src / "deep").mkdir()
        (src / "deep" / "y.txt").write_text("yyy")

        put_dir(src, "memory:///putdir/dst")

        mem = fsspec.filesystem("memory")
        assert mem.exists("/putdir/dst/x.txt")
        assert mem.exists("/putdir/dst/deep/y.txt")
        with mem.open("/putdir/dst/x.txt", "rb") as f:
            assert f.read() == b"xxx"

    def test_round_trip_with_materialize(self, tmp_path):
        """put_dir then materialize_dir must round-trip all files byte-exactly."""
        from pragmatiq.storage.cache import materialize_dir, put_dir

        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.bin").write_bytes(b"\x01\x02\x03")
        (src / "nested").mkdir()
        (src / "nested" / "file2.bin").write_bytes(b"\xaa\xbb")

        put_dir(src, "memory:///roundtrip/dir")

        dst = tmp_path / "dst"
        materialize_dir("memory:///roundtrip/dir", dst)

        assert (dst / "file1.bin").read_bytes() == b"\x01\x02\x03"
        assert (dst / "nested" / "file2.bin").read_bytes() == b"\xaa\xbb"


# ------------------------------------------------------------------ #
# pipeline equivalence tests
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_local_vs_remote_pipeline_equivalence(tmp_path):
    """Embeddings produced via remote (memory://) staging must equal local embeddings."""
    # --- LOCAL run ---
    local_raw = tmp_path / "local" / "raw"
    local_tok = tmp_path / "local" / "tok"
    local_runs = tmp_path / "local" / "runs"
    local_emb = tmp_path / "local" / "embeddings.parquet"

    api.synthesize(SYNTH_CFG, out=local_raw, write_report=False)
    api.tokenize(local_raw, local_tok)
    pretrain_result = api.pretrain(
        local_tok,
        "testrun",
        model_size="nano",
        config=TRAIN_CFG,
        runs_root=local_runs,
    )
    local_run_dir = pretrain_result["run_dir"]
    api.embed(local_tok, local_run_dir, out=local_emb)
    local_embs = _read_embeddings(local_emb)

    # --- REMOTE (memory://) run ---
    mem_raw = "memory:///pipeline/raw"
    mem_tok = "memory:///pipeline/tok"
    mem_runs = "memory:///pipeline/runs"
    mem_emb = "memory:///pipeline/embeddings.parquet"

    api.synthesize(SYNTH_CFG, out=mem_raw, write_report=False)
    api.tokenize(mem_raw, mem_tok)
    api.pretrain(
        mem_tok,
        "testrun",
        model_size="nano",
        config=TRAIN_CFG,
        runs_root=mem_runs,
    )
    # After staging, run was uploaded to mem_runs/testrun; stage.input pulls it back for embed
    mem_run = mem_runs.rstrip("/") + "/testrun"
    api.embed(mem_tok, mem_run, out=mem_emb)

    # Read remote embeddings back: download parquet from memory://
    parquet_bytes = storage.read_bytes(mem_emb)
    remote_embs_table = pq.read_table(io.BytesIO(parquet_bytes))
    remote_embs = {
        row["user_id"]: np.array(row["embedding"])
        for _, row in remote_embs_table.to_pandas().iterrows()
    }

    # Assert same user IDs
    assert set(local_embs.keys()) == set(remote_embs.keys()), (
        f"User ID sets differ: local={set(local_embs.keys())}, "
        f"remote={set(remote_embs.keys())}"
    )

    # Assert embeddings are numerically identical (same model, same seed, same data)
    for uid in local_embs:
        np.testing.assert_allclose(
            local_embs[uid],
            remote_embs[uid],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Embedding mismatch for user {uid}",
        )


def test_embed_missing_remote_run_raises(tmp_path):
    """embed() with a non-existent remote run URL must raise a clear error."""
    api.synthesize(dict(n_users=20, seed=1), out=tmp_path / "raw", write_report=False)
    api.tokenize(tmp_path / "raw", tmp_path / "tok")

    with pytest.raises((FileNotFoundError, ValueError, OSError, RuntimeError)):
        api.embed(tmp_path / "tok", "memory:///nonexistent/run")
