"""End-to-end pipeline equivalence test: local vs remote (memory://) staging.

Runs synthesize → tokenize → pretrain (nano, 2 steps) → embed(out=parquet)
with both local and memory:// paths, then asserts the embeddings are identical.
"""
from __future__ import annotations

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

        mem = fsspec.filesystem("memory")
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
    """Staging a checkpoint through memory:// and back must be bit-exact.

    One training run is performed locally; the resulting checkpoint is uploaded to
    memory:// via put_dir, then embed() is called twice:
      1. from the original local run directory
      2. from the memory:// URL (staging materialises it back before inference)

    The two embed calls use the SAME checkpoint and the SAME tokenised data, so
    the outputs must be bit-for-bit identical — any difference would indicate that
    staging corrupted the checkpoint or the tokenised input.
    """
    from pragmatiq.storage.cache import put_dir

    raw = tmp_path / "raw"
    tok = tmp_path / "tok"
    runs = tmp_path / "runs"
    local_emb = tmp_path / "local_embeddings.parquet"
    staged_emb = tmp_path / "staged_embeddings.parquet"

    # Single training run (local)
    api.synthesize(SYNTH_CFG, out=raw, write_report=False)
    api.tokenize(raw, tok)
    pretrain_result = api.pretrain(
        tok,
        "testrun",
        model_size="nano",
        config=TRAIN_CFG,
        runs_root=runs,
    )
    local_run_dir = pretrain_result["run_dir"]

    # Embed from the local run directory
    api.embed(tok, local_run_dir, out=local_emb)
    local_embs = _read_embeddings(local_emb)

    # Upload the SAME checkpoint to memory:// and embed from there
    mem_run = "memory:///pipeline/testrun"
    put_dir(local_run_dir, mem_run)
    api.embed(tok, mem_run, out=staged_emb)
    staged_embs = _read_embeddings(staged_emb)

    # Assert same user IDs
    assert set(local_embs.keys()) == set(staged_embs.keys()), (
        f"User ID sets differ: local={set(local_embs.keys())}, "
        f"staged={set(staged_embs.keys())}"
    )

    # Staging must be bit-exact: same checkpoint bytes → same model weights → same output.
    for uid in local_embs:
        np.testing.assert_array_equal(
            local_embs[uid],
            staged_embs[uid],
            err_msg=f"Embedding mismatch for user {uid} — staging altered the checkpoint",
        )


def test_embed_missing_remote_run_raises(tmp_path):
    """embed() with a non-existent remote run URL must raise a clear error."""
    api.synthesize(dict(n_users=20, seed=1), out=tmp_path / "raw", write_report=False)
    api.tokenize(tmp_path / "raw", tmp_path / "tok")

    with pytest.raises((FileNotFoundError, ValueError, OSError, RuntimeError), match=r"nonexistent"):
        api.embed(tmp_path / "tok", "memory:///nonexistent/run")
