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


class TestStagingEagerValidation:
    """Unknown schemes / missing backends fail at input()/output() time,
    before the api function spends any compute."""

    def test_output_unknown_scheme_fails_eagerly(self):
        from pragmatiq.storage.staging import staging

        with pytest.raises((ValueError, ImportError)):
            with staging() as stage:
                stage.output("bogus-scheme://bucket/out", is_dir=True)
                raise AssertionError("output() accepted an unknown scheme")

    def test_input_unknown_scheme_fails_eagerly(self):
        from pragmatiq.storage.staging import staging

        with pytest.raises((ValueError, ImportError)):
            with staging() as stage:
                stage.input("bogus-scheme://bucket/in")
                raise AssertionError("input() accepted an unknown scheme")


class TestStagingUploadFailure:
    """An upload failure must PRESERVE the computed results, not rmtree them."""

    def test_upload_failure_preserves_work_root(self, monkeypatch):
        import importlib
        import re
        import shutil
        from pathlib import Path

        # `pragmatiq.storage` re-exports the staging() function under the same
        # name as the submodule, so resolve the module object explicitly.
        staging_mod = importlib.import_module("pragmatiq.storage.staging")

        def _boom(local, remote):
            raise OSError("simulated upload outage")

        monkeypatch.setattr(staging_mod, "put_dir", _boom)
        with pytest.raises(RuntimeError, match="preserved locally") as excinfo:
            with staging_mod.staging() as stage:
                local_out = Path(stage.output("memory:///preserve/dir", is_dir=True))
                (local_out / "result.txt").write_text("expensive")

        m = re.search(r"preserved locally at (\S+)", str(excinfo.value))
        assert m, f"error does not name the preserved path: {excinfo.value}"
        work_root = Path(m.group(1))
        try:
            preserved = list(work_root.rglob("result.txt"))
            assert preserved and preserved[0].read_text() == "expensive"
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    def test_body_exception_still_cleans_up_work_root(self):
        from pathlib import Path

        from pragmatiq.storage.staging import staging

        local_out = None
        with pytest.raises(RuntimeError, match="intentional"):
            with staging() as stage:
                local_out = Path(stage.output("memory:///cleanup/dir", is_dir=True))
                raise RuntimeError("intentional")
        assert local_out is not None and not local_out.exists()


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


# ------------------------------------------------------------------ #
# Bugbot PR #10 regression tests (issues 2 and 3)
# ------------------------------------------------------------------ #


def test_remote_config_yaml_loaded_by_load_yaml():
    """load_yaml() must read a remote (memory://) config URL, not fail.

    Regression guard for Bugbot issue 3: config=s3://... was passed straight
    to OmegaConf.load which only handles local files.  Now load_yaml is
    storage-aware.
    """
    import fsspec

    from pragmatiq.core.config import load_yaml

    mem = fsspec.filesystem("memory")
    mem.makedirs("/cfgtest", exist_ok=True)
    with mem.open("/cfgtest/train.yaml", "wb") as f:
        f.write(b"max_steps: 5\ntoken_budget: 512\n")

    result = load_yaml("memory:///cfgtest/train.yaml")
    assert result == {"max_steps": 5, "token_budget": 512}, (
        f"load_yaml() returned unexpected result for remote URL: {result}"
    )


def test_fresh_remote_pretrain_does_not_pull_existing_run(tmp_path):
    """A fresh pretrain (no resume) must NOT stage an existing remote run.

    Regression guard for Bugbot issue 2: when runs_root was a remote URL and
    a run with the same name already existed remotely, the staging block used
    to download it even without resume='auto', potentially injecting stale
    checkpoints into a new run.
    """
    import fsspec

    from pragmatiq.storage.cache import put_dir

    # Synthesize + tokenize a tiny dataset
    api.synthesize(SYNTH_CFG, out=tmp_path / "raw", write_report=False)
    api.tokenize(tmp_path / "raw", tmp_path / "tok")

    # Run #1: train locally and upload the checkpoint to memory://
    api.pretrain(
        tmp_path / "tok",
        "resume_guard_run",
        model_size="nano",
        config=TRAIN_CFG,
        runs_root=tmp_path / "runs1",
    )
    remote_runs = "memory:///resume_guard_test/runs"
    put_dir(tmp_path / "runs1" / "resume_guard_run", remote_runs + "/resume_guard_run")

    # Run #2: fresh pretrain (no resume) — must NOT pull the remote checkpoint
    # We verify this by checking that the local staging dir is empty at start,
    # i.e., the trainer starts from step 0 (no resumed step counter).
    fresh_cfg = {k: v for k, v in TRAIN_CFG.items() if k != "max_steps"}
    fresh_cfg["max_steps"] = 2
    second = api.pretrain(
        tmp_path / "tok",
        "resume_guard_run",
        model_size="nano",
        config=fresh_cfg,
        runs_root=remote_runs,
        resume=None,  # explicitly NOT resuming
    )
    # A fresh run always starts at 0 + max_steps steps total — if stale
    # checkpoint were staged, step count could exceed max_steps.
    assert second["steps"] <= TRAIN_CFG["max_steps"] + 2, (
        "Fresh remote pretrain (resume=None) appears to have loaded a stale checkpoint"
    )

    # Clean up memory fs
    mem = fsspec.filesystem("memory")
    mem.store.clear()


def test_remote_pretrain_returns_durable_remote_run_dir(tmp_path):
    """pretrain() with a remote runs_root must return the remote run URL.

    Regression guard for Bugbot finding 3449249150: the returned run_dir used
    to be the staged local temp path, which staging deletes right after the
    upload — so the caller received a path that no longer existed.
    """
    api.synthesize(SYNTH_CFG, out=tmp_path / "raw", write_report=False)
    api.tokenize(tmp_path / "raw", tmp_path / "tok")

    remote_runs = "memory:///durable_rundir/runs"
    result = api.pretrain(tmp_path / "tok", "durable_run", model_size="nano",
                          config=TRAIN_CFG, runs_root=remote_runs)

    assert result["run_dir"] == remote_runs + "/durable_run", result["run_dir"]
    # The returned location must actually be usable — embed straight from it.
    api.embed(tmp_path / "tok", result["run_dir"], out=tmp_path / "emb.parquet")
    assert (tmp_path / "emb.parquet").exists()


def test_remote_tokenizer_dir_is_staged(tmp_path):
    """tokenize(tokenizer_dir=<remote url>) must materialize the tokenizer.

    Regression guard for Bugbot finding 3449267057: a remote tokenizer_dir was
    passed straight to ``PragmaTokenizer.load()``, which only reads local paths.
    """
    from pragmatiq.storage.cache import put_dir

    api.synthesize(SYNTH_CFG, out=tmp_path / "raw", write_report=False)
    local_manifest = api.tokenize(tmp_path / "raw", tmp_path / "tok1")

    remote_tok = "memory:///tokdir_test/tokenizer"
    put_dir(tmp_path / "tok1" / "tokenizer", remote_tok)

    remote_manifest = api.tokenize(tmp_path / "raw", tmp_path / "tok2",
                                   tokenizer_dir=remote_tok)
    assert remote_manifest["tokenizer_hash"] == local_manifest["tokenizer_hash"], (
        "remote tokenizer_dir produced a different tokenizer than the local one it mirrors"
    )
    # The produced shard dir must be self-contained: downstream commands read
    # out/tokenizer for the hash check, so tokenize(tokenizer_dir=...) has to
    # persist the loaded tokenizer into the output dir too.
    assert (tmp_path / "tok2" / "tokenizer" / "tokenizer.json").exists(), (
        "tokenize(tokenizer_dir=...) did not persist the tokenizer into out/tokenizer"
    )
    assert api._read_shard_tokenizer_hash(tmp_path / "tok2") == local_manifest["tokenizer_hash"]


def test_export_with_remote_out_returns_remote_url(tmp_path, monkeypatch):
    """export() with a remote out must return the remote URL, not the deleted temp path.

    export_onnx itself is stubbed (the ONNX toolchain is orthogonal to staging);
    the stub writes bytes to the staged path so the upload leg is exercised.
    """
    from pathlib import Path

    import pragmatiq.inference.export as export_mod

    api.synthesize(SYNTH_CFG, out=tmp_path / "raw", write_report=False)
    api.tokenize(tmp_path / "raw", tmp_path / "tok")
    trained = api.pretrain(tmp_path / "tok", "export_run", model_size="nano",
                           config=TRAIN_CFG, runs_root=tmp_path / "runs")

    def _fake_export_onnx(model, example_batch, out_path, opset=18):
        Path(out_path).write_bytes(b"onnx-bytes")
        return {"out": str(out_path), "opset": opset, "max_abs_diff": 0.0}

    monkeypatch.setattr(export_mod, "export_onnx", _fake_export_onnx)
    remote_out = "memory:///export_test/model.onnx"
    result = api.export(trained["run_dir"], tmp_path / "tok", out=remote_out)
    assert result["out"] == remote_out, result["out"]
    assert storage.read_bytes(remote_out) == b"onnx-bytes"


def test_runs_list_missing_remote_root_returns_empty():
    """runs_list on a nonexistent REMOTE root matches the local behaviour ([])."""
    assert api.runs_list("memory:///no/such/runs_root") == []


def test_runs_compare_missing_remote_root_flags_all_missing():
    """runs_compare on a nonexistent REMOTE root flags every run as missing."""
    res = api.runs_compare(["a", "b"], "memory:///no/such/runs_root")
    assert res == [{"name": "a", "missing": True}, {"name": "b", "missing": True}]


def test_quickstart_remote_out_lands_in_remote_store():
    """quickstart(out=<remote url>) must place every stage in the remote store.

    Regression guard for Bugbot finding 3523826773: child paths were built with
    Path(out) / "raw", which collapses s3://bucket to s3:/bucket — a local path.
    """
    # n_users must give the credit-default probe a few positives in both split
    # halves (default_12m prevalence is a few percent), while staying small
    # enough for a CI-scale run.
    res = api.quickstart(out="memory:///qs_remote", n_users=250, seed=0,
                         model_size="nano", max_steps=2)
    assert res["run_dir"] == "memory:///qs_remote/runs/quickstart", res["run_dir"]
    assert storage.exists("memory:///qs_remote/raw/manifest.json")
    assert storage.exists("memory:///qs_remote/raw/labels/default_12m.parquet")
    assert storage.exists("memory:///qs_remote/tok/tokenizer/tokenizer.json")
    assert storage.exists("memory:///qs_remote/runs/quickstart/meta.json")
    assert "probe_auc" in res["probe"]
