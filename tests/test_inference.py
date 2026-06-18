"""Inference tests: batch embedder, event attribution, ONNX export."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pragmatiq import api
from pragmatiq.data.collate import VarlenCollator
from pragmatiq.data.dataset import ShardDataset
from pragmatiq.inference.benchmark import benchmark_batch_embed, perf_analyzer_command
from pragmatiq.inference.embedder import BatchEmbedder
from pragmatiq.inference.explain import EventAttributor
from pragmatiq.models.pragmatiq import PragmaModel


def _set_tokenizer_state_hash(tok_dir: Path, state: dict) -> None:
    blob = json.dumps(state, sort_keys=True).encode()
    bpe_path = tok_dir / "bpe.json"
    if bpe_path.exists():
        blob += bpe_path.read_text().encode()
    (tok_dir / "tokenizer.hash").write_text(hashlib.sha256(blob).hexdigest())


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("p7")
    api.synthesize({"n_users": 200, "months": 14, "n_merchants": 600, "mule_ring_count": 1,
                    "seed": 6, "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", n_workers=0, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 4000, "n_buckets": 32, "categorical_threshold": 200})
    summary = api.pretrain(work / "tok", "p7", model_size="small",
                           config={"max_steps": 20, "token_budget": 4096, "warmup_steps": 5,
                                   "log_every": 5, "checkpoint_every_min": 1000.0},
                           runs_root=work / "runs")
    return work, summary["run_dir"]


class TestBatchEmbedder:
    def test_embed_records_dict_format(self, trained) -> None:
        """The notebook/serving path: plain dicts with {ts, source, fields} events."""
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        records = [
            {"user_id": "svc_1", "events": [
                {"ts": 1_700_000_000_000_000, "source": "transaction",
                 "fields": {"amount": "42.10", "mcc": "5411", "merchant": "TESCO 1"}},
                {"ts": 1_700_003_600_000_000, "source": "app",
                 "fields": {"screen": "home", "action": "view"}}],
             "attributes": {"country": "GB"},
             "lifelong": [{"key": "account_opened", "ts": 1_699_000_000_000_000}]},
            # unseen key must not raise (global rule 4)
            {"user_id": "svc_2", "events": [
                {"ts": 1_700_000_000_000_000, "source": "transaction",
                 "fields": {"amount": "9.99", "totally_new_key": "xyz"}}],
             "attributes": {}, "lifelong": []},
        ]
        emb = model.embed_records(records)
        assert emb.shape == (2, model.config.dim)
        assert np.isfinite(emb).all()

    def test_embed_to_parquet(self, trained, tmp_path: Path) -> None:
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        out = tmp_path / "emb.parquet"
        stats = BatchEmbedder(model).embed_to_parquet(work / "tok", out)
        assert out.exists()
        assert stats["n_users"] == 200
        assert stats["users_per_sec"] > 0
        import pyarrow.parquet as pq

        t = pq.read_table(out)
        assert t.num_rows == 200
        assert len(t.column("embedding")[0]) == model.config.dim

    def test_benchmark(self, trained) -> None:
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        stats = benchmark_batch_embed(model, work / "tok")
        assert stats["users_per_sec"] > 0
        assert stats["tokens_per_sec"] > 0
        assert "pragmatiq_embedder" in perf_analyzer_command()


class TestApiTokenizerCompatibility:
    @pytest.fixture()
    def mismatched_tok(self, trained, tmp_path: Path) -> Path:
        work, _run_dir = trained
        dst = tmp_path / "tok_mismatch"
        shutil.copytree(work / "tok", dst)
        tok_dir = dst / "tokenizer"
        state_path = tok_dir / "tokenizer.json"
        state = json.loads(state_path.read_text())
        state["config"]["calendar_tz"] = "Europe/London"
        state_path.write_text(json.dumps(state, sort_keys=True))
        _set_tokenizer_state_hash(tok_dir, state)
        return dst

    @pytest.fixture()
    def stale_manifest_tok(self, trained, tmp_path: Path) -> Path:
        """Shards whose tokenizer/ still matches the run but whose shard_manifest.json
        records a different tokenizer_hash."""
        work, _run_dir = trained
        dst = tmp_path / "tok_stale_manifest"
        shutil.copytree(work / "tok", dst)
        manifest_path = dst / "shard_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tokenizer_hash"] = "not_the_training_tokenizer"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return dst

    def test_embed_uses_live_tokenizer_hash_not_stale_manifest(
        self, trained, stale_manifest_tok: Path, monkeypatch
    ) -> None:
        # Regression guard: the shard/run tokenizer match reads the on-disk
        # tokenizer's content hash, NOT shard_manifest.json — so a stale manifest
        # must not reject a shard/run pair whose tokenizers actually agree.
        _work, run_dir = trained

        def fake_embed_users(model, dataset, **kwargs):  # noqa: ANN001, ANN003
            return {"u0": np.zeros(model.config.dim, dtype=np.float32)}

        monkeypatch.setattr("pragmatiq.training.probe.embed_users", fake_embed_users)
        dim = PragmaModel.from_pretrained(run_dir).config.dim
        assert api.embed(stale_manifest_tok, run_dir) == {"n_users": 1, "dim": dim}

    def test_embed_rejects_mismatched_shards_before_embedding(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        _work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("embed_users should not run before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.training.probe.embed_users", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.embed(mismatched_tok, run_dir)

    def test_probe_rejects_mismatched_shards_before_embedding(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("probe embedding should not run before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.training.probe.embed_users", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.probe(mismatched_tok, run_dir, work / "raw" / "labels" / "default_12m.parquet")

    def test_finetune_rejects_mismatched_shards_before_training(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("finetune should not start before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.training.finetuner.LoRAFineTuner.fit", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.finetune(mismatched_tok, run_dir, work / "raw" / "labels" / "default_12m.parquet")

    def test_export_rejects_mismatched_shards_before_export(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        _work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("export should not run before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.inference.export.export_onnx", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.export(run_dir, mismatched_tok)

    def test_benchmark_rejects_mismatched_shards_before_benchmark(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        _work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("benchmark should not run before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.inference.benchmark.benchmark_batch_embed", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.benchmark(run_dir, mismatched_tok)

    def test_gnn_rejects_mismatched_shards_before_embedding(
        self, trained, mismatched_tok: Path, monkeypatch
    ) -> None:
        work, run_dir = trained

        def guard_missing(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("gnn embedding should not run before tokenizer compatibility is checked")

        monkeypatch.setattr("pragmatiq.training.probe.embed_users", guard_missing)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            api.gnn(
                mismatched_tok,
                run_dir,
                work / "raw" / "transfers.parquet",
                work / "raw" / "labels" / "aml.parquet",
                seeds=(0,),
                epochs=1,
            )


class TestEventAttributor:
    def test_topk_events(self, trained) -> None:
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        ds = ShardDataset(work / "tok")
        batch = VarlenCollator()([ds.get(u) for u in ds.user_ids[:3]])
        ds.close()
        attrs = EventAttributor(model, steps=8).attribute(batch, top_k=5)
        assert len(attrs) == 3
        for a in attrs:
            assert len(a.event_indices) <= 5
            assert len(a.scores) == len(a.event_indices)


class TestExport:
    def test_pack_to_dense_matches_native(self, trained) -> None:
        """The dense reformulation reproduces the varlen embeddings exactly."""
        from pragmatiq.inference.export import DenseEmbedder, pack_to_dense

        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        ds = ShardDataset(work / "tok")
        # A ragged batch: varied events/user and tokens/event.
        batch = VarlenCollator()([ds.get(u) for u in ds.user_ids[:5]])
        ds.close()
        import torch

        with torch.no_grad():
            native = model.embed_users(batch)
            dense = DenseEmbedder(model)(**pack_to_dense(batch))
        assert torch.allclose(native, dense, atol=1e-4)

    def test_onnx_export(self, trained, tmp_path: Path) -> None:
        pytest.importorskip("onnxscript", reason="install pragmatiq[serve] for ONNX export")
        ort = pytest.importorskip("onnxruntime", reason="install pragmatiq[serve] for ONNX export")
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        ds = ShardDataset(work / "tok")
        example = VarlenCollator()([ds.get(ds.user_ids[0])])
        ds.close()
        from pragmatiq.inference.export import export_onnx, pack_to_dense

        res = export_onnx(model, example, tmp_path / "model.onnx")
        assert (tmp_path / "model.onnx").exists()
        assert res["opset"] == 18
        # the exported model runs and matches the native embedding for that user
        sess = ort.InferenceSession(str(tmp_path / "model.onnx"))
        feeds = {name: t.numpy() for name, t in pack_to_dense(example).items()}
        onnx_emb = sess.run(None, feeds)[0]
        native = model.embed_users(example).detach().numpy()
        assert np.allclose(onnx_emb, native, atol=1e-3)

    def test_onnx_export_dynamic_shapes(self, trained, tmp_path: Path) -> None:
        """One export serves a second batch with different U/E/L/P axes."""
        pytest.importorskip("onnxscript", reason="install pragmatiq[serve] for ONNX export")
        ort = pytest.importorskip("onnxruntime", reason="install pragmatiq[serve] for ONNX export")
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        ds = ShardDataset(work / "tok")
        example = VarlenCollator()([ds.get(ds.user_ids[0])])
        second = VarlenCollator()([ds.get(u) for u in ds.user_ids[1:5]])
        ds.close()
        from pragmatiq.inference.export import export_onnx, pack_to_dense

        export_onnx(model, example, tmp_path / "model.onnx")
        sess = ort.InferenceSession(str(tmp_path / "model.onnx"))
        feeds = {name: t.numpy() for name, t in pack_to_dense(second).items()}
        onnx_emb = sess.run(None, feeds)[0]
        native = model.embed_users(second).detach().numpy()
        assert onnx_emb.shape == native.shape
        assert np.allclose(onnx_emb, native, atol=1e-3)

    def test_export_rejects_non_cpu_device(self, trained) -> None:
        # ONNX export runs on CPU; a non-CPU device must fail fast with guidance
        # rather than crash deep in dense-tensor construction (no GPU needed here).
        work, run_dir = trained
        with pytest.raises(ValueError, match="CPU"):
            api.export(run_dir, work / "tok", device="cuda")

    def test_onnx_export_self_validates(self, trained, tmp_path: Path, monkeypatch) -> None:
        pytest.importorskip("onnxscript", reason="install pragmatiq[serve] for ONNX export")
        pytest.importorskip("onnxruntime", reason="install pragmatiq[serve] for ONNX export")
        work, run_dir = trained
        model = PragmaModel.from_pretrained(run_dir)
        ds = ShardDataset(work / "tok")
        example = VarlenCollator()([ds.get(ds.user_ids[0])])
        ds.close()
        from pragmatiq.inference.export import export_onnx

        class _WrongSession:  # returns a result that diverges from the native path
            def __init__(self, *a, **k) -> None:
                pass

            def run(self, *a, **k):
                return [np.full((1, model.config.dim), 1e9, dtype="float32")]

        monkeypatch.setattr("onnxruntime.InferenceSession", _WrongSession)
        with pytest.raises(RuntimeError, match="numerically equivalent"):
            export_onnx(model, example, tmp_path / "bad.onnx")

    def test_onnx_export_rejects_embed_mode(self, trained, tmp_path: Path) -> None:
        # ONNX dense export covers the BPE token path. A PRAGMA+Nemotron model carries a
        # frozen text encoder whose contribution the dense graph does not restate, so
        # export must refuse with a pointer to the native Triton path instead of emitting
        # a graph whose embeddings would diverge.
        import dataclasses

        work, run_dir = trained
        cfg = dataclasses.replace(PragmaModel.from_pretrained(run_dir).config,
                                  text_encoder="hash", text_encoder_dim=16)
        embed_model = PragmaModel(cfg)
        ds = ShardDataset(work / "tok")
        example = VarlenCollator()([ds.get(ds.user_ids[0])])
        ds.close()
        from pragmatiq.inference.export import export_onnx

        with pytest.raises(NotImplementedError, match="Triton"):
            export_onnx(embed_model, example, tmp_path / "embed.onnx")


def test_from_pretrained_rejects_unsupported_format(trained, tmp_path: Path) -> None:
    import shutil

    import torch

    work, run_dir = trained
    dst = tmp_path / "run"
    shutil.copytree(run_dir, dst)
    ckpt_path = dst / "checkpoints" / "last.pt"
    ck = torch.load(ckpt_path, weights_only=False)
    ck["format"] = 999
    torch.save(ck, ckpt_path)
    with pytest.raises(ValueError, match="checkpoint format"):
        PragmaModel.from_pretrained(dst)


class TestTritonServingContract:
    """The Triton python-backend model.py request→response cycle, Docker-free: a
    records_json request must yield the [n_users, dim] embedding matrix. (End-to-end
    container serving is validated on a GPU box by scripts/deploy_serving.sh.)"""

    @staticmethod
    def _load_model_module():
        import importlib.util

        path = (Path(__file__).resolve().parent.parent / "deploy" / "triton" /
                "model_repository" / "pragmatiq_embedder" / "1" / "model.py")
        spec = importlib.util.spec_from_file_location("triton_pragmatiq_model", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_records_json_request_returns_embeddings(self, trained) -> None:
        import json
        import types

        work, run_dir = trained
        mod = self._load_model_module()

        captured: dict[str, object] = {}

        class _OutTensor:
            def __init__(self, name, arr):
                self.name, self.arr = name, arr

        class _Response:
            def __init__(self, output_tensors):
                self.output_tensors = output_tensors

        def _get_input(request, name):
            return types.SimpleNamespace(as_numpy=lambda: np.array([request["payload"].encode()]))

        mod.pb_utils = types.SimpleNamespace(
            get_input_tensor_by_name=_get_input, Tensor=_OutTensor, InferenceResponse=_Response)

        records = [
            {"user_id": "svc_1", "events": [
                {"ts": 1_700_000_000_000_000, "source": "transaction",
                 "fields": {"amount": "42.10", "mcc": "5411", "merchant": "TESCO 1"}}],
             "attributes": {"country": "GB"}, "lifelong": []},
            {"user_id": "svc_2", "events": [
                {"ts": 1_700_000_000_000_000, "source": "app",
                 "fields": {"screen": "home"}}],
             "attributes": {}, "lifelong": []},
        ]
        model = mod.TritonPythonModel()
        model.initialize({"model_config": json.dumps(
            {"parameters": {"run_dir": {"string_value": str(run_dir)}}}),
            "model_instance_kind": "CPU"})
        responses = model.execute([{"payload": json.dumps(records)}])
        captured["emb"] = responses[0].output_tensors[0].arr
        emb = captured["emb"]
        assert emb.shape == (2, model.model.config.dim)
        assert emb.dtype == np.float32 and np.isfinite(emb).all()
        model.finalize()
