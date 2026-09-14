"""Additional validation legs (1.1.0): quickstart timing, ONNX export on the pod,
serving request caps, attention-backend report, bf16-vs-fp32 embedding agreement.

Every leg returns a plain dict that lands in the JSON evidence file; none of
them raises — a failure is recorded as ``{"passed": False, "error": ...}`` so
the harness keeps going and the verdict reports it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import REPO_ROOT


def _guard(fn: Any) -> Any:
    """Wrap a leg so any exception becomes a failed record instead of a crash."""
    def run(*a: Any, **kw: Any) -> dict[str, Any]:
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"[{fn.__name__}] ERROR: {exc}", flush=True)
            return {"passed": False, "error": str(exc)}
    run.__name__ = fn.__name__
    return run


@_guard
def quickstart_timing(out_dir: Path, *, n_users: int = 2000, max_steps: int = 80,
                      timeout_sec: float = 3600.0) -> dict[str, Any]:
    """Wall-clock of ``pragmatiq quickstart --n-users N --max-steps S``."""
    work = out_dir / f"quickstart_{n_users}_{max_steps}"
    cmd = [sys.executable, "-m", "pragmatiq.cli", "quickstart", "--out", str(work),
           "--n-users", str(n_users), "--max-steps", str(max_steps)]
    log = out_dir / "leg_logs" / f"quickstart_{n_users}_{max_steps}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("wb") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=fh, timeout=timeout_sec, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    print(f"[quickstart] n_users={n_users} max_steps={max_steps} rc={rc} {elapsed:.0f}s", flush=True)
    return {"n_users": n_users, "max_steps": max_steps, "returncode": rc,
            "wall_time_s": round(elapsed, 1), "passed": rc == 0}


@_guard
def export_on_device(run_dir: Path, shard_dir: Path, out_dir: Path) -> dict[str, Any]:
    """ONNX export with ``device="auto"`` (the graph is built on CPU regardless)."""
    from pragmatiq import api

    t0 = time.time()
    res = api.export(str(run_dir), str(shard_dir), out=str(out_dir / "pragmatiq_embedder.onnx"),
                     device="auto")
    return {"passed": True, "out": res.get("out"), "opset": res.get("opset"),
            "max_abs_diff": res.get("max_abs_diff"), "seconds": round(time.time() - t0, 1)}


@_guard
def serving_request_caps(run_dir: Path, records: list[dict]) -> dict[str, Any]:
    """The runtime refuses oversized requests and chunked == whole on the same records."""
    import numpy as np

    from pragmatiq.inference.serve import runtime as serve_runtime

    env_keys = ("PRAGMATIQ_SERVE_MAX_RECORDS", "PRAGMATIQ_SERVE_TOKEN_BUDGET")
    saved = {k: os.environ.get(k) for k in env_keys}
    try:
        rt = serve_runtime.load(str(run_dir))
        os.environ["PRAGMATIQ_SERVE_MAX_RECORDS"] = "1"
        rejected = False
        try:
            rt.embed(records)
        except ValueError as exc:
            rejected = "at most 1" in str(exc)
        os.environ["PRAGMATIQ_SERVE_MAX_RECORDS"] = "100000"
        os.environ["PRAGMATIQ_SERVE_TOKEN_BUDGET"] = "1"  # one record per forward
        chunked = rt.embed(records)
        os.environ.pop("PRAGMATIQ_SERVE_TOKEN_BUDGET")
        whole = rt.embed(records)
        max_abs = float(np.abs(chunked - whole).max())
        rt.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    passed = rejected and max_abs < 1e-2
    print(f"[serve-caps] rejected_oversized={rejected} chunked_vs_whole_max_abs={max_abs:.2e}",
          flush=True)
    return {"passed": passed, "rejected_oversized": rejected, "device": rt.device,
            "chunked_vs_whole_max_abs": max_abs}


@_guard
def attention_backend_report() -> dict[str, Any]:
    """Which kernel varlen attention takes on this host, with and without flash disabled."""
    import torch

    from pragmatiq.models import layers

    has_cuda = torch.cuda.is_available()
    dev = "cuda" if has_cuda else "cpu"
    out: dict[str, Any] = {
        "cuda": has_cuda, "flash_importable": bool(layers._HAS_FLASH),
        "bf16_backend": layers.attention_backend(dev, torch.bfloat16),
        "fp32_backend": layers.attention_backend(dev, torch.float32),
    }
    saved = os.environ.get("PRAGMATIQ_DISABLE_FLASH")
    os.environ["PRAGMATIQ_DISABLE_FLASH"] = "1"
    try:
        out["bf16_backend_flash_disabled"] = layers.attention_backend(dev, torch.bfloat16)
    finally:
        if saved is None:
            os.environ.pop("PRAGMATIQ_DISABLE_FLASH", None)
        else:
            os.environ["PRAGMATIQ_DISABLE_FLASH"] = saved
    out["passed"] = out["bf16_backend_flash_disabled"] == "sdpa" and (
        out["bf16_backend"] == "flash" if (has_cuda and out["flash_importable"]) else True)
    print(f"[attention] {out}", flush=True)
    return out


@_guard
def precision_agreement(run_dir: Path, shard_dir: Path, label_path: Path,
                        *, max_users: int | None = 20_000) -> dict[str, Any]:
    """bf16 vs fp32: embed throughput and the probe ROC-AUC delta on the same users.

    On a CPU host both legs are fp32 (bf16 is downgraded), so the delta is 0 and
    the leg only proves the plumbing.
    """
    from pragmatiq.data.dataset import ShardDataset
    from pragmatiq.inference.benchmark import benchmark_batch_embed
    from pragmatiq.models.pragmatiq import PragmaModel
    from pragmatiq.training.probe import (
        EmbeddingProbe,
        _load_label_table,
        cutoffs_from_labels,
        embed_users,
    )

    model = PragmaModel.from_pretrained(str(run_dir))
    device = str(next(model.parameters()).device)
    out: dict[str, Any] = {"device": device}
    for prec in ("fp32", "bf16"):
        stats = benchmark_batch_embed(model, shard_dir, device=device, precision=prec,
                                      max_users=max_users)
        out[f"{prec}_users_per_sec"] = stats["users_per_sec"]
        out[f"{prec}_tokens_per_sec"] = stats["tokens_per_sec"]
        out[f"{prec}_resolved"] = stats["precision"]
    ds = ShardDataset(shard_dir)
    uids, _, eval_us = _load_label_table(label_path)
    have = set(ds.index.order)
    uids = [u for u in uids if u in have][: (max_users or len(uids))]
    cutoffs = cutoffs_from_labels(uids, eval_us) if eval_us is not None else None
    aucs: dict[str, float] = {}
    embs: dict[str, dict[str, Any]] = {}
    for prec in ("fp32", "bf16"):
        emb = embed_users(model, ds, device=device, user_ids=uids, cutoffs=cutoffs, precision=prec)
        embs[prec] = emb
        aucs[prec] = EmbeddingProbe(seed=0).run(emb, label_path).auc
    ds.close()
    # Direct closeness of the two embedding sets (the probe AUC on an
    # undertrained model is noisy at the 0.01 level; the cosine is not).
    import numpy as np

    common = [u for u in embs["fp32"] if u in embs["bf16"]]
    a = np.stack([embs["fp32"][u] for u in common]).astype(np.float64)
    b = np.stack([embs["bf16"][u] for u in common]).astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    out["mean_cosine"] = float(cos.mean())
    out["min_cosine"] = float(cos.min())
    out["fp32_probe_auc"], out["bf16_probe_auc"] = aucs["fp32"], aucs["bf16"]
    out["abs_auc_delta"] = abs(aucs["fp32"] - aucs["bf16"])
    # The embeddings themselves are the check; the probe AUC on a 300-step model
    # moves by several hundredths between two numerically identical embedding
    # sets (GBDT on near-random features), so it is reported, not gated.
    out["passed"] = out["mean_cosine"] >= 0.99
    print(f"[precision] fp32 auc={aucs['fp32']:.4f} bf16 auc={aucs['bf16']:.4f} "
          f"delta={out['abs_auc_delta']:.4f} mean_cosine={out['mean_cosine']:.4f}", flush=True)
    return out
