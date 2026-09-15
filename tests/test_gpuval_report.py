"""GPU validation evidence: acceptance-table logic and the README block renderer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def report():
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("gpuval.report", _SCRIPTS / "gpuval" / "report.py",
                                                  submodule_search_locations=[])
    import gpuval.report as mod  # noqa: PLC0415

    assert spec is not None
    return mod


def _evidence() -> dict:
    return {
        "meta": {"tag": "t1", "started": "2026-09-15T10:00:00", "gpu_name": "NVIDIA A100 80GB PCIe",
                 "gpu_count": 1, "torch": "2.8.0+cu128", "flash_attn": "2.8.3", "cuda": "12.8",
                 "n_users": 50000, "steps": 300, "model_size": "small"},
        "legs": {
            "training": [
                {"devices": 1, "returncode": 0, "tokens_per_sec": 310000.0, "gpu_mem_gb": 2.1,
                 "efficiency": 1.0, "model_size": "small"},
                {"devices": 2, "returncode": 0, "tokens_per_sec": 500000.0, "gpu_mem_gb": 2.2,
                 "efficiency": 0.81, "model_size": "small"},
            ],
            "finetune": [
                {"devices": 1, "returncode": 0, "best_val_auc": 0.71, "epochs_run": 3, "wall_time_s": 900.0,
                 "epoch_stats": [{"phase": "train", "tokens_per_sec": 100000.0, "seconds": 100.0,
                                  "data_wait_seconds": 30.0},
                                 {"phase": "val", "tokens_per_sec": 300000.0, "seconds": 5.0},
                                 {"phase": "train", "tokens_per_sec": 95000.0, "seconds": 90.0,
                                  "data_wait_seconds": 1.0}]},
            ],
            "serving": [
                {"device": "cpu", "concurrency": 1, "req_s": 9.0, "p50_ms": 100.0, "p99_ms": 250.0},
                {"device": "cuda", "concurrency": 1, "req_s": 30.0, "p50_ms": 10.0, "p99_ms": 90.0},
            ],
            "flash_check": {"skipped": False, "passed": True, "max_abs_diff": 3e-3, "tol": 1e-2},
            "precision": {"passed": True, "abs_auc_delta": 0.004, "mean_cosine": 0.9998, "bf16_probe_auc": 0.701,
                          "fp32_probe_auc": 0.705, "bf16_users_per_sec": 5000.0, "fp32_users_per_sec": 3000.0},
            "serve_caps": {"passed": True, "rejected_oversized": True, "chunked_vs_whole_max_abs": 1e-4},
            "export": {"passed": True, "max_abs_diff": 1e-5},
            "attention": {"passed": True, "bf16_backend": "flash", "bf16_backend_flash_disabled": "sdpa"},
            "quickstart_fast": {"passed": True, "wall_time_s": 95.0},
        },
        "utilisation": [{"label": "finetune_d1", "gpu": {"mean_util_pct": 72.0}},
                        {"label": "pretrain_d1", "gpu": {"mean_util_pct": 40.0}}],
    }


def test_acceptance_table_all_green(report) -> None:
    rows = report.acceptance_table(_evidence())
    by = {r["check"]: r for r in rows}
    assert all(r["passed"] for r in rows), [r for r in rows if not r["passed"]]
    assert by["DDP scaling efficiency (max devices)"]["value"] == 0.81
    assert by["finetune d=1 epoch-2 tok/s vs epoch-1"]["value"] == 0.95
    assert by["finetune d=1 loader wait share (last epoch)"]["value"] == pytest.approx(0.011, abs=0.001)
    assert not any("GPU mean util" in k for k in by)  # utilisation is reported, not gated
    assert by["serving GPU req/s > CPU req/s (concurrency 1)"]["value"] == pytest.approx(3.33, abs=0.01)


def test_acceptance_table_flags_failures_and_skips(report) -> None:
    ev = _evidence()
    ev["legs"]["finetune"][0]["epoch_stats"][2]["tokens_per_sec"] = 60000.0  # -40%
    ev["legs"]["precision"]["mean_cosine"] = 0.9
    ev["legs"]["precision"]["passed"] = False
    for st in ev["legs"]["finetune"][0]["epoch_stats"]:
        if st.get("phase") == "train":
            st["data_wait_seconds"] = st["seconds"] * 0.9  # host-bound: the GPU waits on the loader
    ev["legs"]["flash_check"] = {"skipped": True, "skip_reason": "no CUDA"}
    del ev["legs"]["export"]
    rows = {r["check"]: r for r in report.acceptance_table(ev)}
    assert rows["finetune d=1 epoch-2 tok/s vs epoch-1"]["passed"] is False
    assert rows["bf16 vs fp32 embeddings mean cosine (|ΔROC-AUC| reported)"]["passed"] is False
    assert rows["finetune d=1 loader wait share (last epoch)"]["passed"] is False
    assert "flash-attn ≡ SDPA max abs diff" not in rows  # skipped → not listed as a failure
    assert "ONNX export on the pod" not in rows


def test_evidence_json_and_readme_block(report, tmp_path: Path) -> None:
    path = report.write_evidence_json(tmp_path, "t1", _evidence())
    assert path.name == "gpu-validation-t1.json"
    data = json.loads(path.read_text())
    assert data["all_passed"] is True and len(data["acceptance"]) >= 8
    block = report.render_readme_block(path)
    assert "NVIDIA A100 80GB PCIe × 1" in block and "flash-attn 2.8.3" in block
    assert "| small | 1 | 310,000 | 2.1 | 100% |" in block
    assert "| 1 | 3 | 900 s | 0.710 | 100,000 → 95,000 |" in block
    assert "| cuda | 1 | 30.0 | 10 | 90 |" in block
    assert "|Δ| = 0.0040" in block and "3.00e-03" in block and "95 s" in block
    assert block.endswith("checks passed. Evidence: `docs/benchmarks/gpu-validation-t1.json`.\n")

    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n## Running on GPU\n\n<!-- GPU_VALIDATION_RESULTS -->\n\n_not measured_\n\n## Next\n\ntext\n")
    assert report.write_readme_block(path, readme)
    text = readme.read_text()
    assert "_not measured_" not in text and "## Next\n\ntext\n" in text
    assert text.count("<!-- GPU_VALIDATION_RESULTS -->") == 1
    assert report.write_readme_block(path, readme)  # idempotent
    assert readme.read_text() == text
    other = tmp_path / "no_marker.md"
    other.write_text("plain\n")
    assert not report.write_readme_block(path, other) and other.read_text() == "plain\n"


def test_nan_becomes_null(report, tmp_path: Path) -> None:
    ev = _evidence()
    ev["legs"]["training"][0]["gpu_mem_gb"] = float("nan")
    path = report.write_evidence_json(tmp_path, "nan", ev)
    assert json.loads(path.read_text())["legs"]["training"][0]["gpu_mem_gb"] is None
