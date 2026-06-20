# Task 3 (W2): Extras Retaxonomy + Slim-Serve Enforcement — Report

## Status: DONE

---

## pyproject.toml Before / After

### Before (core `dependencies`)
```
torch>=2.0, numpy>=1.24, pyarrow>=14.0, pandas>=2.0, lmdb>=1.4, tokenizers>=0.15,
typer>=0.9, omegaconf>=2.3, pyyaml>=6.0, scikit-learn>=1.3,
torch-geometric>=2.4,   ← REMOVED
lightning>=2.2,          ← REMOVED
matplotlib>=3.9,         ← REMOVED
tqdm>=4.66
```

### After (core `dependencies`)
```
torch>=2.0, numpy>=1.24, pyarrow>=14.0, pandas>=2.0, lmdb>=1.4, tokenizers>=0.15,
typer>=0.9, omegaconf>=2.3, pyyaml>=6.0, scikit-learn>=1.3, tqdm>=4.66
```
(torch-geometric, lightning, matplotlib removed from core)

### Before `[project.optional-dependencies]`
```toml
serve   = ["onnx>=1.16", "onnxscript>=0.1", "onnxruntime>=1.18", "tritonclient[http]>=2.40"]
demo    = ["streamlit>=1.30", "plotly>=5.18"]
extras  = ["transformers>=4.40", "wandb>=0.16", "tensorboard>=2.16"]
gbdt    = ["lightgbm>=4.0"]
full    = ["pragmatiq[serve,demo,extras,gbdt]"]
dev     = ["pytest>=7.4", "pytest-cov", "ruff>=0.4", "mypy>=1.8", "types-PyYAML"]
```

### After `[project.optional-dependencies]`
```toml
data     = ["matplotlib>=3.9"]
train    = ["lightning>=2.2", "matplotlib>=3.9"]
serve    = ["onnx>=1.16", "onnxscript>=0.1", "onnxruntime>=1.18", "tritonclient[http]>=2.40"]
aml      = ["torch-geometric>=2.4"]
text     = ["transformers>=4.40"]
tracking = ["wandb>=0.16", "tensorboard>=2.16"]
gbdt     = ["lightgbm>=4.0"]
demo     = ["streamlit>=1.30", "plotly>=5.18"]
full     = ["pragmatiq[data,train,serve,aml,text,tracking,gbdt,demo]"]
dev      = ["pytest>=7.4", "pytest-cov", "ruff>=0.4", "mypy>=1.8", "types-PyYAML"]
```

Key changes:
- `[extras]` group split into `[text]` (transformers) + `[tracking]` (wandb + tensorboard)
- `[data]` added for matplotlib
- `[train]` added for lightning + matplotlib
- `[aml]` added for torch-geometric
- `[serve]` is now truly slim: onnx/onnxscript/onnxruntime/tritonclient only
- `[full]` references all 8 extras groups (was 4)

---

## Part B — 7 Heavy-Import Sites Wired

| File | Line (approx) | Extra | Package | Change |
|---|---|---|---|---|
| `pragmatiq/training/pretrainer.py` | 203 | `train` | `lightning` | Wrapped `from lightning.fabric import Fabric` in try/except → MissingExtraError |
| `pragmatiq/models/gnn.py` | 163 | `aml` | `torch_geometric` | Wrapped `from torch_geometric.nn import SAGEConv` → MissingExtraError |
| `pragmatiq/models/text_encoder.py` | 75 | `text` | `transformers` | Replaced generic ImportError message → MissingExtraError |
| `pragmatiq/training/probe.py` | 156 | `gbdt` | `lightgbm` | Replaced generic ImportError message → MissingExtraError |
| `pragmatiq/data/synthetic/report.py` | 22 | `data` | `matplotlib` | Wrapped `import matplotlib.pyplot` in `_fig_to_b64` → MissingExtraError |
| `pragmatiq/data/synthetic/report.py` | 103 | `data` | `matplotlib` | Wrapped `import matplotlib` + `import matplotlib.pyplot` in `write_realism_report` → MissingExtraError |
| `pragmatiq/inference/export.py` | 321 | `serve` | `onnxscript` | Replaced generic ImportError message → MissingExtraError |
| `pragmatiq/inference/export.py` | 355 | `serve` | `onnxruntime` | Wrapped `import onnxruntime` → MissingExtraError |
| `pragmatiq/experiments/tracking.py` | 42 | `tracking` | `wandb` | Split: ImportError → MissingExtraError (re-raised); other init failures stay silent |

All imports remain lazy (inside functions). MissingExtraError subclasses ImportError, so existing `except ImportError` handlers remain unaffected.

For `tracking.py`: when wandb is installed but init fails (e.g., network/auth), the original silent fallback is preserved. Only the missing-package case now gives a clear error message.

---

## Part C — errors.py Minors

- `MissingExtraError` now subclasses both `PragmatiqError` and `ImportError`:
  `class MissingExtraError(PragmatiqError, ImportError)`
- MRO: `MissingExtraError → PragmatiqError → ImportError → Exception` (valid diamond through Exception)
- Fixed UP037 ruff warning: removed string quotes from return type annotation (covered by `from __future__ import annotations`)

Sanity check output:
```
True True
pragmatiq[train] is required for this feature: pip install 'pragmatiq[train]' (missing: lightning)
```

---

## Part D — Slim-Serve Boundary Test Design

### File: `tests/boundaries/test_serve_import_safe.py`

**Strategy:** subprocess + import-blocker (not in-process, so main pytest interpreter is not disturbed).

**How the fixture is built (cheapest viable approach):**

1. Generate 10 synthetic users (14 months minimum, per WorldConfig constraint) using `WorldConfig(n_users=10, months=14, ...)` — fast.
2. Fit a `PragmaTokenizer` with tiny vocab (512 tokens, 8 buckets).
3. Construct a nano `PragmaModel(ModelConfig.preset("small", tok.vocab_size))`.
4. Attach tokenizer directly: `model._tokenizer = tok` — this is exactly what `from_pretrained()` does after hash check. No checkpoint needed.

**Import blocker:** A `sys.meta_path` finder inserted at index 0 that raises `ImportError` for any module in `{lightning, torch_geometric, transformers, lightgbm}` (exact name or prefix match).

**The call:** `model.embed_records(records)` with 2 plain-dict records. Asserts `shape == (2, cfg.dim)` and `np.isfinite(emb).all()`.

**Absence check:** After the call, asserts `sys.modules` contains none of the blocked module names/prefixes.

**Test assertion:** Subprocess must return code 0 and stdout must contain `"SLIM_OK"`. On failure, stderr is included in the assertion message.

This test runs GREEN in the `.venv` (where heavy modules ARE installed) because the blocker makes them unimportable in the child process regardless of what the parent process has loaded.

---

## Part E — Gate + CI Changes

### New gate: `scripts/gates/gate_serve_slim.sh`
- Sources `_env.sh` for interpreter resolution.
- Runs `"$PY" -m pytest tests/boundaries/test_serve_import_safe.py -q`.
- Documents the stronger CI check (fresh venv + `! python -c "import lightning"`).

### CI changes (`.github/workflows/ci.yml`)

| Job | Before | After |
|---|---|---|
| `quality` | `.[dev]` | `.[dev,train]` (adds lightning for PreTrainer smoke tests) |
| `tests` | `.[dev,serve]` | `.[dev,full]` (all deps present for full 385-test suite) |
| `gates` | `.[dev,serve]` | `.[dev,full]` (gates exercise all paths) |
| `serve-slim` | NEW | `.[serve]` only + pytest; runs boundary test + `! import lightning` |
| `python-compat` | `.[dev]` | `.[dev]` (smoke tests don't need lightning/matplotlib) |
| `package-smoke` | no pytest | no pytest (unchanged) |

---

## Part F — Doc/Help References Updated

| File | Before | After |
|---|---|---|
| `pragmatiq/cli.py:78` | `[extras] extra` | `[tracking] extra` |
| `pragmatiq/training/pretrainer.py:161` | `.[extras]` | `.[tracking]` |
| `pragmatiq/models/text_encoder.py:15` | `[extras] extra` | `[text] extra` |
| `pragmatiq/models/text_encoder.py:65` | `[extras] extra` | `[text] extra` |
| `README.md:103` | `.[extras]` blurb | Updated with full new taxonomy |
| `README.md:733` | `pip install -e ".[extras]"` | `pip install -e ".[text]"` |
| `README.md:912-915` | `.[extras]` (×2) | `.[tracking]` (×2) |
| `scripts/runpod_launch.py:61` | `".[extras]"` | `".[text]"` |

---

## Verify Outputs

### 1. Boundary test
```
.venv/bin/python -m pytest tests/boundaries/test_serve_import_safe.py -q
1 passed in 1.86s
```

### 2. Contract tests
```
.venv/bin/python -m pytest tests/contract -q
83 passed in 1.50s
```

### 3. Focused module tests (gnn, text_encoder, nemotron)
```
22 passed in 31.16s
```

### 4. Focused test (inference + models + training)
```
83 passed, 1 skipped in 606.62s (0:10:06)
(skip: bf16-vs-fp32 Newton-Schulz only diverges on CUDA)
```

### 5. Full suite
```
386 passed, 1 skipped, 18 warnings in 1380.47s (0:23:00)
(1 extra test from new boundary test; skip: bf16 Newton-Schulz CUDA-only)
No failures.
```

### 6. MissingExtraError sanity
```python
>>> from pragmatiq.core.errors import MissingExtraError, PragmatiqError
>>> e = MissingExtraError.for_extra('train', 'lightning')
>>> isinstance(e, ImportError), isinstance(e, PragmatiqError)
True True
>>> str(e)
"pragmatiq[train] is required for this feature: pip install 'pragmatiq[train]' (missing: lightning)"
```

### 7. Gate script
```
bash scripts/gates/gate_serve_slim.sh
=== slim-serve boundary (import-blocker subprocess) ===
. 1 passed in 1.98s
SERVE-SLIM CHECKS GREEN
```

---

## Self-Review Checklist

- [x] Boundary test actually calls `embed_records` (not a stub or mock)
- [x] Boundary test asserts blocked modules absent from `sys.modules`
- [x] All 7 import sites remain lazy (inside functions, not at module level)
- [x] CI full-suite job uses `.[dev,full]` (lightning/torch-geometric/etc. present)
- [x] New `serve-slim` CI job installs `.[serve]` only + asserts `! import lightning`
- [x] `MissingExtraError` messages are clear and point to correct `[extra]`
- [x] `MissingExtraError` subclasses both `PragmatiqError` and `ImportError`
- [x] All `[extras]` references updated to new taxonomy in user-facing code + docs
- [x] Contract suite: 83/83 passed
- [x] `quality` CI job updated to `.[dev,train]` (PreTrainer smoke tests need lightning)
- [x] ruff: only pre-existing violations remain (I001 in generate.py, labels.py, tokenizer.py, pragmatiq.py, validate.py — pre-existing before this task); fixed the UP037 in errors.py
