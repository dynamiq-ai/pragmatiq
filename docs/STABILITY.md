# pragmatiq Public API Stability Contract

> **FROZEN as of pragmatiq 1.0.0** (2026-06-21). SemVer applies from
> this release forward. The contract below is enforced automatically by
> `tests/contract/` on every CI run and gate 9. No item in this document
> may change without a deliberate version bump and a matching update here.

This document is the **frozen surface** for pragmatiq's public API.
Changes to any item listed here require a deliberate version bump per the
SemVer policy below.  The `tests/contract/` test suite enforces this
automatically on every CI run and gate 9.

---

## SemVer Policy

- **MAJOR**: change a frozen signature / return-dict key / CLI command-or-param
  name / serving contract; or a checkpoint-format break.
- **MINOR**: additive — new api function, new optional param with default, new
  return key, new CLI command, new extra, new integration adapter.
- **PATCH**: internals / bugfix / perf.
- **Internal import paths are NOT part of the contract** (modules may move
  between minors).
- `# GUESS` default *values* are not contract (changing a default is MINOR, for
  new runs only); shipped checkpoints embed identically forever (checkpoint
  format + tokenizer-hash guard frozen).

## E. Serving Contract (W4 — frozen)

The serving wire format is defined once in `pragmatiq/inference/serve/contract.py`
and pinned by `tests/contract/test_serving_contract.py`.  All adapters (Triton,
REST, gRPC, cloud) MUST use the constants from that module.

| Symbol | Value | Notes |
|---|---|---|
| `INPUT_NAME` | `"records_json"` | Input tensor name (BYTES / JSON-encoded list of dicts) |
| `OUTPUT_NAME` | `"embeddings"` | Output tensor name (float32) |
| Output shape | `[n_users, dim]` | C-contiguous float32; `dim` comes from model config |

Renaming `INPUT_NAME` or `OUTPUT_NAME`, or changing the output dtype, is a
**MAJOR** contract break and requires a version bump.

---

## A. `pragmatiq.api` Public Functions

The following 15 functions must exist, be importable, and be callable from
`pragmatiq.api`.  Their **parameter names, order, kinds, and default values**
are frozen.  Adding a new optional parameter (with a default) is MINOR and does
not break this contract; renaming, removing, or reordering a required parameter
is MAJOR.

### Function Signatures

#### `synthesize`

```python
def synthesize(
    config: str | Path | dict[str, Any] | None = None,
    out: str | Path = "data/synth",
    n_users: int | None = None,
    seed: int | None = None,
    n_workers: int = 0,
    write_report: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
```

#### `tokenize`

```python
def tokenize(
    data_dir: str | Path,
    out: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    tokenizer_dir: str | Path | None = None,
    max_users: int | None = None,
    rows_per_shard: int = 4096,
    n_workers: int = 0,
) -> dict[str, Any]:
```

#### `pretrain`

```python
def pretrain(
    shard_dir: str | Path,
    run_name: str,
    model_size: str = "small",
    config: str | Path | dict[str, Any] | None = None,
    runs_root: str | Path = "runs",
    resume: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
```

#### `finetune`

```python
def finetune(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    device: str = "auto",
    **overrides: Any,
) -> dict[str, Any]:
```

#### `embed`

```python
def embed(
    shard_dir: str | Path,
    run: str | Path,
    out: str | Path | None = None,
    token_budget: int = 16_384,
    device: str = "auto",
) -> dict[str, Any]:
```

#### `probe`

```python
def probe(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    device: str = "auto",
    token_budget: int = 16_384,
    seed: int = 0,
    with_baseline: bool = True,
    probe_model: str = "gbdt",
) -> dict[str, Any]:
```

#### `uplift`

```python
def uplift(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    device: str = "auto",
    token_budget: int = 16_384,
    seed: int = 0,
    learner: str = "t",
) -> dict[str, Any]:
```

#### `export`

```python
def export(
    run: str | Path,
    shard_dir: str | Path,
    out: str | Path = "pragmatiq_embedder.onnx",
    device: str = "cpu",
) -> dict[str, Any]:
```

#### `benchmark`

```python
def benchmark(
    run: str | Path,
    shard_dir: str | Path,
    device: str = "auto",
    out: str | Path = "deploy/benchmarks/RESULTS.md",
    max_users: int | None = None,
) -> dict[str, Any]:
```

#### `gnn`

```python
def gnn(
    shard_dir: str | Path,
    run: str | Path,
    transfers_path: str | Path,
    aml_label_path: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str = "auto",
    epochs: int = 150,
) -> dict[str, Any]:
```

#### `validate`

```python
def validate(data_dir: str | Path) -> dict[str, Any]:
```

#### `quickstart`

```python
def quickstart(
    out: str | Path = "runs/quickstart",
    n_users: int = 50_000,
    seed: int = 0,
    model_size: str = "nano",
    max_steps: int = 400,
    n_workers: int = 0,
) -> dict[str, Any]:
```

#### `runs_list`

```python
def runs_list(runs_root: str | Path = "runs") -> list[dict[str, Any]]:
```

#### `runs_compare`

```python
def runs_compare(names: list[str], runs_root: str | Path = "runs") -> list[dict[str, Any]]:
```

#### `calibrate`

```python
def calibrate(
    stats: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
```

---

## B. Documented Return-Dict Keys

The keys listed below are frozen.  Adding a new key to a return dict is MINOR
(additive); removing or renaming a key is MAJOR.

| Function | Frozen return keys |
|---|---|
| `embed` | `n_users`, `dim` |
| `probe` | `probe_model`, `probe_auc`, `probe_pr_auc`, `probe_accuracy`, `n_test`, `prevalence` |
| `probe` (with_baseline=True) | + `baseline_auc`, `baseline_pr_auc` |
| `uplift` | `qini`, `qini_oracle`, `ate`, `n_train`, `n_test`, `treated_frac` |
| `pretrain` | `run`, `run_dir`, `steps`, `last_metrics` |
| `finetune` | `best_val_auc`, `epochs_run`, `n_adapted`, `val_auc_history` |
| `validate` | `ok`, `errors`, `warnings`, `summary` |
| `quickstart` | `run_dir`, `probe`, `message` |
| `runs_list` | `list[dict]` (contents vary by run) |
| `runs_compare` | `list[dict]` (missing runs flagged with `{name, missing: True}`) |

For `synthesize`, `tokenize`, `calibrate`, `export`, `benchmark`, and `gnn`,
the return dict is a manifest / stats dict.  The exact keys are recorded below
as observed from the implementation; they are frozen for contract purposes.

| Function | Observed return keys |
|---|---|
| `synthesize` | `n_users`, `months`, `seed`, `n_merchants`, plus label metadata |
| `tokenize` | `n_users`, `n_shards`, `vocab_size`, `tokenizer_hash`, plus shard metadata |
| `calibrate` | keys mirror the `WorldConfig` fields (calibrated priors) |
| `export` | ONNX-specific metadata (file path, model size, opset, etc.) |
| `benchmark` | throughput stats (users/sec, latency percentiles, device info, etc.) |
| `gnn` | AML ablation results per arm (AUC, F1, etc.) |

---

## C. CLI Command Tree

The following command paths and parameter names are frozen.

### Root callback

Option: `--verbose` / `--quiet` (Python param name: `verbose`, default `True`).

### Top-level commands

| Command path | Parameter names (Python identifiers) |
|---|---|
| `tokenize` | `data_dir`, `out`, `config`, `tokenizer_dir`, `max_users`, `n_workers` |
| `pretrain` | `shard_dir`, `run_name`, `model_size`, `config`, `runs_root`, `resume`, `wandb` |
| `probe` | `shard_dir`, `run`, `label`, `device`, `probe_model`, `seed` |
| `uplift` | `shard_dir`, `run`, `label`, `device`, `learner` |
| `finetune` | `shard_dir`, `run`, `label`, `config`, `device` |
| `embed` | `shard_dir`, `run`, `out`, `device` |
| `quickstart` | `out`, `n_users`, `model_size`, `max_steps`, `n_workers` |
| `validate` | `data_dir` |
| `export` | `run`, `shard_dir`, `out` |
| `benchmark` | `run`, `shard_dir`, `device`, `out` |
| `gnn` | `shard_dir`, `run`, `transfers`, `aml_label`, `seeds`, `epochs`, `device` |

### Sub-app: `synth`

| Command path | Parameter names |
|---|---|
| `synth generate` | `out`, `config`, `n_users`, `seed`, `n_workers`, `report` |
| `synth calibrate` | `stats`, `config`, `out` |

### Sub-app: `runs`

| Command path | Parameter names |
|---|---|
| `runs list` | `runs_root` |
| `runs compare` | `names`, `runs_root` |

---

## D. Model API

### `PragmaModel.from_pretrained`

```python
@classmethod
def from_pretrained(
    cls,
    run: str | Path,
    device: str = "cpu",
    checkpoint: str = "last.pt",
) -> PragmaModel:
```

Required parameters: `run`.
Optional parameters: `device` (default `"cpu"`), `checkpoint` (default `"last.pt"`).

### `PragmaModel.embed_records`

```python
@torch.no_grad()
def embed_records(self, records: list[dict[str, Any]]) -> np.ndarray:
```

Required parameters: `records`.
Return type: `np.ndarray` of shape `[N, dim]`.

---

## Checkpoint Format Contract

The checkpoint format version (`CKPT_FORMAT = 2`) is frozen.  A checkpoint
written by the current version must be loadable by `from_pretrained` in the
same major version.  A format bump increments MAJOR.

The tokenizer hash embedded in every checkpoint is verified on load;
`from_pretrained` raises `ValueError` on a hash mismatch (global rule 3).

---

## Attribution

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
