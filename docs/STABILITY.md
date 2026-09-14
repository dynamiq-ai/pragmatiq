# pragmatiq Public API Stability Contract

> **Current surface: pragmatiq 1.1.0.** This document records the public
> surface as it ships and the policy under which it changes. It is enforced by
> `tests/contract/` on every CI run and by gate 9.

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

---

## Pre-2.0 policy

pragmatiq is used by a small number of teams and still moving quickly, so the
1.x line follows a deliberately loose version of SemVer:

- **A minor release may break.** Signatures, defaults, CLI parameters, module
  paths, the shard format and the generator output can change in `1.(x+1).0`
  when carrying compatibility would cost more than it buys.
- **Every break is listed.** `CHANGELOG.md` records each one under a
  *Breaking* heading with a one-line migration; a break that is not listed
  there is a bug.
- **The contract tests pin the current surface**, not a historical one. They
  are updated deliberately, in the same PR as the change, so that no signature
  or default drifts by accident — a red contract test is the checklist item
  "did you mean to break this, and did you write it down?".
- **PATCH releases never break** anything listed here.
- **What is stable across the whole 1.x line:** the serving wire format
  (section E), the checkpoint format (`CKPT_FORMAT = 2`) and its tokenizer-hash
  guard, and the fact that a checkpoint trained on 1.0.x loads and embeds
  identically on 1.1.x against its own copied tokenizer.
- Internal import paths are never contract; `# GUESS` default *values* may
  change in a minor release (recorded in the changelog) and only affect new runs,
  because every run embeds its resolved config.

From 2.0.0 on, the surface below freezes under strict SemVer.

## Breaks in 1.1.0

Recorded here as well as in the changelog so a reader of this file sees them:

| Item | 1.0.x | 1.1.0 | Migration |
| --- | --- | --- | --- |
| `PragmaModel.from_pretrained(device=)` | `"cpu"` | `"auto"` (CUDA when visible) | pass `device="cpu"` or set `PRAGMATIQ_DEVICE=cpu` |
| `api.export(device=)` | `"cpu"`, other values rejected | `"auto"`, any value accepted (graph built on CPU) | none |
| `api.benchmark(out=)` / `pragmatiq benchmark --out` | `deploy/benchmarks/RESULTS.md` | `benchmark_results.md` | pass `--out` |
| `pragmatiq pretrain --config` | none (dataclass defaults) | `auto` | pass `--config configs/pretrain.yaml` |
| `FineTuneConfig.token_budget` | `16384` | `None` → 16 384 on CPU, device-sized on CUDA | set it explicitly |
| Shard format | event cap applied at encode time | full history in shards; cap in the manifest, applied at collate time | re-tokenize |
| Tokenizer content hash | — | new `max_counter_distinct` field | re-tokenize or reuse the run's tokenizer via `--tokenizer-dir` |
| Generator output for a seed | v1 | v2 (see the changelog) | regenerate |
| Serving device | CPU unless `PRAGMATIQ_SERVE_GPU=1` | CUDA when visible; `PRAGMATIQ_SERVE_CPU=1` pins CPU | drop `PRAGMATIQ_SERVE_GPU`, use the CPU compose/config for CPU hosts |
| `torch` floor | `>=2.0` | `>=2.6` | upgrade |
| `typer` floor | `>=0.9` | `>=0.12` | upgrade |
| Removed modules | — | see the changelog *Removed* list | — |

## E. Serving Contract (stable across 1.x)

The serving wire format is defined once in `pragmatiq/inference/serve/contract.py`
and pinned by `tests/contract/test_serving_contract.py`.  All adapters (Triton,
REST, gRPC, cloud) MUST use the constants from that module.

| Symbol | Value | Notes |
|---|---|---|
| `INPUT_NAME` | `"records_json"` | Input tensor name (BYTES / JSON-encoded list of dicts) |
| `OUTPUT_NAME` | `"embeddings"` | Output tensor name (float32) |
| Output shape | `[n_users, dim]` | C-contiguous float32; `dim` comes from model config |

`decode_request` validates each record (a dict with a non-empty string
`user_id` and an `events` list) and raises `ValueError` otherwise. The runtime
(`pragmatiq.inference.serve`) exposes `Runtime`, `load`, `resolve_serve_device`
and `request_limits()` → `(PRAGMATIQ_SERVE_MAX_RECORDS, PRAGMATIQ_SERVE_TOKEN_BUDGET)`,
defaults `(1024, 16384)`; a request above the record cap raises `ValueError`.
Device policy: `PRAGMATIQ_SERVE_CPU=1` → CPU; else a Triton `KIND_GPU` instance
→ its assigned GPU; else CUDA when visible; else CPU.

Renaming `INPUT_NAME` or `OUTPUT_NAME`, or changing the output dtype, is a
**MAJOR** contract break and requires a version bump.

---

## A. `pragmatiq.api` Public Functions

The following functions exist, are importable and callable from `pragmatiq.api`,
and are listed in `api.__all__`. Their **parameter names, order, kinds, and
default values** are pinned by the contract tests. Adding a new optional
parameter (with a default) is additive; renaming, removing, or reordering a
required parameter, or changing a default, is a break that the changelog must
record (pre-2.0 policy above).

Two functions are new in 1.1.0 and pinned from here on: `info()` (the facts
`pragmatiq info` prints) and `pretrain_plan(...)` (the resolved training
config that `pragmatiq pretrain --show-config` prints, same parameters as
`pretrain` minus `resume`).

### Function Signatures

#### `synthesize`

```python
def synthesize(
    config: str | Path | dict[str, Any] | None = None,
    out: str | Path = "data/synth",
    n_users: int | None = None,
    seed: int | None = None,
    n_workers: int = 0,
    write_report: bool | None = None,
    **overrides: Any,
) -> dict[str, Any]:
```

`write_report=None` (the default) is auto: the realism report is written when
`matplotlib` (the `data` extra) is importable and skipped with a logged warning
otherwise; `True` forces it, `False` skips it.

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
    staleness_window: str | int | None = None,
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
    device: str = "auto",
) -> dict[str, Any]:
```

#### `benchmark`

```python
def benchmark(
    run: str | Path,
    shard_dir: str | Path,
    device: str = "auto",
    out: str | Path = "benchmark_results.md",
    max_users: int | None = None,
    precision: str = "auto",
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

#### `info` (1.1.0)

```python
def info() -> dict[str, Any]:
```

#### `pretrain_plan` (1.1.0)

```python
def pretrain_plan(
    shard_dir: str | Path,
    model_size: str = "small",
    config: str | Path | dict[str, Any] | None = "auto",
    **overrides: Any,
) -> dict[str, Any]:
```

---

## B. Documented Return-Dict Keys

The keys listed below are pinned.  Adding a new key to a return dict is
additive; removing or renaming a key is a break the changelog must record.

| Function | Pinned return keys |
|---|---|
| `embed` | `n_users`, `dim` |
| `probe` | `probe_model`, `probe_auc`, `probe_pr_auc`, `probe_accuracy`, `n_test`, `prevalence`, `staleness_window`, `staleness_us` |
| `probe` (with_baseline=True) | + `baseline_auc`, `baseline_pr_auc` |
| `uplift` | `qini`, `qini_oracle`, `ate`, `n_train`, `n_test`, `treated_frac` |
| `pretrain` | `run`, `run_dir`, `steps`, `last_metrics` |
| `finetune` | `best_val_auc`, `epochs_run`, `n_adapted`, `val_auc_history`, `epoch_stats`, `token_budget` |
| `validate` | `ok`, `errors`, `warnings`, `summary` |
| `quickstart` | `run_dir`, `probe`, `message` |
| `runs_list` | `list[dict]` (contents vary by run) |
| `runs_compare` | `list[dict]` (missing runs flagged with `{name, missing: True}`) |

For `synthesize`, `tokenize`, `calibrate`, `export`, `benchmark`, and `gnn`,
the return dict is a manifest / stats dict.  The exact keys are recorded below
as observed from the implementation; they are pinned for contract purposes.

| Function | Observed return keys |
|---|---|
| `synthesize` | `n_users`, `months`, `seed`, `n_merchants`, plus label metadata |
| `tokenize` | `n_users`, `n_shards`, `vocab_size`, `tokenizer_hash`, plus shard metadata |
| `calibrate` | keys mirror the `WorldConfig` fields (calibrated priors) |
| `export` | ONNX-specific metadata (file path, model size, opset, etc.) |
| `benchmark` | throughput stats (`device`, `precision`, `users_per_sec`, `tokens_per_sec`, `usd_per_million_users`, …) |
| `info` | `version`, `python`, `torch`, `device`, `cuda`, `flash_attn`, `extras`, … (what `pragmatiq info` prints) |
| `pretrain_plan` | the resolved `TrainConfig` / `ModelConfig` fields |
| `gnn` | AML ablation results per arm (AUC, F1, etc.) |

---

## C. CLI Command Tree

The following command paths and parameter names are pinned by `tests/contract/test_cli_commands.py`.

### Root callback

Options: `--verbose` / `--quiet` (Python param name: `verbose`, default `True`)
and `--version` (prints the package version and exits).

### Top-level commands

| Command path | Parameter names (Python identifiers) |
|---|---|
| `tokenize` | `data_dir`, `out`, `config`, `tokenizer_dir`, `max_users`, `n_workers` |
| `info` | — |
| `pretrain` | `shard_dir`, `run_name`, `model_size`, `config` (default `auto`), `runs_root`, `resume`, `wandb`, `show_config` |
| `probe` | `shard_dir`, `run`, `label`, `device`, `probe_model`, `seed`, `staleness_window` |
| `uplift` | `shard_dir`, `run`, `label`, `device`, `learner` |
| `finetune` | `shard_dir`, `run`, `label`, `config`, `device` |
| `embed` | `shard_dir`, `run`, `out`, `device` |
| `quickstart` | `out`, `n_users`, `model_size`, `max_steps`, `n_workers` |
| `validate` | `data_dir` |
| `export` | `run`, `shard_dir`, `out`, `device` |
| `benchmark` | `run`, `shard_dir`, `device`, `out`, `precision` |
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
    device: str = "auto",
    checkpoint: str = "last.pt",
) -> PragmaModel:
```

Required parameters: `run`.
Optional parameters: `device` (default `"auto"` — CUDA when visible or
`PRAGMATIQ_DEVICE`, else CPU), `checkpoint` (default `"last.pt"`).

### `PragmaModel.embed_records`

```python
def embed_records(
    self,
    records: list[dict[str, Any]],
    precision: str = "auto",
    token_budget: int | None = None,
) -> np.ndarray:
```

Required parameters: `records`. `precision` is `auto` (bf16 on CUDA, fp32 on
CPU), `bf16` or `fp32`; `token_budget` splits the request into forward passes
of at most that many tokens (`None` = one forward). Runs under
`torch.inference_mode`. Return type: `np.ndarray` of shape `[N, dim]`.

---

## Checkpoint Format Contract

The checkpoint format version is `CKPT_FORMAT = 2` (`pragmatiq/models/pragmatiq.py`),
unchanged in 1.1.0: checkpoints written by 1.0.x load on 1.1.x. 1.1.0 adds the
Python `random` state to the RNG block and loads with `weights_only=True`
(falling back, with a warning, for files that need a full unpickle); neither
changes the format version. A format bump increments MAJOR.

The tokenizer hash embedded in every checkpoint is verified on load;
`from_pretrained` raises `ValueError` on a hash mismatch (global rule 3).

---

## Attribution

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
