# Changelog

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

All notable changes to pragmatiq are documented in this file. This project
follows [Semantic Versioning](https://semver.org) with the pre-2.0 caveat in
[`docs/STABILITY.md`](docs/STABILITY.md): a minor release may break, and every
break is listed under *Breaking* with a migration line.

## [1.1.0] — GPU-first, generator v2, leaner tree

The GPU is the default target, the synthetic book is more realistic, the data
path and trainer shed their host-side stalls, and ~3k lines of dead or
unmaintained code are gone. This is a **minor release that breaks** where
carrying compatibility would have cost more than it was worth; every break is
listed under *Breaking* with a migration line, and `docs/STABILITY.md` now
states the pre-2.0 policy that allows this.

### Breaking

- **Re-tokenize your shards.** The per-user event cap (`max_events_per_user`,
  6500) is no longer applied at encode time; shards keep the full history and
  the cap is applied when a batch is collated, *after* the eval-point cut, from
  a `max_events_per_user` entry in the shard manifest. `TokenizerConfig` also
  gained `max_counter_distinct`, which changes the tokenizer content hash.
  *Migration:* run `pragmatiq tokenize` again; models trained on 1.0.x shards
  keep loading against their own copied tokenizer, but new shards need a new
  fit or `--tokenizer-dir` pointing at the run's tokenizer.
- **Generator v2 replaces v1.** The same seed produces a different (more
  realistic) book than 1.0.x: amounts are in the transaction currency, rent
  and subscriptions no longer earn interchange, bank holidays are computed,
  paydays vary per user, trading keeps market hours, overdraft fees feed the
  balance, merchants follow the user's country. *Migration:* regenerate
  synthetic data and any result table derived from it; label prevalences move
  slightly (credit ≈ 2.5 %, churn ≈ 10 %, `ltv_positive` ≈ 73 % at the
  default config).
- **`PragmaModel.from_pretrained(run, device="auto")`** — the default device
  was `"cpu"`. *Migration:* pass `device="cpu"` or set `PRAGMATIQ_DEVICE=cpu`
  to keep CPU inference on a GPU host. `embed_users`, `BatchEmbedder`,
  `benchmark_batch_embed`, `LoRAFineTuner` default to `"auto"` too.
- **Serving is GPU-first; `PRAGMATIQ_SERVE_GPU` is gone.** The Triton config
  ships `KIND_GPU`, `deploy/docker-compose.yaml` reserves the host GPUs, and
  the backend picks CUDA whenever it is visible. *Migration:* for a CPU-only
  host use `deploy/docker-compose.cpu.yaml` or `scripts/deploy_serving.sh`
  (which overlays `deploy/triton/config.cpu.pbtxt` when `nvidia-smi` is
  absent), or set `PRAGMATIQ_SERVE_CPU=1`. Requests are now validated (string
  `user_id`, `events` list per record) and capped at
  `PRAGMATIQ_SERVE_MAX_RECORDS` (1024) users.
- **Dependency floors:** `torch>=2.6` (dynamo ONNX exporter, `weights_only`
  checkpoint loading), `typer>=0.12`. The Triton image is based on NGC
  `tritonserver:25.06-py3` (torch 2.8). *Migration:* upgrade torch.
- **`pragmatiq benchmark` writes `benchmark_results.md`** in the current
  directory instead of `deploy/benchmarks/RESULTS.md`; `api.benchmark(out=...)`
  default changed accordingly. `api.export(device=...)` defaults to `"auto"`
  and no longer rejects non-CPU values (the graph is always built on CPU).
- **`pragmatiq pretrain` defaults to `--config auto`** (batch and schedule
  sized from the data and device). *Migration:* pass `--config
  configs/pretrain.yaml` for the fixed defaults.
- **`FineTuneConfig.token_budget` defaults to `None`** (16 384 on CPU, sized
  from device memory on CUDA). *Migration:* set it explicitly to pin a budget.
- **Module moves (internal paths are not contract, listed for grep):**
  `pragmatiq/progress.py` → `pragmatiq/core/progress.py`;
  `pragmatiq/experiments/` → `pragmatiq/runs/` (`Run`, `MetricLogger`,
  `compare_runs`); `tests/baselines/credit_gbdt.py` →
  `scripts/baselines/credit_gbdt.py`; `MissingExtraError` lives in
  `pragmatiq.core.errors` (re-exported by `integrations`).
- **Removed** (see below): the event-attribution module, the multitask
  inference module, the offline bundle, the Azure ML and Nebius adapter
  stubs, `storage.artifacts`, unused storage/env helpers, `configs/model/*`,
  the `fraud` / `aml_gnn` fine-tune YAMLs, `configs/pretrain_nano.yaml`,
  `docs/audit/`, `sbom/`.

### Added

- **`pragmatiq info`** (versions, resolved device, CUDA, flash-attn,
  importable extras, run/data roots; `api.info()`), **`pragmatiq --version`**,
  and **`pragmatiq pretrain --show-config`** (`api.pretrain_plan`) to print the
  resolved training plan without training.
- **Inference precision policy** (`pragmatiq.core.env`): `inference_context`
  wraps every inference forward in `torch.inference_mode` plus bf16 autocast
  on CUDA (which is what routes attention through the flash varlen kernel);
  `precision="auto|bf16|fp32"` on `embed_records`, `embed_users`,
  `BatchEmbedder`, `benchmark` (`--precision`), serving; the
  `PRAGMATIQ_INFERENCE_PRECISION` and `PRAGMATIQ_DEVICE` environment pins.
- **`embed_records(records, token_budget=...)`** splits large requests into
  bounded forward passes; the serving runtime uses it with
  `PRAGMATIQ_SERVE_TOKEN_BUDGET` (16 384). `request_limits()` exposes the caps.
- **Event-staleness evaluation** (paper §3.4.2): `api.probe(...,
  staleness_window="6h")` / `pragmatiq probe --staleness-window`, applied to
  the probe and the raw-count baseline alike; `scripts/benchmarks/
  staleness_probe.py` sweeps 0 / 1h / 6h / 1d / 3d into the README
  `STALENESS_PROBE_RESULTS` block.
- **Generator v2 realism**: FX-converted amounts (`FX_PER_GBP`), card-only
  interchange (`monthly_card_spend`), computed England & Wales bank holidays
  (`uk_bank_holidays`, `easter_sunday`), per-user payday rules, market-hours
  equity trading with 24/7 crypto, overdraft fees debited from the balance,
  clamped mule windows, home-country merchant pools and trip-country travel
  spend, cross-border online orders. Tests cover each property.
- **Prefetching shard loader** (`ShardDataLoader(prefetch=, pin_memory=)`)
  used by pretraining (`TrainConfig.prefetch_batches`), fine-tuning
  (`FineTuneConfig.prefetch_batches`), batch embedding and the benchmark; the
  sampler position it checkpoints is exact across prefetch depths.
- **Fine-tune `epoch_stats`** (batches, users, tokens, seconds, tokens/s per
  epoch and phase) in the `finetune` result, plus `token_budget`; a
  single-process fallback with a warning when several GPUs are visible but
  `lightning` is not installed.
- **`TrainConfig.accelerator`** (`auto|cpu|cuda`) so an explicit CPU run on a
  GPU host is possible.
- **`TokenizerConfig.max_counter_distinct`** (1 000 000) bounds `fit()`
  memory on continuous numeric keys; a key that saturates and then stops
  parsing as a number raises with a pointer to `force_numeric` /
  `force_categorical`.
- **Typed errors** `ConfigError` and `DataContractError` in
  `pragmatiq.core.errors`, raised where a mistyped config or a broken data
  contract used to surface as a bare `ValueError`.
- **`api.__all__`**, read by `scripts/docs_facts.py`.
- **GPU validation package** `scripts/gpuval/` (monitoring, legs, training,
  serving, flash check, report) replacing the monolithic `scripts/validate_gpu.py`;
  new legs: quickstart timing, export on the GPU image, request-cap check,
  fine-tune epoch stats, attention-backend report; results land in
  `docs/benchmarks/gpu-validation-1.1.0.json` and render into the README
  `GPU_VALIDATION_RESULTS` block, drift-checked in CI.
- `deploy/triton/config.cpu.pbtxt`, `deploy/docker-compose.cpu.yaml`, the
  `flash` extra (`flash-attn>=2.4.1` floor), the `gpu` pytest marker.
- CI runs every gate in `scripts/gates/` (including `gate_serve_slim`,
  `gate_storage`).

### Changed

- **Varlen attention** (`pragmatiq/models/layers.py`): the SDPA fallback
  scatters with `index_copy_` / `index_select` (deterministic under
  `torch.use_deterministic_algorithms`, no more crash on CUDA); the segment
  layout (positions, mask, RoPE tables) is built once per encoder forward and
  threaded through the blocks; segments are grouped into length buckets so one
  6,500-event history no longer pads every other segment in the batch to its
  width (the padded path was O(n_seg × max_len²) on heavy books — a GPU
  fine-tune without flash-attn ran at 1% utilisation); `attention_backend()` /
  `flash_available()` report the active kernel; `PRAGMATIQ_DISABLE_FLASH=1`
  forces SDPA.
- **Trainer metrics** log `tokens_per_sec_window` (the rate over the last log
  interval) next to the cumulative `tokens_per_sec`; the GPU validation sweep
  reports the steady-state median of the window rate.
- **`quickstart`** pins `devices=1`: a nano smoke run gains nothing from DDP and
  was 2.5× slower on an 8-GPU host.
- **Shard cache** (`ShardDataset`): sized in bytes (a quarter of RAM, up to
  16 GiB) instead of four shards; `cache_shards=` still pins a count.
- **Fine-tune `epoch_stats`** carry `data_wait_seconds` (time spent waiting on
  the loader) next to `tokens_per_sec`, so a slow epoch can be attributed to the
  host or the device from the result dict alone.
- **GPU validation harness**: serving legs warm up and send 64+ requests per
  concurrency level; `scripts/benchmarks/refresh_results.py` regenerates every
  README/notebook result table on one pod; `scripts/validate_gpu.py
  --render-json` re-renders a validation JSON into the README block; the RunPod
  launcher pins the release torch build the flash-attn wheel targets and drops
  the image's nightly torchvision/torchaudio.
- **Collator** vectorized with `np.repeat` / `np.diff`; `PackedBatch` carries
  host-side `max_len_event` / `max_len_history` / `max_len_profile` so the
  model never syncs to size a block; `PackedBatch.pin_memory()` and
  `to(device, non_blocking=)`.
- **Trainer**: one host read per micro-batch, per-type metrics only on logged
  steps, a single stacked `isfinite` over all grads, Python `random` state in
  checkpoints, checkpoints loaded with `weights_only=True` (fallback with a
  warning for foreign files), resume-config check tolerant of keys added by a
  newer version.
- **Tokenizer fit** keeps numeric samples as float64 chunks with the same
  reservoir replay (byte-identical bins); per-day timezone offsets with DST
  transition bisection; `LargeListArray` shard columns; incremental LMDB
  profile puts; `UserIndex.meta_many` / `ShardDataset.get_many` in one
  transaction; `validate` vectorized with `pyarrow.compute`.
- **Fine-tuning** on CUDA runs bf16 autocast in the single-process path too,
  sizes `token_budget` from device memory, and prefetches batches.
- **ONNX export** requires torch >= 2.6 and says so; `export` accepts any
  `device` and builds on CPU.
- **Triton image** installs every runtime dependency from `pyproject.toml`
  (the hand-maintained list once shipped without `fsspec`) and smoke-imports
  the serving runtime at build time.
- **Docs**: README rewritten around the GPU-first story (Hardware table,
  Running on GPU, Synthetic data realism, event staleness, serving device
  policy, gates table); `docs/STABILITY.md` rewritten to the pre-2.0 policy;
  `RELEASING.md` documents the feature branch → PR → `main` flow with
  `develop` mirroring `main`.
- Smaller fixes: LoRA target matcher anchored on the leaf module name; the
  GNN keeps the best trained state even below chance (`best_val=-1.0`);
  the hash text encoder memoizes; the fine-tuner caches its parameter list and
  skips probability computation on train batches; the CPU intra-op thread cap
  is a no-op on CUDA; `pyyaml` and `pandas` left the core dependencies
  (`pandas` in the `aml` / `demo` / `dev` extras); `tritonclient` moved to a
  `triton-client` extra.

### Removed

- `pragmatiq/inference/explain.py` (integrated-gradients event attribution)
  and the notebook / README material that referenced it.
- `pragmatiq/inference/multitask.py` (the benchmark script
  `scripts/benchmarks/multitask_probe.py` renders its own table).
- `deploy/offline/` (offline bundle) and the `offline_mode` /
  `telemetry_enabled` env flags nothing read.
- `integrations/azure` and `integrations/nebius` stub adapters; the runbook
  in `docs/INTEGRATIONS.md` covers those platforms with the generic Triton image.
- `pragmatiq/storage/artifacts.py`, `storage.fs.open_file` / `makedirs`,
  `storage.cache.local_path` and `PRAGMATIQ_CACHE_DIR`.
- `configs/model/*.yaml`, `configs/finetune/fraud.yaml`,
  `configs/finetune/aml_gnn.yaml`, `configs/pretrain_nano.yaml`.
- `docs/audit/` historical records, the `sbom/` README (folded into
  `SECURITY.md`), `scripts/validate_gpu.py` (replaced by `scripts/gpuval/`).
- `PRAGMATIQ_SERVE_GPU`.

## [1.0.0] — 1.0 production release

First stable release. The public API, CLI, serving contract, and checkpoint
format are now **frozen** under the SemVer policy in `docs/STABILITY.md`.

### BREAKING (install behavior)

`torch-geometric`, `lightning`, and `matplotlib` are no longer installed by
default. They moved to optional extras so that a plain `pip install pragmatiq`
gives a slim inference-capable install:

- **training** now requires `pip install 'pragmatiq[train]'`
- **AML GraphSAGE** requires `pip install 'pragmatiq[aml]'`
- **serving / ONNX export** requires `pip install 'pragmatiq[serve]'`

**Migration:** `pip install 'pragmatiq[full]'` reproduces the old all-in
install with every optional dependency. No change to the Python API, CLI
command names, `from_pretrained` / `embed_records`, the serving contract, or
the checkpoint format.

### Added

- **Public-API stability contract** — `docs/STABILITY.md` (frozen at 1.0.0)
  enumerates the 15 `pragmatiq.api.*` functions, `PragmaModel.from_pretrained`
  / `embed_records`, the full CLI command tree, the serving wire format, and
  the checkpoint-format version. `tests/contract/` enforces the contract on
  every CI run and gate 9.
- **`# GUESS` hyperparameter catalog** — README section "Paper-silent (`#
  GUESS`) hyperparameters" documents all 9 unique paper-silent defaults (13
  source markers), their config keys, and one-line rationale. These defaults
  are embedded in every run's `run.yaml` / `meta.json` so shipped checkpoints
  reproduce regardless of future default changes.
- **Pluggable object-store storage** (`fsspec`) — `pragmatiq.storage`
  abstracts run/checkpoint/shard I/O over any fsspec-compatible backend (local
  file, S3, GCS, Azure Blob). Use `[s3]`, `[gcs]`, or `[azure]` extras.
- **Serving glue extracted** — `pragmatiq.inference.serve` owns the single
  serving contract (`records_json → embeddings [n_users, dim]`); the Triton
  `model.py` and REST/gRPC adapters delegate to it.
- **Cloud-adapter seams** — `integrations/` holds real SageMaker and
  Databricks adapters plus documented stubs for two further platforms
  (removed in 1.1.0); see `docs/INTEGRATIONS.md`.
- **`apps/` UI seam** — the Streamlit demo relocated to `apps/demo`; a thin
  `apps/` namespace provides a stable hook for future UIs.
- **BYOC hardening** — verified no-phone-home behavior, offline / air-gapped
  install path, locked dependencies (`uv.lock`), SBOM generation
  (`scripts/supply_chain/gen_sbom.sh`), and license + vulnerability scan in CI.
- **RELEASING.md** updated with the 1.0 release procedure (uv.lock
  regeneration, SBOM, full validation, tag + build + publish steps).

### Fixed

- **Fine-tune epochs batch only the labeled split** — the single-process
  fine-tune path now uses the same subset sampler as the DDP path instead of
  iterating the entire shard set and discarding unlabeled users. Epoch cost is
  now proportional to the label table, not the dataset (a 3-epoch fine-tune of
  a 100k-user shard set with partial labels dropped from ~21 h to well under
  an hour per epoch on one GPU).
- **GPU LoRA fine-tuning runs bf16 autocast** — the single-process CUDA path
  now matches the DDP path's mixed precision, routing attention through the
  flash varlen kernel. The previous fp32/SDPA path retained O(L^2) attention
  scores through the frozen backbone's backward and could exhaust an 80 GB GPU
  on the `large` preset for a single long-history user. CPU fine-tuning is
  unchanged (fp32, byte-identical).

- **Slim install can synthesize out of the box** — `api.synthesize` /
  `pragmatiq synth generate` now default `write_report` to *auto*: the realism
  report is written when matplotlib (the `[data]` extra) is installed and
  skipped with a logged warning otherwise. An explicit `write_report=True`
  still raises `MissingExtraError` when matplotlib is absent.
- **Remote `runs_root` pretrain returns a durable path** — `api.pretrain`
  with a remote (e.g. `s3://`) `runs_root` now returns the remote run URL
  instead of the staged local temp directory that staging deletes on exit.
- **Remote `tokenizer_dir` is staged** — `api.tokenize(tokenizer_dir="s3://…")`
  materializes the tokenizer locally before loading instead of failing.
- **Databricks `register()` reports the real model version** — the returned
  Unity Catalog URI uses the version the registry assigned (previously
  hardcoded `/1`).
- **Staging preserves computed results when an upload fails** — a remote-output
  upload error no longer deletes the freshly computed local results; the error
  names the preserved directory. Remote backends are also validated eagerly, so
  a missing cloud extra or unknown scheme fails before compute, not after.
- **`tokenize(tokenizer_dir=...)` produces a self-contained shard dir** — the
  loaded tokenizer is saved into `out/tokenizer`, so downstream commands accept
  the output (previously they rejected it with a missing-tokenizer error).
- **CLI accepts remote URLs** — path-like CLI options no longer mangle
  `s3://...` to `s3:/...`; `pragmatiq quickstart`, `runs list/compare`, and
  `export` handle remote roots end to end, and `pretrain` rejects invalid
  `--resume` values instead of silently starting a fresh run.
- **SageMaker `package()` builds a bootable Triton bundle** — the model.tar.gz
  now contains the Triton model repository (config + backend) wired to
  `PRAGMATIQ_RUN`, and adapter healthchecks speak the KServe v2 envelope
  (`encode_v2_request`/`decode_v2_response` added to the serving contract).
- **Databricks pyfunc is loadable outside the repo** — the wrapper moved into
  the shipped package (`pragmatiq.inference.serve.pyfunc`), `register()` pins
  `pip_requirements`, and `predict()` accepts the DataFrame input Databricks
  Model Serving actually delivers.

### One default change (otherwise no API changes)

The Python API (`pragmatiq.api.*`), CLI command names, `from_pretrained` /
`embed_records`, the serving wire contract, and the checkpoint format are
unchanged from 0.1.0b4, with one deliberate default change recorded above:
`synthesize(write_report=…)` defaults to ``None`` (auto) instead of ``True``.
All existing code and shipped checkpoints continue to work without
modification.

## [0.1.0b4] — Hardening and the SageMaker guide

Surgical reliability fixes from a pre-launch validation pass, plus a new deployment
tutorial. No change to the public API or the foundation-model architecture.

### Added
- "Run pragmatiq on Amazon SageMaker" tutorial: a managed training job for
  pretraining and a NVIDIA Triton real-time endpoint for serving, on synthetic data
  or your own.
- A topology-only GraphSAGE control (arm `e`) reported alongside the AML ablation arms.
- A full `pytest` CI job, so the finetuner / uplift / multitask / inference unit
  tests run in CI rather than only the smoke subset.

### Changed
- The AML relational-recovery gate uses a noise-aware margin — the mean paired
  per-seed difference must exceed the cross-seed standard deviation — instead of a
  fixed `0.01` a within-noise gap could clear.
- The LoRA fine-tuner stratifies its validation split by label, so rare-positive
  tasks keep both classes held out (no silent single-class split).
- LightGBM moved to an optional `[gbdt]` extra; the default probe (scikit-learn
  `HistGradientBoosting`) needs no extra. The `probe` CLI gained a `--seed` flag.

## [0.1.0b3] — Validation hardening

Strengthens release-readiness validation and reproducibility safety. No change to
the public API or the foundation-model architecture.

### Added
- `py.typed` marker (PEP 561) so downstream type-checkers see pragmatiq's inline
  types, plus a CI packaging smoke check that builds the wheel and sdist and
  asserts the version metadata and typing marker are present.
- Realism metrics emitted as machine-readable JSON alongside the HTML report, and
  label-table schema validation that flags unexpected or missing columns.

### Changed
- Embedding, probing, fine-tuning, export, and the AML GNN verify that a tokenized
  shard directory was encoded by the same tokenizer as the training run (content
  hash) before running; resuming a run refuses architecture, optimizer, masking,
  data, or schedule changes while still allowing operational knobs such as
  `max_steps`.
- A merged LoRA layer inherits the base layer's device and dtype.

## [0.1.0b2] — Public beta

First public (beta) release: an end-to-end, CPU-capable toolkit for behavioral
banking foundation models.

### Added
- **Synthetic data** — a deterministic, agent-based banking simulator (events,
  profiles, transfers, and causal fraud / credit-default / AML / churn / LTV
  labels with strict eval-point truncation), an HTML realism report, and
  `synth calibrate` for fitting the generator to aggregate statistics.
- **Tokenizer & data pipeline** — a key–value–time tokenizer (percentile-binned
  numerics, categorical/BPE values, calendar + log-second time features, `[UNK]`
  fallbacks), parquet sharding with an LMDB user index, and padding-free varlen
  batching under a token budget.
- **Model** — PRAGMA-style encoders with shared/tied key–value embeddings,
  TimeRoPE continuous-time positions, per-event independent encoding, profile and
  history encoders, and a tied MLM head; `small`/`medium`/`large` presets plus a
  CPU-friendly `nano` size.
- **Training & adaptation** — pretraining on Lightning Fabric (Muon + AdamW,
  cosine schedule, fully resumable checkpoints), a gradient-boosting probe
  (ROC-AUC + PR-AUC vs a same-classifier raw-count baseline; logistic/LightGBM
  selectable), LoRA fine-tuning, and a registry for swappable heads, maskers, and
  value encoders.
- **Hands-off scale** — gradient accumulation, multi-node DDP, and a `config="auto"`
  sizer that picks the batch and schedule from the data + device, so a run scales
  from 1M to 26M records without tuning.
- **PRAGMA+Nemotron variant** — an optional, switchable text pathway that embeds
  high-cardinality text fields with a frozen text encoder and reconstructs them
  with MSE; off by default, so the BPE path is byte-identical.
- **AML over the transfer graph** — a GraphSAGE node classifier and the
  four-arm relational-recovery ablation (isolated embedding vs graph-aware
  pragmatiq vs graph-aware hand-crafted, with a no-graph control); see
  MODEL_CARD.md and `notebooks/04`.
- **Inference & serving** — a batch embedder, integrated-gradients event
  attribution, ONNX export, a Triton serving image that installs pragmatiq, a
  turnkey `deploy_serving.sh`, monitoring (Prometheus + Grafana), and a Streamlit demo.
- **Documentation** — a modern docs/educational site (Next.js + Fumadocs) at
  pragmatiq.getdynamiq.ai, with interactive visualizers and a facts drift-check.
- **Engineering** — every path runs on CPU (CUDA and flash-attn as accelerations),
  a typed public API, notebooks, a model card, and CI (ruff + mypy + pytest +
  acceptance gates).
