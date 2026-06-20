# Changelog

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

All notable changes to pragmatiq are documented in this file. This project
follows [Semantic Versioning](https://semver.org); 0.x releases are pre-1.0 and
the public API may change. From **1.0.0** onward the public API is frozen and
SemVer applies (see [`docs/STABILITY.md`](docs/STABILITY.md)).

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
  Databricks adapters plus documented stubs for Azure ML and Nebius; see
  `docs/INTEGRATIONS.md`.
- **`apps/` UI seam** — the Streamlit demo relocated to `apps/demo`; a thin
  `apps/` namespace provides a stable hook for future UIs.
- **BYOC hardening** — verified no-phone-home behavior, offline / air-gapped
  install path, locked dependencies (`uv.lock`), SBOM generation
  (`scripts/supply_chain/gen_sbom.sh`), and license + vulnerability scan in CI.
- **RELEASING.md** updated with the 1.0 release procedure (uv.lock
  regeneration, SBOM, full validation, tag + build + publish steps).

### No API changes

The Python API (`pragmatiq.api.*`), CLI command names, `from_pretrained` /
`embed_records`, the serving wire contract, and the checkpoint format are
unchanged from 0.1.0b4. All existing code and shipped checkpoints continue to
work without modification.

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

First public (beta) release: an end-to-end, CPU-first toolkit for behavioral
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
- **Engineering** — CPU-first throughout (CUDA and flash-attn are accelerations),
  a typed public API, notebooks, a model card, and CI (ruff + mypy + pytest +
  acceptance gates).
