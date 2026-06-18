# Changelog

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

All notable changes to pragmatiq are documented in this file. This project
follows [Semantic Versioning](https://semver.org); 0.x releases are pre-1.0 and
the public API may change.

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
  five-arm relational-recovery ablation with isolated, graph-aware, no-graph,
  and topology-only controls; see MODEL_CARD.md and `notebooks/04`.
- **Inference & serving** — a batch embedder, integrated-gradients event
  attribution, ONNX export, a Triton serving image that installs pragmatiq, a
  turnkey `deploy_serving.sh`, monitoring (Prometheus + Grafana), and a Streamlit demo.
- **Documentation** — a modern docs/educational site (Next.js + Fumadocs) at
  pragmatiq.getdynamiq.ai, with interactive visualizers and a facts drift-check.
- **Engineering** — CPU-first throughout (CUDA and flash-attn are accelerations),
  a typed public API, notebooks, a model card, and CI (ruff + mypy + pytest +
  acceptance gates).
