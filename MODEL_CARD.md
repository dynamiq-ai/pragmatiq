# pragmatiq Model Card

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

## Model details

- **Architecture.** A PRAGMA-style bidirectional transformer encoder stack over
  key–value–time tokenized banking event sequences. Three encoders: a profile
  encoder (`[USR]` marker over static attributes and lifelong items), an event
  encoder that encodes each event *independently* (block-diagonal varlen
  attention, an `[EVT]` marker per event, calendar embedding), and a history
  encoder over the per-event vectors with rotary time embeddings (TimeRoPE)
  whose continuous position is `8·ln(1 + Δt/8)` log-seconds. Pre-norm, GELU
  MLP (ffn = 4d), dropout 0.1. Keys and values share one embedding table, tied
  to the output projection.
- **Pretraining objective.** Masked language modeling over event *values*: the
  key is kept, the value is predicted. Selection unions three modes (token
  p=0.15, whole-event p=0.10, key p=0.10); 10% of selected positions become
  `[UNK]` and are excluded from the loss. The MLM head concatenates
  `[ẑ_e(token), z_h(event), z_h(USR)] ∈ R^{3d}`, projects `3d → d` with a single
  `Linear`, and scores against the tied embedding table with cross-entropy and
  label smoothing 0.1. Gradient clipping is per-optimizer (Muon and AdamW
  each clipped to 1.0).
- **PRAGMA+Nemotron variant (optional, off by default).** High-cardinality text
  fields can instead be embedded by a *frozen* text encoder into a single vector;
  masked text tokens are then reconstructed with MSE (`loss = CE + λ·MSE`) rather
  than predicted as sub-word ids. Default behavior is byte-identical to the BPE path.
- **Sizes.** `small` (d=192, ~9.1M params), `medium` (d=512, ~94M), `large`
  (d=1024, ~940M) at a ~28k vocabulary; see the README size table.
- **Output.** The history encoder's `[USR]` slot is the user embedding
  (`z_h[USR]`), the primary artifact consumed downstream.
- **Provenance binding.** Checkpoints embed the tokenizer content hash and the
  resolved training config; `PragmaModel.from_pretrained()` refuses to load a
  checkpoint against a mismatched tokenizer. Unseen keys/values at inference
  map to `[UNK]` with a logged warning, never an exception.

## Intended use

Behavioral user embeddings for retail-banking ML tasks, consumed by downstream
classifiers rather than acting as a decision system themselves:

- **Probes** (`pragmatiq probe`): a classifier on frozen embeddings — the cheap
  evaluation path. The default head is gradient boosting (`HistGradientBoostingClassifier`;
  `logistic` and `lightgbm` are selectable), reporting ROC-AUC and PR-AUC against a
  raw-count baseline that uses the same classifier.
- **LoRA fine-tuning** (`pragmatiq finetune`): frozen backbone, rank-8 adapters
  plus a classification head on `z_h[USR]`, early stopping; `merge_lora()`
  folds adapters back for export. The default adapters target the attention
  `qkv` and output projections plus the MLP — a deliberate superset of the
  paper's QKV+MLP placement.
- **AML via GNN** (`pragmatiq gnn`): GraphSAGE over a money-transfer graph with
  pragmatiq embeddings as node features — mule rings are a relational fan-in/
  fan-out pattern that isolated per-user embeddings cannot see.

Target tasks demonstrated on the synthetic benchmark: fraud (account takeover),
credit default (12 months), churn (6 months), AML (mule-ring membership), and
LTV. Per-event attribution for any of these is available via integrated
gradients (`pragmatiq/inference/explain.py`).

**Out of scope.** Shipped checkpoints and the quickstart model are trained on
synthetic data only and must not be used for real credit, fraud, or AML
decisions. Even after pretraining on real data, the model produces features
for governed downstream models — it is not a standalone decisioning system.

## Training data

**Fully synthetic — no real customer data, no PII, at any stage.** The training
corpus comes from pragmatiq's agent-based causal simulator
(`pragmatiq/data/synthetic/`):

- A **world** with a calendar (paydays, weekends, holidays, seasonality,
  inflation drift), a 50k-merchant universe (Zipf popularity, MCCs, noisy
  display names), and a transfer graph with injected mule rings.
- **Personas** drawn from a configurable archetype mixture (student, salaried,
  freelancer, family, pensioner, high-net-worth, trader, dormant, mule,
  fraud victim), each with latent traits (income level, spend propensity,
  financial stress, tech savviness, risk appetite, sociability, churn hazard,
  fraud vulnerability) sampled from per-archetype priors.
- **Per-user simulation**: a lifecycle Markov chain, recurring
  salary/rent/subscription series, a non-homogeneous Poisson spending process,
  Hawkes-burst app sessions, trading and communications processes, P2P
  transfers, and balance tracking with overdraft events.
- **Episode injection**: account-takeover fraud episodes, multi-month financial
  stress arcs, and mule episodes — recorded in a latent log that the label
  oracle reads.

Generation is deterministic: the same seed yields byte-identical parquet
output regardless of worker count (CI-enforced).

## Label design (no leakage)

Labels are *consequences* of latent traits and simulated behavior, not inputs
to it, and the oracle enforces a strict temporal split:

- user-level labels (`default_12m`, `churn_6m`, `ltv_positive`) are computed
  only from what happens strictly **after** the task's eval point;
- eligibility (who gets a label row) is computed only from what happens
  **before** it;
- event-level labels (`fraud`, `recurring`) are exact event memberships;
  `aml` is mule-ring membership; `comm_uplift` stores both potential outcomes.

Two difficulty knobs keep ceilings realistic: `trait_noise` blurs the
trait-to-behavior mapping and `label_noise` flips a small fraction of binary
labels. An acceptance check enforces that a gradient-boosted-tree baseline on
hand-crafted features scores in a realistic band on credit (~0.75–0.85 AUC) and
*fails the build* above 0.95, which would indicate leakage.

## Reproducibility

From a fixed seed, **CPU runs are byte-identical** — data generation, weight
init, dropout, the masking stream, and tokenized shards (any worker count) are
all seeded, and the resume test is checked bit-exactly in CI. Because GPU
kernels reduce in a different order, **CPU and GPU outputs are never
bit-identical to each other**; compare a target against itself.

The opt-in `deterministic: true` flag (default off, so default behaviour and
throughput are unchanged) makes the **GPU** path reproducible on fixed hardware.
It enables `torch.use_deterministic_algorithms`, the cuDNN deterministic path,
flash-attn's deterministic backward, and trains in fp32 instead of bf16-mixed.
With it on:

- the **GPU forward / embedding** is reproducible run-to-run on the same
  hardware;
- **GPU training is bit-exact in fp32** (same seed → same loss curve);
- **GPU bf16 training is not bit-exact** — the SDPA/flash bf16 backward has no
  deterministic implementation upstream (a known PyTorch limitation), so a
  deterministic bf16 run is only run-to-run stable to ~1e-3, which is why
  deterministic mode selects fp32.

## Limitations

- **Synthetic data is a stand-in.** The generator reproduces qualitative
  banking structure (long-tailed activity, day/night cycles, Zipf merchants,
  payday effects), not any real book. Expect distribution shift; treat shipped
  metrics as a pipeline check, not a performance claim. To approximate your
  own book without sharing raw data, fit the generator to aggregate statistics
  with `pragmatiq synth calibrate --stats aggregates.yaml` (moment matching;
  example aggregates in `configs/data/aggregates.example.yaml`), then retrain.
- Episode templates are stylized: real fraud, laundering, and financial
  distress are more varied than the injected account-takeover / mule-ring /
  stress-arc patterns.
- **AML ablation finding.** The AML benchmark reports five arms on identical
  splits: **(a)** a probe on isolated pragmatiq embeddings; **(b)** GraphSAGE
  over transfers with pragmatiq node features; **(c)** GraphSAGE with
  hand-crafted node features and transfer edge attributes; **(d)** logistic
  regression on the same hand-crafted node features without a graph; and **(e)**
  GraphSAGE with the hand-crafted node features and topology only. The synthetic
  mules are multi-hop layered laundering chains: their amounts and counterparty
  degree are drawn to *match ordinary accounts*, so 1-hop degree is not a trivial
  oracle, and the discriminative signal is the multi-hop layering chain. The
  **gated claim** is *relational recovery*: a GraphSAGE over the transfer graph
  recovers money-mule rings a probe on the isolated per-user embedding cannot,
  so the AML signal lives in the multi-hop transfer structure an isolated
  embedding misses. The learned per-user embedding, no-graph control, and
  edge-attribute contribution are reported, not gated. The isolated embedding is
  expected to be weak on this relational task, so recovering the multi-hop
  laundering signal in a learned per-user representation remains the open
  challenge. This is
  consistent with the PRAGMA paper's own observation that AML is a setting where
  the model underperforms because it processes user histories in isolation; the
  GNN is pragmatiq's standalone extension that probes that gap. See
  `notebooks/04_aml_gnn.ipynb`.
- **Serving path.** ONNX export (`pragmatiq export`) emits a faithful **dense
  reformulation** of the model: the same weights run over padded tensors, so the
  exported graph reproduces the native embeddings (validated against onnxruntime
  on export, shape-dynamic in the user/event/token axes). The **Triton python
  backend** (`deploy/triton/`) remains the high-throughput path because it runs
  the native varlen model and skips the padding the dense graph materializes — a
  deployment choice, not a fidelity gap. Pick Triton for throughput, ONNX for
  portability.
- Long histories are capped by default (per-event ≤24 tokens, profile ≤200
  tokens, ≤6500 most-recent events per user); set those caps to `None` to encode
  histories in full. Cost grows with token count under the token-budget batching,
  and very long users dominate batches.
- Hyperparameters the paper does not specify (learning rates, token budget,
  vocab size, RoPE base, etc.) are documented guesses — see the README GUESS
  table.

## Ethical considerations

- **Synthetic-only stance.** No real customer data was used to build, tune, or
  evaluate this repository, and none is distributed with it. This is a
  deliberate choice: it makes the full pipeline reproducible and shareable
  without privacy risk, and the calibration path keeps it that way (banks
  share population-level aggregates, never records).
- **Fairness in financial ML.** Models trained on behavioral banking data can
  encode proxies for protected attributes. The synthetic personas carry no
  protected attributes, so this repository cannot demonstrate or audit
  real-world disparate impact — that obligation transfers to whoever trains on
  real data. Run subgroup performance and calibration analyses on your own
  population before any deployment, and comply with applicable credit and
  consumer-protection regulation (e.g. adverse-action explanation
  requirements).
- **Explainability is a tool, not a defense.** Integrated-gradients event
  attribution highlights which events drove a score; it does not by itself
  satisfy model-governance or recourse obligations.
- **Misuse.** Embeddings that summarize a person's financial behavior are
  sensitive by construction. Apply your institution's data-protection controls
  to embedding stores exactly as you would to the raw event data.

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
