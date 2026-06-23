# pragmatiq Correctness Audit

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

A paranoid, evidence-based correctness audit of the ML implementation after the 1.0 restructure and the
multi-GPU work, to confirm nothing subtle broke — masking, embeddings, MLM, LoRA, the Nemotron text branch,
the gradient-boosting probe, the optimizer, or the GPU/CPU/multi-GPU paths.

**Method:** independent adversarial re-verification — for each claim, ≥2 separate agents each prompted to
*refute* it (3 dedicated tracers on the highest-risk item), corroborated against `docs/SPEC.md` and the paper,
followed by a full re-run of the test suite.

## Bottom line

The core model is correct and matches the PRAGMA paper, and the 1.0 restructure changed none of its math. The
only real defects were in the **newly-added multi-GPU code** (3 bugs, all fixed below). CPU and single-GPU
training were never affected.

## Per-subsystem verdict (≥2 independent adversarial verifiers each, high confidence)

| Subsystem | Verdict | Notes |
|---|---|---|
| **Core math vs pre-restructure baseline** | ✅ byte-identical | `masking.py`, `models/{embeddings,layers,heads}.py`, `optim.py` = 0 diff lines; the schema move is a pure rename; storage staging + serving extraction are behavior-preserving (local≡remote equivalence test) |
| **Masking + MLM** | ✅ matches paper | token/event/key 0.15/0.10/0.10 + union/priority; 10% [UNK] excluded from loss (label −100); MLM head 3d concat `[ẑ_token, z_history(event), z_history(USR)]` → Linear(3d→d) → tied logits; CE label-smoothing 0.1 |
| **Architecture + TimeRoPE + embeddings + attention** | ✅ matches paper | 3 encoders (profile/history RoPE, event no-RoPE + calendar); z_h[USR] = user embedding; continuous-time RoPE kept fp32 under bf16; tied key+value embedding; flash↔SDPA selection + cu_seqlens block-diagonal |
| **Optimizer + schedule + single-process grad-accum** | ✅ matches paper | Muon 2-D / AdamW split; Newton-Schulz; WarmupCosine pure function of global step; accum ≡ single-batch |
| **Probe / GBDT / uplift / AML + tokenizer + Nemotron-MSE** | ✅ matches paper | no-leakage eval_ts truncation (probe + baseline); Qini uplift; AML arms; percentile binner + [UNK]-never-KeyError + sha256 hash; frozen-encoder MSE branch |
| **DDP fine-tune (world==1) + TF32 gating** | ✅ after fixes | world==1 byte-identical; see fixes below |

**Empirical:** real-GPU runs show pretraining reducing MLM loss (≈7.3 → 4.1 over a short sweep; full runs reach
~3.4) with MLM-accuracy climbing — it trains; multi-GPU scaling is positive (≈1.85× on 2 GPUs).

## Bugs found and fixed — all in new multi-GPU code (each with a red-green test)

1. **grad-accumulation × DDP desync/deadlock** (critical). With `world_size>1` and `grad_accum_steps>1`, the
   gradient all-reduce fired per micro-batch and ranks with different per-rank contributing counts applied
   different rescale factors → replicas could desynchronize (and deadlock). **Fix:** one all-reduce per window
   via `fabric.no_backward_sync`; every rank issues equal backwards (a graph-connected zero loss for empty
   micro-batches); a global rescale `(world·accum)/global_contributing` that equals the single-process accum
   step. A 2-rank gloo test asserts no deadlock, byte-identical replicas, and a <1e-5 match to single-process.
2. **NaN/grad-skip DDP deadlock** (important). The window's new collective could hang if a non-finite loss hit
   one rank only. **Fix:** the skip decision is collective — one combined all-reduce of (contributing, skip,
   grads-nonfinite); any rank's non-finite makes all ranks skip together. The test deadlocks against the
   pre-fix code and passes with the fix.
3. **TF32 deterministic-mode leak** (important). The deterministic branch of `seed_everything` never reset
   `float32_matmul_precision`, so TF32 could leak across in-process calls. **Fix:** reset to `highest`; tested.

## Intentional divergences (allowlist — not bugs)

AML GNN `b<c` (documented limitation); the gradient-boosting probe default vs the paper's linear probe; the
LoRA target superset (qkv + out + ffn); the `hash` text-encoder CI stand-in; ONNX dense-not-varlen export; the
`# GUESS` paper-silent hyperparameters; GPU bf16 non-bit-exactness (deterministic fp32 is bit-exact). See
`MODEL_CARD.md` / `README.md`.

## Test evidence

Full CPU suite green (incl. the new DDP grad-accum, NaN-divergence, TF32-reset, and text-encoder-absent
tests); single-process grad-accum equivalence, bit-exact resume, and varlen ≡ padded equivalence all remain
green; `ruff` + `mypy` clean.
