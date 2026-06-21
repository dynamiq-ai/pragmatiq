# pragmatiq Correctness Audit — Verdict (2026-06-21)

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

**Scope:** a paranoid, evidence-based correctness audit of the ML implementation after the 1.0 restructure
(W0–W8) and the multi-GPU work, prompted by concern that something subtle (or an AI-introduced hallucination)
broke the model — masking, embeddings, MLM, LoRA, the Nemotron text branch, the gradient-boosting probe, the
optimizer, or the GPU/CPU/CUDA/multi-GPU paths.

**Method:** 3 Opus exploration audits (implementation map / paper-fidelity / change-impact) → a 15-agent
adversarial re-verification Workflow (≥2 independent Opus agents per claim, each prompted to *refute*
correctness; 3 dedicated tracers on the one genuine concern) → re-run of the full test suite + gates.

## Bottom line

**The pragmatiq library is sound.** The core model matches the PRAGMA paper/SPEC and the restructure did not
change its math. The only real defects were in the **newly-added multi-GPU code** (2 bugs) and in **the GPU
*validation tooling*** (2 script bugs that caused paid-run overruns) — all now **fixed and verified**. CPU and
single-GPU training were never affected.

## Per-subsystem verdict (≥2 independent adversarial Opus verifiers each, high confidence)

| Subsystem | Verdict | Evidence |
|---|---|---|
| **Core math vs pre-restructure baseline** | ✅ byte-identical | `masking.py`, `models/{embeddings,layers,heads}.py`, `optim.py` = 0 diff lines; blob hashes match; schema move = pure rename; staging/serving extraction behavior-preserving (local≡remote test) |
| **Masking + MLM** | ✅ matches paper | 0.15/0.10/0.10 + union/priority; 10% [UNK] excluded (label −100); MLM head 3d concat `[ẑ_e, z_h(event), z_h(USR)]` → Linear(3d→d) → tied logits; CE label-smoothing 0.1 |
| **Architecture + TimeRoPE + embeddings + attention** | ✅ matches paper | 3 encoders (profile/history RoPE, event no-RoPE+calendar), z_h[USR]=user emb, continuous-time RoPE kept fp32 under bf16, tied key+value embedding, flash↔SDPA selection + cu_seqlens block-diagonal |
| **Optimizer + schedule + single-process grad-accum** | ✅ matches paper | Muon 2-D / AdamW split, Newton-Schulz, WarmupCosine pure-of-step, per-optimizer clip, accum≡single-batch |
| **Probe / GBDT / uplift / AML + tokenizer + Nemotron-MSE** | ✅ matches paper | no-leakage eval_ts truncation (probe+baseline), GroupShuffleSplit uplift, AML gated c>a & c>d, percentile binner + [UNK]-never-KeyError + sha256 hash, frozen-encoder MSE branch |
| **DDP fine-tune (world==1) + TF32 gating** | ✅ after fixes | world==1 byte-identical; see fixed bugs below |

**Empirical:** real-H100 runs showed pretraining **reducing MLM loss 7.28 → 3.43** (MLM-acc → ~0.40) — it trains.

## Confirmed bugs — all FIXED (with red-green tests)

1. **grad-accumulation × DDP desync/deadlock** (CRITICAL; 3/3 tracers, high conf). With `world_size>1` AND
   `grad_accum_steps>1`, the all-reduce fired per micro-batch and ranks with different `contributing` counts
   applied different rescale factors → replicas desync (+ possible deadlock). README advertises DDP+accum, so
   it was a live path. **Fix** (`pretrainer.py`, commit `01f62de`): one all-reduce per window via
   `fabric.no_backward_sync`; every rank issues `accum` backwards (zero-graph loss routed through the same
   tied-embedding forward for empty micro-batches); global rescale `(world·accum)/global_contributing` →
   numerically equals the single-process accum step. New `tests/test_ddp_grad_accum.py` (2-rank gloo) asserts
   no deadlock, replicas byte-identical, and single-process match <1e-5, exercising an empty micro-batch.
2. **NaN/grad-skip DDP deadlock** (IMPORTANT; found in review of fix #1). The new unconditional all-reduce
   could hang if a non-finite loss hit one rank only (per-rank data+seed) and that rank early-returned past
   the collective. **Fix** (`pretrainer.py`, commit `e7de8c2`): the skip decision is now collective — one
   combined `all_reduce([contributing, skip, grads_nonfinite])`; any rank's non-finite makes all ranks skip
   together. The new test *deadlocks against the pre-fix code* and passes (~10s) with the fix.
3. **TF32 deterministic-mode leak** (IMPORTANT; 2/2 agents). The non-deterministic branch set matmul
   precision `'high'` but the deterministic branch never reset it → a deterministic run after a
   non-deterministic one in-process silently lost fp32 bit-exactness. **Fix** (`pretrainer.py`, `01f62de`):
   the deterministic branch resets `set_float32_matmul_precision('highest')`; tested in `TestDeterminism`.

## Doc reconciliations + guardrails (commit `6188a19`)
- README `warmup_steps` default framed consistently (100 dataclass / 500 yaml, yaml wins).
- `docs/SPEC.md` GUESS list updated to the 9-item set; probe default noted as gbdt (intentional); `p_unk`
  framed as a documented default not a paper requirement. (SPEC.md is the gitignored local design spec.)
- DDP fine-tune AUC de-dup made deterministic (average duplicate-user probs) — rank-order-independent metric.
- New test: the frozen text encoder is absent from `state_dict()` (guards a fragile non-`nn.Module` invariant).

## GPU validation tooling fixes (commit `ccd30fd`) — the overrun root cause
Two paid 8×H100 runs overran (~$67 + ~$24.5) due to bugs in the *validation scripts* (not pragmatiq):
- `scripts/validate_gpu.py`: a per-leg subprocess **pipe deadlock** (captured stdout filled the OS pipe → the
  leg blocked on write). **Fixed** by redirecting leg stdout to a file (no pipe can fill); a `--high-output-test`
  mode proves a 300KB-emitting leg completes without hanging.
- `scripts/runpod_launch.py`: the budget cap (a subprocess timeout) didn't fire when the SSH command was
  wedged. **Fixed** with an independent **watchdog daemon** that DELETEs the pod at `create_ts+cap` and kills
  the SSH process regardless of its state.

## Intentional divergences (allowlist — do NOT re-flag as bugs)
AML GNN `b<c` (documented limitation); gbdt probe default vs the paper's linear probe; LoRA targets superset
(qkv+out+ffn); the `hash` text-encoder CI stand-in; ONNX dense-not-varlen export; the `# GUESS` paper-silent
hyperparameters; GPU bf16 non-bit-exactness (deterministic fp32 is bit-exact). See `MODEL_CARD.md` / `README.md`.

## Test evidence
Full CPU suite green at HEAD `ccd30fd`: **545 passed, 1 skipped** (CUDA-only Newton-Schulz), 18 pre-existing
warnings; `ruff` + `mypy` clean. Includes the new DDP grad-accum + NaN-divergence + TF32-reset +
text-encoder-absent tests. Single-process grad-accum equivalence, bit-exact resume, padding-equivalence
(varlen ≡ padded), and probe>baseline all remain green and unchanged.

## Not done (gated, optional)
A clean 1→2→4→8 GPU scaling curve + DDP-finetune + serving req/s + a flash-attn≡SDPA numeric check — deferred:
it needs a paid GPU run, which is now safe to attempt (tooling fixed, watchdog hard-caps the spend) but is held
pending an explicit go (cumulative GPU spend so far ≈ $91).
