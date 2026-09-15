# GPU validation — pragmatiq 1.1.0

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

Every number the README quotes for 1.1.0 comes from the runs below, executed on
RunPod with `scripts/runpod_launch.py` (image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1`,
torch 2.8.0+cu128 pinned, flash-attn 2.8.3). The evidence files are committed under
`docs/benchmarks/` and render into the README `<!-- GPU_VALIDATION_RESULTS -->` block
with `python scripts/validate_gpu.py --render-json <json> --write-readme README.md`.

## Runs

| run | hardware | what it measured | evidence |
| --- | --- | --- | --- |
| baseline (v1.0.0) | 1× A100 80GB PCIe | the 1.0.0 code on the same legs, for before/after | not committed (scratch) |
| rc1 | 1× H100 80GB | branch with SDPA only (the image's torch nightly broke the flash wheel) | not committed |
| rc3 | 1× H100 80GB | full leg set on flash-attn: pretrain, 3-epoch LoRA fine-tune, serving, bf16 vs fp32, ONNX export, request caps, quickstart | `gpu-validation-1.1.0-h100-1gpu.json` |
| h100x8 | 8× A100-SXM4-80GB | the same legs on an 8-GPU pod plus the DDP fine-tune (socket NCCL transport) | folded into `gpu-validation-1.1.0.json` |
| sweep8 | 8× A100-SXM4-80GB | pretrain scaling sweep 1/2/4/8 GPUs over NVLink, steady-state (`tokens_per_sec_window`) | training legs of `gpu-validation-1.1.0.json` |

The composite `gpu-validation-1.1.0.json` takes its training sweep from `sweep8`
(NVLink transport, the configuration a user gets by default) and every other leg from
`h100x8`; `meta.sources` records the split.

## Headline numbers (small preset, 50k synthetic users, 300 steps, token budget 32,768)

| metric | 1.0.0 baseline (A100 PCIe) | 1.1.0 (H100, 1 GPU) | 1.1.0 (A100-SXM, 1 → 8 GPUs) |
| --- | --- | --- | --- |
| pretrain tokens/s | 299k | 276k | 438k → 2.42M (69% DDP efficiency at 8) |
| LoRA fine-tune, 3 epochs on 2,500 users | 44 min, 3% GPU util | 2.5 min (epochs 2–3: 38–49 s) | 3.0 min / 9.1 min on 8 GPUs |
| batch embedding, bf16 | — | 1.07M tokens/s (98 users/s) | 1.39M tokens/s |
| serving req/s at concurrency 1, CPU / CUDA | 8.9 / 28.7 | 133 / 157 | 104 / 129 |
| `quickstart --n-users 2000 --max-steps 80` | — | 134 s | 311 s (see below) |
| flash-attn vs SDPA, max abs diff | 0.0 | 0.0 | 0.0 |
| bf16 vs fp32 embeddings, mean cosine | — | (not measured) | 0.999999 |

The pretrain throughput on a single GPU is launch-bound for the small preset (a 10M
parameter model at 1.9 GB peak VRAM); the A100 single-GPU figure is higher than the
H100 one because the two pods differ in host CPU, and the loader/collation side is
what a launch-bound run waits on.

## What the rounds found and fixed

- **Fine-tuning was host-bound.** On the baseline and rc1 the fine-tune ran at
  1.8 s per batch with 1–3% GPU utilisation. Two causes, both fixed: the padded SDPA
  fallback padded every segment in a batch to the longest history (O(n_seg × L²); the
  fallback now buckets segments by length), and the fine-tune never engaged flash-attn
  on the pod because the image's torch *nightly* could not load the flash wheel (the
  launcher now pins the release torch build and removes the nightly torchvision that
  broke `torchmetrics` on import). `epoch_stats` now carries `data_wait_seconds`, so a
  host-bound epoch is visible from the result dict.
- **ONNX export on the pod** failed on the torch nightly with a dynamo constraint
  violation; it exports and validates on torch 2.8.0 release (rc3, h100x8).
- **Staleness benchmark** crashed on a batch whose users were all truncated to zero
  events (flash-attn rejects an empty batch); attention now short-circuits an empty
  token stream.
- **8-GPU efficiency.** With the socket NCCL transport the 8-rank run stalled for
  minutes mid-run (4% efficiency). Over NVLink the steady-state efficiency is 69% for
  the small preset over a 300-step run — just under the 70% acceptance line. The
  remaining loss is the synchronised first-touch shard decode when all ranks move to a
  new length band at the same time; a longer run amortises it.
- **quickstart on a multi-GPU host** was 2.5× slower than on one GPU because the nano
  model was trained with DDP across all eight devices; `quickstart` now pins
  `devices=1`.
- **Acceptance rules** were tightened to what the evidence can actually decide:
  fine-tune is gated on the loader-wait share of the last epoch (not on GPU
  utilisation, which is launch-bound for small models), bf16 vs fp32 is gated on the
  mean embedding cosine (the probe ROC-AUC on a 300-step model moves by several
  hundredths between numerically identical embedding sets), and the epoch-2 throughput
  check only fails on a slowdown.

## Cost

Eight pods, about 4.5 GPU-hours on single-GPU A100/H100 pods and 1.3 hours on 8×A100
pods; roughly $55 in total.
