# pragmatiq — guide for contributors and AI coding assistants

pragmatiq is an independent open-source implementation by Dynamiq, inspired by
the PRAGMA paper (arXiv 2604.08649). It turns user histories made of timestamped
key–value events into embeddings for probes, LoRA fine-tuning, AML graph work,
and serving. It is GPU-first and CPU-complete: every path picks CUDA when a
device is visible and runs the same code in fp32 on a CPU when none is.

## Commands

```bash
pip install -e ".[dev]"            # editable install + test deps
pytest tests/ -x -q                # full suite
pytest tests/test_<module>.py -q   # one module
bash scripts/gates/gate_1.sh       # one acceptance gate (see the list below)
ruff check . && mypy pragmatiq     # lint + types
pragmatiq info                     # resolved device / precision / flash-attn / extras
```

Gates: `gate_1` (synthetic data) … `gate_8` (nano end-to-end + packaging),
`gate_9_contract` (public-API / CLI / serving contract), `gate_serve_slim`,
`gate_storage`, `gate_integrations`, `gate_10_byoc`;
`scripts/gates/run_full_validation.sh` runs them all and CI runs every one.
GPU validation lives in `scripts/gpuval/` (launched via `scripts/runpod_launch.py`).

Gate scripts honor `PRAGMATIQ_GATE_FULL=1` for full-scale runs (100k users,
8 cores); the default is CI scale (small N with throughput extrapolation).

## Non-negotiable rules

1. All logic lives in the library (`pragmatiq/`); `cli.py` (Typer) only parses
   args and calls `pragmatiq/api.py` functions.
1. Every randomized component takes an explicit seed; the same seed produces
   byte-identical output (CI-enforced for the generator).
1. Checkpoints store model + optimizers + LR scheduler + sampler position + RNG
   states, and embed the tokenizer hash + resolved config. `from_pretrained()`
   refuses to run with a mismatched tokenizer (clear error message).
1. Unseen keys/values at inference map to `[UNK]` with a logged warning — never a
   KeyError.
1. GPU-first, CPU-complete: never a CUDA-only path without a CPU branch and a
   test. CPU is fp32 and byte-stable; CUDA runs bf16 autocast at inference and
   bf16-mixed in training. `device="auto"` (and `PRAGMATIQ_DEVICE`) resolve the
   device in `pragmatiq.core.env`; use flash-attn varlen when available, else
   fall back to SDPA over segments built from `cu_seqlens`.
1. Write tests alongside each module; type-hint the public API; docstring every
   public function.
1. Public API (`pragmatiq/api.py`) exposes
   `synthesize / tokenize / pretrain / finetune / embed / probe`; notebooks use
   `PragmaModel.from_pretrained(run)` and `model.embed_records(list_of_dicts)`
   (plain dicts, no shard pipeline needed interactively).
1. Extend via `pragmatiq/registry.py` decorators — `@register_head(name)`,
   `@register_masker(name)`, `@register_value_encoder(name)` — configs reference
   components by name so engineers customize without forking.

Hyperparameters the paper does not specify are marked `# GUESS`, exposed in
config, and documented in the README. Keep comments and docstrings focused on
what the code does and the design reasoning behind it.

## Attribution (must appear in every README/doc)

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
