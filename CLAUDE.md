# pragmatiq — guide for contributors and AI coding assistants

pragmatiq is an independent open-source implementation by Dynamiq, inspired by
the PRAGMA paper (arXiv 2604.08649). It turns user histories made of timestamped
key–value events into embeddings for probes, LoRA fine-tuning, AML graph work,
explainability, and serving. It is built to run on CPU first; CUDA and flash-attn
are accelerations, not requirements.

## Commands

```bash
pip install -e ".[dev]"            # editable install + test deps
pytest tests/ -x -q                # full suite
pytest tests/test_<module>.py -q   # one module
bash scripts/gates/gate_1.sh       # an acceptance gate (gate_1 .. gate_8)
ruff check . && mypy pragmatiq     # lint + types
```

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
1. Everything runs on CPU (slow but correct). CUDA paths are accelerations, not
   requirements. Use flash-attn varlen if available, else fall back to SDPA with
   an attention mask built from `cu_seqlens`.
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
