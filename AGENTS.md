# pragmatiq — project memory

pragmatiq is an independent open-source implementation inspired by PRAGMA
(arXiv 2604.08649) by Dynamiq.
The ONLY spec is **docs/SPEC.md** — read it from disk; never trust chat context
over the file. Subagents must read docs/SPEC.md before reviewing anything.

## Current phase

> **current phase: 8 (polish). All gates 1-8 green at CI scale; gate-6 also
> confirmed at full scale on GPU (12k accounts × 3 seeds: a≈0.50, c≈0.85). The
> phase-6 AML ablation demonstrates relational recovery: money mules are ordinary
> accounts whose only signal is ring membership in the transfer graph, so an
> isolated embedding is at chance (a≈0.50) while a graph-aware model recovers them
> (c≈0.85). On these structurally-distinctive synthetic rings the signal is
> largely transfer-graph degree, so hand-crafted degree is a strong baseline and
> (b)>(c) is not claimed. Gate 8 is the CPU pipeline smoke test; gate 5 is the
> performance gate. See MODEL_CARD.md and notebooks/04.**

Phases & gates: see docs/SPEC.md. Gate scripts live in `scripts/gates/gate_N.sh`.
Never start phase N+1 with a red gate N.

## Public code hygiene (non-negotiable)

This repository is public, production-grade open source. Everything that ships in
git — code comments, docstrings, commit messages, README, MODEL_CARD, CHANGELOG,
notebooks, gate scripts — must read as a clean first version written by its
authors, not as a record of how it was fixed:

- Comments and docstrings explain WHAT the code does and its forward-looking design
  rationale. They never describe prior behavior, bug history, or any review/audit
  process — avoid "was X, now Y", "previously/old/used to", "fixed", "byte-identical
  to before", an AUC a removed bug scored, "the audit/verification found …", etc.
- Commit messages and the CHANGELOG describe v1 capabilities, not defect fixes.
- This is v1: there is no prior release. Do NOT add backwards-compatibility shims,
  legacy fallbacks, or "for older …" branches.
- Keep working notes, edit history, and learnings under `outputs/` (gitignored) —
  never in tracked files.

## Commands

```bash
pip install -e ".[dev]"            # editable install + test deps
pytest tests/ -x -q                # full suite
pytest tests/test_<module>.py -q   # one module
bash scripts/gates/gate_1.sh       # phase-1 acceptance gate (etc.)
ruff check . && mypy pragmatiq     # lint + types
```

Gate scripts honor `PRAGMATIQ_GATE_FULL=1` for full-scale runs (100k users,
8 cores); default is CI scale (small N, throughput extrapolation).

## Non-negotiable global rules

1. All logic lives in the library (`pragmatiq/`); `cli.py` (Typer) only parses args
   and calls `pragmatiq/api.py` functions.
1. Every randomized component takes an explicit seed; same seed → byte-identical
   output (CI-enforced for the generator).
1. Checkpoints store model + optimizer + LR scheduler + sampler position + RNG
   states, and embed the tokenizer hash + resolved config. `from_pretrained()`
   refuses to run with a mismatched tokenizer (clear error message).
1. Unseen keys/values at inference map to [UNK] with a logged warning — never a
   KeyError.
1. Everything must run on CPU (slow but correct). CUDA paths are accelerations,
   not requirements. Use flash-attn varlen if available, else fall back to SDPA
   with an attention mask built from cu_seqlens.
1. Write tests alongside each module; type-hint the public API; docstrings on
   every public function.
1. Public API layering: `pragmatiq/api.py` exposes `synthesize / tokenize / pretrain / finetune / embed / probe`; notebooks get
   `PragmaModel.from_pretrained(run)` and `model.embed_records(list_of_dicts)`
   (plain dicts, no shard pipeline needed interactively).
1. Extension points via `pragmatiq/registry.py` decorators —
   `@register_head(name)`, `@register_masker(name)`, `@register_value_encoder(name)`
   — configs reference components by name so engineers customize without forking.

## Attribution (must appear in every README/doc)

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
