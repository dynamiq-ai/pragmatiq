# Contributing to pragmatiq

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

The single source of truth for what this project does is
[`docs/SPEC.md`](docs/SPEC.md). Read the relevant phase section before
changing anything; reviews are conducted against the spec, not preferences.

## Setup

Python 3.11+ required.

```bash
pip install -e ".[dev]"        # editable install + test/lint/type deps
pip install -e ".[dev,gnn]"    # + torch-geometric for the AML GNN work
```

## Tests, lint, types

```bash
pytest tests/ -x -q                # full suite
pytest tests/test_<module>.py -q   # one module
ruff check . && mypy pragmatiq     # lint + types (CI-enforced)
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on Python 3.11 and
3.12, gates 1–4, and a nano CPU quickstart end to end. All of it must be green
on a PR.

## The phase + gate workflow

Development proceeds phase by phase (synthetic data → tokenizer → sharding →
model → training → AML GNN → serving → polish; see `docs/SPEC.md`). Each phase
has an acceptance gate:

```bash
bash scripts/gates/gate_1.sh   # ... gate_2.sh .. gate_6.sh
```

- Gates run at CI scale by default (small N, throughput extrapolation); set
  `PRAGMATIQ_GATE_FULL=1` for full-scale runs (e.g. 100k users on 8 cores for
  gate 1). They are plain bash — runnable outside Claude Code.
- **Never start phase N+1 with a red gate N.** If your change breaks an
  earlier phase's gate, fixing that comes first.
- The current phase is tracked in `CLAUDE.md`; update it when a gate turns
  green.

## Commit messages

Phase work is committed as:

```
phase-N: <summary> [gate-N green]
```

Only claim `[gate-N green]` if you actually ran the gate script and it passed;
note reviewer (spec-guardian) approval in the commit body for phase-final
commits. Deferred WARNs need a written justification in the commit message.

## The `.claude/` subagent workflow

This repo is developed with Claude Code, and the workflow is checked in so any
contributor's session inherits it:

- `.claude/agents/` — subagent definitions: **spec-guardian** (reviews diffs
  against `docs/SPEC.md`; read-only), **test-runner** (runs pytest/gate
  scripts and triages failures), **paper-fidelity-reviewer** (checks
  model/tokenizer/masking exactness at gates 2 and 4),
  **data-realism-analyst** (statistically validates generator output at
  gate 1), **docs-writer** (README/MODEL_CARD/docstrings; never edits library
  code).
- `.claude/commands/` — slash commands: `/phase N` starts a phase in plan mode
  from the spec on disk; `/gate N` runs the full gate (test-runner, then
  spec-guardian plus the phase's specialist reviewer in parallel, fix
  BLOCKERs, repeat until clean, then commit).

Subagents read `docs/SPEC.md` from disk — never trust chat context over the
file. If you contribute without Claude Code, the same standards apply: run the
gate script, self-review against the spec, keep the commit convention.

## Code style and ground rules

The non-negotiable rules live in `CLAUDE.md` and `docs/SPEC.md`; the ones that
shape most reviews:

- **No logic in `cli.py`.** The Typer CLI only parses arguments and calls
  `pragmatiq/api.py` functions. New functionality means a typed API function
  first, then (optionally) a thin CLI command.
- **Determinism.** Every randomized component takes an explicit seed; the same
  seed must produce byte-identical output (CI-enforced for the generator).
- **Checkpoints are complete.** Model + both optimizers + scheduler + sampler
  position + RNG states, plus the tokenizer hash and resolved config.
- **Never `KeyError` on unseen input.** Unseen keys/values map to `[UNK]` with
  a logged warning.
- **CPU always works.** CUDA/flash-attn paths are accelerations with an exact
  SDPA fallback built from `cu_seqlens` — keep the padding-equivalence test
  green.
- **Tests alongside modules** (`tests/test_<module>.py`), type hints on the
  public API, docstrings on every public function.
- **Extend via the registry**, not by forking: `@register_head(name)`,
  `@register_masker(name)`, `@register_value_encoder(name)` in
  `pragmatiq/registry.py`, referenced from configs by name.
- Formatting is ruff's job (line length 110, py311 target); mypy settings are
  in `pyproject.toml`. Hyperparameters the paper does not specify are marked
  `# GUESS`, exposed in config, and documented in the README GUESS table —
  update that table when you change a config default.

## Docs

Every README/doc carries the attribution line verbatim (see the top of this
file). Documentation describes what the code actually does — read the module
before writing about it, and keep the README quickstart runnable.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
