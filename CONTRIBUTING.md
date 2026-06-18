# Contributing to pragmatiq

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

Thanks for your interest in contributing. The [README](README.md) is the best
overview of what the project does; [`docs/architecture.md`](docs/architecture.md)
explains how the pieces fit together. Please read the relevant module before
changing it — documentation and reviews focus on what the code actually does.

## Setup

Python 3.11+ required.

```bash
pip install -e ".[dev]"        # editable install + test/lint/type deps
```

The full pipeline — including the gradient-boosting probe and the AML transfer-graph
GraphSAGE work — installs with the line above; the optional extras (`serve`, `demo`,
`extras`, `full`) add focused tooling.

## Tests, lint, types

```bash
pytest tests/ -x -q                # full suite
pytest tests/test_<module>.py -q   # one module
ruff check . && mypy pragmatiq     # lint + types (CI-enforced)
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on Python 3.11 and
3.12, the acceptance gates, and a nano CPU quickstart end to end. All of it must
be green on a PR.

## Acceptance gates

`scripts/gates/gate_1.sh` … `gate_8.sh` are end-to-end integration checks for the
synthetic generator, tokenizer, sharding, model, training, AML GNN, serving, and
the packaged pipeline. They are plain bash and run locally:

```bash
bash scripts/gates/gate_1.sh
```

Gates run at CI scale by default (small N with throughput extrapolation); set
`PRAGMATIQ_GATE_FULL=1` for full-scale runs (for example 100k users on 8 cores
for `gate_1.sh`). If a change affects one of these areas, run the corresponding
gate script before opening a PR.

## Commit messages and PRs

Use a concise, imperative subject line that describes the capability or change.
Keep each PR focused, include tests for new behavior, and make sure the full
suite, ruff, mypy, and any relevant gate pass before requesting review.

Open pull requests against **`develop`** (the default branch); **`main`** is the
release branch. Releases are cut by a `develop` → `main` pull request that bumps
the version, which tags the commit and publishes to PyPI automatically — see
[RELEASING.md](RELEASING.md).

## Code style and ground rules

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
