# Task 1 Report: Public-API Stability Contract Harness

**Date**: 2026-06-20
**Branch**: restructure/1.0-production
**Status**: DONE

---

## What Was Built

Six deliverables, all additive (no production code touched):

1. **`tests/contract/__init__.py`** — Package marker with module-level docstring.

2. **`tests/contract/test_public_api.py`** — 45 tests covering:
   - All 15 public API functions are present and callable (A).
   - Required params for each function: names, order (parametrized; 15 tests).
   - Optional params for each function: names, order (parametrized; 15 tests).
   - Default values for optional params (parametrized; 15 tests).
   - Return-dict key assertions for `validate`, `runs_list`, `runs_compare` (cheap).
   - `DOCUMENTED_RETURN_KEYS` table for all expensive functions (B).

3. **`tests/contract/test_cli_commands.py`** — 23 tests covering:
   - All 15 command paths present in the live Typer app.
   - Each command's parameter names and order (parametrized; 15 tests).
   - Root callback `--verbose/--quiet` param.
   - CLI smoke tests (help exits 0, synth/runs sub-apps render).
   - `_walk_typer_app` helper that introspects the live Typer app.

4. **`tests/contract/test_model_api.py`** — 15 tests covering:
   - `PragmaModel.from_pretrained` importability, required/optional params, defaults.
   - `PragmaModel.embed_records` importability, required params, return annotation.
   - Module-level import sanity.

5. **`docs/STABILITY.md`** — Full frozen surface document with:
   - SemVer policy (verbatim from brief).
   - All 15 API function signatures.
   - Return-dict key table (A, B, C, D).
   - CLI command tree with parameter names.
   - Model API signatures.
   - Checkpoint format contract.
   - TODO for W4 serving contract (one-line note).

6. **`scripts/gates/gate_9_contract.sh`** — Runs `pytest tests/contract -q`, follows `_env.sh` + gate_*.sh conventions exactly.

---

## Implementation Approach

- Used `inspect.signature` for all signature pinning (required, optional, defaults).
- Used `_walk_typer_app()` helper that introspects `app.registered_commands` and `app.registered_groups` recursively — no subprocess, no CLI invocation needed for param pinning.
- Golden values were read directly from `pragmatiq/api.py`, `pragmatiq/cli.py`, and `pragmatiq/models/pragmatiq.py`.
- Additive-change semantics: tests filter actual params to the pinned set before comparing, so new optional params appended at the end pass cleanly.
- Default comparison handles Path vs str: string defaults are compared as `str(actual)` to handle `Path("runs") != "runs"`.

---

## Safety Net Demonstration

Temporarily monkey-patched `api_module.embed` with a renamed param (`tok_budget` instead of `token_budget`) to confirm the contract test would catch it:

```
FAILURE DETECTED (expected): ['out', 'token_budget', 'device']
Actual pinned params: ['out', 'device']
Contract test would FAIL - safety net works!
Reverted rename - tests pass again
```

---

## Test Results

- **Contract suite**: `pytest tests/contract -q` → **83 passed in 2.19s**
- **Gate 9**: `bash scripts/gates/gate_9_contract.sh` → **CONTRACT CHECKS GREEN**
- **Full suite** (Python 3.10 env; pre-existing failures unrelated to this task):
  - Pre-existing: `datetime.UTC` not in Python 3.10 (`test_naive_datetime_warns_once`)
  - Pre-existing: scipy numpy ABI conflict (`test_generator_signal.py` collection error)
  - Pre-existing: 37 other failures in `test_tokenize_parallel`, `test_tokenizer`, `test_training` related to the Python 3.10 environment (pragmatiq requires ≥3.11)
  - Contract tests: all 83 pass in both Python 3.10 and pass cleanly (no 3.11 features used in contract code)

---

## Files Created

| File | Purpose |
|---|---|
| `tests/contract/__init__.py` | Package marker |
| `tests/contract/test_public_api.py` | API signature + return-key contracts |
| `tests/contract/test_cli_commands.py` | CLI command path + param-name contracts |
| `tests/contract/test_model_api.py` | PragmaModel.from_pretrained/embed_records contracts |
| `docs/STABILITY.md` | Frozen surface document + SemVer policy |
| `scripts/gates/gate_9_contract.sh` | Gate script |

---

## Self-Review

- **Completeness vs brief**: All 6 deliverables implemented. All golden sets match what's in the production files. SemVer policy included verbatim. W4 TODO included.
- **YAGNI**: No extra functionality beyond the spec. No production code touched.
- **Output pristine**: `pytest tests/contract -q` produces only passing dots + summary, no warnings.
- **Additive semantics verified**: Optional params can be extended without breaking; the filtering logic (`actual_pinned = [n for n in actual_optional_names if n in set(expected)]`) handles this correctly.
- **Determinism**: All tests are pure introspection + cheap API calls; no network, no training runs.
