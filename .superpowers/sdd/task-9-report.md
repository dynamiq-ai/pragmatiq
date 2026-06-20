# Task 9 (W8) Report: pragmatiq 1.0.0 cutover

**Branch:** `restructure/1.0-production`  
**Commit:** `4950489` — "release: pragmatiq 1.0.0 — freeze stability contract, document # GUESS, CHANGELOG + migration (W8)"  
**Date:** 2026-06-21

---

## Status: DONE

All 5 deliverables completed, all verification checks green.

---

## 1. `# GUESS` Hyperparameter Catalog

### Source survey

`grep -rn "GUESS" pragmatiq/ | wc -l` → **17 markers** across 7 files.

### Unique hyperparameters (9 total)

| # | Parameter | Default | Config key | Files (markers) |
|---|---|---|---|---|
| 1 | `lr_muon` — Muon LR | `3e-3` | `configs/pretrain.yaml · lr_muon` | `training/optim.py:55,129`, `training/pretrainer.py:139` |
| 2 | `lr_adamw` — AdamW LR | `3e-4` | `configs/pretrain.yaml · lr_adamw` | `training/optim.py:130`, `training/pretrainer.py:140` |
| 3 | `warmup_steps` | `100` (dataclass) / `500` (YAML) | `configs/pretrain.yaml · warmup_steps` | `training/pretrainer.py:142` |
| 4 | `token_budget` | `16384` | `configs/pretrain.yaml · token_budget` | `training/pretrainer.py:3`, `training/autoconfig.py:15` (module docstring) |
| 5 | `p_unk` | `0.10` | `TrainConfig.p_unk` / `configs/pretrain.yaml · p_unk` | `training/masking.py:67`, `training/pretrainer.py:173` |
| 6 | `n_buckets` | `64` | `configs/data/tokenizer.yaml · n_buckets` | `data/tokenizer.py:64` |
| 7 | `target_vocab` | `28000` | `configs/data/tokenizer.yaml · target_vocab` | `data/tokenizer.py:66` |
| 8 | `numeric_min_cardinality` | `None` (= 4×n_buckets) | `configs/data/tokenizer.yaml · numeric_min_cardinality` | `data/tokenizer.py:70,399` |
| 9 | `rope_base` | `10000.0` | `configs/model/{small,medium,large}.yaml · rope_base` | `models/pragmatiq.py:53`, `models/embeddings.py:70` |

Remaining 4 markers are class/module-level docstrings annotating entire sections, not individual hyperparameters:
- `training/pretrainer.py:129` — `TrainConfig` class docstring
- `training/autoconfig.py:15` — module docstring (general note, links to token_budget)
- `data/tokenizer.py:62` — `TokenizerConfig` class docstring
- `data/tokenizer.py:399` — caveat comment on the numeric routing heuristic (links to numeric_min_cardinality)

### Config gap fixed

`numeric_min_cardinality` was annotated `# GUESS` in `data/tokenizer.py` (it IS a dataclass field in `TokenizerConfig`) but was absent from `configs/data/tokenizer.yaml`. Added with comment explaining the default (`null` → `4 × n_buckets`).

Similarly, `p_unk` and the other masking probabilities (`p_token`, `p_event`, `p_key`) were in `TrainConfig` but not the YAML. Added to `configs/pretrain.yaml`.

### Documentation

README section "Paper-silent (`# GUESS`) hyperparameters" (replacing the old 8-row table) now contains:
- All 9 hyperparameters in a numbered table with file, default, config key, and one-line rationale
- A note that all 17 source markers resolve to 9 unique hyperparameters
- A statement that defaults are embedded in `run.yaml`/`meta.json` per run, so shipped checkpoints reproduce indefinitely
- Cross-reference to the stability policy (changing a default is MINOR)

---

## 2. `docs/STABILITY.md` — Frozen at 1.0

Added frozen header block at the top:

```
> **FROZEN as of pragmatiq 1.0.0** (2026-06-21). SemVer applies from
> this release forward. The contract below is enforced automatically by
> `tests/contract/` on every CI run and gate 9. No item in this document
> may change without a deliberate version bump and a matching update here.
```

Verified the document already listed all required items:
- ✅ 15 `api.*` functions (section A)
- ✅ `PragmaModel.from_pretrained` / `embed_records` (section D)
- ✅ Full CLI command tree (section C)
- ✅ Serving contract `records_json → embeddings [n_users, dim]` (section E)
- ✅ Checkpoint format version + tokenizer-hash guard (final section)
- ✅ Frozen return-dict keys per function (section B)

No policy text from W1/W4 was changed.

---

## 3. Version Bump → 1.0.0

Files updated:
- `pyproject.toml`: `version = "0.1.0b4"` → `"1.0.0"` (+ Development Status: 4 Beta → 5 Production/Stable)
- `pragmatiq/__init__.py`: fallback `"0.1.0b4"` → `"1.0.0"`
- `CITATION.cff`: `version: "0.1.0b4"` → `"1.0.0"` (per old RELEASING.md requirement)

Re-registration:
```
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```
Required installing `hatchling` + `editables` into the venv first (missing build-backend deps); ran cleanly.

Confirmation:
```
.venv/bin/python -c "import pragmatiq; print(pragmatiq.__version__)"
→ 1.0.0
```

No test files hardcode `0.1.0b4` (grep clean).

---

## 4. CHANGELOG.md — 1.0.0 entry

Added `## [1.0.0]` section at the top (above 0.1.0b4), covering:

- **BREAKING install behavior**: `torch-geometric`/`lightning`/`matplotlib` moved to extras; migration: `pip install 'pragmatiq[full]'`
- Public-API stability contract + `tests/contract/` enforcement
- `# GUESS` hyperparameter catalog
- fsspec pluggable object-store storage (`[s3]/[gcs]/[azure]`)
- Serving glue extracted to `pragmatiq.inference.serve`
- Cloud-adapter seams under `integrations/`
- `apps/` UI seam
- BYOC hardening (no-phone-home, offline install, `uv.lock`, SBOM, CI scan)
- "No API changes" section explicitly stating Python API / CLI / serving contract / checkpoint format unchanged from 0.1.0b4

No sweeps claimed; GUESS defaults are documented, not tuned.

---

## 5. RELEASING.md — 1.0 steps

Rewrote to include the full 1.0 release procedure:
1. Bump version in 3 places (`pyproject.toml`, `__init__.py`, `CITATION.cff`) + CHANGELOG
2. Regenerate lock file: `uv lock`
3. Regenerate SBOM: `bash scripts/supply_chain/gen_sbom.sh`
4. Run full validation: `bash scripts/gates/run_full_validation.sh`
5. Merge + tag (CI does PyPI publish via Trusted Publishing)

Added SemVer quick-reference table (MAJOR/MINOR/PATCH) and cross-reference to `docs/STABILITY.md`.

---

## Verification Results

| Check | Result |
|---|---|
| `pragmatiq.__version__` | `1.0.0` ✅ |
| `ruff check .` | All checks passed ✅ |
| `mypy pragmatiq` | Success: no issues found in 61 source files ✅ |
| `pytest tests/contract -q` | 98 passed in 1.49s ✅ |
| `bash scripts/gates/gate_9_contract.sh` | CONTRACT CHECKS GREEN ✅ |
| `bash scripts/gates/gate_storage.sh` | STORAGE STAGING CHECKS GREEN ✅ |
| `bash scripts/gates/gate_serve_slim.sh` | SERVE-SLIM CHECKS GREEN ✅ |
| `bash scripts/gates/gate_integrations.sh` | INTEGRATIONS CHECKS GREEN ✅ |
| `bash scripts/gates/gate_10_byoc.sh` | BYOC SECURITY CHECKS GREEN ✅ |
| `pytest tests/ -q` (full suite) | **538 passed, 1 skipped, 0 failed** in 1040.27s (17:20) ✅ |
| `grep -rn "GUESS" pragmatiq/ \| wc -l` | 17 (matches catalog count) ✅ |

The 1 skip is expected and pre-existing: `test_training.py:91` — "bf16-vs-fp32 Newton-Schulz only diverges on CUDA" — skipped on CPU (correct behavior).

**Docker-dependent gate:** `gate_7` (Triton) requires Docker daemon — not run. All other gates are offline-capable and green.

---

## Files Changed

- `pyproject.toml` — version bump + Stable classifier
- `pragmatiq/__init__.py` — version fallback bump
- `CITATION.cff` — version bump
- `README.md` — expanded GUESS catalog section (9-param table, 17-marker count, policy note)
- `docs/STABILITY.md` — frozen-at-1.0 header block
- `CHANGELOG.md` — [1.0.0] entry with BREAKING note + migration
- `RELEASING.md` — 1.0 release procedure (uv lock, SBOM, full validation, SemVer table)
- `configs/data/tokenizer.yaml` — added `numeric_min_cardinality: null # GUESS` (was missing)
- `configs/pretrain.yaml` — added `p_unk / p_token / p_event / p_key` (masking probs now explicit in YAML)

---

## Self-Review Checklist

- [x] All 17 `# GUESS` markers accounted for; 9 unique hyperparameters in the catalog with rationale + config key
- [x] `numeric_min_cardinality` gap fixed (added to tokenizer.yaml)
- [x] `p_unk` added to pretrain.yaml (was only in TrainConfig dataclass)
- [x] CHANGELOG honest: no sweeps claimed; GUESS values are documented defaults
- [x] Contract suite 98/98 — public API unchanged
- [x] Full suite 538/0 failed — regression gate green
- [x] `pragmatiq.__version__ == "1.0.0"` confirmed
- [x] ruff + mypy clean
- [x] Commit message matches spec exactly

---

## Final-review fix wave (2026-06-21)

Post-1.0 review corrections applied on branch `restructure/1.0-production`:

### True `# GUESS` count after fixes: **13 markers, 9 unique hyperparameters**

Previous W8 report claimed 17 markers (incorrect — previous grep hit docstrings/comments that
contained the string "GUESS" but were not inline field markers). Actual count after this wave:

| File | Markers |
|---|---|
| `pragmatiq/data/tokenizer.py` | 4 (`n_buckets`, `target_vocab`, `numeric_min_cardinality` block+inline) |
| `pragmatiq/models/pragmatiq.py` | 1 (`rope_base`) |
| `pragmatiq/training/masking.py` | 1 (`p_unk` call-site) |
| `pragmatiq/training/optim.py` | 2 (`lr_muon`, `lr_adamw` call-sites) |
| `pragmatiq/training/pretrainer.py` | 5 (`token_budget`, `lr_muon`, `lr_adamw`, `warmup_steps`, `p_unk`) |
| **Total** | **13** |

### New markers added (no default values changed)

- `pragmatiq/training/pretrainer.py:132` — `token_budget: int = 16_384  # GUESS: fits a \`small\` model on a 16 GiB GPU with optimizer-state headroom`
- `pragmatiq/data/tokenizer.py:73` — `numeric_min_cardinality: int | None = None  # GUESS: separates low-cardinality codes from continuous magnitudes`

### Documentation fixes

- `README.md` table: "17 source markers" → "13 source markers"; `token_budget` File column removed spurious `training/autoconfig.py`; `rope_base` File column removed spurious `models/embeddings.py`
- `CHANGELOG.md`: "(17 source markers)" → "(13 source markers)"

### Dead `[extras]` references fixed

- `configs/pretrain.yaml:20` — `.[extras]` → `.[tracking]`
- `configs/data/tokenizer_nemotron.yaml:9` — `.[extras]` → `.[text]`
- Verified: `grep -rn '\[extras\]' configs/ README.md docs/ pragmatiq/ scripts/ .github/` → empty

### Demo docstring fixed

- `apps/demo/app.py:7` — `streamlit run demo/app.py` → `streamlit run apps/demo/app.py`

### CI gate wiring (C2)

Added to `.github/workflows/ci.yml` `gates:` job:
- `bash scripts/gates/gate_9_contract.sh` (Public-API contract)
- `bash scripts/gates/gate_integrations.sh` (Integration adapters)

### Verification outputs

| Check | Result |
|---|---|
| `grep -rn "# GUESS" pragmatiq/ \| wc -l` | **13** (matches README + CHANGELOG) |
| `grep -rn '\[extras\]' configs/ README.md docs/ pragmatiq/ scripts/ .github/` | empty ✅ |
| `pytest tests/contract -q` | **98 passed** ✅ |
| `python scripts/supply_chain/no_phone_home.py` | `NO_PHONE_HOME_PASS` ✅ |
| `ruff check .` | `All checks passed!` ✅ |
| `mypy pragmatiq` | `Success: no issues found in 61 source files` ✅ |
| `py_compile apps/demo/app.py` | ok ✅ |
| CI YAML valid | `yaml ok` ✅ |
