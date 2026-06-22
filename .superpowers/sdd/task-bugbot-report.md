# Bugbot PR #10 — Fix Report

**Status:** All 4 issues confirmed real and fixed. Tests green. Ruff clean. Mypy clean.

## Issues

| # | Severity | File | Real? | Fix |
|---|----------|------|-------|-----|
| 1 | HIGH | `integrations/nebius/_adapter.py` | Yes | Fixed `_BATCH_JOB_TMPL` command: replaced `--run-dir`/`--output` with positional `shard_dir_mount`, `--run`, `--out`; added `shard_dir_mount` to `common_kw`; fixed `manifest()` `command` list the same way |
| 2 | MEDIUM | `pragmatiq/api.py` L229 | Yes | Added `and resume == "auto"` guard so fresh remote pretrain never stages an existing remote run dir |
| 3 | MEDIUM | `pragmatiq/core/config.py` | Yes | `load_yaml` now calls `storage.read_text()` + `OmegaConf.load(StringIO(...))` for remote URLs; local paths unchanged |
| 4 | LOW | `deploy/triton/.../model.py` L51 | Yes | `finalize()` now calls `self.runtime.close()` before clearing the reference |

## Tests added

- `test_integrations_stubs.py::TestNebiusBatchEmbedCliFlags` — 4 assertions for issue 1 (both YAML and manifest)
- `test_storage_pipeline.py::test_remote_config_yaml_loaded_by_load_yaml` — issue 3
- `test_storage_pipeline.py::test_fresh_remote_pretrain_does_not_pull_existing_run` — issue 2

## Test results

- `test_integrations_stubs.py` + sagemaker + databricks: **72 passed**
- `test_storage_pipeline.py` + storage + contract: **164 passed**
- `test_inference.py`: **19 passed** (Triton finalize tested by existing `TestTritonServingContract.test_records_json_request_returns_embeddings`)
- `ruff check .`: clean
- `mypy pragmatiq`: clean (61 files)

## Concerns

None. All fixes are minimal, targeted, and backward-compatible.

## Report path
`/Users/vitalii.duk/dynamiq/claude-code/pragmatiq/.superpowers/sdd/task-bugbot-report.md`
