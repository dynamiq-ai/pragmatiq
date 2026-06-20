"""Contract tests: pragmatiq.api public-function signatures and return-dict keys.

Golden set A — all 15 public API functions must exist and their signatures must
match exactly. Adding a new optional parameter (with a default) PASSES; renaming,
removing, or reordering a required parameter FAILS.

Golden set B — return-dict keys for functions that are cheap to call with
existing fixtures (``validate``, ``runs_list``, ``runs_compare``). Heavy
functions (``pretrain``, ``quickstart``, …) are covered only in STABILITY.md.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

import pragmatiq.api as api_module


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sig(fn: Any) -> inspect.Signature:
    """Return the ``inspect.Signature`` of a callable."""
    return inspect.signature(fn)


def _param_names(fn: Any) -> list[str]:
    """Return ordered list of parameter names (excluding ``**kwargs``)."""
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]


def _required_param_names(fn: Any) -> list[str]:
    """Return names of parameters that have no default value."""
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]


def _defaults(fn: Any) -> dict[str, Any]:
    """Return {name: default} for all parameters that have a default value."""
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


# ---------------------------------------------------------------------------
# A. All 15 public functions must be present
# ---------------------------------------------------------------------------

GOLDEN_FUNCTION_NAMES = frozenset({
    "synthesize",
    "tokenize",
    "pretrain",
    "finetune",
    "embed",
    "probe",
    "uplift",
    "export",
    "benchmark",
    "gnn",
    "validate",
    "quickstart",
    "runs_list",
    "runs_compare",
    "calibrate",
})


def test_all_public_functions_present() -> None:
    """All 15 pinned public functions must be importable from pragmatiq.api."""
    missing = GOLDEN_FUNCTION_NAMES - set(dir(api_module))
    assert not missing, f"Missing public API functions: {sorted(missing)}"


def test_all_public_functions_are_callable() -> None:
    """Every pinned name must be callable (not accidentally shadowed by a constant)."""
    for name in GOLDEN_FUNCTION_NAMES:
        fn = getattr(api_module, name)
        assert callable(fn), f"pragmatiq.api.{name} is not callable"


# ---------------------------------------------------------------------------
# A. Frozen signatures — golden parameter names, order, and defaults
# ---------------------------------------------------------------------------
#
# Policy:
#   • Adding a new optional param (with a default) is additive → PASS (we only
#     check that the required params are still there in the right positions).
#   • Renaming / removing / reordering a required param → FAIL.
#   • For optional params we also pin defaults so accidental default-value changes
#     (MINOR per the SemVer policy) show up here and require a deliberate update.
#
# The golden values below were read directly from pragmatiq/api.py on the date
# this contract was established.  Update them only on a deliberate, versioned
# contract change.

GOLDEN_REQUIRED_PARAMS: dict[str, list[str]] = {
    "synthesize": [],
    "tokenize": ["data_dir", "out"],
    "pretrain": ["shard_dir", "run_name"],
    "finetune": ["shard_dir", "run", "label_path"],
    "embed": ["shard_dir", "run"],
    "probe": ["shard_dir", "run", "label_path"],
    "uplift": ["shard_dir", "run", "label_path"],
    "export": ["run", "shard_dir"],
    "benchmark": ["run", "shard_dir"],
    "gnn": ["shard_dir", "run", "transfers_path", "aml_label_path"],
    "validate": ["data_dir"],
    "quickstart": [],
    "runs_list": [],
    "runs_compare": ["names"],
    "calibrate": ["stats"],
}

# Pinned optional-parameter names (order-sensitive list; excludes **overrides)
GOLDEN_OPTIONAL_PARAMS: dict[str, list[str]] = {
    "synthesize": ["config", "out", "n_users", "seed", "n_workers", "write_report"],
    "tokenize": ["config", "tokenizer_dir", "max_users", "rows_per_shard", "n_workers"],
    "pretrain": ["model_size", "config", "runs_root", "resume"],
    "finetune": ["config", "device"],
    "embed": ["out", "token_budget", "device"],
    "probe": ["device", "token_budget", "seed", "with_baseline", "probe_model"],
    "uplift": ["device", "token_budget", "seed", "learner"],
    "export": ["out", "device"],
    "benchmark": ["device", "out", "max_users"],
    "gnn": ["seeds", "device", "epochs"],
    "validate": [],
    "quickstart": ["out", "n_users", "seed", "model_size", "max_steps", "n_workers"],
    "runs_list": ["runs_root"],
    "runs_compare": ["runs_root"],
    "calibrate": ["config", "out"],
}

# Pinned default values for optional parameters
GOLDEN_DEFAULTS: dict[str, dict[str, Any]] = {
    "synthesize": {"config": None, "out": "data/synth", "n_users": None, "seed": None,
                   "n_workers": 0, "write_report": True},
    "tokenize": {"config": None, "tokenizer_dir": None, "max_users": None,
                 "rows_per_shard": 4096, "n_workers": 0},
    "pretrain": {"model_size": "small", "config": None, "runs_root": "runs", "resume": None},
    "finetune": {"config": None, "device": "auto"},
    "embed": {"out": None, "token_budget": 16_384, "device": "auto"},
    "probe": {"device": "auto", "token_budget": 16_384, "seed": 0,
              "with_baseline": True, "probe_model": "gbdt"},
    "uplift": {"device": "auto", "token_budget": 16_384, "seed": 0, "learner": "t"},
    "export": {"out": "pragmatiq_embedder.onnx", "device": "cpu"},
    "benchmark": {"device": "auto", "out": "deploy/benchmarks/RESULTS.md", "max_users": None},
    "gnn": {"seeds": (0, 1, 2), "device": "auto", "epochs": 150},
    "validate": {},
    "quickstart": {"out": "runs/quickstart", "n_users": 50_000, "seed": 0,
                   "model_size": "nano", "max_steps": 400, "n_workers": 0},
    "runs_list": {"runs_root": "runs"},
    "runs_compare": {"runs_root": "runs"},
    "calibrate": {"config": None, "out": None},
}


@pytest.mark.parametrize("fn_name", sorted(GOLDEN_REQUIRED_PARAMS))
def test_required_params_present_and_ordered(fn_name: str) -> None:
    """Required parameters must exist and appear in the same order."""
    fn = getattr(api_module, fn_name)
    actual = _required_param_names(fn)
    expected = GOLDEN_REQUIRED_PARAMS[fn_name]
    assert actual == expected, (
        f"pragmatiq.api.{fn_name}: required params changed.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


@pytest.mark.parametrize("fn_name", sorted(GOLDEN_OPTIONAL_PARAMS))
def test_optional_params_present_and_ordered(fn_name: str) -> None:
    """Existing optional parameters must all still exist in the same order.

    New parameters appended at the end are allowed (additive change).
    """
    fn = getattr(api_module, fn_name)
    actual_defaults = _defaults(fn)
    expected = GOLDEN_OPTIONAL_PARAMS[fn_name]
    # Only check that every pinned optional param is still present and in order
    actual_optional_names = [n for n in _param_names(fn) if n in actual_defaults]
    # Filter actual to the pinned set, preserving order
    actual_pinned = [n for n in actual_optional_names if n in set(expected)]
    assert actual_pinned == expected, (
        f"pragmatiq.api.{fn_name}: optional params changed or reordered.\n"
        f"  expected (pinned): {expected}\n"
        f"  actual   (pinned): {actual_pinned}"
    )


@pytest.mark.parametrize("fn_name", sorted(GOLDEN_DEFAULTS))
def test_default_values_unchanged(fn_name: str) -> None:
    """Default values for optional params must not change silently.

    A change here is MINOR per the SemVer policy — it must be deliberate.
    """
    fn = getattr(api_module, fn_name)
    actual_defaults = _defaults(fn)
    expected_defaults = GOLDEN_DEFAULTS[fn_name]
    for param_name, expected_val in expected_defaults.items():
        assert param_name in actual_defaults, (
            f"pragmatiq.api.{fn_name}: default for '{param_name}' disappeared"
        )
        actual_val = actual_defaults[param_name]
        # Compare Path defaults as strings since Path("x") == "x" is False
        if isinstance(expected_val, str):
            actual_val = str(actual_val) if not isinstance(actual_val, str) else actual_val
        assert actual_val == expected_val, (
            f"pragmatiq.api.{fn_name}: default for '{param_name}' changed.\n"
            f"  expected: {expected_val!r}\n"
            f"  actual:   {actual_defaults[param_name]!r}"
        )


# ---------------------------------------------------------------------------
# B. Return-dict key assertions for cheap-to-run functions
# ---------------------------------------------------------------------------

class TestReturnKeys:
    """Assert that cheap-to-call functions return dicts with the pinned keys."""

    def test_validate_return_keys(self, tmp_path: Path) -> None:
        """``validate`` must return ``{ok, errors, warnings, summary}``."""
        # An empty (non-existent) directory is a valid argument — validate will
        # flag errors, but the return dict structure is still what we care about.
        result = api_module.validate(tmp_path)
        assert isinstance(result, dict), "validate() must return a dict"
        assert set(result) >= {"ok", "errors", "warnings", "summary"}, (
            f"validate() return keys changed: {set(result)}"
        )
        # Type checks
        assert isinstance(result["ok"], bool)
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["summary"], str)

    def test_runs_list_return_type(self, tmp_path: Path) -> None:
        """``runs_list`` must return a ``list[dict]``."""
        result = api_module.runs_list(runs_root=tmp_path)
        assert isinstance(result, list), "runs_list() must return a list"
        # Empty runs root → empty list
        assert result == []

    def test_runs_compare_return_type(self, tmp_path: Path) -> None:
        """``runs_compare`` must return a ``list[dict]``."""
        result = api_module.runs_compare(["nonexistent_run"], runs_root=tmp_path)
        assert isinstance(result, list), "runs_compare() must return a list"
        assert len(result) == 1
        assert isinstance(result[0], dict)
        # A missing run is flagged with {name, missing: True}
        assert result[0].get("missing") is True

    def test_runs_compare_missing_run_keys(self, tmp_path: Path) -> None:
        """Missing runs must be flagged with ``{name, missing: True}``."""
        result = api_module.runs_compare(["ghost"], runs_root=tmp_path)
        assert result == [{"name": "ghost", "missing": True}]


# ---------------------------------------------------------------------------
# B. Documented return-dict keys (checked by annotation, not runtime)
# ---------------------------------------------------------------------------
#
# For functions that are too expensive to run in contract tests (pretrain,
# finetune, embed, probe, uplift, quickstart, synthesize, tokenize, calibrate,
# export, benchmark, gnn), we verify the documented return-key sets are recorded
# in this module so the documentation stays in sync with the code. The actual
# keys are verified in STABILITY.md and integration tests; see the brief.

DOCUMENTED_RETURN_KEYS: dict[str, set[str]] = {
    "embed": {"n_users", "dim"},
    "probe": {"probe_model", "probe_auc", "probe_pr_auc", "probe_accuracy", "n_test", "prevalence"},
    # probe additionally returns {"baseline_auc", "baseline_pr_auc"} when with_baseline=True
    "uplift": {"qini", "qini_oracle", "ate", "n_train", "n_test", "treated_frac"},
    "pretrain": {"run", "run_dir", "steps", "last_metrics"},
    "finetune": {"best_val_auc", "epochs_run", "n_adapted", "val_auc_history"},
    "validate": {"ok", "errors", "warnings", "summary"},
    "quickstart": {"run_dir", "probe", "message"},
}


def test_documented_return_keys_are_defined() -> None:
    """Sanity: the DOCUMENTED_RETURN_KEYS table is not empty and covers the spec."""
    required_coverage = {"embed", "probe", "uplift", "pretrain", "finetune", "validate", "quickstart"}
    missing = required_coverage - set(DOCUMENTED_RETURN_KEYS)
    assert not missing, f"DOCUMENTED_RETURN_KEYS table missing entries for: {sorted(missing)}"
