"""Contract tests: CLI command paths and parameter names.

Pins the frozen CLI surface so that internal restructuring cannot silently
rename, remove, or reorder commands or their parameters.

Policy (per STABILITY.md):
- MAJOR: remove a command path, rename a param, remove a param.
- MINOR: add a new command, add a new optional param.
- PATCH: help text changes, internal behaviour.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helper: walk a Typer app and collect (command_path, [param_names])
# ---------------------------------------------------------------------------

def _walk_typer_app(
    typer_app: Any,
    prefix: str = "",
) -> dict[str, list[str]]:
    """Return ``{command_path: [param_names]}`` for every leaf command.

    ``ctx``-typed parameters (``typer.Context``) are excluded because they are
    internal wiring, not CLI params.
    """
    result: dict[str, list[str]] = {}

    for cmd in typer_app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "")
        path = f"{prefix} {name}".strip() if prefix else name
        if cmd.callback:
            sig = inspect.signature(cmd.callback)
            params = [
                pname
                for pname, p in sig.parameters.items()
                if pname != "ctx"
            ]
            result[path] = params

    for group in typer_app.registered_groups:
        sub_prefix = f"{prefix} {group.name}".strip() if prefix else group.name
        result.update(_walk_typer_app(group.typer_instance, prefix=sub_prefix))

    return result


# ---------------------------------------------------------------------------
# C. Golden command tree
# ---------------------------------------------------------------------------
#
# Top-level commands: tokenize, pretrain, probe, uplift, finetune, embed,
# quickstart, validate, export, benchmark, gnn.
# Sub-apps: synth generate, synth calibrate, runs list, runs compare.
# Root callback option: --verbose/--quiet (param name "verbose").
#
# Parameter names are the Python callback parameter names (which Typer converts
# to --kebab-case CLI options, but the contract pins the Python identifiers that
# the callbacks accept, because those are what code inspection sees).

GOLDEN_COMMAND_PATHS: frozenset[str] = frozenset({
    "tokenize",
    "pretrain",
    "probe",
    "uplift",
    "finetune",
    "embed",
    "quickstart",
    "validate",
    "export",
    "benchmark",
    "gnn",
    "synth generate",
    "synth calibrate",
    "runs list",
    "runs compare",
})

GOLDEN_COMMAND_PARAMS: dict[str, list[str]] = {
    "tokenize": ["data_dir", "out", "config", "tokenizer_dir", "max_users", "n_workers"],
    "pretrain": ["shard_dir", "run_name", "model_size", "config", "runs_root", "resume", "wandb"],
    "probe": ["shard_dir", "run", "label", "device", "probe_model", "seed"],
    "uplift": ["shard_dir", "run", "label", "device", "learner"],
    "finetune": ["shard_dir", "run", "label", "config", "device"],
    "embed": ["shard_dir", "run", "out", "device"],
    "quickstart": ["out", "n_users", "model_size", "max_steps", "n_workers"],
    "validate": ["data_dir"],
    "export": ["run", "shard_dir", "out"],
    "benchmark": ["run", "shard_dir", "device", "out"],
    "gnn": ["shard_dir", "run", "transfers", "aml_label", "seeds", "epochs", "device"],
    "synth generate": ["out", "config", "n_users", "seed", "n_workers", "report"],
    "synth calibrate": ["stats", "config", "out"],
    "runs list": ["runs_root"],
    "runs compare": ["names", "runs_root"],
}


@pytest.fixture(scope="module")
def cli_command_map() -> dict[str, list[str]]:
    """Return the live command map from the Typer app."""
    from pragmatiq.cli import app
    return _walk_typer_app(app)


def test_all_command_paths_present(cli_command_map: dict[str, list[str]]) -> None:
    """Every pinned command path must exist in the live CLI app."""
    missing = GOLDEN_COMMAND_PATHS - set(cli_command_map)
    assert not missing, (
        f"CLI command paths disappeared: {sorted(missing)}"
    )


def test_no_accidental_command_removal(cli_command_map: dict[str, list[str]]) -> None:
    """The live app must not contain extra unknown command paths beyond the golden set."""
    extra = set(cli_command_map) - GOLDEN_COMMAND_PATHS
    assert not extra, f"Added unexpected CLI commands: {sorted(extra)}"


@pytest.mark.parametrize("cmd_path", sorted(GOLDEN_COMMAND_PARAMS))
def test_command_param_names(cmd_path: str, cli_command_map: dict[str, list[str]]) -> None:
    """Each command's parameter list must match the golden set (order-sensitive).

    Additive changes (new params appended) are not tested here — only the
    pinned params are checked to still be present and in the same order.
    """
    assert cmd_path in cli_command_map, f"CLI command path '{cmd_path}' not found"
    actual = cli_command_map[cmd_path]
    expected = GOLDEN_COMMAND_PARAMS[cmd_path]
    # Filter actual to only the pinned params (preserving order)
    actual_pinned = [p for p in actual if p in set(expected)]
    assert actual_pinned == expected, (
        f"CLI '{cmd_path}': param names/order changed.\n"
        f"  expected (pinned): {expected}\n"
        f"  actual   (pinned): {actual_pinned}"
    )


def test_root_callback_verbose_option() -> None:
    """The root --verbose/--quiet option must exist on the app callback with default True."""
    from pragmatiq.cli import _setup
    sig = inspect.signature(_setup)
    assert "verbose" in sig.parameters, (
        "Root callback lost the 'verbose' parameter (--verbose/--quiet option)"
    )
    verbose_param = sig.parameters["verbose"]
    # The default is wrapped in a typer.OptionInfo object; extract the actual value
    option_info = verbose_param.default
    assert hasattr(option_info, "default"), (
        "Root callback 'verbose' parameter missing OptionInfo.default"
    )
    assert option_info.default is True, (
        f"Root callback 'verbose' parameter default must be True, got {option_info.default}"
    )


# ---------------------------------------------------------------------------
# Smoke: CLI help exits 0 and lists both sub-apps
# ---------------------------------------------------------------------------

class TestCLISmoke:
    """Fast smoke tests using Typer's CliRunner (no subprocess)."""

    @staticmethod
    def _runner():
        """Return a (CliRunner, app) pair, suppressing the mix_stderr kwarg if unsupported."""
        from typer.testing import CliRunner

        from pragmatiq.cli import app
        try:
            return CliRunner(mix_stderr=False), app
        except TypeError:
            return CliRunner(), app

    def test_help_exits_zero(self) -> None:
        """``pragmatiq --help`` must exit 0 and mention both sub-apps."""
        runner, app = self._runner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "synth" in result.output
        assert "runs" in result.output

    def test_synth_help_exits_zero(self) -> None:
        """``pragmatiq synth --help`` must exit 0."""
        runner, app = self._runner()
        result = runner.invoke(app, ["synth", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "calibrate" in result.output

    def test_runs_help_exits_zero(self) -> None:
        """``pragmatiq runs --help`` must exit 0."""
        runner, app = self._runner()
        result = runner.invoke(app, ["runs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "compare" in result.output
