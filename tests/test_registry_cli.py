"""Tests for the component registry (rule 8) and the thin Typer CLI (rule 1)."""

from __future__ import annotations

import json

import pytest

from pragmatiq import registry


class TestRegistry:
    def test_register_and_get(self) -> None:
        reg = registry.Registry("widget")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo
        assert "foo" in reg
        assert reg.names() == ["foo"]

    def test_unknown_raises_with_known_names(self) -> None:
        reg = registry.Registry("widget")
        reg.register("a")(type("A", (), {}))
        with pytest.raises(KeyError, match="registered: a"):
            reg.get("missing")

    def test_duplicate_name_rejected(self) -> None:
        reg = registry.Registry("widget")
        reg.register("dup")(type("A", (), {}))
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup")(type("B", (), {}))

    def test_reregister_same_class_ok(self) -> None:
        reg = registry.Registry("widget")
        cls = type("A", (), {})
        reg.register("x")(cls)
        reg.register("x")(cls)  # idempotent
        assert reg.get("x") is cls

    def test_builtin_components_registered(self) -> None:
        # importing the modules registers the built-in components by name
        import pragmatiq.data.tokenizer  # noqa: F401
        import pragmatiq.models.heads  # noqa: F401
        import pragmatiq.training.masking  # noqa: F401

        assert registry.get_value_encoder("percentile_binner") is not None
        assert registry.get_head("mlm") is not None
        assert registry.get_head("classification") is not None
        assert registry.get_masker("pragma") is not None


class TestCLI:
    """The CLI must only parse args and call api.py; here we smoke the wiring."""

    def _runner(self):
        from typer.testing import CliRunner

        from pragmatiq.cli import app

        # Keep stderr (progress/log lines) out of the JSON we parse from
        # stdout. click < 8.2 merges the streams unless told otherwise; the
        # kwarg is gone in click >= 8.2 where streams are always separate.
        try:
            return CliRunner(mix_stderr=False), app
        except TypeError:
            return CliRunner(), app

    def test_help_lists_commands(self) -> None:
        runner, app = self._runner()
        res = runner.invoke(app, ["--help"])
        assert res.exit_code == 0
        assert "synth" in res.output and "runs" in res.output

    def test_synth_generate_invokes_api(self, tmp_path) -> None:
        runner, app = self._runner()
        out = tmp_path / "ds"
        res = runner.invoke(app, [
            "synth", "generate", "--out", str(out), "--n-users", "40",
            "--seed", "1", "--no-report",
        ])
        assert res.exit_code == 0, res.output
        manifest = json.loads(res.stdout)
        assert manifest["n_users"] == 40
        assert (out / "events.parquet").exists()

    def test_runs_list_empty(self, tmp_path) -> None:
        runner, app = self._runner()
        res = runner.invoke(app, ["runs", "list", "--runs-root", str(tmp_path)])
        assert res.exit_code == 0
        assert json.loads(res.stdout) == []

    def test_runs_compare_missing_flagged(self, tmp_path) -> None:
        runner, app = self._runner()
        res = runner.invoke(app, ["runs", "compare", "ghost", "--runs-root", str(tmp_path)])
        assert res.exit_code == 0
        out = json.loads(res.stdout)
        assert out == [{"name": "ghost", "missing": True}]
