"""AST-based import-boundary test for apps/.

Statically asserts that no Python file under apps/ imports the training stack.
No Streamlit launch is required; the check is pure AST parsing.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Extend this set to forbid additional prefixes.
_FORBIDDEN_PREFIXES: frozenset[str] = frozenset({"pragmatiq.training"})

_APPS_DIR = Path(__file__).parent.parent.parent / "apps"


def _imports_in_file(path: Path) -> list[str]:
    """Return all top-level module names imported by *path* (AST walk)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_demo_app_exists_and_parses() -> None:
    """apps/demo/app.py must exist and be valid Python (cheap integrity check)."""
    demo = _APPS_DIR / "demo" / "app.py"
    assert demo.is_file(), f"Missing demo app: {demo}"
    # ast.parse raises SyntaxError on invalid Python
    ast.parse(demo.read_text(encoding="utf-8"), filename=str(demo))


def test_apps_no_training_import() -> None:
    """No file under apps/ may import pragmatiq.training or any sub-module."""
    violations: list[str] = []
    py_files = list(_APPS_DIR.rglob("*.py"))
    assert py_files, f"No Python files found under {_APPS_DIR}"

    for path in py_files:
        for module in _imports_in_file(path):
            for prefix in _FORBIDDEN_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(f"{path.relative_to(_APPS_DIR.parent)}: imports {module!r}")

    assert not violations, (
        "apps/ must not import the training stack "
        f"(forbidden prefixes: {_FORBIDDEN_PREFIXES}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )
