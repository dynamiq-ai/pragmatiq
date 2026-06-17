#!/usr/bin/env python3
"""Fail if the docs site's facts.json has drifted from the code.

The documentation site renders its model-size table, config-knob tables, gate list, and
Python API reference from ``website/data/facts.json`` (emitted by ``scripts/docs_facts.py``).
This check re-derives those facts from the installed library and diffs them against the
committed JSON, so a change to a default, a model size, or an API signature that is not
reflected in the docs fails CI.

Usage:
    python scripts/docs_drift_check.py     # exit 0 if in sync, 1 (with a diff) otherwise
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

# docs_facts.py lives alongside this script; make it importable from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_facts import OUT, build_facts  # noqa: E402


def main() -> int:
    if not OUT.exists():
        print(f"ERROR: {OUT} is missing — run `python scripts/docs_facts.py`.", file=sys.stderr)
        return 1

    committed = json.dumps(json.loads(OUT.read_text()), indent=2, sort_keys=True)
    current = json.dumps(build_facts(), indent=2, sort_keys=True)
    if committed == current:
        print("docs facts in sync with the code.")
        return 0

    diff = difflib.unified_diff(
        committed.splitlines(), current.splitlines(),
        fromfile="committed website/data/facts.json", tofile="regenerated from code", lineterm="",
    )
    print("\n".join(diff))
    print(
        "\nERROR: website/data/facts.json is stale. Run `python scripts/docs_facts.py` and "
        "commit the result (and update any prose that quotes these numbers).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
