#!/usr/bin/env python3
"""Emit canonical pragmatiq facts for the documentation site.

The docs site (``website/``) renders its model-size table, config-knob tables, gate
list, and Python API reference from ``website/data/facts.json`` so those numbers cannot
drift from the code. This script is the single source: it introspects the library and
writes that JSON. ``scripts/docs_drift_check.py`` re-runs it and fails CI if the
committed JSON is stale.

Usage:
    python scripts/docs_facts.py            # write website/data/facts.json
    python scripts/docs_facts.py --stdout   # print to stdout (used by the drift check)
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "website" / "data" / "facts.json"

# Public API surface documented on the site (mirrors pragmatiq/api.py's __all__ intent).
API_FUNCTIONS = (
    "synthesize", "tokenize", "pretrain", "finetune", "embed", "probe", "uplift",
    "gnn", "quickstart",
)
MODEL_SIZES = ("nano", "small", "medium", "large")


def _default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return field.default_factory()  # type: ignore[misc]
        except Exception:
            return None
    return None  # required field (no default)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    return str(v)


def _config_fields(cls: type) -> list[dict[str, Any]]:
    out = []
    for f in dataclasses.fields(cls):
        out.append(
            {
                "name": f.name,
                "type": _type_str(f.type),
                "default": _jsonable(_default(f)),
                "required": f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING,  # type: ignore[misc]
            }
        )
    return out


def _type_str(t: Any) -> str:
    s = t if isinstance(t, str) else getattr(t, "__name__", str(t))
    return re.sub(r"\s+", " ", str(s)).strip()


def _model_sizes() -> list[dict[str, Any]]:
    from pragmatiq.models.pragmatiq import ModelConfig

    rows = []
    for name in MODEL_SIZES:
        c = ModelConfig.preset(name, vocab_size=28000)
        rows.append(
            {
                "name": name,
                "dim": c.dim,
                "n_heads": c.n_heads,
                "depth_profile": c.depth_profile,
                "depth_event": c.depth_event,
                "depth_history": c.depth_history,
            }
        )
    return rows


def _gates() -> list[dict[str, str]]:
    gates = []
    for path in sorted((REPO / "scripts" / "gates").glob("gate_*.sh")):
        title = ""
        for line in path.read_text().splitlines():
            m = re.match(r"#\s*(Gate\s*\d+.*)", line)
            if m:
                title = m.group(1).strip()
                break
        gates.append({"id": path.stem, "title": title})
    return gates


def _api() -> list[dict[str, str]]:
    from pragmatiq import api

    out = []
    for fn_name in API_FUNCTIONS:
        fn = getattr(api, fn_name, None)
        if fn is None:
            continue
        doc = inspect.getdoc(fn) or ""
        summary = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
        out.append(
            {
                "name": fn_name,
                "signature": f"{fn_name}{inspect.signature(fn)}",
                "summary": summary,
            }
        )
    return out


def build_facts() -> dict[str, Any]:
    from pragmatiq.data.tokenizer import TokenizerConfig
    from pragmatiq.training.pretrainer import TrainConfig

    return {
        "model_sizes": _model_sizes(),
        "train_config": _config_fields(TrainConfig),
        "tokenizer_config": _config_fields(TokenizerConfig),
        "gates": _gates(),
        "api": _api(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="print JSON instead of writing the file")
    args = ap.parse_args()
    facts = build_facts()
    text = json.dumps(facts, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        print(text, end="")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)} ({len(facts['api'])} API fns, "
          f"{len(facts['model_sizes'])} sizes, {len(facts['gates'])} gates)")


if __name__ == "__main__":
    main()
