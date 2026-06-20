"""Contract tests: PragmaModel.from_pretrained and embed_records signatures.

Pins the two public model-API methods so that refactoring PragmaModel's
internals or moving the class cannot silently change the notebook-facing
surface (``from_pretrained`` / ``embed_records``).
"""

from __future__ import annotations

import inspect
from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _param_info(fn: Any) -> dict[str, inspect.Parameter]:
    """Return {name: Parameter} for a callable."""
    return dict(inspect.signature(fn).parameters)


# ---------------------------------------------------------------------------
# D. PragmaModel.from_pretrained signature
# ---------------------------------------------------------------------------
#
# Golden signature (read from pragmatiq/models/pragmatiq.py on contract date):
#
#   @classmethod
#   def from_pretrained(
#       cls, run: str | Path, device: str = "cpu", checkpoint: str = "last.pt"
#   ) -> PragmaModel:
#
# Pinned required params: ["run"]  (cls is implicit for classmethod)
# Pinned optional params: ["device", "checkpoint"]
# Pinned defaults: device="cpu", checkpoint="last.pt"

GOLDEN_FROM_PRETRAINED_REQUIRED: list[str] = ["run"]
GOLDEN_FROM_PRETRAINED_OPTIONAL: list[str] = ["device", "checkpoint"]
GOLDEN_FROM_PRETRAINED_DEFAULTS: dict[str, Any] = {
    "device": "cpu",
    "checkpoint": "last.pt",
}


def test_from_pretrained_importable() -> None:
    """PragmaModel.from_pretrained must be importable and a classmethod."""
    from pragmatiq.models.pragmatiq import PragmaModel

    assert hasattr(PragmaModel, "from_pretrained"), "PragmaModel.from_pretrained not found"
    # Should be callable (classmethod descriptor returns a bound method)
    assert callable(PragmaModel.from_pretrained), "PragmaModel.from_pretrained not callable"


def test_from_pretrained_required_params() -> None:
    """from_pretrained required parameters must not be renamed or removed."""
    from pragmatiq.models.pragmatiq import PragmaModel

    params = _param_info(PragmaModel.from_pretrained)
    # Exclude 'cls' (implicit for classmethods; shows up in some Python versions)
    actual_required = [
        name
        for name, p in params.items()
        if name != "cls"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    assert actual_required == GOLDEN_FROM_PRETRAINED_REQUIRED, (
        f"PragmaModel.from_pretrained: required params changed.\n"
        f"  expected: {GOLDEN_FROM_PRETRAINED_REQUIRED}\n"
        f"  actual:   {actual_required}"
    )


def test_from_pretrained_optional_params() -> None:
    """from_pretrained optional parameters must not be renamed or removed."""
    from pragmatiq.models.pragmatiq import PragmaModel

    params = _param_info(PragmaModel.from_pretrained)
    actual_optional = [
        name
        for name, p in params.items()
        if name != "cls"
        and p.default is not inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    # Filter to pinned set, preserving order
    actual_pinned = [n for n in actual_optional if n in set(GOLDEN_FROM_PRETRAINED_OPTIONAL)]
    assert actual_pinned == GOLDEN_FROM_PRETRAINED_OPTIONAL, (
        f"PragmaModel.from_pretrained: optional params changed.\n"
        f"  expected (pinned): {GOLDEN_FROM_PRETRAINED_OPTIONAL}\n"
        f"  actual   (pinned): {actual_pinned}"
    )


def test_from_pretrained_defaults() -> None:
    """from_pretrained default values must not change silently."""
    from pragmatiq.models.pragmatiq import PragmaModel

    params = _param_info(PragmaModel.from_pretrained)
    for param_name, expected_default in GOLDEN_FROM_PRETRAINED_DEFAULTS.items():
        assert param_name in params, (
            f"PragmaModel.from_pretrained: parameter '{param_name}' disappeared"
        )
        actual_default = params[param_name].default
        assert actual_default == expected_default, (
            f"PragmaModel.from_pretrained: default for '{param_name}' changed.\n"
            f"  expected: {expected_default!r}\n"
            f"  actual:   {actual_default!r}"
        )


# ---------------------------------------------------------------------------
# D. PragmaModel.embed_records signature
# ---------------------------------------------------------------------------
#
# Golden signature (read from pragmatiq/models/pragmatiq.py on contract date):
#
#   @torch.no_grad()
#   def embed_records(self, records: list[dict[str, Any]]) -> np.ndarray:
#
# Pinned required params: ["records"]  (self is implicit)
# Pinned optional params: []
# Return type annotation: np.ndarray

GOLDEN_EMBED_RECORDS_REQUIRED: list[str] = ["records"]
GOLDEN_EMBED_RECORDS_OPTIONAL: list[str] = []


def test_embed_records_importable() -> None:
    """PragmaModel.embed_records must be importable and callable."""
    from pragmatiq.models.pragmatiq import PragmaModel

    assert hasattr(PragmaModel, "embed_records"), "PragmaModel.embed_records not found"
    assert callable(PragmaModel.embed_records), "PragmaModel.embed_records not callable"


def test_embed_records_required_params() -> None:
    """embed_records required parameters must not be renamed or removed."""
    from pragmatiq.models.pragmatiq import PragmaModel

    params = _param_info(PragmaModel.embed_records)
    actual_required = [
        name
        for name, p in params.items()
        if name not in ("self",)
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    assert actual_required == GOLDEN_EMBED_RECORDS_REQUIRED, (
        f"PragmaModel.embed_records: required params changed.\n"
        f"  expected: {GOLDEN_EMBED_RECORDS_REQUIRED}\n"
        f"  actual:   {actual_required}"
    )


def test_embed_records_optional_params() -> None:
    """embed_records must have no pinned optional parameters (it's a simple call)."""
    from pragmatiq.models.pragmatiq import PragmaModel

    params = _param_info(PragmaModel.embed_records)
    actual_optional = [
        name
        for name, p in params.items()
        if name not in ("self",)
        and p.default is not inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    # Pinned set is empty — any new optional params are additive (allowed)
    # We only verify that no required param was accidentally given a default
    missing_required = [
        n for n in GOLDEN_EMBED_RECORDS_REQUIRED
        if n in actual_optional
    ]
    assert not missing_required, (
        f"PragmaModel.embed_records: formerly required params now have defaults: {missing_required}"
    )


def test_embed_records_return_annotation() -> None:
    """embed_records must declare a return-type annotation (np.ndarray)."""
    import numpy as np
    from pragmatiq.models.pragmatiq import PragmaModel

    sig = inspect.signature(PragmaModel.embed_records)
    ret = sig.return_annotation
    # The annotation may be the np.ndarray type itself or its string repr
    assert ret is not inspect.Parameter.empty, (
        "PragmaModel.embed_records: return type annotation removed"
    )
    if isinstance(ret, str):
        assert "ndarray" in ret, (
            f"PragmaModel.embed_records: unexpected return annotation {ret!r}"
        )
    else:
        assert ret is np.ndarray, (
            f"PragmaModel.embed_records: return annotation changed to {ret!r}"
        )


# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------

def test_pragmamodel_importable_from_models() -> None:
    """PragmaModel must remain importable from pragmatiq.models.pragmatiq."""
    from pragmatiq.models.pragmatiq import PragmaModel  # noqa: F401


def test_pragmamodel_public_notebook_api() -> None:
    """The two notebook-facing methods must both exist on PragmaModel."""
    from pragmatiq.models.pragmatiq import PragmaModel

    assert hasattr(PragmaModel, "from_pretrained")
    assert hasattr(PragmaModel, "embed_records")
