"""Unit tests for integrations.databricks — fully OFFLINE.

No cloud SDKs (mlflow / databricks-sdk) are required.  All testable adapter
methods (manifest, package, pyfunc-predict logic) run without any network or
Databricks credentials.

The nano-model+tokenizer pattern mirrors tests/contract/test_serving_contract.py
so the two test suites stay in sync.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Nano model fixture — reuses the same build as the contract test suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nano_runtime():
    """Return a Runtime wrapping a tiny nano PragmaModel (10 users, CPU).

    Built identically to the nano_model_and_records fixture in
    tests/contract/test_serving_contract.py.
    """
    from pragmatiq.data.synthetic import WorldConfig, generate
    from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig
    from pragmatiq.inference.serve.runtime import Runtime
    from pragmatiq.models import ModelConfig, PragmaModel

    tmp = Path(tempfile.mkdtemp(prefix="pragmatiq-db-test-"))
    generate(
        WorldConfig(
            n_users=10,
            months=14,
            n_merchants=30,
            seed=999,
            mule_ring_count=0,
            eval_month_credit=2,
            eval_month_short=8,
        ),
        tmp / "raw",
        n_workers=0,
        write_report=False,
    )
    tok = PragmaTokenizer(
        TokenizerConfig(target_vocab=512, n_buckets=8, categorical_threshold=20, seed=0)
    ).fit(tmp / "raw")

    cfg = ModelConfig.preset("small", tok.vocab_size)
    model = PragmaModel(cfg).eval()
    model._tokenizer = tok  # attach tokenizer exactly as from_pretrained does

    return Runtime(model=model, device="cpu")


@pytest.fixture(scope="module")
def sample_records():
    """Two minimal user records for embedding tests."""
    return [
        {
            "user_id": "db_test_1",
            "events": [
                {
                    "ts": 1_700_000_000_000_000,
                    "source": "transaction",
                    "fields": {"amount": "9.99", "mcc": "5411", "merchant": "STORE A"},
                }
            ],
            "attributes": {},
            "lifelong": [],
        },
        {
            "user_id": "db_test_2",
            "events": [
                {
                    "ts": 1_700_003_600_000_000,
                    "source": "app",
                    "fields": {"screen": "home", "action": "view"},
                }
            ],
            "attributes": {},
            "lifelong": [],
        },
    ]


# ---------------------------------------------------------------------------
# Import-cleanness test — no mlflow/databricks-sdk leaked into sys.modules
# ---------------------------------------------------------------------------


def test_databricks_adapter_import_does_not_load_mlflow() -> None:
    """Importing integrations.databricks must not pull mlflow or databricks-sdk."""
    import integrations.databricks  # noqa: F401

    bad = {
        m
        for m in sys.modules
        if m.split(".")[0] in {"mlflow", "databricks"}
    }
    assert not bad, (
        f"Importing integrations.databricks leaked cloud SDK modules: {sorted(bad)}"
    )


# ---------------------------------------------------------------------------
# manifest() structure
# ---------------------------------------------------------------------------


def test_databricks_manifest_returns_dict() -> None:
    """manifest() returns a dict."""
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    m = adapter.manifest()
    assert isinstance(m, dict)


def test_databricks_manifest_has_required_keys() -> None:
    """manifest() contains the keys needed to register in Unity Catalog."""
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    m = adapter.manifest()

    required = {"model_uri", "pyfunc_entry", "signature"}
    missing = required - m.keys()
    assert not missing, f"manifest() is missing keys: {missing}"


def test_databricks_manifest_model_uri_format() -> None:
    """manifest()['model_uri'] follows the 'catalog.schema.model' pattern."""
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    m = adapter.manifest()
    uri = m["model_uri"]
    parts = uri.split(".")
    assert len(parts) == 3, f"model_uri should be 'catalog.schema.model', got {uri!r}"
    assert parts == ["main", "pragmatiq", "embedder"]


def test_databricks_manifest_signature() -> None:
    """manifest()['signature'] describes inputs and outputs."""
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    m = adapter.manifest()
    sig = m["signature"]
    assert "inputs" in sig
    assert "outputs" in sig


def test_databricks_adapter_name() -> None:
    """DatabricksAdapter.name is 'databricks'."""
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    assert adapter.name == "databricks"


# ---------------------------------------------------------------------------
# PyfuncWrapper.predict — offline test using the nano runtime
# ---------------------------------------------------------------------------


def test_pyfunc_wrapper_predict_shape(nano_runtime, sample_records) -> None:
    """PyfuncWrapper.predict returns a 2-D float32 array [n_users, dim]."""
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    result = wrapper.predict(context=None, model_input=sample_records)

    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
    assert result.ndim == 2, f"Expected 2-D output, got shape {result.shape}"
    assert result.shape[0] == len(sample_records)
    assert result.dtype == np.float32


def test_pyfunc_wrapper_predict_finite(nano_runtime, sample_records) -> None:
    """PyfuncWrapper.predict output contains only finite values."""
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    result = wrapper.predict(context=None, model_input=sample_records)
    assert np.isfinite(result).all(), "predict() returned non-finite values"


def test_pyfunc_wrapper_predict_from_json_bytes(nano_runtime, sample_records) -> None:
    """PyfuncWrapper.predict also accepts JSON-encoded bytes (contract wire format)."""
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper
    from pragmatiq.inference.serve.contract import encode_request

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    payload = encode_request(sample_records)
    result = wrapper.predict(context=None, model_input=payload)

    assert result.ndim == 2
    assert result.shape[0] == len(sample_records)
    assert result.dtype == np.float32


def test_pyfunc_wrapper_class_importable_without_mlflow() -> None:
    """PragmaPyfuncWrapper class must be importable even when mlflow is absent."""
    # mlflow is absent in this env — import must not raise
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper  # noqa: F401

    assert PragmaPyfuncWrapper is not None


# ---------------------------------------------------------------------------
# package() — assembles local artifact directory
# ---------------------------------------------------------------------------


def _make_fake_run_dir() -> Path:
    """Create a temporary run-dir that looks like a real pragmatiq run."""
    tmp = Path(tempfile.mkdtemp(prefix="pragmatiq-db-run-"))
    (tmp / "checkpoints").mkdir()
    (tmp / "checkpoints" / "last.pt").write_bytes(b"FAKE_CHECKPOINT")
    (tmp / "tokenizer").mkdir()
    (tmp / "tokenizer" / "config.json").write_text('{"vocab_size": 512}')
    return tmp


def test_databricks_package_returns_artifact() -> None:
    """package() returns an Artifact with kind='databricks-pyfunc'."""
    from integrations._base import Artifact
    from integrations.databricks import DatabricksAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-db-dest-"))
    dest = str(dest_dir / "pyfunc_artifact")

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    artifact = adapter.package(run_dir, dest=dest, image="unused-for-pyfunc")

    assert isinstance(artifact, Artifact)
    assert artifact.kind == "databricks-pyfunc"
    assert artifact.path_or_uri == dest


def test_databricks_package_creates_directory() -> None:
    """package() creates the destination directory."""
    from integrations.databricks import DatabricksAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-db-dest2-"))
    dest = str(dest_dir / "pyfunc_artifact")

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    adapter.package(run_dir, dest=dest, image="unused")

    assert Path(dest).exists(), "package() did not create the destination directory"


def test_databricks_package_stages_run_dir() -> None:
    """package() copies the run dir into the artifact directory."""
    from integrations.databricks import DatabricksAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-db-dest3-"))
    dest = str(dest_dir / "pyfunc_artifact")

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    adapter.package(run_dir, dest=dest, image="unused")

    dest_path = Path(dest)
    # Should contain a run_dir sub-directory with checkpoints
    run_sub = dest_path / "run_dir"
    assert run_sub.exists(), f"package() did not stage run_dir into dest. dest contents: {list(dest_path.iterdir())}"
    assert (run_sub / "checkpoints" / "last.pt").exists()


def test_databricks_package_artifact_details() -> None:
    """Artifact.details includes run_dir and catalog info."""
    from integrations.databricks import DatabricksAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-db-dest4-"))
    dest = str(dest_dir / "pyfunc_artifact")

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    artifact = adapter.package(run_dir, dest=dest, image="unused")

    assert "run_dir" in artifact.details
    assert "model_uri" in artifact.details


# ---------------------------------------------------------------------------
# Live-op guard: register() raises MissingExtraError when mlflow absent
# ---------------------------------------------------------------------------


def test_databricks_register_raises_missing_extra_when_mlflow_absent() -> None:
    """register() must raise MissingExtraError (or ImportError) when mlflow is absent."""
    from integrations._base import MissingExtraError
    from integrations.databricks import DatabricksAdapter

    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    # mlflow is NOT installed in this env
    with pytest.raises((MissingExtraError, ImportError), match="mlflow"):
        adapter.register(artifact_path="dbfs:/artifacts/pyfunc_artifact")
