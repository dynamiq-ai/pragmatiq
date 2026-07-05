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


def test_pyfunc_wrapper_lives_in_shipped_package() -> None:
    """The wrapper class must resolve from the pragmatiq wheel, not the repo tree.

    MLflow pickles the class by reference (module + qualname); a class under
    the repo-only 'integrations' package would be unpicklable on any serving
    cluster, where only the pragmatiq wheel is installed.
    """
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper

    assert PragmaPyfuncWrapper.__module__ == "pragmatiq.inference.serve.pyfunc"


# ---------------------------------------------------------------------------
# PyfuncWrapper.predict — Databricks Model Serving DataFrame inputs ([F15])
# ---------------------------------------------------------------------------


def test_pyfunc_wrapper_predict_from_dataframe_records_json_column(
    nano_runtime, sample_records
) -> None:
    """A DataFrame with a records_json column (the signature's input name) decodes.

    Databricks Model Serving converts the 'dataframe_records' JSON form into a
    pandas DataFrame before calling predict — list(DataFrame) would yield
    column names, never records.
    """
    import pandas as pd

    from integrations.databricks._pyfunc import PragmaPyfuncWrapper
    from pragmatiq.inference.serve.contract import encode_request

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    df = pd.DataFrame({"records_json": [encode_request(sample_records).decode("utf-8")]})
    result = wrapper.predict(context=None, model_input=df)

    assert result.ndim == 2
    assert result.shape[0] == len(sample_records)
    assert result.dtype == np.float32


def test_pyfunc_wrapper_predict_from_dataframe_of_record_fields(
    nano_runtime, sample_records
) -> None:
    """A DataFrame with one column per record field converts back to records."""
    import pandas as pd

    from integrations.databricks._pyfunc import PragmaPyfuncWrapper

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    df = pd.DataFrame(sample_records)
    result = wrapper.predict(context=None, model_input=df)

    assert result.ndim == 2
    assert result.shape[0] == len(sample_records)
    assert result.dtype == np.float32


def test_pyfunc_wrapper_predict_from_numpy_json_array(nano_runtime, sample_records) -> None:
    """MLflow's tensor-input path ({"inputs": ...}) delivers an ndarray of JSON strings."""
    import json

    from integrations.databricks._pyfunc import PragmaPyfuncWrapper

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    arr = np.array([json.dumps(sample_records)], dtype=object)
    result = wrapper.predict(context=None, model_input=arr)

    assert result.ndim == 2
    assert result.shape[0] == len(sample_records)


def test_pyfunc_wrapper_predict_list_of_dicts_unchanged(nano_runtime, sample_records) -> None:
    """The plain list[dict] path must keep working alongside the DataFrame path."""
    from integrations.databricks._pyfunc import PragmaPyfuncWrapper
    from pragmatiq.inference.serve.contract import encode_request

    wrapper = PragmaPyfuncWrapper(runtime=nano_runtime)
    from_list = wrapper.predict(context=None, model_input=sample_records)
    from_bytes = wrapper.predict(context=None, model_input=encode_request(sample_records))
    np.testing.assert_array_equal(from_list, from_bytes)


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


# ---------------------------------------------------------------------------
# register() version handling (Bugbot PR #10 finding 3460867078)
# ---------------------------------------------------------------------------


def _install_fake_mlflow(monkeypatch, *, info_version, registry_versions=()) -> dict:
    """Install a minimal in-memory mlflow stub sufficient for register().

    Returns the dict that captures the kwargs passed to ``pyfunc.log_model``
    so tests can assert on pip_requirements etc.
    """
    import sys
    import types
    from contextlib import contextmanager

    fake = types.ModuleType("mlflow")
    fake.__path__ = []  # mark as a package so `import mlflow.pyfunc` resolves

    class _PythonModel:
        pass

    class _Info:
        registered_model_version = info_version

    captured: dict = {}

    def _log_model(**kwargs):
        captured.update(kwargs)
        return _Info()

    pyfunc = types.ModuleType("mlflow.pyfunc")
    pyfunc.PythonModel = _PythonModel
    pyfunc.log_model = _log_model
    fake.pyfunc = pyfunc

    @contextmanager
    def _start_run():
        yield None

    fake.start_run = _start_run

    class _ModelVersion:
        def __init__(self, version: str) -> None:
            self.version = version

    class _Client:
        def search_model_versions(self, _filter: str):
            return [_ModelVersion(str(v)) for v in registry_versions]

    fake.MlflowClient = _Client
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", pyfunc)
    return captured


def test_databricks_register_uses_returned_model_version(monkeypatch) -> None:
    """register() must report the version the registry assigned, not a hardcoded /1."""
    from integrations.databricks import DatabricksAdapter

    _install_fake_mlflow(monkeypatch, info_version="7")
    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    uri = adapter.register(str(_make_fake_run_dir()))
    assert uri == "models:/main.pragmatiq.embedder/7"


def test_databricks_register_falls_back_to_registry_lookup(monkeypatch) -> None:
    """Older mlflow (no version on ModelInfo) → the newest registry version wins."""
    from integrations.databricks import DatabricksAdapter

    _install_fake_mlflow(monkeypatch, info_version=None, registry_versions=(1, 3, 2))
    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    uri = adapter.register(str(_make_fake_run_dir()))
    assert uri == "models:/main.pragmatiq.embedder/3"


def test_databricks_register_empty_registry_falls_back_to_version_1(monkeypatch) -> None:
    """No ModelInfo version + empty registry search must yield /1, not ValueError.

    A just-registered model has at least one version, so an empty
    search_model_versions() result is registry eventual-consistency — the
    initial version is 1.
    """
    from integrations.databricks import DatabricksAdapter

    _install_fake_mlflow(monkeypatch, info_version=None, registry_versions=())
    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    uri = adapter.register(str(_make_fake_run_dir()))
    assert uri == "models:/main.pragmatiq.embedder/1"


def test_databricks_register_pins_pragmatiq_version(monkeypatch) -> None:
    """register() must pass pip_requirements pinning the installed pragmatiq version.

    The pickled wrapper class resolves from the pragmatiq wheel
    (pragmatiq.inference.serve.pyfunc); without the pin, the serving cluster
    has no dependency telling it to install the library at all.
    """
    import pragmatiq
    from integrations.databricks import DatabricksAdapter

    captured = _install_fake_mlflow(monkeypatch, info_version="1")
    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    adapter.register(str(_make_fake_run_dir()))
    assert captured["pip_requirements"] == [f"pragmatiq=={pragmatiq.__version__}"]


def test_databricks_register_logs_shipped_wrapper_class(monkeypatch) -> None:
    """The python_model logged by register() must come from the shipped package."""
    from integrations.databricks import DatabricksAdapter

    captured = _install_fake_mlflow(monkeypatch, info_version="1")
    adapter = DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder")
    adapter.register(str(_make_fake_run_dir()))
    logged = captured["python_model"]
    # The dynamic subclass is defined in (and its wrapper resolves from) the
    # shipped module, not the repo-only integrations tree.
    assert type(logged).__module__ == "pragmatiq.inference.serve.pyfunc"
