"""Unit tests for integrations.sagemaker — fully OFFLINE.

No cloud SDKs (boto3 / sagemaker) are required.  All testable adapter
methods (manifest, package, request-building) run without any network or
AWS credentials.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers — build a minimal fake run-dir (checkpoints/last.pt + tokenizer/)
# ---------------------------------------------------------------------------


def _make_fake_run_dir() -> Path:
    """Create a temporary run-dir that looks like a real pragmatiq run.

    Structure::

        run_dir/
          checkpoints/last.pt   (dummy bytes)
          tokenizer/             (empty dir, mirrors reality)
    """
    tmp = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-test-"))
    (tmp / "checkpoints").mkdir()
    (tmp / "checkpoints" / "last.pt").write_bytes(b"FAKE_CHECKPOINT")
    (tmp / "tokenizer").mkdir()
    (tmp / "tokenizer" / "config.json").write_text('{"vocab_size": 512}')
    return tmp


# ---------------------------------------------------------------------------
# Import-cleanness test — no boto3/sagemaker leaked into sys.modules
# ---------------------------------------------------------------------------


def test_sagemaker_adapter_import_does_not_load_boto3() -> None:
    """Importing integrations.sagemaker must not pull boto3 or the sagemaker SDK.

    Runs in a fresh subprocess to avoid false-negatives from sibling tests that
    may have already imported boto3/botocore into the same process's sys.modules.
    The subprocess starts with a clean interpreter, so any cloud SDK present in
    sys.modules after the import is genuinely caused by the adapter itself.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import integrations.sagemaker, sys; "
                "leaked = [m for m in sys.modules if m.split('.')[0] in {'boto3', 'botocore', 'sagemaker'}]; "
                "assert not leaked, f'Importing integrations.sagemaker leaked cloud SDK modules: {sorted(leaked)}'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-cleanness check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# manifest() structure
# ---------------------------------------------------------------------------


def test_sagemaker_manifest_returns_dict() -> None:
    """manifest() returns a dict (not None, not a string)."""
    from integrations.sagemaker import SageMakerAdapter

    adapter = SageMakerAdapter(image="123456789012.dkr.ecr.us-east-1.amazonaws.com/pragmatiq:latest")
    m = adapter.manifest()
    assert isinstance(m, dict)


def test_sagemaker_manifest_has_required_keys() -> None:
    """manifest() contains the keys callers need to deploy to SageMaker."""
    from integrations.sagemaker import SageMakerAdapter

    adapter = SageMakerAdapter(image="123456789012.dkr.ecr.us-east-1.amazonaws.com/pragmatiq:latest")
    m = adapter.manifest()

    required = {"model", "endpoint_config"}
    missing = required - m.keys()
    assert not missing, f"manifest() is missing keys: {missing}"


def test_sagemaker_manifest_model_has_image() -> None:
    """manifest()['model'] includes the container image URI."""
    from integrations.sagemaker import SageMakerAdapter

    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/pragmatiq:latest"
    adapter = SageMakerAdapter(image=image)
    m = adapter.manifest()
    assert m["model"]["image"] == image


def test_sagemaker_manifest_endpoint_config_has_instance_type() -> None:
    """manifest()['endpoint_config'] includes an instance_type."""
    from integrations.sagemaker import SageMakerAdapter

    adapter = SageMakerAdapter(
        image="123456789012.dkr.ecr.us-east-1.amazonaws.com/pragmatiq:latest",
        instance_type="ml.g4dn.xlarge",
    )
    m = adapter.manifest()
    assert m["endpoint_config"]["instance_type"] == "ml.g4dn.xlarge"


def test_sagemaker_manifest_env_vars_present() -> None:
    """manifest()['model'] includes an env block with PRAGMATIQ_RUN."""
    from integrations.sagemaker import SageMakerAdapter

    adapter = SageMakerAdapter(image="example.amazonaws.com/pragmatiq:latest")
    m = adapter.manifest()
    env = m["model"].get("env", {})
    assert "PRAGMATIQ_RUN" in env, "manifest model.env must contain PRAGMATIQ_RUN"


def test_sagemaker_adapter_name() -> None:
    """SageMakerAdapter.name is 'sagemaker'."""
    from integrations.sagemaker import SageMakerAdapter

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    assert adapter.name == "sagemaker"


# ---------------------------------------------------------------------------
# package() — produces a real .tar.gz with the correct layout
# ---------------------------------------------------------------------------


def test_sagemaker_package_creates_tarball() -> None:
    """package() creates a .tar.gz file at the specified dest path."""
    from integrations.sagemaker import SageMakerAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-dest-"))
    dest = str(dest_dir / "model.tar.gz")

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    adapter.package(run_dir, dest=dest, image="example.amazonaws.com/img:tag")

    assert Path(dest).exists(), "package() did not create the .tar.gz file"


def test_sagemaker_package_returns_artifact() -> None:
    """package() returns an Artifact with kind='sagemaker-model-tar'."""
    from integrations._base import Artifact
    from integrations.sagemaker import SageMakerAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-dest-"))
    dest = str(dest_dir / "model.tar.gz")

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    artifact = adapter.package(run_dir, dest=dest, image="example.amazonaws.com/img:tag")

    assert isinstance(artifact, Artifact)
    assert artifact.kind == "sagemaker-model-tar"
    assert artifact.path_or_uri == dest


def test_sagemaker_package_tarball_contains_checkpoint() -> None:
    """The .tar.gz must contain a checkpoints/last.pt entry."""
    from integrations.sagemaker import SageMakerAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-dest-"))
    dest = str(dest_dir / "model.tar.gz")

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    adapter.package(run_dir, dest=dest, image="example.amazonaws.com/img:tag")

    with tarfile.open(dest, "r:gz") as tf:
        names = tf.getnames()

    # The checkpoint must be somewhere inside the archive
    has_checkpoint = any("last.pt" in n for n in names)
    assert has_checkpoint, f"Archive does not contain checkpoints/last.pt. Members: {names}"


def test_sagemaker_package_tarball_contains_tokenizer() -> None:
    """The .tar.gz must contain an entry from the tokenizer/ directory."""
    from integrations.sagemaker import SageMakerAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-dest-"))
    dest = str(dest_dir / "model.tar.gz")

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    adapter.package(run_dir, dest=dest, image="example.amazonaws.com/img:tag")

    with tarfile.open(dest, "r:gz") as tf:
        names = tf.getnames()

    has_tokenizer = any("tokenizer" in n for n in names)
    assert has_tokenizer, f"Archive does not contain tokenizer/ entries. Members: {names}"


def test_sagemaker_package_artifact_details() -> None:
    """Artifact.details must include run_dir and image keys."""
    from integrations.sagemaker import SageMakerAdapter

    run_dir = _make_fake_run_dir()
    dest_dir = Path(tempfile.mkdtemp(prefix="pragmatiq-sm-dest-"))
    dest = str(dest_dir / "model.tar.gz")

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    artifact = adapter.package(run_dir, dest=dest, image="example.amazonaws.com/img:tag")

    assert "image" in artifact.details
    assert "run_dir" in artifact.details


# ---------------------------------------------------------------------------
# Contract request-building — offline, no boto3
# ---------------------------------------------------------------------------


def test_sagemaker_request_builder_produces_bytes() -> None:
    """The contract encode_request helper produces bytes for SageMaker invocation."""
    from pragmatiq.inference.serve.contract import encode_request

    records = [{"user_id": "u1", "events": [], "attributes": {}, "lifelong": []}]
    payload = encode_request(records)
    assert isinstance(payload, bytes)


def test_sagemaker_request_roundtrip_via_contract() -> None:
    """encode_request → decode_request is identity for the SageMaker wire format."""
    from pragmatiq.inference.serve.contract import decode_request, encode_request

    records = [
        {"user_id": "sm_test_1", "events": [{"ts": 1_700_000_000_000_000, "source": "tx", "fields": {}}]},
    ]
    raw = encode_request(records)
    recovered = decode_request(raw)
    assert recovered == records


# ---------------------------------------------------------------------------
# Live-op guard: push() raises MissingExtraError when boto3 absent
# ---------------------------------------------------------------------------


def test_sagemaker_push_raises_missing_extra_when_boto3_absent(monkeypatch) -> None:
    """push() must raise MissingExtraError (or ImportError) when boto3 is absent.

    Simulates boto3 absence via monkeypatch so the test is env-robust (works
    whether or not boto3 is installed in the current environment).
    """
    from integrations._base import MissingExtraError
    from integrations.sagemaker import SageMakerAdapter

    # Block boto3 so the adapter's lazy import fails — same signal as absent package.
    monkeypatch.setitem(sys.modules, "boto3", None)

    adapter = SageMakerAdapter(image="example.amazonaws.com/img:tag")
    with pytest.raises((MissingExtraError, ImportError)):
        adapter.push(
            artifact_path="s3://bucket/model.tar.gz",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        )
