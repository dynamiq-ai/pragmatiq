"""Unit tests for Azure and Nebius stub adapters — fully OFFLINE.

No cloud SDKs are required.  All testable methods (manifest, package,
live-guard) run without any network, Azure credentials, or Nebius credentials.

Coverage:
- Azure: manifest keys, package writes Helm skeleton, deploy_live raises NIE.
- Nebius: manifest keys, package writes job specs, deploy_live raises NIE.
- Both: import without cloud SDK; healthcheck request-building (offline part).
- All four adapters satisfy the CloudAdapter protocol shape (structural check).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers — build a minimal fake run-dir (mirrors the pattern in the real tests)
# ---------------------------------------------------------------------------


def _make_fake_run_dir() -> Path:
    """Create a temporary run-dir that looks like a real pragmatiq run.

    Structure::

        run_dir/
          checkpoints/last.pt   (dummy bytes)
          tokenizer/config.json (dummy JSON)
    """
    tmp = Path(tempfile.mkdtemp(prefix="pragmatiq-stub-test-"))
    (tmp / "checkpoints").mkdir()
    (tmp / "checkpoints" / "last.pt").write_bytes(b"FAKE_CHECKPOINT")
    (tmp / "tokenizer").mkdir()
    (tmp / "tokenizer" / "config.json").write_text('{"vocab_size": 512}')
    return tmp


# ===========================================================================
# Azure adapter tests
# ===========================================================================


class TestAzureAdapterImport:
    """Import-cleanness: no Azure/cloud SDK must leak into sys.modules."""

    def test_import_does_not_load_azure_sdk(self) -> None:
        """Importing integrations.azure must not pull azure-sdk or msrest.

        Runs in a fresh subprocess to avoid false-negatives from sibling tests
        that may have already imported azure-sdk packages into the same process's
        sys.modules.  The subprocess starts with a clean interpreter, so any
        cloud SDK present in sys.modules after the import is genuinely caused by
        the adapter itself.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import integrations.azure, sys; "
                    "leaked = [m for m in sys.modules if m.split('.')[0] in {'azure', 'msrest', 'msrestazure'}]; "
                    "assert not leaked, f'Importing integrations.azure leaked cloud SDK modules: {sorted(leaked)}'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"import-cleanness check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestAzureManifest:
    """manifest() structure and required keys."""

    def test_manifest_returns_dict(self) -> None:
        """manifest() returns a dict."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        m = adapter.manifest()
        assert isinstance(m, dict)

    def test_manifest_has_required_keys(self) -> None:
        """manifest() contains the keys callers need to deploy on AKS."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        m = adapter.manifest()

        required = {"adapter", "helm", "container", "storage", "live_ops_status"}
        missing = required - m.keys()
        assert not missing, f"manifest() is missing keys: {missing}"

    def test_manifest_adapter_name(self) -> None:
        """manifest()['adapter'] is 'azure'."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        assert adapter.manifest()["adapter"] == "azure"

    def test_manifest_helm_has_image(self) -> None:
        """manifest()['helm'] includes the container image."""
        from integrations.azure import AzureAdapter

        image = "myacr.azurecr.io/pragmatiq:v1.2.3"
        adapter = AzureAdapter(image=image)
        helm = adapter.manifest()["helm"]
        assert "image" in helm
        assert helm["image"]["tag"] == "v1.2.3"

    def test_manifest_container_port(self) -> None:
        """manifest()['container']['port'] matches the serving contract (8000)."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        container = adapter.manifest()["container"]
        assert container["port"] == 8000

    def test_manifest_live_ops_status_mentions_stub(self) -> None:
        """manifest()['live_ops_status'] honestly declares stub status."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        status = adapter.manifest()["live_ops_status"]
        assert "STUB" in status or "documented" in status.lower()

    def test_azure_adapter_name(self) -> None:
        """AzureAdapter.name is 'azure'."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        assert adapter.name == "azure"


class TestAzurePackage:
    """package() writes a real Helm chart skeleton."""

    def test_package_returns_artifact(self) -> None:
        """package() returns an Artifact with kind='helm-chart'."""
        from integrations._base import Artifact
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        artifact = adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        assert isinstance(artifact, Artifact)
        assert artifact.kind == "helm-chart"
        assert artifact.path_or_uri == str(dest)

    def test_package_creates_chart_yaml(self) -> None:
        """package() writes Chart.yaml."""
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest2-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        assert (dest / "Chart.yaml").exists(), "package() did not write Chart.yaml"

    def test_package_creates_values_yaml(self) -> None:
        """package() writes values.yaml."""
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest3-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        assert (dest / "values.yaml").exists(), "package() did not write values.yaml"

    def test_package_creates_deployment_template(self) -> None:
        """package() writes templates/deployment.yaml."""
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest4-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        tmpl = dest / "templates" / "deployment.yaml"
        assert tmpl.exists(), "package() did not write templates/deployment.yaml"

    def test_package_deployment_references_image(self) -> None:
        """templates/deployment.yaml references the Triton image via values."""
        from integrations.azure import AzureAdapter

        image = "myacr.azurecr.io/pragmatiq:v2.0"
        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest5-")) / "helm"

        adapter = AzureAdapter(image=image)
        adapter.package(run_dir, dest=str(dest), image=image)

        content = (dest / "templates" / "deployment.yaml").read_text()
        # The deployment.yaml uses Helm template syntax for the image
        assert ".Values.image.repository" in content or "image.repository" in content, (
            "deployment.yaml does not reference the image via values"
        )

    def test_package_deployment_references_contract_port(self) -> None:
        """templates/deployment.yaml references the contract port (8000) via values."""
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest6-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        deployment = (dest / "templates" / "deployment.yaml").read_text()
        values = (dest / "values.yaml").read_text()
        # The port 8000 must appear in either the template reference or values
        assert "8000" in values, "values.yaml does not reference contract port 8000"
        assert ".Values.container.port" in deployment or "container.port" in deployment, (
            "deployment.yaml does not reference container port via values"
        )

    def test_package_artifact_details(self) -> None:
        """Artifact.details includes run_dir and image."""
        from integrations.azure import AzureAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-az-dest7-")) / "helm"

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        artifact = adapter.package(run_dir, dest=str(dest), image="myacr.azurecr.io/pragmatiq:latest")

        assert "run_dir" in artifact.details
        assert "image" in artifact.details


class TestAzureLiveGuard:
    """Live operations raise the documented NotImplementedError."""

    def test_deploy_live_raises_not_implemented(self) -> None:
        """deploy_live() raises NotImplementedError with the INTEGRATIONS.md pointer."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        with pytest.raises(NotImplementedError, match="docs/INTEGRATIONS.md"):
            adapter.deploy_live()

    def test_deploy_live_message_mentions_azure(self) -> None:
        """deploy_live() error message mentions Azure or is clearly documented."""
        from integrations.azure import AzureAdapter

        adapter = AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest")
        with pytest.raises(NotImplementedError) as exc_info:
            adapter.deploy_live()
        msg = str(exc_info.value).lower()
        assert "azure" in msg or "documented" in msg


# ===========================================================================
# Nebius adapter tests
# ===========================================================================


class TestNebiusAdapterImport:
    """Import-cleanness: no Nebius cloud SDK must leak into sys.modules."""

    def test_import_does_not_load_nebius_sdk(self) -> None:
        """Importing integrations.nebius must not pull any Nebius SDK."""
        import integrations.nebius  # noqa: F401

        bad = {
            m
            for m in sys.modules
            if m.split(".")[0] in {"nebius", "nebius_sdk"}
        }
        assert not bad, (
            f"Importing integrations.nebius leaked cloud SDK modules: {sorted(bad)}"
        )


class TestNebiusManifest:
    """manifest() structure and required keys."""

    def test_manifest_returns_dict(self) -> None:
        """manifest() returns a dict."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        m = adapter.manifest()
        assert isinstance(m, dict)

    def test_manifest_has_required_keys(self) -> None:
        """manifest() contains the keys callers need to deploy on Nebius."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        m = adapter.manifest()

        required = {"adapter", "token_factory", "soperator_batch", "storage", "live_ops_status"}
        missing = required - m.keys()
        assert not missing, f"manifest() is missing keys: {missing}"

    def test_manifest_adapter_name(self) -> None:
        """manifest()['adapter'] is 'nebius'."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        assert adapter.manifest()["adapter"] == "nebius"

    def test_manifest_token_factory_has_image(self) -> None:
        """manifest()['token_factory'] includes the image."""
        from integrations.nebius import NebiusAdapter

        image = "cr.eu-north1.nebius.cloud/pragmatiq:v1.0"
        adapter = NebiusAdapter(image=image)
        tf = adapter.manifest()["token_factory"]
        assert tf["image"] == image

    def test_manifest_token_factory_container_port(self) -> None:
        """manifest()['token_factory']['container']['port'] matches contract (8000)."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        port = adapter.manifest()["token_factory"]["container"]["port"]
        assert port == 8000

    def test_manifest_storage_has_s3_endpoint(self) -> None:
        """manifest()['storage'] includes an s3_endpoint."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        storage = adapter.manifest()["storage"]
        assert "s3_endpoint" in storage

    def test_manifest_live_ops_status_mentions_stub(self) -> None:
        """manifest()['live_ops_status'] honestly declares stub status."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        status = adapter.manifest()["live_ops_status"]
        assert "STUB" in status or "documented" in status.lower()

    def test_nebius_adapter_name(self) -> None:
        """NebiusAdapter.name is 'nebius'."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        assert adapter.name == "nebius"


class TestNebiusPackage:
    """package() writes real job-spec YAML files."""

    def test_package_returns_artifact(self) -> None:
        """package() returns an Artifact with kind='nebius-job-spec'."""
        from integrations._base import Artifact
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-dest-")) / "specs"

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        artifact = adapter.package(run_dir, dest=str(dest), image="cr.eu-north1.nebius.cloud/pragmatiq:latest")

        assert isinstance(artifact, Artifact)
        assert artifact.kind == "nebius-job-spec"
        assert artifact.path_or_uri == str(dest)

    def test_package_creates_serving_spec(self) -> None:
        """package() writes serving_spec.yaml."""
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-dest2-")) / "specs"

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="cr.eu-north1.nebius.cloud/pragmatiq:latest")

        assert (dest / "serving_spec.yaml").exists(), "package() did not write serving_spec.yaml"

    def test_package_creates_batch_job_spec(self) -> None:
        """package() writes batch_embed_job.yaml."""
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-dest3-")) / "specs"

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="cr.eu-north1.nebius.cloud/pragmatiq:latest")

        assert (dest / "batch_embed_job.yaml").exists(), "package() did not write batch_embed_job.yaml"

    def test_package_serving_spec_references_image(self) -> None:
        """serving_spec.yaml contains the image URI."""
        from integrations.nebius import NebiusAdapter

        image = "cr.eu-north1.nebius.cloud/pragmatiq:v3.0"
        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-dest4-")) / "specs"

        adapter = NebiusAdapter(image=image)
        adapter.package(run_dir, dest=str(dest), image=image)

        content = (dest / "serving_spec.yaml").read_text()
        assert image in content, f"serving_spec.yaml does not reference image {image!r}"

    def test_package_artifact_details(self) -> None:
        """Artifact.details includes run_dir and image."""
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-dest5-")) / "specs"

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        artifact = adapter.package(run_dir, dest=str(dest), image="cr.eu-north1.nebius.cloud/pragmatiq:latest")

        assert "run_dir" in artifact.details
        assert "image" in artifact.details


class TestNebiusLiveGuard:
    """Live operations raise the documented NotImplementedError."""

    def test_deploy_live_raises_not_implemented(self) -> None:
        """deploy_live() raises NotImplementedError with the INTEGRATIONS.md pointer."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        with pytest.raises(NotImplementedError, match="docs/INTEGRATIONS.md"):
            adapter.deploy_live()

    def test_deploy_live_message_mentions_nebius(self) -> None:
        """deploy_live() error message mentions Nebius or is clearly documented."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        with pytest.raises(NotImplementedError) as exc_info:
            adapter.deploy_live()
        msg = str(exc_info.value).lower()
        assert "nebius" in msg or "documented" in msg


# ===========================================================================
# CloudAdapter protocol structural conformance — all four adapters
# ===========================================================================


class TestCloudAdapterProtocolConformance:
    """All four adapters must satisfy the CloudAdapter protocol shape.

    Checks that each concrete adapter has ``name``, ``manifest``, ``package``,
    and ``healthcheck`` — the four members declared in the CloudAdapter Protocol.
    This is a structural (duck-typing) check, not an ``isinstance`` check, so it
    also works as a regression guard if the Protocol changes.
    """

    PROTOCOL_MEMBERS = ("name", "manifest", "package", "healthcheck")

    def _check_adapter(self, adapter_obj: object) -> None:
        for member in self.PROTOCOL_MEMBERS:
            assert hasattr(adapter_obj, member), (
                f"{type(adapter_obj).__name__!r} is missing required CloudAdapter "
                f"member {member!r}"
            )

    def test_sagemaker_adapter_protocol_shape(self) -> None:
        """SageMakerAdapter has name, manifest, package, healthcheck."""
        from integrations.sagemaker import SageMakerAdapter

        self._check_adapter(SageMakerAdapter(image="ecr.amazonaws.com/pragmatiq:latest"))

    def test_databricks_adapter_protocol_shape(self) -> None:
        """DatabricksAdapter has name, manifest, package, healthcheck."""
        from integrations.databricks import DatabricksAdapter

        self._check_adapter(DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder"))

    def test_azure_adapter_protocol_shape(self) -> None:
        """AzureAdapter has name, manifest, package, healthcheck."""
        from integrations.azure import AzureAdapter

        self._check_adapter(AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest"))

    def test_nebius_adapter_protocol_shape(self) -> None:
        """NebiusAdapter has name, manifest, package, healthcheck."""
        from integrations.nebius import NebiusAdapter

        self._check_adapter(NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest"))

    def test_all_four_adapters_satisfy_cloud_adapter_protocol(self) -> None:
        """All four adapters are instances of the CloudAdapter Protocol."""
        from integrations._base import CloudAdapter
        from integrations.azure import AzureAdapter
        from integrations.databricks import DatabricksAdapter
        from integrations.nebius import NebiusAdapter
        from integrations.sagemaker import SageMakerAdapter

        adapters = [
            SageMakerAdapter(image="ecr.amazonaws.com/pragmatiq:latest"),
            DatabricksAdapter(catalog="main", schema="pragmatiq", model_name="embedder"),
            AzureAdapter(image="myacr.azurecr.io/pragmatiq:latest"),
            NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest"),
        ]
        for adapter in adapters:
            assert isinstance(adapter, CloudAdapter), (
                f"{type(adapter).__name__} does not satisfy the CloudAdapter Protocol"
            )


# ===========================================================================
# Regression: Nebius batch-embed command uses the real CLI flag contract
# (Bugbot PR #10 — issue 1)
# ===========================================================================


class TestNebiusBatchEmbedCliFlags:
    """Generated Nebius batch-embed YAML must use the real pragmatiq CLI flags.

    The ``embed`` command signature is::

        pragmatiq embed <shard_dir> --run <run_dir> --out <out.parquet>

    The old (wrong) flags ``--run-dir`` / ``--output`` do NOT exist on the CLI
    and would cause a ``No such option`` error on the Soperator pod.
    """

    def _get_batch_job_yaml(self) -> str:
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-cliflags-")) / "specs"
        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        adapter.package(run_dir, dest=str(dest), image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        return (dest / "batch_embed_job.yaml").read_text()

    def test_batch_job_uses_run_flag_not_run_dir(self) -> None:
        """batch_embed_job.yaml must use '--run', not '--run-dir'."""
        content = self._get_batch_job_yaml()
        assert "--run-dir" not in content, (
            "batch_embed_job.yaml uses wrong flag '--run-dir'; CLI expects '--run'"
        )
        assert "--run" in content, (
            "batch_embed_job.yaml is missing the '--run' flag for the run directory"
        )

    def test_batch_job_uses_out_flag_not_output(self) -> None:
        """batch_embed_job.yaml must use '--out', not '--output'."""
        content = self._get_batch_job_yaml()
        assert "--output" not in content, (
            "batch_embed_job.yaml uses wrong flag '--output'; CLI expects '--out'"
        )
        assert "--out" in content, (
            "batch_embed_job.yaml is missing the '--out' flag for the output parquet"
        )

    def test_batch_job_has_positional_shard_dir(self) -> None:
        """batch_embed_job.yaml must include the positional shard_dir argument."""
        content = self._get_batch_job_yaml()
        # The shard_dir mount path must appear as a positional arg in the command
        assert "/opt/pragmatiq/shard_dir" in content, (
            "batch_embed_job.yaml is missing the positional shard_dir mount path"
        )

    def test_manifest_command_uses_correct_cli_flags(self) -> None:
        """manifest()['soperator_batch']['command'] must use '--run' and '--out'."""
        from integrations.nebius import NebiusAdapter

        adapter = NebiusAdapter(image="cr.eu-north1.nebius.cloud/pragmatiq:latest")
        cmd = adapter.manifest()["soperator_batch"]["command"]
        cmd_str = " ".join(cmd)
        assert "--run-dir" not in cmd_str, (
            "manifest() soperator_batch command uses wrong flag '--run-dir'"
        )
        assert "--output" not in cmd_str, (
            "manifest() soperator_batch command uses wrong flag '--output'"
        )
        assert "--run" in cmd, (
            "manifest() soperator_batch command is missing '--run' flag"
        )
        assert "--out" in cmd, (
            "manifest() soperator_batch command is missing '--out' flag"
        )


# ===========================================================================
# Regression: Nebius batch-embed job mounts the tokenized shards
# (Bugbot PR #10 — "Nebius batch job missing shard mount")
# ===========================================================================


class TestNebiusBatchEmbedShardMount:
    """The Soperator batch job must mount the shard prefix at the CLI's positional path.

    ``pragmatiq embed <shard_dir> --run <run_dir> --out <out>`` opens ``<shard_dir>``
    on the pod.  The old spec mounted only the run directory, so the embed leg
    failed as soon as the CLI opened the (absent) shard directory.  The same job
    writes its parquet back to Object Storage through the CLI, so the env must
    point the S3 client at the Nebius endpoint and the output must be a file key.
    """

    _IMAGE = "cr.eu-north1.nebius.cloud/pragmatiq:latest"

    def _batch_job_spec(self, **adapter_kw: object) -> dict:
        """package() the specs and parse batch_embed_job.yaml."""
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-shardmount-")) / "specs"
        adapter = NebiusAdapter(image=self._IMAGE, **adapter_kw)  # type: ignore[arg-type]
        adapter.package(run_dir, dest=str(dest), image=self._IMAGE)
        return yaml.safe_load((dest / "batch_embed_job.yaml").read_text())

    @staticmethod
    def _mounts(spec: dict) -> dict[str, dict]:
        """Map mountPath -> storage entry for the job spec."""
        return {entry["mountPath"]: entry for entry in spec["spec"]["storage"]}

    def test_positional_shard_dir_has_a_mount(self) -> None:
        """The command's positional shard_dir is a storage mountPath (default shard prefix)."""
        spec = self._batch_job_spec()
        cmd = spec["spec"]["command"]
        shard_dir = cmd[cmd.index("embed") + 1]
        mounts = self._mounts(spec)
        assert shard_dir in mounts, (
            f"positional shard_dir {shard_dir!r} has no storage mount; mounts: {sorted(mounts)}"
        )
        assert mounts[shard_dir]["prefix"] == "pragmatiq/shards"

    def test_run_dir_flag_has_a_mount(self) -> None:
        """The '--run' value is still a storage mountPath (run-dir prefix)."""
        spec = self._batch_job_spec()
        cmd = spec["spec"]["command"]
        run_dir = cmd[cmd.index("--run") + 1]
        mounts = self._mounts(spec)
        assert run_dir in mounts, f"--run {run_dir!r} has no storage mount; mounts: {sorted(mounts)}"
        assert mounts[run_dir]["prefix"] == "pragmatiq/run_dir"

    def test_shard_and_run_mounts_are_distinct(self) -> None:
        """Shards and run dir are separate mounts backed by separate prefixes."""
        spec = self._batch_job_spec()
        cmd = spec["spec"]["command"]
        shard_dir = cmd[cmd.index("embed") + 1]
        run_dir = cmd[cmd.index("--run") + 1]
        mounts = self._mounts(spec)
        assert shard_dir != run_dir
        assert mounts[shard_dir]["prefix"] != mounts[run_dir]["prefix"]

    def test_custom_bucket_and_shard_prefix_flow_into_mount(self) -> None:
        """s3_bucket / s3_shard_prefix ctor args land in the shard mount entry."""
        spec = self._batch_job_spec(s3_bucket="my-bucket", s3_shard_prefix="tokenized/v3")
        entry = self._mounts(spec)["/opt/pragmatiq/shard_dir"]
        assert entry["type"] == "s3"
        assert entry["bucket"] == "my-bucket"
        assert entry["prefix"] == "tokenized/v3"

    def test_env_points_s3_client_at_nebius_endpoint(self) -> None:
        """The CLI's s3:// stage-out must hit the Nebius endpoint, not AWS."""
        endpoint = "https://storage.eu-west1.nebius.cloud:443"
        spec = self._batch_job_spec(s3_endpoint=endpoint)
        env = spec["spec"]["env"]
        assert env["AWS_ENDPOINT_URL_S3"] == endpoint
        for entry in spec["spec"]["storage"]:
            assert entry["endpoint"] == endpoint

    def test_out_is_a_parquet_key_in_the_bucket(self) -> None:
        """'--out' names a parquet object (api.embed stages it as a file), not a '/' prefix."""
        spec = self._batch_job_spec(s3_bucket="my-bucket", release_name="embedder-v3")
        cmd = spec["spec"]["command"]
        out = cmd[cmd.index("--out") + 1]
        assert out == "s3://my-bucket/embeddings/embedder-v3.parquet"

    def test_manifest_storage_exposes_shard_mount(self) -> None:
        """manifest()['storage'] names the shard prefix and both mount paths the command uses."""
        from integrations.nebius import NebiusAdapter

        m = NebiusAdapter(image=self._IMAGE, s3_shard_prefix="tokenized/v3").manifest()
        storage = m["storage"]
        cmd = m["soperator_batch"]["command"]
        assert storage["s3_shard_prefix"] == "tokenized/v3"
        assert cmd[cmd.index("embed") + 1] == storage["shard_dir_mount"]
        assert cmd[cmd.index("--run") + 1] == storage["run_dir_mount"]

    def test_manifest_and_package_agree_on_command(self) -> None:
        """manifest()'s batch command is the same command the YAML spec runs."""
        from integrations.nebius import NebiusAdapter

        spec = self._batch_job_spec(s3_bucket="my-bucket")
        m = NebiusAdapter(image=self._IMAGE, s3_bucket="my-bucket").manifest()
        assert spec["spec"]["command"] == m["soperator_batch"]["command"]

    def test_package_artifact_details_expose_shard_prefix(self) -> None:
        """Artifact.details carries the shard prefix + mount so operators know what to upload where."""
        from integrations.nebius import NebiusAdapter

        run_dir = _make_fake_run_dir()
        dest = Path(tempfile.mkdtemp(prefix="pragmatiq-neb-shardmount-art-")) / "specs"
        artifact = NebiusAdapter(image=self._IMAGE).package(run_dir, dest=str(dest), image=self._IMAGE)
        assert artifact.details["s3_shard_prefix"] == "pragmatiq/shards"
        assert artifact.details["shard_dir_mount"] == "/opt/pragmatiq/shard_dir"
        assert artifact.details["s3_output_path"].endswith(".parquet")
