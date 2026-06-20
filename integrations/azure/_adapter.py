"""Azure adapter implementation (STUB + Helm chart skeleton, offline-testable).

Azure Kubernetes Service (AKS) can host the NVIDIA Triton Inference Server
container via a standard Kubernetes ``Deployment`` + ``Service``.  This adapter
generates the Helm chart skeleton that an operator customises and deploys with
``helm install``.

Live-deploy status: DOCUMENTED STUB
------------------------------------
The offline artifact generation (``manifest()``, ``package()``) is real and
produces a valid Helm chart skeleton that an operator can use immediately.
The LIVE AKS deploy (pushing the image to ACR, creating the AKS cluster,
running ``helm install``) is documented in ``docs/INTEGRATIONS.md`` but NOT
implemented here — ``deploy_live()`` raises ``NotImplementedError`` with a
pointer to the runbook.

Helm chart layout
-----------------
``package()`` writes the following skeleton to *dest*::

    dest/
    ├── Chart.yaml          (Helm chart metadata)
    ├── values.yaml         (default values: image, port, run-dir mount, ...)
    └── templates/
        └── deployment.yaml (Deployment + Service referencing image/port/health)

The chart uses ``values.yaml`` variables throughout so an operator can override
image, replica count, PV claim name, etc. without editing templates.

Contract port
-------------
The container port and health path follow the serving contract defined in
``pragmatiq.inference.serve.contract`` (port 8000, path ``/v2/health/ready``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations._base import Artifact

# ---------------------------------------------------------------------------
# Serving contract constants (mirrored here to avoid a runtime import in the
# offline path; values are stable and match pragmatiq.inference.serve.contract)
# ---------------------------------------------------------------------------

_CONTRACT_PORT = 8000
_CONTRACT_HEALTH_PATH = "/v2/health/ready"
_CONTRACT_INFER_PATH = "/v2/models/pragmatiq_embedder/infer"

# Default Azure / Kubernetes settings
_DEFAULT_NAMESPACE = "pragmatiq"
_DEFAULT_REPLICAS = 1

# Helm chart templates
_CHART_YAML_TMPL = """\
apiVersion: v2
name: pragmatiq-embedder
description: pragmatiq PRAGMA embedder on Azure Kubernetes Service (AKS)
type: application
version: 0.1.0
appVersion: "{app_version}"
keywords:
  - pragmatiq
  - embedder
  - triton
  - aks
home: https://github.com/dynamiq/pragmatiq
"""

_VALUES_YAML_TMPL = """\
# Default values for pragmatiq-embedder Helm chart.
# Override with: helm install pragmatiq-embedder . -f my-values.yaml

replicaCount: {replicas}

image:
  repository: "{image_repository}"
  tag: "{image_tag}"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: {contract_port}

container:
  port: {contract_port}
  healthPath: "{health_path}"
  inferPath: "{infer_path}"

# Mount the run directory from an Azure Blob-backed PVC (abfs://) or a
# pre-populated Azure Disk.  Operator must create the PVC before installing.
storage:
  runDir: "/opt/pragmatiq/run_dir"
  pvcName: "pragmatiq-run-pvc"

env:
  PRAGMATIQ_RUN: "/opt/pragmatiq/run_dir"
  PRAGMATIQ_SERVE_GPU: "1"

resources:
  requests:
    cpu: "2"
    memory: "8Gi"
  limits:
    cpu: "8"
    memory: "32Gi"
    # nvidia.com/gpu: "1"   # uncomment for GPU node pool

nodeSelector: {{}}
tolerations: []
affinity: {{}}
"""

_DEPLOYMENT_YAML_TMPL = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pragmatiq-embedder
  namespace: {namespace}
  labels:
    app: pragmatiq-embedder
spec:
  replicas: {{{{ .Values.replicaCount }}}}
  selector:
    matchLabels:
      app: pragmatiq-embedder
  template:
    metadata:
      labels:
        app: pragmatiq-embedder
    spec:
      containers:
        - name: pragmatiq-embedder
          image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag }}}}"
          imagePullPolicy: {{{{ .Values.image.pullPolicy }}}}
          ports:
            - name: http
              containerPort: {{{{ .Values.container.port }}}}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: {{{{ .Values.container.healthPath }}}}
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: {{{{ .Values.container.healthPath }}}}
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
          env:
            {{{{- range $k, $v := .Values.env }}}}
            - name: {{{{ $k }}}}
              value: "{{{{ $v }}}}"
            {{{{- end }}}}
          volumeMounts:
            - name: run-dir
              mountPath: {{{{ .Values.storage.runDir }}}}
          resources:
            {{{{- toYaml .Values.resources | nindent 12 }}}}
      volumes:
        - name: run-dir
          persistentVolumeClaim:
            claimName: {{{{ .Values.storage.pvcName }}}}
      nodeSelector:
        {{{{- toYaml .Values.nodeSelector | nindent 8 }}}}
      tolerations:
        {{{{- toYaml .Values.tolerations | nindent 8 }}}}
      affinity:
        {{{{- toYaml .Values.affinity | nindent 8 }}}}
---
apiVersion: v1
kind: Service
metadata:
  name: pragmatiq-embedder
  namespace: {namespace}
spec:
  type: {{{{ .Values.service.type }}}}
  selector:
    app: pragmatiq-embedder
  ports:
    - name: http
      port: {{{{ .Values.service.port }}}}
      targetPort: http
      protocol: TCP
"""


class AzureAdapter:
    """Stub adapter for deploying pragmatiq on Azure Kubernetes Service (AKS).

    ``manifest()`` and ``package()`` are fully offline.  ``deploy_live()``
    raises ``NotImplementedError`` — live AKS deploy is documented in
    ``docs/INTEGRATIONS.md``.  ``healthcheck()`` builds the request offline
    and requires ``requests`` for the live leg.

    Args:
        image: The container image URI for the Triton serving image
               (e.g. from Azure Container Registry).
        namespace: Kubernetes namespace to deploy into.
        replicas: Initial replica count for the Deployment.
        release_name: Helm release name used in ``manifest()`` metadata.
    """

    #: Short adapter identifier used in logging and artifact kind strings.
    name: str = "azure"

    def __init__(
        self,
        image: str,
        *,
        namespace: str = _DEFAULT_NAMESPACE,
        replicas: int = _DEFAULT_REPLICAS,
        release_name: str = "pragmatiq-embedder",
    ) -> None:
        self._image = image
        self._namespace = namespace
        self._replicas = replicas
        self._release_name = release_name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_image(self) -> tuple[str, str]:
        """Split ``registry/repo:tag`` into ``(repository, tag)``."""
        if ":" in self._image:
            repo, tag = self._image.rsplit(":", 1)
        else:
            repo, tag = self._image, "latest"
        return repo, tag

    # ------------------------------------------------------------------
    # OFFLINE: manifest()
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return a declarative Azure AKS deploy spec as a plain ``dict``.

        Describes what an operator needs to deploy the pragmatiq embedder on
        AKS via Helm.  Fully offline — no Azure SDK required.

        Returns:
            A dict with:

            * ``"adapter"`` — adapter name ``"azure"``.
            * ``"helm"`` — Helm chart metadata (release name, namespace, image).
            * ``"container"`` — contract port and health/infer paths.
            * ``"storage"`` — expected PVC mount paths for the run directory.
            * ``"live_ops_status"`` — honest stub declaration.
        """
        repo, tag = self._split_image()
        return {
            "adapter": self.name,
            "helm": {
                "release_name": self._release_name,
                "namespace": self._namespace,
                "replicas": self._replicas,
                "image": {
                    "repository": repo,
                    "tag": tag,
                },
            },
            "container": {
                "port": _CONTRACT_PORT,
                "health_path": _CONTRACT_HEALTH_PATH,
                "infer_path": _CONTRACT_INFER_PATH,
            },
            "storage": {
                "run_dir_mount": "/opt/pragmatiq/run_dir",
                "pvc_name": "pragmatiq-run-pvc",
                "note": (
                    "Back the PVC with Azure Blob Storage (abfs://) via "
                    "the Azure Blob CSI Driver, or pre-copy to Azure Disk."
                ),
            },
            "env": {
                "PRAGMATIQ_RUN": "/opt/pragmatiq/run_dir",
                "PRAGMATIQ_SERVE_GPU": "1",
            },
            "live_ops_status": (
                "STUB — live AKS deploy is documented, not implemented; "
                "see docs/INTEGRATIONS.md"
            ),
        }

    # ------------------------------------------------------------------
    # OFFLINE: package()
    # ------------------------------------------------------------------

    def package(
        self,
        run_dir: str | Path,
        *,
        dest: str,
        image: str,
    ) -> Artifact:
        """Write a minimal Helm chart skeleton to *dest* and return an Artifact.

        Produces a ready-to-customise Helm chart::

            dest/
            ├── Chart.yaml
            ├── values.yaml
            └── templates/
                └── deployment.yaml

        The chart references the Triton image URI and the serving contract's
        container port / health path via ``values.yaml`` variables.

        This method is fully offline — it uses stdlib ``pathlib`` for file
        writes and does NOT push to Azure Container Registry or AKS.

        Args:
            run_dir: Path to the trained run directory (used only in artifact
                     details; the chart references it via the PVC mount path).
            dest: Local directory path where the Helm chart will be written.
            image: Container image URI; overrides the instance image for this
                   call and is recorded in artifact details.

        Returns:
            An :class:`~integrations._base.Artifact` with
            ``kind="helm-chart"`` and ``path_or_uri=dest``.
        """
        run_dir = Path(run_dir)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "templates").mkdir(exist_ok=True)

        # Resolve image for this call (may differ from self._image)
        if ":" in image:
            repo, tag = image.rsplit(":", 1)
        else:
            repo, tag = image, "latest"

        # 1. Chart.yaml
        (dest_path / "Chart.yaml").write_text(
            _CHART_YAML_TMPL.format(app_version=tag),
            encoding="utf-8",
        )

        # 2. values.yaml
        (dest_path / "values.yaml").write_text(
            _VALUES_YAML_TMPL.format(
                replicas=self._replicas,
                image_repository=repo,
                image_tag=tag,
                contract_port=_CONTRACT_PORT,
                health_path=_CONTRACT_HEALTH_PATH,
                infer_path=_CONTRACT_INFER_PATH,
            ),
            encoding="utf-8",
        )

        # 3. templates/deployment.yaml
        (dest_path / "templates" / "deployment.yaml").write_text(
            _DEPLOYMENT_YAML_TMPL.format(namespace=self._namespace),
            encoding="utf-8",
        )

        return Artifact(
            kind="helm-chart",
            path_or_uri=str(dest_path),
            details={
                "run_dir": str(run_dir),
                "image": image,
                "namespace": self._namespace,
                "replicas": self._replicas,
                "contract_port": _CONTRACT_PORT,
                "health_path": _CONTRACT_HEALTH_PATH,
                "files": ["Chart.yaml", "values.yaml", "templates/deployment.yaml"],
            },
        )

    # ------------------------------------------------------------------
    # DOCUMENTED STUB: deploy_live()
    # ------------------------------------------------------------------

    def deploy_live(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Live AKS deploy is documented, not implemented.

        The Helm chart produced by :meth:`package` is a real, ready-to-use
        skeleton.  Completing the deployment requires:

        1. Pushing the Triton image to Azure Container Registry (ACR).
        2. Creating an AKS cluster with the right node pool (GPU or CPU).
        3. Running ``helm install <release> <chart-dir> --namespace pragmatiq``.

        See ``docs/INTEGRATIONS.md`` for the full step-by-step runbook.

        Raises:
            NotImplementedError: Always — live AKS deploy is not automated.
        """
        raise NotImplementedError(
            "Azure live deploy is documented, not implemented; "
            "see docs/INTEGRATIONS.md"
        )

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): healthcheck()
    # ------------------------------------------------------------------

    def healthcheck(self, endpoint: str) -> bool:
        """Hit the AKS endpoint with a contract-compliant payload.

        The request payload is built offline via
        ``pragmatiq.inference.serve.contract.encode_request`` so every adapter
        speaks the same wire format.  The live HTTP call requires ``requests``.

        Args:
            endpoint: The full HTTPS URL of the AKS endpoint
                      (e.g. ``"http://<aks-lb-ip>:8000"``).

        Returns:
            ``True`` if the endpoint returned a valid response.

        Raises:
            MissingExtraError: If the ``requests`` package is not installed.
        """
        from integrations._base import _require

        _require("requests", "requests")
        import requests  # noqa: PLC0415 — intentionally lazy

        from pragmatiq.inference.serve.contract import encode_request

        records = [{"user_id": "healthcheck", "events": [], "attributes": {}, "lifelong": []}]
        payload = encode_request(records)

        infer_url = endpoint.rstrip("/") + _CONTRACT_INFER_PATH
        response = requests.post(
            infer_url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        response.raise_for_status()
        return True
