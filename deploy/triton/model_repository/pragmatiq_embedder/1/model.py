"""Triton python-backend model: native varlen pragmatiq embedder (Phase 7).

Requests carry a JSON array of plain user records (the same dicts accepted by
``PragmaModel.embed_records``); the response is the ``[n_users, dim]`` embedding
matrix. Running the native PyTorch model serves the no-padding varlen path for
high throughput; the ONNX export trades that for a portable dense graph.
"""

import json
import os

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except Exception:  # allows import-time linting without the Triton runtime
    pb_utils = None


class TritonPythonModel:
    """Loads a trained pragmatiq model once and embeds batched record requests."""

    def initialize(self, args):
        import torch

        from pragmatiq.models.pragmatiq import PragmaModel

        params = json.loads(args["model_config"]).get("parameters", {})
        run_dir = params.get("run_dir", {}).get("string_value", os.environ.get("PRAGMATIQ_RUN", "/models/run"))
        # Respect Triton's assigned instance kind/device (a KIND_CPU instance must
        # run on CPU even if a GPU is visible); fall back to availability if the
        # runtime doesn't report a kind.
        kind = args.get("model_instance_kind", "")
        if kind == "GPU" and torch.cuda.is_available():
            self.device = f"cuda:{args.get('model_instance_device_id', '0')}"
        elif kind == "CPU":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PragmaModel.from_pretrained(run_dir, device=self.device)

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "records_json").as_numpy()
            records = json.loads(raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0]))
            emb = self.model.embed_records(records).astype(np.float32)
            out = pb_utils.Tensor("embeddings", emb)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        self.model = None
