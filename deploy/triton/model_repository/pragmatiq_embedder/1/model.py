"""Triton python-backend model: native varlen pragmatiq embedder.

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
        # Device selection. CPU-first (global rule 5): with nothing set the model
        # runs on CPU even where a GPU is visible. Setting PRAGMATIQ_SERVE_GPU=1
        # opts into CUDA serving when a GPU is actually present — this is the
        # switch the deploy script sets so "serving on CUDA" is true end to end.
        # A Triton-assigned GPU instance also pins to its device.
        serve_gpu = os.environ.get("PRAGMATIQ_SERVE_GPU", "") == "1"
        kind = args.get("model_instance_kind", "")
        if kind == "GPU" and torch.cuda.is_available():
            self.device = f"cuda:{args.get('model_instance_device_id', '0')}"
        elif serve_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
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
