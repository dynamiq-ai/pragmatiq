"""Triton python-backend model: thin delegate to pragmatiq.inference.serve.

Requests carry a JSON array of plain user records (the same dicts accepted by
``PragmaModel.embed_records``); the response is the ``[n_users, dim]`` embedding
matrix. Running the native PyTorch model serves the no-padding varlen path for
high throughput; the ONNX export trades that for a portable dense graph.

All serving logic (device resolution, model loading, request/response encoding)
lives in ``pragmatiq.inference.serve`` (runtime.py + contract.py).  This file
is intentionally kept thin: it reads Triton args, delegates to the library, and
wraps the result in a ``pb_utils.Tensor``.  ``pb_utils`` is ONLY imported here,
never in the library.
"""

import json
import os

try:
    import triton_python_backend_utils as pb_utils
except Exception:  # allows import-time linting without the Triton runtime
    pb_utils = None

from pragmatiq.inference.serve import decode_request, load
from pragmatiq.inference.serve.contract import INPUT_NAME, OUTPUT_NAME


class TritonPythonModel:
    """Loads a trained pragmatiq model once and embeds batched record requests."""

    def initialize(self, args):
        params = json.loads(args["model_config"]).get("parameters", {})
        run_dir = params.get("run_dir", {}).get("string_value", os.environ.get("PRAGMATIQ_RUN", "/models/run"))
        kind = args.get("model_instance_kind", "")
        device_id = args.get("model_instance_device_id", "0")
        self.runtime = load(
            run_dir,
            instance_kind=kind,
            instance_device_id=device_id,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, INPUT_NAME).as_numpy()[0]
            records = decode_request(raw)
            emb = self.runtime.embed(records)
            out = pb_utils.Tensor(OUTPUT_NAME, emb)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        self.runtime = None
