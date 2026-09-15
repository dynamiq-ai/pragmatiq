# Triton serving image for the pragmatiq embedder.
#
# The python-backend model (model_repository/pragmatiq_embedder/1/model.py) imports
# `pragmatiq`, so the stock `tritonserver` image cannot serve it as-is — this image
# installs the package into Triton's Python. The package is installed with `--no-deps`
# so the image's CUDA build of torch is left untouched (a plain `pip install .` could
# resolve a different torch and break CUDA); every other runtime dependency is then
# installed from the same list pyproject.toml declares, so the image can never drift
# from the package (a hand-maintained list once shipped without fsspec, and the model
# died on its first load).
#
# Base image: NGC Triton 25.06 ships PyTorch 2.8 (pragmatiq needs torch >= 2.6).
# Verify a different tag's torch with:
#   docker run --rm nvcr.io/nvidia/tritonserver:<tag> python3 -c "import torch; print(torch.__version__)"
#
# Build (default model):    docker build -f deploy/triton/Dockerfile -t pragmatiq-triton .
# Build (Nemotron variant): docker build -f deploy/triton/Dockerfile \
#                             --build-arg EXTRAS=nemotron -t pragmatiq-triton:nemotron .
ARG TRITON_TAG=25.06-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

ARG EXTRAS=""
WORKDIR /opt/pragmatiq

COPY pyproject.toml README.md ./
COPY pragmatiq ./pragmatiq

# Base dependencies straight from pyproject (minus torch, which the image provides).
RUN pip install --no-cache-dir --no-deps . \
 && python3 - <<'PY' > /tmp/runtime-reqs.txt \
 && pip install --no-cache-dir -r /tmp/runtime-reqs.txt
import tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
print("\n".join(d for d in deps if not d.startswith("torch")))
PY

# The PRAGMA+Nemotron variant's frozen text encoder needs transformers; add it only
# when building for that variant so the default serving image stays lean.
RUN if echo "$EXTRAS" | grep -qw nemotron; then pip install --no-cache-dir "transformers>=4.40"; fi

# Sanity-check at build time that the serving runtime imports and torch is new enough
# (fails the build early rather than at first inference).
RUN python3 -c "import torch, pragmatiq; from pragmatiq.inference.serve import load, request_limits; \
    from pragmatiq.models.pragmatiq import PragmaModel; \
    v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); \
    assert v >= (2, 6), f'torch {torch.__version__} < 2.6'; print('torch', torch.__version__)"
