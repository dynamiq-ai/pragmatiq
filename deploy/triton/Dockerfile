# Triton serving image for the pragmatiq embedder (Phase 7).
#
# The python-backend model (model_repository/pragmatiq_embedder/1/model.py) imports
# `pragmatiq`, so the stock `tritonserver` image cannot serve it as-is — this image
# installs the package into Triton's Python. We install with `--no-deps` plus only the
# runtime dependencies the embedder needs, so the image's CUDA build of torch is left
# untouched (a plain `pip install .` could resolve a different torch and break CUDA),
# and training-only deps (lightning, matplotlib) are skipped.
#
# Build (default model):    docker build -f deploy/triton/Dockerfile -t pragmatiq-triton .
# Build (Nemotron variant): docker build -f deploy/triton/Dockerfile \
#                             --build-arg EXTRAS=nemotron -t pragmatiq-triton:nemotron .
ARG TRITON_TAG=24.08-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

ARG EXTRAS=""
WORKDIR /opt/pragmatiq

COPY pyproject.toml README.md ./
COPY pragmatiq ./pragmatiq

RUN pip install --no-cache-dir --no-deps . \
 && pip install --no-cache-dir \
      "numpy>=1.24" "pyarrow>=14.0" "pandas>=2.0" "lmdb>=1.4" "tokenizers>=0.15" \
      "omegaconf>=2.3" "pyyaml>=6.0" "scikit-learn>=1.3" "typer>=0.9" "tqdm>=4.66"

# The PRAGMA+Nemotron variant's frozen text encoder needs transformers; add it only
# when building for that variant so the default serving image stays lean.
RUN if echo "$EXTRAS" | grep -qw nemotron; then pip install --no-cache-dir "transformers>=4.40"; fi

# Sanity-check that the backend can import the package at build time (fails the build
# early rather than at first inference if a runtime dep is missing).
RUN python3 -c "import pragmatiq; from pragmatiq.models.pragmatiq import PragmaModel"
