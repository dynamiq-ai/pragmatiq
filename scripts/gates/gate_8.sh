#!/usr/bin/env bash
# Acceptance check — nano end-to-end on CPU, validate, packaging.
#
#   1. validate unit tests + `pragmatiq validate` with actionable errors
#   2. nano end-to-end on CPU (synth -> tokenize -> nano pretrain -> probe),
#      target < 10 min; smoke test: asserts the pipeline runs and returns a
#      finite probe metric (the training check is the performance bar, not this
#      nano run)
#   3. packaging sanity: version tracks the package metadata, CLI entrypoint resolves,
#      the PEP 561 marker is present, and docs carry the required attribution line
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== validate module + actionable errors ==="
$PY -m pytest tests/test_validate.py -q

echo "=== nano end-to-end quickstart on CPU (<10 min) ==="
$PY - <<'EOF'
import time, tempfile
from pathlib import Path
from pragmatiq import api
t0 = time.time()
out = Path(tempfile.mkdtemp())
res = api.quickstart(out=out, n_users=2000, model_size="nano", max_steps=80, n_workers=2)
dt = time.time() - t0
print("  " + res["message"])
print(f"  elapsed {dt:.0f}s")
assert dt < 600, f"nano e2e took {dt:.0f}s (> 10 min budget)"
auc = res["probe"]["probe_auc"]
# The nano quickstart is a CPU PIPELINE smoke test (2k users, 80 steps): credit is
# low-prevalence and the model is tiny, so the probe sits near chance and the tiny
# eval set makes any fixed AUC threshold unstable. The training check is the
# performance bar (small model, relative beats-baseline criterion). Here we assert only that the
# end-to-end pipeline ran in budget and returned a finite, in-range metric.
import math
assert math.isfinite(auc) and 0.0 <= auc <= 1.0, f"quickstart returned an invalid probe AUC: {auc}"
print("nano end-to-end OK")
EOF

echo "=== packaging + attribution ==="
$PY - <<'EOF'
import pragmatiq
import tomllib
from importlib.metadata import version as _dist_version
from pathlib import Path
# __version__ tracks the installed distribution metadata, and both must match
# pyproject.toml so stale editable installs and release-metadata drift fail loud.
assert pragmatiq.__version__, "empty package version"
assert pragmatiq.__version__ == _dist_version("pragmatiq"), pragmatiq.__version__
pyproject_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
assert pragmatiq.__version__ == pyproject_version, (pragmatiq.__version__, pyproject_version)
from pragmatiq.cli import app  # CLI entrypoint resolves
need = "not affiliated with or endorsed by Revolut"
assert (Path(pragmatiq.__file__).parent / "py.typed").exists(), "py.typed missing"
docs = [Path("README.md"), Path("MODEL_CARD.md"), *Path("website/content/docs").rglob("*.mdx")]
for doc in docs:
    assert need in doc.read_text(), f"{doc} missing attribution line"
print(f"  version {pragmatiq.__version__}; attribution present in public docs")
print("packaging OK")
EOF

echo ""
echo "POLISH CHECKS GREEN"
