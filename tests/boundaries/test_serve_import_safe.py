"""Slim-serve boundary test.

Proves that the inference / embedding path (PragmaModel.embed_records) works
correctly with lightning, torch_geometric, transformers, and lightgbm BLOCKED,
and that none of those heavy packages get imported as a side-effect.

The test runs a subprocess using the same Python interpreter.  The subprocess
installs a sys.meta_path blocker at index 0 before importing pragmatiq, so the
main test process (which has those modules loaded for other tests) is not
disturbed.

This test must stay GREEN in the .venv where the heavy modules ARE installed
because the blocker makes them unimportable regardless.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# The SCRIPT that runs inside the subprocess
# ---------------------------------------------------------------------------
_BLOCKED = ["lightning", "torch_geometric", "transformers", "lightgbm"]

_SCRIPT = textwrap.dedent("""\
import sys

# ── Step 1: install import blocker at the front of meta_path ──────────────
BLOCKED = {blocked!r}

class _Blocker:
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in BLOCKED:
            raise ImportError(
                f"[slim-serve blocker] {{name!r}} is a heavy dep blocked in [serve] "
                f"install — serve path must not import it."
            )
        return None

sys.meta_path.insert(0, _Blocker())

# ── Step 2: build a tiny nano PragmaModel + fitted tokenizer ──────────────
import tempfile, numpy as np
from pathlib import Path
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig, iter_user_records
from pragmatiq.data.collate import VarlenCollator
from pragmatiq.models import ModelConfig, PragmaModel

tmp = Path(tempfile.mkdtemp())
generate(
    WorldConfig(n_users=10, months=14, n_merchants=30, seed=999,
                mule_ring_count=0, eval_month_credit=2, eval_month_short=8),
    tmp / "raw", n_workers=0, write_report=False,
)
tok = PragmaTokenizer(
    TokenizerConfig(target_vocab=512, n_buckets=8, categorical_threshold=20, seed=0)
).fit(tmp / "raw")

# nano-sized model — fast to construct on CPU
cfg = ModelConfig.preset("small", tok.vocab_size)
model = PragmaModel(cfg).eval()
model._tokenizer = tok  # attach tokenizer exactly as from_pretrained does

# ── Step 3: call embed_records with plain-dict records ────────────────────
records = [
    {{"user_id": "slim_1", "events": [
        {{"ts": 1_700_000_000_000_000, "source": "transaction",
          "fields": {{"amount": "12.50", "mcc": "5411", "merchant": "SHOP A"}}}},
    ], "attributes": {{}}, "lifelong": []}},
    {{"user_id": "slim_2", "events": [
        {{"ts": 1_700_003_600_000_000, "source": "app",
          "fields": {{"screen": "home", "action": "view"}}}},
    ], "attributes": {{}}, "lifelong": []}},
]

emb = model.embed_records(records)
assert emb.shape == (2, cfg.dim), (
    f"Expected shape (2, {{cfg.dim}}), got {{emb.shape}}"
)
assert np.isfinite(emb).all(), "embedding contains non-finite values"

# ── Step 4: assert blocked modules were NOT imported ─────────────────────
leaked = [
    m for m in sys.modules
    if any(m == b or m.startswith(b + ".") for b in BLOCKED)
]
assert not leaked, f"Blocked modules were imported: {{leaked}}"

print("SLIM_OK")
""").format(blocked=_BLOCKED)


def test_serve_slim_boundary() -> None:
    """embed_records works with heavy deps blocked; none of them are imported."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "SLIM_OK" in result.stdout, (
        f"slim-serve boundary test FAILED\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
