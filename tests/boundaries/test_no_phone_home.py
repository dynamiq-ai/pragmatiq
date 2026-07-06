"""No-phone-home boundary test.

Proves that the core embed_records path never opens non-loopback network
connections, even when run with sockets blocked.  Also verifies that
MetricLogger without wandb=True does NOT import wandb.

The test runs a subprocess with socket.socket.connect replaced by a blocker
that raises OSError for any non-loopback address.  Unix-domain sockets
(AF_UNIX) are allowed (TensorBoard uses them on some platforms).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# Subprocess script — run with sockets blocked
# ---------------------------------------------------------------------------

_SCRIPT = textwrap.dedent("""\
import socket as _socket
import sys

# ── Step 1: install socket blocker ────────────────────────────────────────
_real_connect = _socket.socket.connect
_real_connect_ex = _socket.socket.connect_ex

def _is_loopback(address):
    \"\"\"Return True for loopback / Unix-domain addresses.\"\"\"
    if not isinstance(address, tuple):
        # AF_UNIX path or other non-IP address — allow
        return True
    host = str(address[0])
    return host in ("127.0.0.1", "::1", "localhost", "")

def _blocked_connect(self, address):
    if _is_loopback(address):
        return _real_connect(self, address)
    raise OSError(f"NETWORK_BLOCKED: {address}")

def _blocked_connect_ex(self, address):
    if _is_loopback(address):
        return _real_connect_ex(self, address)
    raise OSError(f"NETWORK_BLOCKED: {address}")

_socket.socket.connect = _blocked_connect
_socket.socket.connect_ex = _blocked_connect_ex

# ── Step 2: build a tiny nano PragmaModel + fitted tokenizer ──────────────
import tempfile, numpy as np
from pathlib import Path
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig
from pragmatiq.models import ModelConfig, PragmaModel

tmp = Path(tempfile.mkdtemp())
generate(
    WorldConfig(n_users=10, months=14, n_merchants=50, seed=999,
                mule_ring_count=0, eval_month_credit=2, eval_month_short=8),
    tmp / "raw", n_workers=0, write_report=False,
)
tok = PragmaTokenizer(
    TokenizerConfig(target_vocab=512, n_buckets=8, categorical_threshold=20, seed=0)
).fit(tmp / "raw")

cfg = ModelConfig.preset("small", tok.vocab_size)
model = PragmaModel(cfg).eval()
model._tokenizer = tok

# ── Step 3: call embed_records with plain-dict records ────────────────────
records = [
    {"user_id": "u1", "events": [
        {"ts": 1_700_000_000_000_000, "source": "transaction",
         "fields": {"amount": "12.50", "mcc": "5411", "merchant": "SHOP A"}},
    ], "attributes": {}, "lifelong": []},
    {"user_id": "u2", "events": [
        {"ts": 1_700_003_600_000_000, "source": "app",
         "fields": {"screen": "home", "action": "view"}},
    ], "attributes": {}, "lifelong": []},
]

emb = model.embed_records(records)
assert emb.shape == (2, cfg.dim), f"Expected (2, {cfg.dim}), got {emb.shape}"
assert np.isfinite(emb).all(), "embedding contains non-finite values"

# ── Step 4: assert wandb is NOT imported when not opted in ────────────────
import tempfile as _tf, pathlib as _pl
_log_dir = _pl.Path(_tf.mkdtemp())

from pragmatiq.experiments.tracking import MetricLogger
_logger = MetricLogger(str(_log_dir), tensorboard=False, wandb=False)
_logger.close()

# wandb must not have been imported as a side-effect
wandb_leaked = [m for m in sys.modules if m == "wandb" or m.startswith("wandb.")]
assert not wandb_leaked, f"wandb was imported despite wandb=False: {wandb_leaked}"

print("PHONE_HOME_OK")
""")


def test_no_phone_home() -> None:
    """embed_records works with all non-loopback sockets blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "PHONE_HOME_OK" in result.stdout, (
        f"no-phone-home test FAILED\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Runtime temp-dir cleanup test (in-process)
# ---------------------------------------------------------------------------


def test_runtime_remote_tempdir_cleanup() -> None:
    """Runtime.close() removes the staging temp-dir for remote loads."""
    from pathlib import Path

    from pragmatiq.data.synthetic import WorldConfig, generate
    from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig
    from pragmatiq.inference.serve.runtime import Runtime
    from pragmatiq.models import ModelConfig, PragmaModel

    # Build a nano model so we can construct a real Runtime
    _tmp = Path(tempfile.mkdtemp())
    generate(
        WorldConfig(
            n_users=10,
            months=14,
            n_merchants=50,
            seed=999,
            mule_ring_count=0,
            eval_month_credit=2,
            eval_month_short=8,
        ),
        _tmp / "raw",
        n_workers=0,
        write_report=False,
    )
    tok = PragmaTokenizer(
        TokenizerConfig(target_vocab=512, n_buckets=8, categorical_threshold=20, seed=0)
    ).fit(_tmp / "raw")
    cfg = ModelConfig.preset("small", tok.vocab_size)
    model = PragmaModel(cfg).eval()
    model._tokenizer = tok

    # Simulate what load() does for a remote path: set _staging_dir
    staging = tempfile.mkdtemp(prefix="pragmatiq-test-staging-")
    assert os.path.isdir(staging), "staging dir should exist before close()"

    rt = Runtime(model=model, device="cpu")
    rt._staging_dir = staging  # type: ignore[assignment]

    rt.close()

    assert not os.path.exists(staging), (
        f"Runtime.close() did not remove staging dir: {staging}"
    )
    # Second close() should be a no-op (not raise)
    rt.close()


def test_runtime_context_manager_cleanup() -> None:
    """Runtime used as a context manager cleans up the staging dir on exit."""
    from pathlib import Path

    from pragmatiq.data.synthetic import WorldConfig, generate
    from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig
    from pragmatiq.inference.serve.runtime import Runtime
    from pragmatiq.models import ModelConfig, PragmaModel

    _tmp = Path(tempfile.mkdtemp())
    generate(
        WorldConfig(
            n_users=10,
            months=14,
            n_merchants=50,
            seed=999,
            mule_ring_count=0,
            eval_month_credit=2,
            eval_month_short=8,
        ),
        _tmp / "raw",
        n_workers=0,
        write_report=False,
    )
    tok = PragmaTokenizer(
        TokenizerConfig(target_vocab=512, n_buckets=8, categorical_threshold=20, seed=0)
    ).fit(_tmp / "raw")
    cfg = ModelConfig.preset("small", tok.vocab_size)
    model = PragmaModel(cfg).eval()
    model._tokenizer = tok

    staging = tempfile.mkdtemp(prefix="pragmatiq-test-ctx-")
    assert os.path.isdir(staging)

    with Runtime(model=model, device="cpu") as rt:
        rt._staging_dir = staging  # type: ignore[assignment]

    assert not os.path.exists(staging), (
        f"Runtime.__exit__ did not remove staging dir: {staging}"
    )
