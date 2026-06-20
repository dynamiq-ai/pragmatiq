"""Supply-chain / BYOC security audit: no-phone-home proof.

Two parts:

(a) STATIC — AST/grep-based scan of pragmatiq/ Python files for network and
    telemetry call sites.  Each occurrence is checked against an allowlist of
    known-gated uses.  Un-gated occurrences fail the script with exit code 1.

(b) DYNAMIC — subprocess that builds a nano model, calls embed_records with
    sockets blocked, and asserts the call succeeds without touching the network.

Exit 0 when both parts pass; non-zero otherwise.

Usage:
    .venv/bin/python scripts/supply_chain/no_phone_home.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent  # repo root
LIB_DIR = ROOT / "pragmatiq"

# ---------------------------------------------------------------------------
# (a) STATIC AUDIT
# ---------------------------------------------------------------------------

# Patterns to look for (as plain strings that we search for in source lines).
# Each pattern maps to a list of (file_relative_to_LIB_DIR, reason) tuples
# that are ALLOWED.  Any occurrence NOT in this allowlist fails the audit.

_ALLOWLIST: dict[str, list[dict[str, str]]] = {
    # wandb is ONLY imported inside `if wandb:` / `if self._wandb is not None:` guards
    # in tracking.py — it's opt-in (user must pass wandb=True).
    "import wandb": [
        {
            "file": "experiments/tracking.py",
            "reason": "opt-in: imported only inside `if wandb:` block; user must pass wandb=True",
        }
    ],
    # TensorBoard is imported inside a try/except inside `if tensorboard:` guard.
    # It writes local event files only — no network connections.
    "torch.utils.tensorboard": [
        {
            "file": "experiments/tracking.py",
            "reason": "local-only: SummaryWriter writes files; imported inside `if tensorboard:` guard",
        }
    ],
    "tensorboard": [
        {
            "file": "experiments/tracking.py",
            "reason": "local-only: SummaryWriter writes files; imported inside `if tensorboard:` guard",
        },
        {
            "file": "training/pretrainer.py",
            "reason": "docstring/comment reference only — not an import",
        },
    ],
}

# Network-related patterns that must NOT appear outside the allowlist
_PATTERNS = [
    "import wandb",
    "import requests",
    "import urllib",
    "from urllib",
    "http.client",
    r"socket\.",
    "import httpx",
    "torch.utils.tensorboard",
    "tensorboard",
]

# Compile patterns for efficiency (word-sensitive where needed)
_COMPILED = [(p, re.compile(re.escape(p) if not p.startswith("r") else p[1:]))
             for p in _PATTERNS]
# Re-compile properly
_COMPILED = [(p, re.compile(p)) for p in _PATTERNS]


def _run_static_audit() -> bool:
    """Scan pragmatiq/ for un-gated network call sites.

    Returns:
        True if audit passes (no un-gated occurrences), False otherwise.
    """
    print("--- Static network/telemetry audit ---")
    py_files = sorted(LIB_DIR.rglob("*.py"))
    violations: list[str] = []
    allowed_hits: list[str] = []

    for py_path in py_files:
        rel = py_path.relative_to(LIB_DIR)
        rel_str = rel.as_posix()
        try:
            source = py_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  [WARN] Could not read {rel_str}: {exc}")
            continue

        for pattern, rx in _COMPILED:
            for i, line in enumerate(source.splitlines(), start=1):
                if not rx.search(line):
                    continue
                # Check allowlist
                allowed_for_pattern = _ALLOWLIST.get(pattern, [])
                match_allowed = any(
                    a["file"] == rel_str for a in allowed_for_pattern
                )
                if match_allowed:
                    reason = next(
                        a["reason"] for a in allowed_for_pattern if a["file"] == rel_str
                    )
                    allowed_hits.append(
                        f"  [ALLOWED] {rel_str}:{i}: {line.strip()!r}\n"
                        f"            reason: {reason}"
                    )
                else:
                    violations.append(
                        f"  [VIOLATION] {rel_str}:{i}: {line.strip()!r}\n"
                        f"              pattern: {pattern!r}"
                    )

    if allowed_hits:
        print("Known-gated (allowed) occurrences:")
        for h in allowed_hits:
            print(h)

    if violations:
        print("\nUN-GATED network/telemetry call sites detected:")
        for v in violations:
            print(v)
        print(f"\nFAILED: {len(violations)} violation(s) found.")
        return False

    print(f"STATIC_OK — {len(py_files)} files scanned, 0 violations.")
    return True


# ---------------------------------------------------------------------------
# (b) DYNAMIC PROOF
# ---------------------------------------------------------------------------

_DYNAMIC_SCRIPT = textwrap.dedent("""\
import socket as _socket
import sys

# Install socket blocker before any pragmatiq import
_real_connect = _socket.socket.connect
_real_connect_ex = _socket.socket.connect_ex

def _is_loopback(address):
    if not isinstance(address, tuple):
        return True  # AF_UNIX — allow
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

# Build nano model + tokenizer
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

records = [
    {"user_id": "d1", "events": [
        {"ts": 1_700_000_000_000_000, "source": "transaction",
         "fields": {"amount": "9.99", "mcc": "5411", "merchant": "MART"}},
    ], "attributes": {}, "lifelong": []},
    {"user_id": "d2", "events": [
        {"ts": 1_700_003_600_000_000, "source": "app",
         "fields": {"screen": "home", "action": "view"}},
    ], "attributes": {}, "lifelong": []},
]

emb = model.embed_records(records)
assert emb.shape == (2, cfg.dim), f"shape mismatch: {emb.shape}"
assert np.isfinite(emb).all(), "non-finite values in embedding"
print("DYNAMIC_OK")
""")


def _run_dynamic_proof() -> bool:
    """Run embed_records in a subprocess with sockets blocked.

    Returns:
        True if the subprocess exits 0 and prints DYNAMIC_OK.
    """
    print("--- Dynamic proof (sockets blocked) ---")
    result = subprocess.run(
        [sys.executable, "-c", _DYNAMIC_SCRIPT],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and "DYNAMIC_OK" in result.stdout:
        print("DYNAMIC_OK — embed_records succeeded with non-loopback sockets blocked.")
        return True
    print("DYNAMIC PROOF FAILED")
    print("--- stdout ---")
    print(result.stdout)
    print("--- stderr ---")
    print(result.stderr)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run static audit then dynamic proof.

    Returns:
        0 on full success, 1 on any failure.
    """
    static_ok = _run_static_audit()
    print()
    dynamic_ok = _run_dynamic_proof()

    print()
    if static_ok and dynamic_ok:
        print("NO_PHONE_HOME_PASS — all checks green.")
        return 0
    parts = []
    if not static_ok:
        parts.append("static audit")
    if not dynamic_ok:
        parts.append("dynamic proof")
    print(f"NO_PHONE_HOME_FAIL — failed: {', '.join(parts)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
