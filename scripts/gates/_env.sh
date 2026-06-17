# Shared interpreter resolution for the gate scripts (source this; sets $PY).
#
# Picks a Python >=3.11: $PYTHON if set, else the project venv (.venv), else
# python3. Versions below 3.11 are rejected with a clear error, so the gates
# always run on an interpreter that satisfies pragmatiq's requirements (numpy
# ABI compatibility, datetime.UTC). pragmatiq requires Python 3.11+
# (pyproject requires-python).
_gate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$_gate_root/.venv/bin/python" ]; then
    PY="$_gate_root/.venv/bin/python"
  else
    PY="python3"
  fi
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
  echo "ERROR: '$PY' is not Python >=3.11 ($("$PY" --version 2>&1))." >&2
  echo "       pragmatiq requires Python 3.11+. Set PYTHON=/path/to/python3.11" >&2
  echo "       (e.g. PYTHON=$_gate_root/.venv/bin/python bash scripts/gates/gate_N.sh)." >&2
  exit 1
fi
export PY
