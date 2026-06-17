#!/usr/bin/env bash
# Full-validation orchestrator: runs the wall-clock-sensitive gate first, then
# the remaining gates and the test suite concurrently.
#
# DAG:
#   stage A: gate_1 ALONE — its wall-clock throughput assertion (proj < 600s,
#            linear 8-core extrapolation) needs an unloaded machine; any
#            concurrent load risks a false red.
#   stage B: gate_5, gate_6 and `pytest tests/ -q` as concurrent background
#            jobs, each tee-ing to its own log file.
#
# All stages always run (even after a failure); the script exits nonzero if
# any stage failed and prints a per-stage summary table with durations.
#
# Honored env pass-through (inherited by the child stages):
#   PRAGMATIQ_GATE_FULL       1 = full-scale gates (default: CI scale)
#   PRAGMATIQ_GATE_WORKERS    tokenize workers inside gates 5/6 (default 0)
#   PRAGMATIQ_GATE_SKIP_UNIT  1 = gates 5/6 skip their in-gate unit tests
#                             (the orchestrator's pytest job covers them)
#
# NOTE: keep PRAGMATIQ_WRITE_RESULTS unset during this run — it writes
# README/notebook 04 and must only run after all gates join.
#
# Runnable outside Claude Code: bash scripts/gates/run_full_validation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

LOG_DIR=${PRAGMATIQ_VALIDATION_LOGS:-$(mktemp -d -t pragmatiq-validation.XXXXXX)}
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"
echo "PRAGMATIQ_GATE_FULL=${PRAGMATIQ_GATE_FULL:-0}" \
     "PRAGMATIQ_GATE_WORKERS=${PRAGMATIQ_GATE_WORKERS:-0}" \
     "PRAGMATIQ_GATE_SKIP_UNIT=${PRAGMATIQ_GATE_SKIP_UNIT:-0}"
if [ "${PRAGMATIQ_WRITE_RESULTS:-0}" = "1" ]; then
    echo "WARNING: PRAGMATIQ_WRITE_RESULTS=1 is unsafe during the parallel stage; unsetting for this run."
    unset PRAGMATIQ_WRITE_RESULTS
fi

T_TOTAL_START=$(date +%s)

# ---- stage A: gate_1 alone (wall-clock-sensitive throughput assertion) ----
echo ""
echo "===== stage A: gate_1 (alone) ====="
GATE1_START=$(date +%s)
( set -o pipefail; bash scripts/gates/gate_1.sh 2>&1 | tee "$LOG_DIR/gate_1.log" )
GATE1_STATUS=$?
GATE1_SECS=$(( $(date +%s) - GATE1_START ))

# ---- stage B: gate_5 + gate_6 + pytest, concurrent ----
echo ""
echo "===== stage B: gate_5 / gate_6 / pytest (concurrent) ====="
GATE5_START=$(date +%s)
( set -o pipefail; bash scripts/gates/gate_5.sh 2>&1 | tee "$LOG_DIR/gate_5.log" ) &
GATE5_PID=$!
GATE6_START=$(date +%s)
( set -o pipefail; bash scripts/gates/gate_6.sh 2>&1 | tee "$LOG_DIR/gate_6.log" ) &
GATE6_PID=$!
PYTEST_START=$(date +%s)
( set -o pipefail; "$PY" -m pytest tests/ -q 2>&1 | tee "$LOG_DIR/pytest.log" ) &
PYTEST_PID=$!

wait "$GATE5_PID"; GATE5_STATUS=$?
GATE5_SECS=$(( $(date +%s) - GATE5_START ))
wait "$GATE6_PID"; GATE6_STATUS=$?
GATE6_SECS=$(( $(date +%s) - GATE6_START ))
wait "$PYTEST_PID"; PYTEST_STATUS=$?
PYTEST_SECS=$(( $(date +%s) - PYTEST_START ))

TOTAL_SECS=$(( $(date +%s) - T_TOTAL_START ))

# ---- summary ----
fmt() {  # fmt <name> <status> <secs> <log>
    local verdict="PASS"
    [ "$2" -ne 0 ] && verdict="FAIL($2)"
    printf "%-14s %-9s %6ss   %s\n" "$1" "$verdict" "$3" "$4"
}
echo ""
echo "===== full validation summary ====="
printf "%-14s %-9s %7s   %s\n" "stage" "status" "secs" "log"
fmt "gate_1"  "$GATE1_STATUS"  "$GATE1_SECS"  "$LOG_DIR/gate_1.log"
fmt "gate_5"  "$GATE5_STATUS"  "$GATE5_SECS"  "$LOG_DIR/gate_5.log"
fmt "gate_6"  "$GATE6_STATUS"  "$GATE6_SECS"  "$LOG_DIR/gate_6.log"
fmt "pytest"  "$PYTEST_STATUS" "$PYTEST_SECS" "$LOG_DIR/pytest.log"
echo "-----------------------------------"
echo "total wall-clock: ${TOTAL_SECS}s (stage sum: $((GATE1_SECS + GATE5_SECS + GATE6_SECS + PYTEST_SECS))s)"

EXIT=0
for s in "$GATE1_STATUS" "$GATE5_STATUS" "$GATE6_STATUS" "$PYTEST_STATUS"; do
    [ "$s" -ne 0 ] && EXIT=1
done
if [ "$EXIT" -eq 0 ]; then
    echo "FULL VALIDATION GREEN"
else
    echo "FULL VALIDATION RED"
fi
exit "$EXIT"
