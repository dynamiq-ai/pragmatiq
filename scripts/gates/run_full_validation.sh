#!/usr/bin/env bash
# Full-validation orchestrator: runs the wall-clock-sensitive check first, then
# the remaining checks and the test suite concurrently.
#
# DAG:
#   stage A: gate_1 ALONE — its wall-clock throughput assertion (proj < 600s,
#            linear 8-core extrapolation) needs an unloaded machine; any
#            concurrent load risks a false red.
#   stage B: every other gate script in scripts/gates (the same list CI runs)
#            and `pytest tests/ -q -m "not gpu"` as concurrent background jobs,
#            each tee-ing to its own log file.
#
# All stages always run (even after a failure); the script exits nonzero if
# any stage failed and prints a per-stage summary table with durations.
#
# Honored env pass-through (inherited by the child stages):
#   PRAGMATIQ_GATE_FULL       1 = full-scale checks (default: CI scale)
#   PRAGMATIQ_GATE_WORKERS    tokenize workers inside the data-pipeline gates (default 0)
#   PRAGMATIQ_GATE_SKIP_UNIT  1 = the gates skip their in-script unit tests
#                             (the orchestrator's pytest job covers them)
#
# NOTE: keep PRAGMATIQ_WRITE_RESULTS unset during this run — it writes
# README/notebook 04 and must only run after all checks join.
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

# ---- stage B: every other gate + pytest, concurrent ----
# (indexed arrays only: macOS ships bash 3.2, which has no associative arrays)
STAGES=(gate_2 gate_3 gate_4 gate_5 gate_6 gate_7 gate_8 gate_9_contract gate_integrations gate_10_byoc gate_serve_slim gate_storage pytest)
echo ""
echo "===== stage B: ${STAGES[*]} (concurrent) ====="
PIDS=(); STARTS=(); STATUS=(); SECS=()
for i in "${!STAGES[@]}"; do
    g=${STAGES[$i]}
    STARTS[$i]=$(date +%s)
    if [ "$g" = "pytest" ]; then
        ( set -o pipefail; "$PY" -m pytest tests/ -q -m "not gpu" 2>&1 | tee "$LOG_DIR/pytest.log" ) &
    else
        ( set -o pipefail; bash "scripts/gates/$g.sh" 2>&1 | tee "$LOG_DIR/$g.log" ) &
    fi
    PIDS[$i]=$!
done
for i in "${!STAGES[@]}"; do
    wait "${PIDS[$i]}"; STATUS[$i]=$?
    SECS[$i]=$(( $(date +%s) - STARTS[$i] ))
done

TOTAL_SECS=$(( $(date +%s) - T_TOTAL_START ))

# ---- summary ----
fmt() {  # fmt <name> <status> <secs> <log>
    local verdict="PASS"
    [ "$2" -ne 0 ] && verdict="FAIL($2)"
    printf "%-18s %-9s %6ss   %s\n" "$1" "$verdict" "$3" "$4"
}
echo ""
echo "===== full validation summary ====="
printf "%-18s %-9s %7s   %s\n" "stage" "status" "secs" "log"
fmt "gate_1"  "$GATE1_STATUS"  "$GATE1_SECS"  "$LOG_DIR/gate_1.log"
EXIT=0
[ "$GATE1_STATUS" -ne 0 ] && EXIT=1
SUM=$GATE1_SECS
for i in "${!STAGES[@]}"; do
    fmt "${STAGES[$i]}" "${STATUS[$i]}" "${SECS[$i]}" "$LOG_DIR/${STAGES[$i]}.log"
    [ "${STATUS[$i]}" -ne 0 ] && EXIT=1
    SUM=$((SUM + SECS[$i]))
done
echo "-----------------------------------"
echo "total wall-clock: ${TOTAL_SECS}s (stage sum: ${SUM}s)"
if [ "$EXIT" -eq 0 ]; then
    echo "FULL VALIDATION GREEN"
else
    echo "FULL VALIDATION RED"
fi
exit "$EXIT"
