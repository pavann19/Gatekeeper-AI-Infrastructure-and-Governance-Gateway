#!/usr/bin/env bash
# Orchestrates the perf-baseline runbook end to end: full test suite, the
# closed-loop benchmark at concurrency 1/4/16, and a py-spy flamegraph --
# everything logged to disk so it can be reviewed after the fact rather than
# watched live. Safe to run unattended: every heavy step checks free RAM
# first and SKIPS (not crashes) if there isn't enough, recording why in the
# summary. Does NOT touch the judge/Ollama probe -- see run_judge_probe.sh,
# a separate, explicitly-run script for that, because it crashed this
# project's dev machine once already (see docs/perf/01-baseline-analysis.md
# section 6).
#
#   bash scripts/run_perf_baseline.sh
#
# Everything lands under _evidence/perf/run_<timestamp>/:
#   00_summary.txt        PASS/SKIP/FAIL per step, read this first
#   01_pytest.log
#   02_server.log
#   03_bench_c1.log  03_bench_c4.log  03_bench_c16.log
#   04_pyspy.log
#   <git-sha>-<concurrency>.json   (also copied here from _evidence/perf/)
#   <git-sha>-c16-flamegraph.svg   (ditto)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="_evidence/perf/run_${TS}"
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/00_summary.txt"
: > "$SUMMARY"

log_step() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

free_gb() {
    powershell -NoProfile -Command \
        '$o=Get-CimInstance Win32_OperatingSystem; "{0:N2}" -f ($o.FreePhysicalMemory/1MB)' \
        2>/dev/null | tr -d '\r'
}

require_gb() {
    # require_gb <min_gb> <step_name> -- returns 1 (caller should skip) if
    # free RAM is below the threshold, after logging why.
    local min="$1" name="$2"
    local free
    free="$(free_gb)"
    log_step "RAM check for '$name': ${free} GB free (need >= ${min} GB)"
    awk -v f="$free" -v m="$min" 'BEGIN{exit !(f>=m)}'
}

SERVER_PID=""
cleanup() {
    # Always attempt this (idempotent, -ErrorAction SilentlyContinue) rather
    # than pre-checking with bash's `kill -0` -- SERVER_PID may be a real
    # Windows PID resolved via PowerShell (see Step 2), which MSYS's kill
    # does not reliably recognise as a job to probe.
    if [ -n "$SERVER_PID" ]; then
        log_step "cleanup: stopping server pid $SERVER_PID (if still running)"
        powershell -NoProfile -Command "Stop-Process -Id $SERVER_PID -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
    fi
    rm -f audit.db audit.db-wal audit.db-shm 2>/dev/null
}
trap cleanup EXIT

log_step "=== perf baseline run started, output dir: $RUN_DIR ==="
log_step "git HEAD: $(git rev-parse HEAD 2>/dev/null) dirty=$( [ -n "$(git status --porcelain)" ] && echo true || echo false )"

# --- Step 1: full pytest suite -------------------------------------------
# NOTE: this has been observed to segfault (Windows access violation inside
# torch/storage.py during transformers weight loading) even above this
# threshold -- it is a known, probabilistic native-library crash risk on
# this hardware (heap fragmentation/allocation-timing dependent, not a hard
# RAM cutoff), not something this script can fully prevent. A FAIL here
# from a segfault does not necessarily mean the code regressed -- check
# whether a full run has passed cleanly recently (docs/perf/ records when
# it last did) before assuming otherwise.
log_step "--- Step 1/4: full pytest suite ---"
if require_gb 5.5 "pytest suite"; then
    rm -f audit.db audit.db-wal audit.db-shm
    if PYTHONPATH=. python -m pytest tests/ -q --timeout=300 > "$RUN_DIR/01_pytest.log" 2>&1; then
        tail -3 "$RUN_DIR/01_pytest.log" | tee -a "$SUMMARY"
        log_step "Step 1: PASS"
    else
        tail -20 "$RUN_DIR/01_pytest.log" | tee -a "$SUMMARY"
        log_step "Step 1: FAIL -- see $RUN_DIR/01_pytest.log"
    fi
    rm -f audit.db audit.db-wal audit.db-shm
else
    log_step "Step 1: SKIP (insufficient RAM)"
fi

# --- Step 2: start the server ---------------------------------------------
log_step "--- Step 2/4: start API server ---"
SERVER_OK=0
if require_gb 3.0 "server start"; then
    RATE_LIMIT_AUTHENTICATED_RPM=100000 PYTHONPATH=. python -m uvicorn api.main:app \
        --host 127.0.0.1 --port 8000 > "$RUN_DIR/02_server.log" 2>&1 &
    BASH_PID=$!
    log_step "bash job pid: $BASH_PID (not necessarily py-spy's target -- see below), waiting up to 150s for /health ..."
    for i in $(seq 1 50); do
        if curl -s -m 2 http://127.0.0.1:8000/health 2>/dev/null | grep -q status; then
            SERVER_OK=1
            break
        fi
        sleep 3
    done
    if [ "$SERVER_OK" = "1" ]; then
        # Git-Bash/MSYS's $! is the shell job's PID, which is NOT reliably the
        # real Windows PID of python.exe -- py-spy needs the latter or it
        # fails with "Failed to open process ... os error 87". Resolve the
        # true PID from the process actually holding port 8000 instead.
        REAL_PID="$(powershell -NoProfile -Command \
            "(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)" \
            2>/dev/null | tr -d '\r')"
        if [ -n "$REAL_PID" ]; then
            SERVER_PID="$REAL_PID"
            log_step "resolved real server pid (owns :8000): $SERVER_PID"
        else
            SERVER_PID="$BASH_PID"
            log_step "WARNING: could not resolve the real pid from port 8000; falling back to $BASH_PID (py-spy may fail)"
        fi
        curl -s http://127.0.0.1:8000/health | tee -a "$SUMMARY"
        echo | tee -a "$SUMMARY"
        log_step "Step 2: PASS"
    else
        SERVER_PID="$BASH_PID"  # still clean this up on exit even though it never got healthy
        tail -30 "$RUN_DIR/02_server.log" | tee -a "$SUMMARY"
        log_step "Step 2: FAIL -- server did not become healthy, see $RUN_DIR/02_server.log"
    fi
else
    log_step "Step 2: SKIP (insufficient RAM)"
fi

# --- Step 3: benchmark at concurrency 1, 4, 16 -----------------------------
if [ "$SERVER_OK" = "1" ]; then
    log_step "--- Step 3/4: closed-loop benchmark (concurrency 1, 4, 16) ---"

    KEY_FILE="$REPO/.bench_api_key"
    if [ -f "$KEY_FILE" ]; then
        KEY="$(cat "$KEY_FILE")"
        log_step "reusing benchmark API key from $KEY_FILE"
    else
        ISSUE_OUT="$(PYTHONPATH=. python scripts/manage_api_keys.py issue \
            --capability GENERAL --tenant benchmark \
            --key-id "load-test-${TS}" 2>&1)"
        echo "$ISSUE_OUT" >> "$RUN_DIR/00_summary.txt"
        KEY="$(echo "$ISSUE_OUT" | sed -n 's/.*key[[:space:]]*:[[:space:]]*\(gk_[A-Za-z0-9_-]*\).*/\1/p' | head -1)"
        if [ -n "$KEY" ]; then
            echo "$KEY" > "$KEY_FILE"
            log_step "issued new benchmark API key, saved to $KEY_FILE (gitignored -- do not commit)"
        fi
    fi

    if [ -z "${KEY:-}" ]; then
        log_step "Step 3: FAIL -- could not obtain an API key"
    else
        BENCH_OK=1
        for c in 1 4 16; do
            if ! require_gb 1.0 "benchmark concurrency=$c"; then
                log_step "Step 3 (concurrency=$c): SKIP (insufficient RAM)"
                BENCH_OK=0
                continue
            fi
            if [ "$c" = "16" ]; then
                # Start py-spy so it overlaps this run -- see Step 4 note.
                if require_gb 1.0 "py-spy" && command -v py-spy >/dev/null 2>&1; then
                    SHA_SHORT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
                    FLAME="$RUN_DIR/${SHA_SHORT}-c16-flamegraph.svg"
                    log_step "--- Step 4/4: py-spy record (40s, overlapping concurrency=16) ---"
                    py-spy record -o "$FLAME" -d 40 -r 100 --pid "$SERVER_PID" \
                        > "$RUN_DIR/04_pyspy.log" 2>&1 &
                    PYSPY_PID=$!
                else
                    log_step "Step 4: SKIP (py-spy unavailable or insufficient RAM)"
                    PYSPY_PID=""
                fi
            fi
            PYTHONPATH=. python benchmarks/load/assess_bench.py \
                --api-key "$KEY" --concurrency "$c" --duration 30 --warmup 5 \
                --judge-label disabled_unreachable \
                > "$RUN_DIR/03_bench_c${c}.log" 2>&1
            tail -6 "$RUN_DIR/03_bench_c${c}.log" | tee -a "$SUMMARY"
            if [ -n "${PYSPY_PID:-}" ]; then
                wait "$PYSPY_PID" 2>/dev/null
                tail -5 "$RUN_DIR/04_pyspy.log" | tee -a "$SUMMARY"
                if [ -s "$FLAME" ]; then
                    log_step "Step 4: PASS -- $FLAME"
                else
                    log_step "Step 4: FAIL -- no flamegraph written, see $RUN_DIR/04_pyspy.log (often a stale/wrong pid; re-run)"
                fi
            fi
        done
        [ "$BENCH_OK" = "1" ] && log_step "Step 3: PASS" || log_step "Step 3: PARTIAL (see per-level logs)"
        # Only the JSONs, and only here -- each concurrency level writes a
        # uniquely-named file (-1/-4/-16.json) so re-copying all of them is
        # safe. The flamegraph is NOT re-copied: py-spy already writes it
        # straight into $RUN_DIR under one fixed name, and copying
        # _evidence/perf/*.svg here would silently overwrite that fresh file
        # with whatever unrelated stale .svg happens to share that name in
        # the shared root folder -- confirmed to actually happen (a two-day-
        # old flamegraph clobbered a fresh one this way once already).
        cp _evidence/perf/*.json "$RUN_DIR/" 2>/dev/null
    fi
else
    log_step "Step 3+4: SKIP (server not healthy)"
fi

log_step "=== run finished. Review $SUMMARY first, then the per-step .log files. ==="
