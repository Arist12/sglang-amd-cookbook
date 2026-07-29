#!/usr/bin/env bash
# chain-after-search.sh -- wait for the parameter search to finish, then run the
# supplemental sweeps and the accuracy gate.
#
# Exists so the whole night runs unattended from a single trigger. It waits on the
# search by polling for the process rather than by being launched from inside it,
# so it does not care how the search terminates (normal exit, deadline, crash).
#
# Both chained scripts are resumable and skip anything already in results.csv, so
# a duplicate trigger is harmless.
set -uo pipefail

W=/sgl-workspace/workspace
RUN_DIR="${1:-}"
[[ -z "${RUN_DIR}" ]] && RUN_DIR="$(ls -td "${W}"/grid_results/2*/ 2>/dev/null | head -1)"
RUN_DIR="${RUN_DIR%/}"
LOG="${RUN_DIR}/chain.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"; }

log "chain watcher started (pid $$), run dir ${RUN_DIR}"

# Wait for the search to exit. Bounded so a stuck search cannot hold the night.
for _ in $(seq 1 720); do
  if ! pgrep -f 'bash /sgl-workspace/workspace/grid-k3\.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'bash grid-k3\.sh' >/dev/null 2>&1; then
    log "search process gone"
    break
  fi
  sleep 30
done

# Let the search's own teardown finish releasing the GPUs and the lock.
sleep 60

log "=== supplements ==="
GL_BUDGET_MIN="${SUPP_BUDGET_MIN:-55}" bash "${W}/grid-k3-supp.sh" "${RUN_DIR}" \
  >> "${RUN_DIR}/supp.log" 2>&1
log "supplements exited rc=$?"

log "=== accuracy gate ==="
GL_BUDGET_MIN="${ACC_BUDGET_MIN:-330}" bash "${W}/accuracy-k3.sh" "${RUN_DIR}" \
  >> "${RUN_DIR}/accuracy.log" 2>&1
log "accuracy exited rc=$?"

log "chain complete"
