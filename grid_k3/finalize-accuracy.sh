#!/usr/bin/env bash
# finalize-accuracy.sh -- rescore the degeneration probes and rewrite the verdict
# once the accuracy gate has finished.
#
# The first degeneration probe used a 0.25 8-gram repetition threshold with no
# corroborating signal, which flagged a perfectly coherent reasoning trace (29%
# phrase repetition while summarising repetitive source text, longest consecutive
# token run of 1) and reported FAIL. The thresholds in gridtools.py now require
# repetition *and* consecutive runs, or one signal far out. Completion text was
# saved, so the verdict can be recomputed without re-running any eval.
set -uo pipefail

W=/sgl-workspace/workspace
RUN_DIR="${1:-}"
[[ -z "${RUN_DIR}" ]] && RUN_DIR="$(ls -td "${W}"/grid_results/2*/ 2>/dev/null | head -1)"
RUN_DIR="${RUN_DIR%/}"
ACC="${RUN_DIR}/accuracy"
LOG="${RUN_DIR}/finalize.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"; }

log "finalizer started (pid $$), run dir ${RUN_DIR}"

# Wait for the accuracy gate to exit.
for _ in $(seq 1 900); do
  pgrep -f 'accuracy-k3\.sh' >/dev/null 2>&1 || { log "accuracy gate finished"; break; }
  sleep 30
done
sleep 20

for lane in nospec dspark; do
  det="${ACC}/details-${lane}.jsonl"
  if [[ -s "${det}" ]]; then
    log "rescoring ${lane}"
    python3 "${W}/gridtools.py" gibberish --details "${det}" \
      > "${ACC}/gibberish-${lane}.txt" 2>> "${LOG}" || true
    log "  $(tr '\n' ' ' < "${ACC}/gibberish-${lane}.txt")"
  else
    log "no details for ${lane}, skipping rescore"
  fi
done

python3 "${W}/gate-k3.py" "${ACC}" > "${RUN_DIR}/accuracy_gate.md" 2>> "${LOG}" || true
python3 "${W}/gridtools.py" summarize --csv "${RUN_DIR}/results.csv" \
  --out "${RUN_DIR}/summary.md" >> "${LOG}" 2>&1 || true
python3 "${W}/gridtools.py" report --csv "${RUN_DIR}/results.csv" \
  --out "${RUN_DIR}/report.md" >> "${LOG}" 2>&1 || true

log "finalize complete"
