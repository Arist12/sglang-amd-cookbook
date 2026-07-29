#!/usr/bin/env bash
# grid-k3-supp.sh -- follow-up sweeps the main phases showed were needed.
#
#   GL_BUDGET_MIN=60 ./grid-k3-supp.sh [run_dir]
#
# Each block exists because a main-phase result raised a question it could not
# answer. Everything writes into the main run's results.csv, tagged p2b / p3b.
#
# p2b, baseline repeats: phase 2's winner was --mamba-ssm-dtype bfloat16 at 7977
#   tok/s against a baseline measured once at 7876 -- a 1.3% margin. That knob
#   changes SSM state precision, so it should not ship on a margin we cannot
#   separate from boot-to-boot variation. Two more baseline runs give it a spread.
#
# p2b, radix-off shared prefix: phase 2 measured radix-ON no-spec on the
#   shared-prefix workload (12,470 tok/s default strategy, 12,561 lazy) but the main
#   grid only runs W3 for radix candidates, so there is nothing to divide by.
#
# p3b, gamma=2: the draft-window sweep found a step, not a slope -- gamma 5 (2118)
#   and gamma 7 (2142) are identical while gamma 3 gives 3606, +68%, and improves
#   TTFT and TPOT at the same time. A step that sharp means the optimum may sit
#   below 3, and one probe settles it.
#
# p3b, gamma=3 + ReplaySSM: the two DSpark wins are orthogonal in mechanism.
#   gamma 3 halves the verify window (throughput); ReplaySSM removes the per-draft
#   state snapshot, which left throughput flat but cut median TTFT from 11,250 ms to
#   3,168 ms with zero queueing (latency). Combining them is the obvious candidate
#   for the interactive recipe, and coordinate descent never tries it.
set -uo pipefail

W=/sgl-workspace/workspace
source "${W}/gridlib.sh"

RUN_DIR="${1:-}"
[[ -z "${RUN_DIR}" ]] && RUN_DIR="$(ls -td "${W}"/grid_results/2*/ 2>/dev/null | head -1)"
RUN_DIR="${RUN_DIR%/}"
[[ -d "${RUN_DIR}" ]] || { echo "no run dir: ${RUN_DIR}"; exit 1; }

gl_init "${RUN_DIR}"
TOOLS="${W}/gridtools.py"
MF_NOSPEC="$(cat "${GL_RUN_DIR}/memfrac-nospec.txt" 2>/dev/null || echo 0.93)"
MF_DSPARK="$(cat "${GL_RUN_DIR}/memfrac-dspark.txt" 2>/dev/null || echo 0.92)"

COST_W2=600
COST_W1=200
COST_W3=400

cfg() {
  local label="$1" lane="$2" points="$3" mode="$4"; shift 4
  local pt all_have=1
  for pt in ${points}; do
    gl_have "${label}" "${pt%%:*}" "${pt##*:}" || { all_have=0; break; }
  done
  if (( all_have )); then gl_log "SKIP ${label} (already recorded)"; return 0; fi

  local need; need="$(gl_boot_cost)"
  for pt in ${points}; do
    case "${pt%%:*}" in
      w1) need=$(( need + COST_W1 )) ;;
      w3) need=$(( need + COST_W3 )) ;;
      *)  need=$(( need + COST_W2 )) ;;
    esac
  done
  gl_budget "${need}" || return 2

  gl_reset
  local kv
  for kv in "$@"; do
    case "${kv}" in *=default|*=NA|*=) continue ;; esac
    export "${kv?}"
  done
  GL_LABEL="${label}"; GL_LANE="${lane}"; GL_PHASE="${PHASE}"

  if ! gl_launch "${mode}"; then gl_row "${GL_BOOT_STATUS}"; return 1; fi
  local rc=0
  for pt in ${points}; do
    gl_have "${label}" "${pt%%:*}" "${pt##*:}" && continue
    gl_bench "${pt%%:*}" "${pt##*:}" || rc=1
    gl_alive || { gl_log "  server gone, abandoning remaining points"; rc=1; break; }
  done
  gl_teardown
  return ${rc}
}

# --------------------------------------------------------------------------- #
# Ordered by value, because the budget guard cuts from the end.
PHASE=p3b
gl_log "########## P3b DSpark follow-ups, mem-fraction ${MF_DSPARK} ##########"

cfg "p3b-g2" "dspark" "w2:48" dspark "MEM_FRAC=${MF_DSPARK}" \
    "EXTRA_ARGS=--speculative-dspark-block-size 2"

cfg "p3b-g3-replayssm" "dspark" "w2:48 w2:96" dspark "MEM_FRAC=${MF_DSPARK}" \
    "EXTRA_ARGS=--speculative-dspark-block-size 3 --enable-linear-replayssm-spec"

PHASE=p2b
gl_log "########## P2b no-spec follow-ups, mem-fraction ${MF_NOSPEC} ##########"

cfg "p2b-noradix-gsp" "nospec" "w3:32" nospec "MEM_FRAC=${MF_NOSPEC}"

for rep in 2 3; do
  cfg "p2b-base-rep${rep}" "nospec" "w2:128" nospec "MEM_FRAC=${MF_NOSPEC}"
done

# Latency curve for whichever DSpark follow-up won, so the interactive recipe is
# chosen on TPOT rather than on aggregate throughput.
PHASE=p3b
res="$(python3 "${TOOLS}" best --csv "${GL_CSV}" --lane dspark --workload w2 \
       --phase p3b --metric total_tps --mode max 2>/dev/null)"
if [[ -n "${res}" ]]; then
  eval "${res}"
  gl_log "P3b winner: ${label} at c${conc} (${total_tps} tok/s)"
  cfg "p3b-lat-win" "dspark" "w1:1 w1:8" dspark \
      "MEM_FRAC=${mem_frac}" "RADIX=${radix}" "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" \
      "CHUNKED_PREFILL=${chunked_prefill}" "EXTRA_ARGS=${extra_args}"
fi

python3 "${TOOLS}" summarize --csv "${GL_CSV}" --out "${GL_RUN_DIR}/summary.md" || true
python3 "${TOOLS}" report --csv "${GL_CSV}" --out "${GL_RUN_DIR}/report.md" || true
gl_teardown
gl_log "########## supplements done ##########"
