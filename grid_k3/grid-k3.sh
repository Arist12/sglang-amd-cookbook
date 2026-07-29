#!/usr/bin/env bash
# grid-k3.sh -- overnight launch-parameter search for Kimi-K3 on 8x MI355X.
#
#   GL_BUDGET_H=14 ./grid-k3.sh all [run_dir]
#   ./grid-k3.sh p1              # bisect --mem-fraction-static only
#   ./grid-k3.sh p2              # no-spec throughput lane
#   ./grid-k3.sh p3              # DSpark interactive lane
#   ./grid-k3.sh p4              # confirm the finalists
#
# Method, and why it is not a cartesian grid: the binding constraint per lane is
# already known from the scheduler telemetry of earlier runs (no-spec saturates on
# MLA KV at ~90 concurrent with full-token usage 0.99; DSpark saturates on the KDA
# state pool at 48 with mamba usage 0.98). So phase 1 bisects the one monotone
# scalar that moves those ceilings (--mem-fraction-static, which had 35 GB/GPU of
# slack), and phases 2-3 coordinate-descend the discrete scheduling knobs from the
# resulting operating point. Phase 4 re-runs the finalists because single-run
# spreads of 15% have already been observed on this workload.
#
# Resumable: every point already present in results.csv is skipped, so the script
# can be re-run after a crash or a reboot and will continue where it stopped.
set -uo pipefail

W=/sgl-workspace/workspace
source "${W}/gridlib.sh"

PHASES="${1:-all}"
RUN_DIR="${2:-}"
if [[ -z "${RUN_DIR}" ]]; then
  # Reuse the newest run dir so re-invocations resume instead of starting over.
  RUN_DIR="$(ls -td "${W}"/grid_results/*/ 2>/dev/null | head -1)"
  [[ -z "${RUN_DIR}" ]] && RUN_DIR="${W}/grid_results/$(date '+%Y%m%d_%H%M%S')"
fi
RUN_DIR="${RUN_DIR%/}"

gl_init "${RUN_DIR}"
TOOLS="${W}/gridtools.py"

# Rough per-point costs, used only for the budget guard. The boot cost is measured
# rather than assumed (see gl_boot_cost).
COST_W2=600
COST_W1=200
COST_W3=400
COST_STRESS=420

# --------------------------------------------------------------------------- #
# config runner
# --------------------------------------------------------------------------- #
# cfg <label> <lane> <points> <mode> KEY=VAL...
#   points: space-separated "<workload>:<conc>" run in order against one server
# Returns 0 if every point succeeded, 1 on launch failure, 2 if out of budget.
cfg() {
  local label="$1" lane="$2" points="$3" mode="$4"; shift 4

  # Skip only when every requested point is already recorded.
  local pt all_have=1
  for pt in ${points}; do
    gl_have "${label}" "${pt%%:*}" "${pt##*:}" || { all_have=0; break; }
  done
  if (( all_have )); then
    gl_log "SKIP ${label} (already recorded)"
    return 0
  fi

  local need; need="$(gl_boot_cost)"
  for pt in ${points}; do
    case "${pt%%:*}" in
      w1) need=$(( need + COST_W1 )) ;;
      w2) need=$(( need + COST_W2 )) ;;
      w3) need=$(( need + COST_W3 )) ;;
      stress) need=$(( need + COST_STRESS )) ;;
    esac
  done
  gl_budget "${need}" || return 2

  gl_reset
  local kv
  for kv in "$@"; do
    # "default"/"NA" come from CSV round-trips and mean "knob not set"; exporting
    # them would put the literal string on the command line.
    case "${kv}" in
      *=default|*=NA|*=) continue ;;
    esac
    export "${kv?}"
  done
  GL_LABEL="${label}"; GL_LANE="${lane}"; GL_PHASE="${PHASE}"

  if ! gl_launch "${mode}"; then
    gl_row "${GL_BOOT_STATUS}"
    return 1
  fi

  local rc=0
  for pt in ${points}; do
    if gl_have "${label}" "${pt%%:*}" "${pt##*:}"; then
      gl_log "  skip ${pt} (recorded)"
      continue
    fi
    gl_bench "${pt%%:*}" "${pt##*:}" || rc=1
    # No point continuing the curve against a dead server.
    gl_alive || { gl_log "  server gone, abandoning remaining points"; rc=1; break; }
  done
  gl_teardown
  return ${rc}
}

incumbent_file() { echo "${GL_RUN_DIR}/incumbent-$1.env"; }

# Persist / restore the coordinate-descent incumbent so a restart resumes.
save_incumbent() {
  local lane="$1"; shift
  printf '%s\n' "$@" > "$(incumbent_file "${lane}")"
  gl_log "INCUMBENT[${lane}] $*"
}

# best_of <lane> <workload> <metric> <max|min> <phase-csv> -- echoes k=v lines
best_of() {
  python3 "${TOOLS}" best --csv "${GL_CSV}" --lane "$1" --workload "$2" \
    --metric "$3" --mode "$4" ${5:+--phase "$5"} 2>/dev/null
}

# --------------------------------------------------------------------------- #
# Phase 1 -- bisect --mem-fraction-static
# --------------------------------------------------------------------------- #
# available_gpu_mem is very nearly linear in mem-fraction with slope -288 GB per
# unit (the card is 288 GiB), so the anchor measurement predicts the target
# directly; the probes that follow only have to confirm and refine. Pure blind
# bisection would spend the same probes learning a slope we can compute.
HBM_GB=288
TARGET_AVAIL=7.0     # middle of the 5-8 GB the tuning doc calls healthy
MIN_AVAIL=5.0        # below this, activations have no room
CLIMB_AVAIL=9.5      # above this, there is still slack worth taking

probe_memfrac() {
  local mode="$1" memfrac="$2" conc="$3"
  local label="p1-${mode}-mf${memfrac}"
  cfg "${label}" "mem-${mode}" "stress:${conc}" "${mode}" "MEM_FRAC=${memfrac}"
  local rc=$?
  P1_AVAIL=""
  P1_OK=0
  if (( rc == 2 )); then return 2; fi
  # Accept only if it booted, survived the peak-activation point, and still has
  # room for activations.
  local row
  row="$(awk -F',' -v l="${label}" '$4==l && $11=="stress"{print; exit}' "${GL_CSV}")"
  if [[ -z "${row}" ]]; then return 1; fi
  local status; status="$(echo "${row}" | cut -d, -f16)"
  P1_AVAIL="$(echo "${row}" | cut -d, -f37)"
  if [[ "${status}" == "OK" && "${P1_AVAIL}" != "NA" ]]; then
    if awk -v a="${P1_AVAIL}" -v m="${MIN_AVAIL}" 'BEGIN{exit !(a>=m)}'; then
      P1_OK=1
    else
      gl_log "  REJECT mf=${memfrac}: avail=${P1_AVAIL}GB < ${MIN_AVAIL}GB"
    fi
  else
    gl_log "  REJECT mf=${memfrac}: status=${status}"
  fi
  return 0
}

phase1_lane() {
  local mode="$1" conc="$2" anchor="${3:-0.85}"
  local best="" avail_at_best=""

  gl_log "=== P1 ${mode}: anchor probe at mem-fraction ${anchor} ==="
  probe_memfrac "${mode}" "${anchor}" "${conc}" || return 0
  if (( P1_OK )); then best="${anchor}"; avail_at_best="${P1_AVAIL}"; fi
  local anchor_avail="${P1_AVAIL}"

  if [[ -z "${anchor_avail}" || "${anchor_avail}" == "NA" ]]; then
    gl_log "P1 ${mode}: anchor gave no capacity reading, keeping ${anchor}"
    echo "${anchor}" > "${GL_RUN_DIR}/memfrac-${mode}.txt"
    return 0
  fi

  local pred
  pred="$(awk -v a="${anchor}" -v av="${anchor_avail}" -v t="${TARGET_AVAIL}" -v h="${HBM_GB}" \
    'BEGIN{p=a+(av-t)/h; if(p>0.97)p=0.97; if(p<a+0.01)p=a+0.01; printf "%.2f", p}')"
  gl_log "=== P1 ${mode}: anchor avail=${anchor_avail}GB -> predicted mem-fraction ${pred} ==="

  local cand="${pred}" probes=0
  while (( probes < 4 )); do
    probes=$(( probes + 1 ))
    probe_memfrac "${mode}" "${cand}" "${conc}"
    local rc=$?
    (( rc == 2 )) && break
    if (( P1_OK )); then
      best="${cand}"; avail_at_best="${P1_AVAIL}"
      # Still slack? climb. Otherwise we are at the knee.
      if awk -v a="${P1_AVAIL}" -v c="${CLIMB_AVAIL}" 'BEGIN{exit !(a>c)}'; then
        cand="$(awk -v c="${cand}" 'BEGIN{printf "%.2f", c+0.01}')"
        awk -v c="${cand}" 'BEGIN{exit !(c>0.97)}' && break
      else
        break
      fi
    else
      cand="$(awk -v c="${cand}" 'BEGIN{printf "%.2f", c-0.01}')"
      awk -v c="${cand}" -v a="${anchor}" 'BEGIN{exit !(c<=a)}' && break
    fi
  done

  [[ -z "${best}" ]] && best="${anchor}"
  echo "${best}" > "${GL_RUN_DIR}/memfrac-${mode}.txt"
  gl_log "=== P1 ${mode}: WINNER mem-fraction ${best} (avail=${avail_at_best:-?}GB) ==="
}

phase1() {
  PHASE=p1
  phase1_lane nospec 96
  phase1_lane dspark 48
}

memfrac_for() {
  local f="${GL_RUN_DIR}/memfrac-$1.txt"
  [[ -f "${f}" ]] && cat "${f}" || echo 0.85
}

# --------------------------------------------------------------------------- #
# Phase 2 -- no-spec throughput lane
# --------------------------------------------------------------------------- #
phase2() {
  PHASE=p2
  local lane=nospec
  local MF; MF="$(memfrac_for nospec)"
  gl_log "########## P2 no-spec lane, mem-fraction ${MF} ##########"

  # Step A: re-saturate concurrency. Concurrency is client-side, so the whole
  # curve runs against one boot -- the cheapest information in the whole search.
  cfg "p2-base-mf${MF}" "${lane}" "w2:96 w2:128 w2:160 w2:192" nospec "MEM_FRAC=${MF}"

  local BC; BC="$(best_conc "${lane}" w2)"
  gl_log "P2 best concurrency so far: ${BC}"

  # Step B: chunked prefill. Fixed at the 16384 default in every prior run, while
  # TTFT at saturation is tens of seconds -- prefill is a large share of the work.
  local cp
  for cp in 8192 32768 65536; do
    cfg "p2-cp${cp}" "${lane}" "w2:${BC}" nospec "MEM_FRAC=${MF}" "CHUNKED_PREFILL=${cp}"
  done

  # Step C: CUDA-graph coverage. If the saturation batch now exceeds 256 the
  # decode path is falling off graph replay entirely.
  local cg
  for cg in 384 512; do
    cfg "p2-cg${cg}" "${lane}" "w2:${BC}" nospec "MEM_FRAC=${MF}" "CUDA_GRAPH_MAX_BS=${cg}"
  done

  # Step D: prefix caching for no-spec -- never A/B'd on this lane. Uncached KDA
  # costs 1 mamba slot/request against a 368 cap, so radix may be nearly free
  # here even though it halves the DSpark batch.
  cfg "p2-radix-default" "${lane}" "w2:${BC} w3:32" nospec "MEM_FRAC=${MF}" "RADIX=1"
  cfg "p2-radix-lazy" "${lane}" "w2:${BC} w3:32" nospec "MEM_FRAC=${MF}" "RADIX=1" \
      "EXTRA_ARGS=--mamba-radix-cache-strategy extra_buffer_lazy"

  # Step E: halve SSM state to convert mamba pool into KV. Accuracy-critical --
  # it changes state precision, so it only ships if the phase-5 gate passes.
  cfg "p2-ssmbf16" "${lane}" "w2:${BC}" nospec "MEM_FRAC=${MF}" \
      "EXTRA_ARGS=--mamba-ssm-dtype bfloat16"

  # Step F: scheduling. Only informative if the incumbent actually queued or
  # retracted; otherwise it cannot move anything.
  local retr; retr="$(awk -F',' -v l="p2-base-mf${MF}" '$4==l && $11=="w2"{s+=$46} END{print s+0}' "${GL_CSV}")"
  local qmax; qmax="$(awk -F',' -v l="p2-base-mf${MF}" '$4==l && $11=="w2"{if($45>m)m=$45} END{print m+0}' "${GL_CSV}")"
  if (( retr > 0 )); then
    gl_log "P2 saw ${retr} retractions -> probing higher schedule-conservativeness"
    cfg "p2-sched1.3" "${lane}" "w2:${BC}" nospec "MEM_FRAC=${MF}" "EXTRA_ARGS=--schedule-conservativeness 1.3"
  elif (( qmax > 0 )); then
    gl_log "P2 queued (max ${qmax}) without retracting -> probing lower schedule-conservativeness"
    cfg "p2-sched0.6" "${lane}" "w2:${BC}" nospec "MEM_FRAC=${MF}" "EXTRA_ARGS=--schedule-conservativeness 0.6"
  else
    gl_log "P2 never queued or retracted -> schedule-conservativeness cannot bind, skipping"
  fi

  # Step G: re-saturate with the tuned config. The knobs above (chunked prefill in
  # particular) move where saturation sits, so the baseline's argmax concurrency is
  # no longer the right operating point to publish.
  local res; res="$(python3 "${TOOLS}" best --csv "${GL_CSV}" --lane "${lane}" \
      --workload w2 --phase p2 --conc "${BC}" --metric total_tps --mode max 2>/dev/null)"
  if [[ -n "${res}" ]]; then
    eval "${res}"
    if [[ "${label}" != "p2-base-mf${MF}" ]]; then
      gl_log "P2 tuned winner at matched c${BC}: ${label} -- re-checking saturation"
      cfg "p2-curve-${label}" "${lane}" "w2:$(( BC + 32 )) w2:$(( BC + 64 ))" nospec \
          "MEM_FRAC=${mem_frac}" "RADIX=${radix}" "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" \
          "CHUNKED_PREFILL=${chunked_prefill}" "EXTRA_ARGS=${extra_args}"
    fi
  fi

  save_p2_winner "${lane}" "${MF}"
}

best_conc() {
  local lane="$1" wl="$2" out
  out="$(python3 "${TOOLS}" best --csv "${GL_CSV}" --lane "${lane}" --workload "${wl}" \
        --metric total_tps --mode max 2>/dev/null | rg -N '^conc=(.*)$' -r '$1' | tr -d "'")"
  [[ -n "${out}" && "${out}" != "NA" ]] && echo "${out}" || echo 96
}

save_p2_winner() {
  local lane="$1" mf="$2"
  local res; res="$(best_of "${lane}" w2 total_tps max p2)"
  if [[ -z "${res}" ]]; then gl_log "P2: no winner found"; return; fi
  eval "${res}"
  save_incumbent "${lane}" "MEM_FRAC=${mem_frac}" "RADIX=${radix}" \
    "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" "CHUNKED_PREFILL=${chunked_prefill}" \
    "EXTRA_ARGS=${extra_args}" "BEST_CONC=${conc}" "BEST_TPS=${total_tps}" "BEST_LABEL=${label}"
}

# --------------------------------------------------------------------------- #
# Phase 3 -- DSpark interactive lane
# --------------------------------------------------------------------------- #
phase3() {
  PHASE=p3
  local lane=dspark
  local MF; MF="$(memfrac_for dspark)"
  gl_log "########## P3 DSpark lane, mem-fraction ${MF} ##########"

  # Step 0/A: the clean single-variable max-running-requests sweep. The existing
  # "small batch is faster" evidence (241 tok/s at 48 vs 379 at 29) came from runs
  # that moved several knobs at once, so it is a hypothesis, not a measurement.
  # Mechanism to confirm: DSpark verify cost scales with batch x 8 while measured
  # accept length on this traffic is only 2.5-3.3.
  cfg "p3-base-mf${MF}" "${lane}" "w2:48" dspark "MEM_FRAC=${MF}"
  local mrr
  for mrr in 16 24 32 40; do
    cfg "p3-mrr${mrr}" "${lane}" "w2:48" dspark "MEM_FRAC=${MF}" \
        "EXTRA_ARGS=--max-running-requests ${mrr}"
  done

  # Step B: draft-window size. gamma 3 raised throughput but cut accept length
  # from 3.0 to 2.55, so the optimum is interior. Never set
  # --speculative-num-draft-tokens directly; it is asserted to equal gamma+1.
  local g
  for g in 3 5; do
    cfg "p3-g${g}" "${lane}" "w2:48" dspark "MEM_FRAC=${MF}" \
        "EXTRA_ARGS=--speculative-dspark-block-size ${g}"
  done

  # Step C: ReplaySSM. Largest capacity lever measured (verify scratch 19.8 GB ->
  # 0, cap 48 -> 128) but it left only 13.5 GB headroom at mem-fraction 0.85, so
  # it gets its own memory point rather than inheriting the lane's.
  cfg "p3-replayssm" "${lane}" "w2:48 w2:96" dspark "MEM_FRAC=0.85" \
      "EXTRA_ARGS=--enable-linear-replayssm-spec --max-running-requests 128"

  # Pick the knob winner at the *same* c48 operating point every candidate ran at.
  # Selecting on the global max would hand the win to whichever config happened to
  # be given a higher-concurrency point (ReplaySSM, here) rather than to the better
  # knob -- an apples-to-oranges comparison.
  local res; res="$(python3 "${TOOLS}" best --csv "${GL_CSV}" --lane "${lane}" \
      --workload w2 --phase p3 --conc 48 --metric total_tps --mode max 2>/dev/null)"
  if [[ -n "${res}" ]]; then
    eval "${res}"
    gl_log "P3 knob winner at matched c48: ${label} (${total_tps} tok/s)"

    # Raising the runnable-batch cap only pays off at concurrency the baseline
    # cannot reach, so give the winner the same higher-concurrency points
    # ReplaySSM already has. Now the global argmax is a fair comparison.
    if [[ "${label}" != "p3-replayssm" ]]; then
      cfg "p3-curve-${label}" "${lane}" "w2:64 w2:96" dspark \
          "MEM_FRAC=${mem_frac}" "RADIX=${radix}" "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" \
          "CHUNKED_PREFILL=${chunked_prefill}" "EXTRA_ARGS=${extra_args}"
    fi
  fi

  # Step D: latency curve. This lane's objective is TPOT, and low-concurrency
  # points are cheap. Run it on the (now fairly chosen) overall winner.
  res="$(best_of "${lane}" w2 total_tps max p3)"
  if [[ -n "${res}" ]]; then
    eval "${res}"
    gl_log "P3 overall winner: ${label} at c${conc} (${total_tps} tok/s)"
    cfg "p3-lat-win" "${lane}" "w1:1 w1:4 w1:8" dspark \
        "MEM_FRAC=${mem_frac}" "RADIX=${radix}" "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" \
        "CHUNKED_PREFILL=${chunked_prefill}" "EXTRA_ARGS=${extra_args}"
    save_incumbent "${lane}" "MEM_FRAC=${mem_frac}" "RADIX=${radix}" \
      "CUDA_GRAPH_MAX_BS=${cuda_graph_max_bs}" "CHUNKED_PREFILL=${chunked_prefill}" \
      "EXTRA_ARGS=${extra_args}" "BEST_CONC=${conc}" "BEST_TPS=${total_tps}" "BEST_LABEL=${label}"
  fi

  # Baseline latency reference, so the tuned config has something to beat.
  cfg "p3-lat-base" "${lane}" "w1:1 w1:4 w1:8" dspark "MEM_FRAC=${MF}"
}

# --------------------------------------------------------------------------- #
# Phase 4 -- confirm the finalists
# --------------------------------------------------------------------------- #
phase4() {
  PHASE=p4
  gl_log "########## P4 confirmation ##########"

  # Repeat runs. Observed spread on this workload was 222-398 tok/s across
  # nominally similar DSpark configs, so a single-run delta under ~15% is not yet
  # distinguishable from noise.
  local lane
  for lane in nospec dspark; do
    if ! gl_read_incumbent "${lane}"; then
      gl_log "P4: no incumbent for ${lane}, skipping"; continue
    fi
    gl_knob_args
    local BC="${GL_INC_CONC:-96}" BL="${GL_INC_LABEL}"
    local mode=nospec; [[ "${lane}" == "dspark" ]] && mode=dspark
    gl_log "P4 ${lane}: confirming ${BL} at c${BC} -- ${GL_KNOBS[*]}"

    local rep
    for rep in 2 3; do
      cfg "p4-${lane}-rep${rep}" "${lane}" "w2:${BC}" "${mode}" "${GL_KNOBS[@]}"
    done
  done

  # Interaction check: coordinate descent cannot see the cross term, so combine
  # the two best single-knob wins per lane.
  phase4_interaction nospec
  phase4_interaction dspark
}

# Combine the top two distinct single-knob wins of a lane into one config.
phase4_interaction() {
  local lane="$1" phase_src=p2 mode=nospec
  [[ "${lane}" == "dspark" ]] && { phase_src=p3; mode=dspark; }

  local res; res="$(best_of "${lane}" w2 total_tps max "${phase_src}")"
  [[ -z "${res}" ]] && return 0
  eval "${res}"
  local best_label="${label}" best_extra="${extra_args}" best_cp="${chunked_prefill}"
  local best_cg="${cuda_graph_max_bs}" best_mf="${mem_frac}" best_rx="${radix}" best_conc="${conc}"

  # Runner-up with a *different* knob signature. Values are shell-quoted because
  # extra_args is multi-word.
  local second
  second="$(python3 - "$GL_CSV" "$lane" "$phase_src" "$best_label" <<'PY'
import csv, shlex, sys
csv_path, lane, phase, best = sys.argv[1:5]
rows = [r for r in csv.DictReader(open(csv_path))
        if r["status"] == "OK" and r["lane"] == lane and r["phase"] == phase
        and r["workload"] == "w2" and r["label"] != best and r["total_tps"] not in ("", "NA")]
rows.sort(key=lambda r: float(r["total_tps"]), reverse=True)
for r in rows:
    for k, col in (("label", "label"), ("cp", "chunked_prefill"), ("cg", "cuda_graph_max_bs"),
                   ("rx", "radix"), ("ea", "extra_args")):
        print(f"{k}={shlex.quote(r[col] or '')}")
    break
PY
)"
  [[ -z "${second}" ]] && return 0
  local label="" cp="" cg="" rx="" ea=""
  eval "${second}"
  gl_log "P4 ${lane} interaction: ${best_label} x ${label}"

  # Merge: take each non-default knob from whichever config set it.
  local m_cp="${best_cp}"; [[ "${m_cp}" == "default" ]] && m_cp="${cp}"
  local m_cg="${best_cg}"; [[ "${m_cg}" == "256" ]] && m_cg="${cg}"
  local m_rx="${best_rx}"; [[ "${m_rx}" == "0" ]] && m_rx="${rx}"
  local m_ea="${best_extra}"
  if [[ -z "${m_ea}" || "${m_ea}" == "NA" ]]; then m_ea="${ea}"; fi
  [[ "${m_cp}" == "default" || "${m_cp}" == "NA" ]] && m_cp=""
  [[ "${m_ea}" == "NA" ]] && m_ea=""

  cfg "p4-${lane}-combo" "${lane}" "w2:${best_conc}" "${mode}" \
      "MEM_FRAC=${best_mf}" "RADIX=${m_rx:-0}" "CUDA_GRAPH_MAX_BS=${m_cg:-256}" \
      ${m_cp:+"CHUNKED_PREFILL=${m_cp}"} ${m_ea:+"EXTRA_ARGS=${m_ea}"}
}

# --------------------------------------------------------------------------- #
finish() {
  python3 "${TOOLS}" summarize --csv "${GL_CSV}" --out "${GL_RUN_DIR}/summary.md" || true
  gl_log "########## search done, $(( ($(date +%s) - ${GL_T0}) / 60 )) min elapsed ##########"
  gl_log "results  ${GL_CSV}"
  gl_log "summary  ${GL_RUN_DIR}/summary.md"
}

GL_T0=$(date +%s)
trap 'gl_log "INTERRUPTED"; gl_teardown; finish; exit 130' INT TERM

case "${PHASES}" in
  p1)  phase1 ;;
  p2)  phase2 ;;
  p3)  phase3 ;;
  p4)  phase4 ;;
  all) phase1; phase2; phase3; phase4 ;;
  *)   echo "usage: $0 {p1|p2|p3|p4|all} [run_dir]"; exit 1 ;;
esac

gl_teardown
finish

# --- chained follow-ups ------------------------------------------------------
# The supplemental sweeps and the accuracy gate run here rather than as separate
# invocations so the whole night is one unattended chain. Both are resumable and
# skip anything already recorded, so running them again by hand is harmless.
# Release the run lock first: the chained scripts call gl_init, which takes it.
exec 9>&- 2>/dev/null || true

GL_BUDGET_MIN="${SUPP_BUDGET_MIN:-55}" bash /sgl-workspace/workspace/grid-k3-supp.sh \
  "${GL_RUN_DIR}" >> "${GL_RUN_DIR}/supp.log" 2>&1 || true

GL_BUDGET_MIN="${ACC_BUDGET_MIN:-330}" bash /sgl-workspace/workspace/accuracy-k3.sh \
  "${GL_RUN_DIR}" >> "${GL_RUN_DIR}/accuracy.log" 2>&1 || true

echo "=== chain complete $(date -Is) ===" >> "${GL_RUN_DIR}/grid.log"
