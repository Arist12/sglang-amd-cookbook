#!/usr/bin/env bash
# gridlib.sh -- config-runner primitives for the Kimi-K3 launch-parameter search.
#
# Source this, then use:
#   gl_init <run_dir>                  set up results dir + CSV header + deadline
#   gl_reset                           clear per-config state (call before gl_launch)
#   gl_launch <mode>                   start a server from the env-configured recipe
#   gl_bench <workload> <conc>         benchmark the live server, append a CSV row
#   gl_row <status>                    append a non-bench row (rejected config)
#   gl_teardown                        kill the server and wait for VRAM to drain
#   gl_have <label> <workload> <conc>  0 if that point is already in the CSV
#   gl_budget <seconds>                0 if that much time remains before deadline
#
# The launch recipe is set through the same env vars serve-k3-ext.sh reads
# (MEM_FRAC, RADIX, AITER, SITU_A8W4, CUDA_GRAPH_MAX_BS, CHUNKED_PREFILL,
# EXTRA_ARGS) plus GL_LABEL / GL_LANE / GL_PHASE for bookkeeping.

W=/sgl-workspace/workspace
SGL=/sgl-workspace/sglang
PORT="${PORT:-30000}"
MODEL=moonshotai/Kimi-K3
TOOLS="${W}/gridtools.py"

# Absolute ceiling, plus the value that actually decides: a boot is declared hung
# when the server log stops growing, not when it takes a long time. A cold page
# cache turns the 1.5 TB weight load from 105 s into tens of minutes, and a
# wall-clock limit cannot tell that apart from a real hang.
BOOT_TIMEOUT="${BOOT_TIMEOUT:-5400}"
BOOT_STALL_TIMEOUT="${BOOT_STALL_TIMEOUT:-480}"
MODEL_HUB="${MODEL_HUB:-/sgl-workspace/models/hub/models--moonshotai--Kimi-K3}"
DRAFT_HUB="${DRAFT_HUB:-/sgl-workspace/models/hub/models--RadixArk--Kimi-K3-DSpark}"
VRAM_IDLE_MB="${VRAM_IDLE_MB:-2000}"
BENCH_FILTER='Calling super\(\)\.encode|^\s*$|it/s\]|aiter\]|Namespace\(|benchmark_args='

CSV_COLS="ts,phase,lane,label,mode,mem_frac,radix,cuda_graph_max_bs,chunked_prefill,extra_args,workload,isl,osl,conc,num_prompts,status,fatal,out_tps,total_tps,req_tps,conc_ach,accept_len,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,mean_itl_ms,mean_e2e_ms,median_e2e_ms,gen_tok,retok_tok,retok_div_pct,cache_hit_pct,max_total_num_tokens,max_running_requests,avail_gpu_mem_gb,mamba_cap,mamba_slots,run_med,run_max,tok_usage_max,mamba_usage_max,sched_accept_med,queue_max,retract_n,cudagraph,boot_s,bench_s,git_sha"

# --------------------------------------------------------------------------- #
gl_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${GL_RUNLOG:-/dev/null}"; }

# Two concurrent runs would tear down each other's servers and misreport the
# result as a crash, so the lock is correctness, not tidiness.
gl_lock() {
  exec 9>/tmp/k3-grid.lock
  # Wait rather than fail: the chained follow-up scripts start right after the
  # search and may briefly overlap its teardown, and dying there would silently
  # drop the rest of the night's work.
  local waited=0 limit="${GL_LOCK_WAIT_S:-600}"
  until flock -n 9; do
    if (( waited >= limit )); then
      echo "another grid run has held /tmp/k3-grid.lock for ${limit}s; refusing to start" >&2
      exit 1
    fi
    sleep 10
    waited=$(( waited + 10 ))
  done
  echo "$$" >&9
}

gl_init() {
  gl_lock
  GL_RUN_DIR="$1"
  mkdir -p "${GL_RUN_DIR}/logs"
  GL_CSV="${GL_RUN_DIR}/results.csv"
  GL_RUNLOG="${GL_RUN_DIR}/grid.log"
  touch "${GL_RUNLOG}"
  [[ -s "${GL_CSV}" ]] || echo "${CSV_COLS}" > "${GL_CSV}"

  GL_GIT_SHA="$(git -C "${SGL}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  local budget_min="${GL_BUDGET_MIN:-$(( ${GL_BUDGET_H:-14} * 60 ))}"
  GL_DEADLINE="${GL_DEADLINE:-$(( $(date +%s) + budget_min * 60 ))}"
  gl_log "run dir      ${GL_RUN_DIR}"
  gl_log "sglang sha   ${GL_GIT_SHA}"
  gl_log "deadline     $(date -d "@${GL_DEADLINE}" '+%Y-%m-%d %H:%M:%S') ($(( (GL_DEADLINE - $(date +%s)) / 60 )) min from now)"
  [[ "${GL_WARM_CACHE:-1}" == "1" ]] && gl_warm_cache
}

gl_remaining() { echo $(( GL_DEADLINE - $(date +%s) )); }

# Observed median boot cost, so the budget guard self-corrects instead of trusting
# a hardcoded estimate. A cold page cache turns a 3-minute boot into 40 minutes
# (the checkpoint is 1.5 TB), which would silently over-commit the schedule.
gl_boot_cost() {
  local med
  med="$(awk -F',' 'NR>1 && $48 ~ /^[0-9]+$/ {print $48}' "${GL_CSV}" 2>/dev/null \
    | sort -n | awk '{a[NR]=$1} END{if(NR)print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2))}')"
  if [[ -n "${med}" ]] && (( med > 260 )); then
    echo $(( med + 60 ))
  else
    echo 260
  fi
}

gl_budget() {
  local need="$1" left
  left="$(gl_remaining)"
  if (( left < need )); then
    gl_log "BUDGET: need ${need}s, only ${left}s left -- skipping"
    return 1
  fi
  return 0
}

# Resume support: a point already recorded (in any status) is not re-run.
gl_have() {
  local label="$1" workload="$2" conc="$3"
  [[ -f "${GL_CSV}" ]] || return 1
  awk -F',' -v l="${label}" -v w="${workload}" -v c="${conc}" \
    '$4==l && $11==w && $14==c {found=1} END{exit !found}' "${GL_CSV}"
}

# --------------------------------------------------------------------------- #
# VRAM / teardown
# --------------------------------------------------------------------------- #
gl_vram_max_mb() {
  rocm-smi --showmeminfo vram 2>/dev/null \
    | rg -No "VRAM Total Used Memory \(B\): (\d+)" -r '$1' \
    | sort -rn | head -1 | awk '{printf "%d", $1/1048576}'
}

gl_teardown() {
  if [[ -n "${GL_PGID:-}" ]]; then
    kill -TERM "-${GL_PGID}" 2>/dev/null || true
  fi
  sleep 5
  pkill -f 'bin/sglang serve' 2>/dev/null || true
  pkill -f 'sglang.launch_server' 2>/dev/null || true
  sleep 3
  if [[ -n "${GL_PGID:-}" ]]; then
    kill -KILL "-${GL_PGID}" 2>/dev/null || true
  fi
  pkill -9 -f 'bin/sglang serve' 2>/dev/null || true
  pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
  GL_PGID=""

  # Launching on top of a server that still holds 194 GB/GPU produces a bogus
  # OOM, so wait for the drain instead of assuming the kill was synchronous.
  local used=""
  for _ in $(seq 1 60); do
    used="$(gl_vram_max_mb)"
    if [[ -n "$used" ]] && (( used < VRAM_IDLE_MB )); then
      return 0
    fi
    sleep 5
  done
  gl_log "WARN: VRAM still ${used:-?} MB after drain wait"
  return 0
}

# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #
# Sets GL_SRV_LOG, GL_BOOT_S, GL_PGID and the capacity vars. Returns 0 when the
# server is serving; on failure GL_BOOT_STATUS / GL_FATAL say why.
gl_launch() {
  local mode="$1"
  GL_MODE="$mode"
  GL_SRV_LOG="${GL_RUN_DIR}/logs/${GL_LABEL}.server.log"
  : > "${GL_SRV_LOG}"

  gl_teardown

  gl_log "launch ${GL_LABEL} mode=${mode} mem=${MEM_FRAC:-0.85} radix=${RADIX:-0} cg=${CUDA_GRAPH_MAX_BS:-256} cp=${CHUNKED_PREFILL:-default} extra='${EXTRA_ARGS:-}'"

  local t0; t0=$(date +%s)
  LOG="${GL_SRV_LOG}" setsid bash "${W}/serve-k3-ext.sh" "${mode}" >/dev/null 2>&1 &
  GL_PGID=$!
  sleep 2

  local elapsed=0 last_size=0 last_progress; last_progress=$(date +%s)
  while (( elapsed < BOOT_TIMEOUT )); do
    if curl -sf --max-time 5 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
      GL_BOOT_S=$(( $(date +%s) - t0 ))
      eval "$(python3 "${TOOLS}" capacity --log "${GL_SRV_LOG}")"
      GL_BOOT_STATUS=OK
      gl_log "  ready in ${GL_BOOT_S}s: max_total_num_tokens=${max_total_num_tokens:-?} max_running_requests=${max_running_requests:-?} avail=${avail_gpu_mem_gb:-?}GB mamba_cap=${mamba_cap:--} slots=${mamba_slots:--}"
      return 0
    fi

    # Read the log's own verdict rather than guessing from an exit status: a
    # rejected arg, a too-low mem-fraction and a real OOM need different rows.
    eval "$(python3 "${TOOLS}" capacity --log "${GL_SRV_LOG}" 2>/dev/null)" || true
    if [[ -n "${boot_status:-}" && "${boot_status}" != "READY" && "${boot_status}" != "NOT_READY" ]]; then
      GL_BOOT_STATUS="${boot_status}"
      GL_FATAL="${fatal:-}"
      GL_BOOT_S=$(( $(date +%s) - t0 ))
      gl_log "  BOOT FAILED (${GL_BOOT_STATUS}) after ${GL_BOOT_S}s: ${GL_FATAL}"
      gl_teardown
      return 1
    fi

    # Only trust "process is gone" once it has had time to exec and appear under
    # its final name; before that a missing match just means it is still starting.
    if (( elapsed > 30 )) && ! pgrep -f 'bin/sglang serve' >/dev/null 2>&1; then
      # The process is gone, so a traceback in the log is now unambiguous and
      # strict classification is safe.
      eval "$(python3 "${TOOLS}" capacity --strict --log "${GL_SRV_LOG}" 2>/dev/null)" || true
      GL_BOOT_STATUS="${boot_status:-EXITED}"
      [[ "${GL_BOOT_STATUS}" == "NOT_READY" ]] && GL_BOOT_STATUS=EXITED
      GL_FATAL="${fatal:-server exited before /health}"
      GL_BOOT_S=$(( $(date +%s) - t0 ))
      gl_log "  BOOT FAILED (${GL_BOOT_STATUS}) after ${GL_BOOT_S}s: ${GL_FATAL}"
      gl_teardown
      return 1
    fi

    # Alive and still writing to the log means still booting. Only silence is a hang.
    local size; size=$(stat -c %s "${GL_SRV_LOG}" 2>/dev/null || echo 0)
    if (( size > last_size )); then
      last_size="${size}"
      last_progress=$(date +%s)
    elif (( $(date +%s) - last_progress > BOOT_STALL_TIMEOUT )); then
      GL_BOOT_STATUS=BOOT_STALLED
      GL_FATAL="server log silent for ${BOOT_STALL_TIMEOUT}s"
      GL_BOOT_S=$(( $(date +%s) - t0 ))
      gl_log "  BOOT STALLED after ${GL_BOOT_S}s (log silent ${BOOT_STALL_TIMEOUT}s)"
      gl_teardown
      return 1
    fi

    sleep 10
    elapsed=$(( $(date +%s) - t0 ))
  done

  GL_BOOT_STATUS=BOOT_TIMEOUT
  GL_FATAL="no /health after ${BOOT_TIMEOUT}s"
  GL_BOOT_S="${BOOT_TIMEOUT}"
  gl_log "  BOOT TIMEOUT after ${BOOT_TIMEOUT}s"
  gl_teardown
  return 1
}

# Pull the checkpoint into page cache with parallel reads. The node has 3 TB of
# RAM against a 1.5 TB checkpoint, and this takes ~12 s at 35 GB/s -- whereas
# letting sglang's loader fault it in cold costs tens of minutes per boot because
# it reads at a much lower queue depth.
gl_warm_cache() {
  local cached_before; cached_before=$(free -g | awk '/Mem/{print $6}')
  local t0; t0=$(date +%s)
  local d
  for d in "${MODEL_HUB}" "${DRAFT_HUB}"; do
    [[ -d "${d}/blobs" ]] || continue
    find "${d}/blobs/" -type f -size +100M -print0 2>/dev/null \
      | xargs -0 -P 16 -I{} dd if={} of=/dev/null bs=16M status=none 2>/dev/null
  done
  gl_log "page cache warmed in $(( $(date +%s) - t0 ))s: ${cached_before} -> $(free -g | awk '/Mem/{print $6}') GB cached"
}

gl_alive() {
  pgrep -f 'bin/sglang serve' >/dev/null 2>&1 \
    && curl -sf --max-time 15 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null
}

# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #
# Protocols kept identical to sweep-highconc.sh / sweep-k3.sh / bench-prefix.sh so
# new numbers stay comparable to the published playbook and to sglang#32548.
# Sets GL_WARGS / GL_ISL / GL_OSL / GL_NP as globals -- must NOT be called through
# command substitution, or the assignments are lost with the subshell.
gl_workload_args() {
  local wl="$1" conc="$2"
  case "$wl" in
    w1)  GL_ISL=1024; GL_OSL=1024; GL_NP=$(( conc * 2 ))
         GL_WARGS="--dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 --num-prompts ${GL_NP} --max-concurrency ${conc} --warmup-requests 2 --flush-cache" ;;
    w2)  GL_ISL=8192; GL_OSL=1024; GL_NP=$(( conc * 2 ))
         GL_WARGS="--dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 --num-prompts ${GL_NP} --max-concurrency ${conc} --warmup-requests 4 --flush-cache" ;;
    w3)  GL_ISL=4096; GL_OSL=256; GL_NP=256
         GL_WARGS="--dataset-name generated-shared-prefix --gsp-num-groups 32 --gsp-prompts-per-group 8 --gsp-system-prompt-len 4096 --gsp-question-len 128 --gsp-output-len 256 --max-concurrency ${conc} --warmup-requests 4 --flush-cache --cache-report" ;;
    # Peak-activation probe for the mem-fraction bisection. ISL 16384 fills a
    # whole prefill chunk and the fan-out loads the decode path at the same
    # concurrency the lane will actually run at, which is the runtime OOM a
    # successful boot cannot rule out. One wave only -- this is a gate, not a
    # measurement, so it must stay cheap.
    stress) GL_ISL=16384; GL_OSL=256; GL_NP="${conc}"
         GL_WARGS="--dataset-name random --random-input-len 16384 --random-output-len 256 --random-range-ratio 1 --num-prompts ${GL_NP} --max-concurrency ${conc} --flush-cache" ;;
    *) return 1 ;;
  esac
}

# gl_bench <workload> <conc> -- run the point, append exactly one CSV row.
# Returns 0 only when the point completed and the server survived.
gl_bench() {
  local wl="$1" conc="$2"
  gl_workload_args "$wl" "$conc" || { gl_log "unknown workload ${wl}"; return 1; }

  local tag="${GL_LABEL}__${wl}_c${conc}"
  local bjson="${GL_RUN_DIR}/logs/${tag}.jsonl"
  local btext="${GL_RUN_DIR}/logs/${tag}.txt"
  local line0; line0=$(wc -l < "${GL_SRV_LOG}")
  local t0; t0=$(date +%s)

  gl_log "  bench ${wl} c${conc} (np=${GL_NP})"
  # shellcheck disable=SC2086
  ( cd "${SGL}" && timeout "${BENCH_TIMEOUT:-5400}" python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" \
      ${GL_WARGS} --output-file "${bjson}" --tag "${tag}" ) 2>&1 \
    | rg -v "${BENCH_FILTER}" > "${btext}" || true
  local bench_s=$(( $(date +%s) - t0 ))
  local line1; line1=$(wc -l < "${GL_SRV_LOG}")

  # Defaults so an early-failing point still writes a well-formed row.
  local status=NO_RESULT
  local out_tps=NA total_tps=NA req_tps=NA conc_ach=NA accept_len=NA
  local mean_ttft_ms=NA median_ttft_ms=NA p99_ttft_ms=NA mean_tpot_ms=NA median_tpot_ms=NA
  local mean_itl_ms=NA mean_e2e_ms=NA median_e2e_ms=NA
  local gen_tok=NA retok_tok=NA retok_div_pct=NA cache_hit_pct=NA
  local completed=NA duration_s=NA
  eval "$(python3 "${TOOLS}" parse-bench --jsonl "${bjson}" --text "${btext}" 2>/dev/null)" || true

  local run_med=NA run_max=NA tok_usage_max=NA tok_usage_med=NA mamba_usage_max=NA
  local mamba_usage_med=NA sched_accept_med=NA gen_tps_med=NA queue_max=NA retract_n=NA
  local cudagraph=NA sched_samples=NA
  eval "$(python3 "${TOOLS}" scrape --log "${GL_SRV_LOG}" --start-line "${line0}" --end-line "${line1}" 2>/dev/null)" || true

  eval "$(python3 "${TOOLS}" capacity --log "${GL_SRV_LOG}" 2>/dev/null)" || true

  # A dead server outranks a parsed result: a run that finished only because the
  # client gave up must not be ranked as a win.
  if ! gl_alive; then
    status=CRASH
    gl_log "  SERVER DIED during ${wl} c${conc}"
  fi

  printf '%s\n' "$(date -Is),${GL_PHASE},${GL_LANE},${GL_LABEL},${GL_MODE},${MEM_FRAC:-0.85},${RADIX:-0},${CUDA_GRAPH_MAX_BS:-256},${CHUNKED_PREFILL:-default},\"${EXTRA_ARGS:-}\",${wl},${GL_ISL},${GL_OSL},${conc},${GL_NP},${status},\"${fatal:-}\",${out_tps},${total_tps},${req_tps},${conc_ach},${accept_len},${mean_ttft_ms},${median_ttft_ms},${p99_ttft_ms},${mean_tpot_ms},${median_tpot_ms},${mean_itl_ms},${mean_e2e_ms},${median_e2e_ms},${gen_tok},${retok_tok},${retok_div_pct},${cache_hit_pct},${max_total_num_tokens:-NA},${max_running_requests:-NA},${avail_gpu_mem_gb:-NA},${mamba_cap:-NA},${mamba_slots:-NA},${run_med},${run_max},${tok_usage_max},${mamba_usage_max},${sched_accept_med},${queue_max},${retract_n},${cudagraph},${GL_BOOT_S:-NA},${bench_s},${GL_GIT_SHA}" >> "${GL_CSV}"

  gl_log "  -> ${status} total_tps=${total_tps} out_tps=${out_tps} ttft_med=${median_ttft_ms}ms tpot=${mean_tpot_ms}ms run=${run_med} tok_use=${tok_usage_max} mamba_use=${mamba_usage_max} q=${queue_max} retract=${retract_n} retok_div=${retok_div_pct}% (${bench_s}s)"

  [[ "${status}" == "OK" ]]
}

# gl_row <status> -- record a config that never reached a bench point.
gl_row() {
  local status="$1"
  printf '%s\n' "$(date -Is),${GL_PHASE},${GL_LANE},${GL_LABEL},${GL_MODE:-NA},${MEM_FRAC:-0.85},${RADIX:-0},${CUDA_GRAPH_MAX_BS:-256},${CHUNKED_PREFILL:-default},\"${EXTRA_ARGS:-}\",none,NA,NA,0,0,${status},\"${GL_FATAL:-}\",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${max_total_num_tokens:-NA},${max_running_requests:-NA},${avail_gpu_mem_gb:-NA},${mamba_cap:-NA},${mamba_slots:-NA},NA,NA,NA,NA,NA,NA,NA,NA,${GL_BOOT_S:-NA},0,${GL_GIT_SHA}" >> "${GL_CSV}"
}

# Read an incumbent recipe file into exported vars, and set GL_INC_LABEL /
# GL_INC_CONC. Deliberately line-based: EXTRA_ARGS holds multi-word values, and
# word-splitting a whole file into positional parameters silently truncates them
# (it drops everything after the first token of the value).
gl_read_incumbent() {
  local lane="$1"
  # Separate statement on purpose: bash expands every argument of `local` before
  # the builtin runs, so referencing ${lane} in the same declaration reads the
  # caller's (empty) variable, not "$1".
  local f="${GL_RUN_DIR}/incumbent-${lane}.env"
  [[ -f "${f}" ]] || return 1
  gl_reset
  GL_INC_LABEL=""; GL_INC_CONC=""
  local kv
  while IFS= read -r kv; do
    [[ -z "${kv}" ]] && continue
    case "${kv}" in
      MEM_FRAC=*)          export MEM_FRAC="${kv#*=}" ;;
      RADIX=*)             export RADIX="${kv#*=}" ;;
      CUDA_GRAPH_MAX_BS=*) export CUDA_GRAPH_MAX_BS="${kv#*=}" ;;
      CHUNKED_PREFILL=*)   [[ "${kv#*=}" != "default" && "${kv#*=}" != "NA" ]] && export CHUNKED_PREFILL="${kv#*=}" ;;
      EXTRA_ARGS=*)        [[ "${kv#*=}" != "NA" ]] && export EXTRA_ARGS="${kv#*=}" ;;
      BEST_LABEL=*)        GL_INC_LABEL="${kv#*=}" ;;
      BEST_CONC=*)         GL_INC_CONC="${kv#*=}" ;;
    esac
  done < "${f}"
  return 0
}

# Echo the currently-exported recipe as quoted KEY=VAL words, safe to expand into
# a cfg call with "${arr[@]}".
gl_knob_args() {
  GL_KNOBS=("MEM_FRAC=${MEM_FRAC:-0.85}" "RADIX=${RADIX:-0}" "CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}")
  [[ -n "${CHUNKED_PREFILL:-}" ]] && GL_KNOBS+=("CHUNKED_PREFILL=${CHUNKED_PREFILL}")
  [[ -n "${EXTRA_ARGS:-}" ]] && GL_KNOBS+=("EXTRA_ARGS=${EXTRA_ARGS}")
}

# Clear per-config state so a failed probe cannot inherit the previous config's
# capacity numbers.
gl_reset() {
  unset max_total_num_tokens max_running_requests avail_gpu_mem_gb mamba_cap mamba_slots
  unset chunked_prefill_size max_prefill_tokens context_len intermediate_ssm_gb fatal boot_status
  unset MEM_FRAC RADIX CUDA_GRAPH_MAX_BS CHUNKED_PREFILL EXTRA_ARGS AITER SITU_A8W4
  GL_BOOT_S=NA; GL_BOOT_STATUS=""; GL_FATAL=""
}
