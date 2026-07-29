#!/usr/bin/env bash
# accuracy-k3.sh -- phase 5 gate for the Kimi-K3 launch-parameter search.
#
#   GL_BUDGET_MIN=360 ./accuracy-k3.sh [run_dir]
#
# Runs the winning config of each lane through exactly the protocol that produced
# the published baselines, so the numbers are directly comparable:
#   GSM8K   run_eval, n=1319, greedy               (baseline 97.49 nospec / 97.64 dspark)
#   AIME25  sgl-eval, avg-of-8, max-tokens 64000   (baseline 93.33+-4.36 / 94.58+-3.05)
#
# Order is deliberate: GSM8K on both lanes first, then AIME25 no-spec, then AIME25
# DSpark (which ran 3.45x slower). If the budget runs out, the cheapest-to-lose
# number is the one that goes missing.
#
# Only knobs that change numerics can produce gibberish -- mem-fraction,
# chunked-prefill, cuda-graph batch, max-running-requests and schedule
# conservativeness are memory/scheduling only. The gate exists for
# --mamba-ssm-dtype, --enable-linear-replayssm-spec and
# --speculative-dspark-block-size, which alter the state-precision / verify path.
set -uo pipefail

W=/sgl-workspace/workspace
source "${W}/gridlib.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td "${W}"/grid_results/2*/ 2>/dev/null | head -1)"
fi
RUN_DIR="${RUN_DIR%/}"
[[ -d "${RUN_DIR}" ]] || { echo "no run dir: ${RUN_DIR}"; exit 1; }

gl_init "${RUN_DIR}"
ACC_DIR="${GL_RUN_DIR}/accuracy"
mkdir -p "${ACC_DIR}"
PHASE=p5

COST_GSM8K=2700
COST_AIME=9000

# --------------------------------------------------------------------------- #
# Resolve the winning launch recipe of a lane into exported env vars.
load_winner() {
  local lane="$1"
  local f="${GL_RUN_DIR}/incumbent-${lane}.env"
  gl_reset
  WIN_LABEL=""; WIN_CONC=""
  if [[ ! -f "${f}" ]]; then
    gl_log "P5: no incumbent for ${lane}; falling back to the published recipe"
    export MEM_FRAC=0.85
    WIN_LABEL="published-baseline"
    return 0
  fi
  local kv
  while read -r kv; do
    [[ -z "${kv}" ]] && continue
    case "${kv}" in
      MEM_FRAC=*)          export MEM_FRAC="${kv#*=}" ;;
      RADIX=*)             export RADIX="${kv#*=}" ;;
      CUDA_GRAPH_MAX_BS=*) export CUDA_GRAPH_MAX_BS="${kv#*=}" ;;
      CHUNKED_PREFILL=*)   [[ "${kv#*=}" != "default" && "${kv#*=}" != "NA" ]] && export CHUNKED_PREFILL="${kv#*=}" ;;
      EXTRA_ARGS=*)        [[ "${kv#*=}" != "NA" ]] && export EXTRA_ARGS="${kv#*=}" ;;
      BEST_LABEL=*)        WIN_LABEL="${kv#*=}" ;;
      BEST_CONC=*)         WIN_CONC="${kv#*=}" ;;
    esac
  done < "${f}"
  gl_log "P5 ${lane} winner: ${WIN_LABEL} (mem=${MEM_FRAC:-0.85} radix=${RADIX:-0} cg=${CUDA_GRAPH_MAX_BS:-256} cp=${CHUNKED_PREFILL:-default} extra='${EXTRA_ARGS:-}')"
}

serve_winner() {
  local lane="$1" mode="$2"
  GL_LABEL="p5-${lane}"; GL_LANE="${lane}"; GL_PHASE=p5
  gl_launch "${mode}"
}

# --------------------------------------------------------------------------- #
run_gsm8k() {
  local lane="$1"
  local out="${ACC_DIR}/gsm8k-${lane}.txt"
  if [[ -s "${out}" ]] && rg -q "GSM8K DONE" "${out}"; then
    gl_log "  skip GSM8K ${lane} (done)"; return 0
  fi
  gl_budget "${COST_GSM8K}" || return 2
  gl_log "  GSM8K ${lane} (n=1319, greedy)"
  ( cd /sgl-workspace/sglang && timeout "${COST_GSM8K}" python3 -m sglang.test.run_eval \
      --port "${PORT}" --eval-name gsm8k \
      --num-examples 1319 --num-threads 32 \
      --max-tokens 8192 --temperature 0 ) > "${out}" 2>&1
  echo "GSM8K DONE" >> "${out}"
  gl_log "  -> $(rg -N 'Accuracy|score' "${out}" | tail -2 | tr '\n' ' ')"
}

run_aime25() {
  local lane="$1"
  local out="${ACC_DIR}/aime25-${lane}.txt"
  if [[ -s "${out}" ]] && rg -q "AIME DONE" "${out}"; then
    gl_log "  skip AIME25 ${lane} (done)"; return 0
  fi
  gl_budget "${COST_AIME}" || return 2
  gl_log "  AIME25 ${lane} (avg-of-8, max-tokens 64000)"
  ( timeout "${COST_AIME}" sgl-eval run aime25 \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model moonshotai/Kimi-K3 --api-key EMPTY \
      --n-repeats 8 --num-threads 48 \
      --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking ) > "${out}" 2>&1
  echo "AIME DONE" >> "${out}"
  gl_log "  -> $(rg -N 'pass@1|stop_rate|error_rate' "${out}" | tail -3 | tr '\n' ' ')"
}

# Degeneration check with real completion text: the evals score answers, but a
# config that produces fluent-looking loops can still score badly for the wrong
# reason. This looks at the text itself.
run_gibberish() {
  local lane="$1"
  local det="${ACC_DIR}/details-${lane}.jsonl"
  local rep="${ACC_DIR}/gibberish-${lane}.txt"
  if [[ -s "${rep}" ]]; then gl_log "  skip gibberish ${lane} (done)"; return 0; fi
  gl_budget 900 || return 2
  gl_log "  gibberish probe ${lane}"
  ( cd /sgl-workspace/sglang && timeout 900 python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat --host 127.0.0.1 --port "${PORT}" \
      --model moonshotai/Kimi-K3 --dataset-name random \
      --random-input-len 1024 --random-output-len 512 --random-range-ratio 1 \
      --num-prompts 32 --max-concurrency 8 \
      --output-file "${det}" --output-details --flush-cache ) > "${rep}.raw" 2>&1
  python3 "${W}/gridtools.py" gibberish --details "${det}" > "${rep}" 2>>"${rep}.raw" || true
  gl_log "  -> $(tr '\n' ' ' < "${rep}")"
}

# --------------------------------------------------------------------------- #
gate() {
  python3 - "${ACC_DIR}" <<'PY'
import os, re, sys

acc = sys.argv[1]
BASE = {
    "nospec": {"gsm8k": 97.49, "aime25": 93.33, "aime_sigma": 4.36},
    "dspark": {"gsm8k": 97.64, "aime25": 94.58, "aime_sigma": 3.05},
}

def read(p):
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""

print("# Phase 5 accuracy gate\n")
verdicts = []
for lane in ("nospec", "dspark"):
    base = BASE[lane]
    print(f"## lane `{lane}`\n")

    g = read(f"{acc}/gsm8k-{lane}.txt")
    m = re.findall(r"(?:Accuracy|score)\D{0,12}([\d.]+)", g)
    if m:
        v = float(m[-1])
        v = v * 100 if v <= 1.0 else v
        d = v - base["gsm8k"]
        ok = abs(d) <= 1.0
        verdicts.append(ok)
        print(f"- GSM8K: **{v:.2f}%** vs baseline {base['gsm8k']}% (delta {d:+.2f} pp) -> "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        print("- GSM8K: no result")

    a = read(f"{acc}/aime25-{lane}.txt")
    m = re.findall(r"pass@1[^\d]{0,40}([\d.]+)", a)
    if m:
        v = float(m[-1])
        v = v * 100 if v <= 1.0 else v
        d = v - base["aime25"]
        ok = abs(d) <= base["aime_sigma"]
        verdicts.append(ok)
        print(f"- AIME25 pass@1 avg-of-8: **{v:.2f}%** vs baseline {base['aime25']}% "
              f"(delta {d:+.2f} pp, 1 sigma = {base['aime_sigma']}) -> {'PASS' if ok else 'FAIL'}")
        for field in ("stop_rate", "truncated", "no_answer", "error_rate", "error"):
            fm = re.search(rf"{field}\D{{0,12}}([\d.]+)", a)
            if fm:
                print(f"  - {field} = {fm.group(1)}")
    else:
        print("- AIME25: no result")

    gb = read(f"{acc}/gibberish-{lane}.txt")
    if gb:
        vm = re.search(r"verdict=(\w+)", gb)
        rm = re.search(r"rep\d+gram_max=([\d.]+)", gb)
        if vm:
            verdicts.append(vm.group(1) == "PASS")
            print(f"- degeneration probe: **{vm.group(1)}** "
                  f"(max n-gram repetition {rm.group(1) if rm else '?'})")
    print()

print("## Verdict\n")
if not verdicts:
    print("INCONCLUSIVE - no results parsed.")
elif all(verdicts):
    print("**PASS** - every measured config is within tolerance of the published baseline.")
else:
    print(f"**FAIL** - {verdicts.count(False)} of {len(verdicts)} checks outside tolerance. "
          "Do not publish the affected config as a recommended recipe.")
PY
}

# --------------------------------------------------------------------------- #
main() {
  gl_log "########## P5 accuracy gate ##########"

  # GSM8K on both lanes first (cheap, greedy, and the sharpest broken-verify signal).
  local lane mode
  for lane in nospec dspark; do
    mode="${lane}"
    load_winner "${lane}"
    if serve_winner "${lane}" "${mode}"; then
      run_gsm8k "${lane}"
      run_gibberish "${lane}"
    else
      gl_log "P5 ${lane}: winner failed to boot (${GL_BOOT_STATUS}) -- ${GL_FATAL}"
      gl_row "${GL_BOOT_STATUS}"
    fi
    gl_teardown
  done

  # Then the expensive sampled eval, no-spec before DSpark.
  for lane in nospec dspark; do
    mode="${lane}"
    local out="${ACC_DIR}/aime25-${lane}.txt"
    if [[ -s "${out}" ]] && rg -q "AIME DONE" "${out}"; then
      gl_log "skip AIME25 ${lane} (done)"; continue
    fi
    gl_budget "$(( COST_AIME + 400 ))" || { gl_log "P5: out of budget before AIME25 ${lane}"; break; }
    load_winner "${lane}"
    if serve_winner "${lane}" "${mode}"; then
      run_aime25 "${lane}"
    else
      gl_log "P5 ${lane}: winner failed to boot for AIME25"
    fi
    gl_teardown
  done

  gate > "${GL_RUN_DIR}/accuracy_gate.md"
  gl_log "gate written to ${GL_RUN_DIR}/accuracy_gate.md"
  cat "${GL_RUN_DIR}/accuracy_gate.md"
}

trap 'gl_log "P5 INTERRUPTED"; gl_teardown; exit 130' INT TERM
main
gl_teardown
