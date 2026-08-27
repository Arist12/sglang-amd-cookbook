#!/usr/bin/env bash
# Round 2 of the determinism bisection.
#
# Round 1 cleared mHC TileLang (22/24 distinct with the torch reference) and aiter
# fused RoPE (24/24 distinct with it off). The MoE config could not be measured:
# --moe-runner-backend triton aborts at startup with
#   "fused silu_and_mul_clamp kernel is CUDA/XPU only; HIP must disable SWIGLU_CLAMP_FUSION"
# so this round disables SGLANG_OPT_SWIGLU_CLAMP_FUSION to reach it, then also tries
# turning aiter off wholesale as the broadest available test.
set -uo pipefail
cd "$(dirname "$0")"

SUMMARY=logs/bisect-summary2.txt
: > "$SUMMARY"

run_cfg() {
  local name="$1"; shift
  echo "################ $name ################" | tee -a "$SUMMARY"
  echo "   env: $*" | tee -a "$SUMMARY"
  docker exec glm53-flash bash -lc 'pkill -f sglang.launch_server; sleep 8' >/dev/null 2>&1
  sleep 8
  env "$@" LOGNAME="serve-bisect-${name}" EXTRA="--disable-cuda-graph" bash serve_glm53.sh >/dev/null
  local ready=0
  for i in $(seq 1 75); do
    sleep 20
    curl -s -m 5 http://127.0.0.1:30000/get_model_info >/dev/null 2>&1 && { ready=1; break; }
    docker exec glm53-flash bash -lc 'pgrep -f sglang.launch_server >/dev/null' 2>/dev/null || break
  done
  if [ "$ready" != 1 ]; then
    echo "  LAUNCH FAILED" | tee -a "$SUMMARY"
    grep -oE "[A-Za-z]*Error: .*" "logs/serve-bisect-${name}.log" 2>/dev/null | tail -2 | tee -a "$SUMMARY"
    return
  fi
  python3 determinism_probe.py 8 8 2>&1 | tail -2 | tee -a "$SUMMARY"
}

run_cfg "moe-triton2" MOE_BACKEND=triton EXTRA_ENV="SGLANG_OPT_SWIGLU_CLAMP_FUSION=0"
run_cfg "aiter-off" EXTRA_ENV="SGLANG_USE_AITER=0 SGLANG_OPT_SWIGLU_CLAMP_FUSION=0"

echo "==================== SUMMARY2 ===================="
cat "$SUMMARY"
