#!/usr/bin/env bash
# Bisect which component makes GLM-5.3-Flash nondeterministic on gfx950.
#
# Baseline measurement: 24/24 distinct outputs over 3 prompts x 8 repeats at
# temperature 0 with the cache flushed between calls, i.e. the first forward pass
# is already irreproducible. Each config below disables one suspect and re-measures
# with the same probe; a config that reports "DETERMINISTIC" identifies the culprit.
#
# --disable-cuda-graph is held constant so graph capture is never a variable.
set -uo pipefail
cd "$(dirname "$0")"

REPEATS="${REPEATS:-8}"
NTOK="${NTOK:-8}"
SUMMARY=logs/bisect-summary.txt
: > "$SUMMARY"

run_cfg() {
  local name="$1"; shift
  echo "################ $name ################" | tee -a "$SUMMARY"
  docker exec glm53-flash bash -lc 'pkill -f sglang.launch_server; sleep 8' >/dev/null 2>&1
  sleep 8
  env "$@" LOGNAME="serve-bisect-${name}" EXTRA="--disable-cuda-graph" bash serve_glm53.sh >/dev/null

  local ready=0
  for i in $(seq 1 75); do
    sleep 20
    if curl -s -m 5 http://127.0.0.1:30000/get_model_info >/dev/null 2>&1; then ready=1; break; fi
    if ! docker exec glm53-flash bash -lc 'pgrep -f sglang.launch_server >/dev/null' 2>/dev/null; then break; fi
  done

  if [ "$ready" != 1 ]; then
    echo "  LAUNCH FAILED - see logs/serve-bisect-${name}.log" | tee -a "$SUMMARY"
    grep -oE "[A-Za-z]*Error: .*" "logs/serve-bisect-${name}.log" 2>/dev/null | tail -2 | tee -a "$SUMMARY"
    return
  fi

  python3 determinism_probe.py "$REPEATS" "$NTOK" 2>&1 | tail -3 | tee -a "$SUMMARY"
}

# 1. aiter MoE runner -> triton. ROCm/aiter has open gfx950 reports of an A8W8
#    B-preshuffle Triton route faulting and of nondeterministic GLM FP8 decode.
run_cfg "moe-triton" MOE_BACKEND=triton

# 2. mHC TileLang kernels -> the torch reference in mhc.py.
run_cfg "mhc-torch" TILELANG_MHC=0

# 3. aiter fused RoPE off; the server log warns this path has lower precision.
run_cfg "rope-noaiter" EXTRA_ENV="USE_ROCM_AITER_ROPE_BACKEND=0"

echo
echo "==================== SUMMARY ===================="
cat "$SUMMARY"
