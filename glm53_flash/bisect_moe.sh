#!/usr/bin/env bash
# Round 3: bisect inside the MoE, now that per-module tensor dumps have localized the
# nondeterminism to model.layers.3.mlp.experts -- identical inputs, identical gate,
# identical top-k, different routed-expert output.
#
# Earlier rounds could not test the MoE at all: SGLANG_USE_AITER=0 does not touch the
# runner when --moe-runner-backend aiter is given explicitly, and
# --moe-runner-backend triton aborts at startup on HIP.
#
# The two candidates that differ between GLM-5.3-Flash and the deterministic GLM-5.2
# control are swiglu_limit=10.0 (absent in GLM-5.2; drives the silu_and_mul_clamp
# fusion that HIP cannot compile) and 288 routed experts under SGLANG_MOE_PADDING=1
# (36 per rank at TP=8).
set -uo pipefail
cd "$(dirname "$0")"

SUMMARY=logs/bisect-moe-summary.txt
: > "$SUMMARY"

run_cfg() {
  local name="$1"; shift
  echo "################ $name ################" | tee -a "$SUMMARY"
  echo "   $*" | tee -a "$SUMMARY"
  docker exec glm53-flash bash -lc 'pkill -f "launch_server"; sleep 8' >/dev/null 2>&1
  sleep 8
  env "$@" LOGNAME="serve-moe-${name}" EXTRA="--disable-cuda-graph" bash serve_glm53.sh >/dev/null
  local ready=0
  for i in $(seq 1 75); do
    sleep 20
    curl -s -m 5 http://127.0.0.1:30000/get_model_info >/dev/null 2>&1 && { ready=1; break; }
    docker exec glm53-flash bash -lc 'pgrep -f "launch_server" >/dev/null' 2>/dev/null || break
  done
  if [ "$ready" != 1 ]; then
    echo "  LAUNCH FAILED" | tee -a "$SUMMARY"
    grep -oE "[A-Za-z]*Error: .*" "logs/serve-moe-${name}.log" 2>/dev/null | tail -2 | tee -a "$SUMMARY"
    return
  fi
  python3 determinism_probe.py 6 8 2>&1 | tail -2 | tee -a "$SUMMARY"
}

run_cfg "noclamp"      EXTRA_ENV="SGLANG_OPT_SWIGLU_CLAMP_FUSION=0"
run_cfg "nopadding"    EXTRA_ENV="SGLANG_MOE_PADDING=0"
run_cfg "noclamp-nopad" EXTRA_ENV="SGLANG_OPT_SWIGLU_CLAMP_FUSION=0 SGLANG_MOE_PADDING=0"

echo "==================== MoE BISECTION ===================="
cat "$SUMMARY"
