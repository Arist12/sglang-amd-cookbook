#!/usr/bin/env bash
# GLM-5.2-FP8 same-build baseline on 8x MI355X, launched from the SAME PR #36507
# container as GLM-5.3-Flash so the throughput/latency comparison carries neither a
# node nor an SGLang-version confound.
#
# Flags are the playbook's canonical AMD recipe, unchanged; only the image differs.
# This run doubles as a ROCm regression check: PR #36507 changes dsa_backend.py by
# +1511 lines and GLM-5.2 shares that backend.
#
# Weights live at /var/tmp/models/GLM-5.2-FP8 (owner qinhang.wu@amd.com), mounted
# read-only at /models/GLM-5.2-FP8. Never write to that path.
set -uo pipefail

NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MODEL="${MODEL:-/models/GLM-5.2-FP8}"
LOGNAME="${LOGNAME:-serve-glm52-$(date -u +%Y%m%dT%H%M%SZ)}"
EXTRA="${EXTRA:-}"

docker exec -d "$NAME" bash -lc "
cd /sgl-workspace/sglang
exec python3 -m sglang.launch_server \
  --model-path '${MODEL}' \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tp ${TP} \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port ${PORT} ${EXTRA} \
  > /results/logs/${LOGNAME}.log 2>&1
"
echo "launching -> ~/glm53-flash-results/logs/${LOGNAME}.log"
