#!/usr/bin/env bash
# GLM-5.2-FP8 high-throughput control on 8x MI355X, launched from the same
# SGLang/AITER stack as GLM-5.3-Flash so the comparison carries neither a node
# nor an engine-version confound.
#
# The published 0.5.17 cell uses a 32768-token prefill chunk. On this 0.5.18
# comparison head that shape aborts while lazily compiling aiter fp8_mqa_logits;
# 8192 is the largest verified direct-control setting and matches GLM-5.3.
#
# Weights live at /var/tmp/models/GLM-5.2-FP8 (owner qinhang.wu@amd.com), mounted
# read-only at /models/GLM-5.2-FP8. Never write to that path.
set -euo pipefail

NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MODEL="${MODEL:-/models/GLM-5.2-FP8}"
LOGNAME="${LOGNAME:-serve-glm52-$(date -u +%Y%m%dT%H%M%SZ)}"
EXTRA="${EXTRA:-}"

docker exec -d "$NAME" bash -lc "
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
cd /sgl-workspace/sglang/python
exec python3 -m sglang.launch_server \
  --model-path '${MODEL}' \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tp-size ${TP} \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.92 \
  --cuda-graph-max-bs 64 \
  --max-running-requests 64 \
  --schedule-policy lpm \
  --num-continuous-decode-steps 2 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port ${PORT} ${EXTRA} \
  > /results/logs/${LOGNAME}.log 2>&1
"
echo "launching -> ~/glm53-flash-results/logs/${LOGNAME}.log"
