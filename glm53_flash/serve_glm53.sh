#!/usr/bin/env bash
# Launch the verified GLM-5.3-Flash high-throughput cell on 8x MI355X.
#
# setup_pr.sh must run first: this recipe needs SGLang #36607, AITER #5060,
# and hybrid_fp8_metadata.patch. The graph tiers intentionally stop at 32; the
# measured concurrency-64 row documents the eager fallback above that boundary.
set -euo pipefail

NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
REVISION="${REVISION:-04c4e9e95c5da8862dced7e5056455116f83a7e0}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CHUNKED="${CHUNKED:-8192}"
EXTRA="${EXTRA:-}"
LOGNAME="${LOGNAME:-serve-glm53-$(date -u +%Y%m%dT%H%M%SZ)}"

docker exec -d "$NAME" bash -lc "
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
cd /sgl-workspace/sglang/python
exec python3 -m sglang.launch_server \
  --model-path '${MODEL}' \
  --revision '${REVISION}' \
  --served-model-name glm-5.3-flash \
  --tp-size ${TP} --ep-size 1 \
  --trust-remote-code \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --quantization fp8 \
  --moe-runner-backend aiter \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 32 \
  --disable-radix-cache \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --chunked-prefill-size ${CHUNKED} \
  --max-running-requests ${MAX_RUNNING} \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port ${PORT} ${EXTRA} \
  > /results/logs/${LOGNAME}.log 2>&1
"
echo "launching -> ~/glm53-flash-results/logs/${LOGNAME}.log"
echo "revision=${REVISION} kv=fp8_e4m3 moe=aiter graph_bs=1,32 extra='${EXTRA}'"
