#!/usr/bin/env bash
# Launch GLM-5.3-Flash on 8x MI355X (gfx950) inside the dev container.
#
# Backend choice is forced by code constraints, not copied from the NVIDIA recipe:
#   - DSA must be tilelang: _check_kpool_tail_backend only accepts fa3/tilelang/trtllm
#     for index_kpool>1 (this model has index_kpool=4); trtllm/fa3 are CUDA-only and
#     the AMD-native `aiter` DSA backend is not on that whitelist.
#   - KV cache must be bfloat16: FP8 KV only pairs with the trtllm DSA backend.
#   - linear-attn must be triton: every other KDA kernel raises on non-CUDA, and
#     TritonKDAKernel is the only one implementing the `lower_bound` safe gate that
#     this checkpoint's linear_lower_bound requires.
#   - MoE runner aiter: deep_gemm is gated to CUDA/MUSA in its configurer.
# No --speculative-* : the DSA nextn draft path is CUDA-only.
set -uo pipefail

NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MOE_BACKEND="${MOE_BACKEND:-aiter}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CHUNKED="${CHUNKED:-8192}"
EXTRA="${EXTRA:-}"
LOGNAME="${LOGNAME:-serve-glm53-$(date -u +%Y%m%dT%H%M%SZ)}"

# Escape hatches, all default-on (canonical path) unless overridden during bring-up.
FUSE_TOPK="${FUSE_TOPK:-1}"          # SGLANG_DSA_FUSE_TOPK=0 avoids _append_kpool_tail_to_topk_kernel
TILELANG_MHC="${TILELANG_MHC:-1}"    # 0 falls back to the torch-native mHC reference path

EXTRA_ENV="${EXTRA_ENV:-}"          # e.g. EXTRA_ENV="USE_ROCM_AITER_ROPE_BACKEND=0"

docker exec -d "$NAME" bash -lc "
export SGLANG_DSA_FUSE_TOPK=${FUSE_TOPK}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${TILELANG_MHC}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${TILELANG_MHC}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
$(for kv in ${EXTRA_ENV}; do echo \"export ${kv}\"; done)
cd /sgl-workspace/sglang
exec python3 -m sglang.launch_server \
  --model-path '${MODEL}' \
  --served-model-name glm-5.3-flash \
  --tp ${TP} \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype bfloat16 \
  --quantization fp8 \
  --moe-runner-backend ${MOE_BACKEND} \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --chunked-prefill-size ${CHUNKED} --max-prefill-tokens ${CHUNKED} \
  --max-running-requests ${MAX_RUNNING} \
  --mem-fraction-static ${MEM_FRAC} \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port ${PORT} ${EXTRA} \
  > /results/logs/${LOGNAME}.log 2>&1
"
echo "launching -> ~/glm53-flash-results/logs/${LOGNAME}.log"
echo "env: SGLANG_DSA_FUSE_TOPK=${FUSE_TOPK} TILELANG_MHC=${TILELANG_MHC} moe=${MOE_BACKEND} extra='${EXTRA}'"
