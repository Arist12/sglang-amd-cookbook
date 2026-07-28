#!/bin/bash
# Launch Kimi-K3 on 8x MI355X (gfx950), TP=8.
#
#   bash test_kimi_k3.sh              # non speculative-decoding
#   MODE=dspark bash test_kimi_k3.sh  # + DSpark speculative decoding
#   PORT=30001 bash test_kimi_k3.sh   # custom port
#
# This is the script that produced the numbers in kimi_k3_playbook.md. It drives
# `sglang serve` directly against a source build (see PROVENANCE below) rather
# than the published Day-0 image, because that is what was measured. To run the
# turnkey image instead, wrap the same command and env in:
#   docker run ... lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727
#
# PROVENANCE of the verified run:
#   sglang  DarkSharpness/sglang-kimi @ amd/kimi-k3 533bff471
#           (== sgl-project/sglang#32541 `kimi-k3` + HIP multi-stream disable)
#   aiter   k3-for-amd 68e42f5f  (carries the K3 opt / FlyDSL / SITU kernels)
#   ROCm    7.2.0,  torch 2.9.1+rocm7.2.0
#
# Upstream: sgl-project/sglang#32541 (model support), #32548 (AMD Day-0 recipe)

set -uo pipefail

MODE="${MODE:-nospec}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.85}"

# Weights resolve out of the local HF cache; staying offline keeps a gated-repo
# revalidation from blocking a 1.56 TB checkpoint at boot.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Mandatory on AMD, not tuning knobs: without the aiter MXFP4 path the routed
# experts are unpacked and the target weights grow 194 GB -> 249 GB per GPU,
# which leaves no room for a KV pool on a 288 GiB card at any mem-fraction.
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A8W4=1

ARGS=(
  --model-path moonshotai/Kimi-K3
  --trust-remote-code
  --tp 8
  --attention-backend triton
  --dtype bfloat16
  --mem-fraction-static "${MEM_FRAC}"
  --cuda-graph-max-bs-decode 256
  --host 0.0.0.0
  --port "${PORT}"
  --disable-radix-cache
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
)

case "${MODE}" in
  nospec) ;;
  dspark)
    ARGS+=(
      --speculative-draft-model-path RadixArk/Kimi-K3-DSpark
      --speculative-algorithm DSPARK
    )
    ;;
  *) echo "Unknown MODE: ${MODE} (nospec|dspark)" >&2; exit 2 ;;
esac

echo "mode : ${MODE}"
echo "port : ${PORT}"
echo "memfr: ${MEM_FRAC}"

exec sglang serve "${ARGS[@]}"
