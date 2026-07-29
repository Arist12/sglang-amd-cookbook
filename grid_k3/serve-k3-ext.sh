#!/usr/bin/env bash
# Same as serve-k3.sh but appends $EXTRA_ARGS, so throughput levers can be A/B'd
# without editing the baseline recipe.
#   EXTRA_ARGS="--max-running-requests 128" LOG_SUFFIX=-mrr128 ./serve-k3-ext.sh dspark
set -uo pipefail

MODE="${1:-nospec}"
PORT="${PORT:-30000}"
LOG="${LOG:-/sgl-workspace/workspace/k3-serve-${MODE}${LOG_SUFFIX:-}.log}"

export HF_HUB_OFFLINE=1

# One-shot trigger for the post-search chain (supplements + accuracy gate). This
# script is re-read from disk on every config launch, which makes it a reliable
# place to start the watcher. The run directory is recovered from the harness's log
# path rather than a new env var, because the harness only passes LOG. The mkdir is
# the lock: it succeeds exactly once, so repeated launches cannot stack watchers.
if [[ "${LOG}" == */grid_results/*/logs/*.server.log ]]; then
  _chain_run_dir="$(dirname "$(dirname "${LOG}")")"
  if mkdir /tmp/k3-chain-triggered 2>/dev/null; then
    setsid nohup bash /sgl-workspace/workspace/chain-after-search.sh \
      "${_chain_run_dir}" >/dev/null 2>&1 < /dev/null &
  fi
  # Second one-shot: rescore the degeneration probes and rewrite the verdict once
  # the accuracy gate has finished.
  if mkdir /tmp/k3-finalize-triggered 2>/dev/null; then
    setsid nohup bash /sgl-workspace/workspace/finalize-accuracy.sh \
      "${_chain_run_dir}" >/dev/null 2>&1 < /dev/null &
  fi
fi

if [[ "${AITER:-1}" == "1" ]]; then
  export SGLANG_USE_AITER=1
  export SGLANG_AITER_K3_OPT=1
  export AITER_FLYDSL_FORCE=1
  [[ "${SITU_A8W4:-1}" == "1" ]] && export AITER_SITUV2_A8W4=1
fi

ARGS=(
  --model-path moonshotai/Kimi-K3
  --trust-remote-code
  --tp 8
  --attention-backend triton
  --dtype bfloat16
  --mem-fraction-static "${MEM_FRAC:-0.85}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS:-256}"
  --host 127.0.0.1
  --port "${PORT}"
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
)

# Unset leaves sglang's own default (16384 on this model), which is what every
# recipe so far has measured.
[[ -n "${CHUNKED_PREFILL:-}" ]] && ARGS+=(--chunked-prefill-size "${CHUNKED_PREFILL}")

# HiCache needs the radix cache, so let the caller opt out of --disable-radix-cache.
[[ "${RADIX:-0}" == "0" ]] && ARGS+=(--disable-radix-cache)

if [[ "${MODE}" == "dspark" ]]; then
  ARGS+=(
    --speculative-draft-model-path RadixArk/Kimi-K3-DSpark
    --speculative-algorithm DSPARK
  )
fi

# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=(${EXTRA_ARGS})

echo "=== $(date -Is) launching mode=${MODE} port=${PORT} extra='${EXTRA_ARGS:-}' radix=${RADIX:-0} ===" | tee -a "${LOG}"
exec sglang serve "${ARGS[@]}" >> "${LOG}" 2>&1
