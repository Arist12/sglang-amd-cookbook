#!/usr/bin/env bash
# Accuracy gates for the measured GLM-5.3-Flash cell.
#
# GSM8K intentionally uses the same in-tree greedy harness as the GLM-5.2 cell.
# AIME25 must use sgl-eval: in-tree run_eval takes the first "Answer:" match from
# a reasoning trace and under-reports these thinking models.
set -euo pipefail

MODE="${1:?usage: eval.sh {smoke|gsm8k|aime25} <tag>}"
TAG="${2:?usage: eval.sh {smoke|gsm8k|aime25} <tag>}"
NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
SERVED="${SERVED:-glm-5.3-flash}"
HOST_OUT="${HOST_OUT:-$HOME/glm53-flash-results/eval/${TAG}}"
CONTAINER_OUT="/results/eval/${TAG}"
mkdir -p "$HOST_OUT"

run() {
  docker exec "$NAME" bash -lc "cd /sgl-workspace/sglang/python && $1"
}

timestamped() {
  local stem="$1"
  shift
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.started"
  "$@"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.finished"
}

case "$MODE" in
  smoke)
    timestamped gsm8k-smoke \
      run "python3 -m sglang.test.run_eval --port ${PORT} --eval-name gsm8k \
        --thinking-mode glm-45 --max-tokens 8192 --temperature 0 \
        --num-examples 100 --num-threads 32" 2>&1 \
      | tee "$HOST_OUT/gsm8k-smoke.txt"
    ;;
  gsm8k)
    timestamped gsm8k-full \
      run "python3 -m sglang.test.run_eval --port ${PORT} --eval-name gsm8k \
        --thinking-mode glm-45 --max-tokens 8192 --temperature 0 \
        --num-examples 1319 --num-threads 32" 2>&1 \
      | tee "$HOST_OUT/gsm8k-full.txt"
    ;;
  aime25)
    timestamped aime25-full \
      run "sgl-eval run aime25 \
        --api-key EMPTY --base-url http://127.0.0.1:${PORT}/v1 \
        --model ${SERVED} --num-threads 32 \
        --n-repeats 16 --max-tokens 64000 \
        --temperature 1.0 --top-p 0.95 --thinking \
        --out-dir '${CONTAINER_OUT}/aime25-full'" 2>&1 \
      | tee "$HOST_OUT/aime25-full.txt"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
