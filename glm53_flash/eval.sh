#!/usr/bin/env bash
# Accuracy evals against a running server, inside the dev container.
#
#   bash eval.sh smoke <tag>    GSM8K 100-question gate (must clear >90% before anything else)
#   bash eval.sh gsm8k <tag>    full 1319, both harnesses (sgl-eval = cookbook's; run_eval = GLM-5.2 baseline's)
#   bash eval.sh aime25 <tag>   sgl-eval AIME25, 16 repeats
#
# AIME25 must use sgl-eval, never the in-tree run_eval harness: its ANSWER_PATTERN
# takes the FIRST "Answer:" match, which lands inside the reasoning trace and
# under-reports a thinking model (91.5% -> 62.5% on the same GLM-5.2 server).
set -uo pipefail

MODE="${1:?usage: eval.sh {smoke|gsm8k|aime25} <tag>}"
TAG="${2:?usage: eval.sh {smoke|gsm8k|aime25} <tag>}"
NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
SERVED="${SERVED:-glm-5.3-flash}"
D="$HOME/glm53-flash-results/eval/${TAG}"
mkdir -p "$D"

run() { docker exec "$NAME" bash -lc "cd /sgl-workspace/sglang && $1"; }

case "$MODE" in
  smoke)
    run "python3 -m sglang.test.run_eval --port ${PORT} --eval-name gsm8k \
      --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 100" 2>&1 \
      | tee "$D/gsm8k-smoke-100.txt"
    ;;
  gsm8k)
    echo "======== GSM8K full 1319 via sgl-eval (cookbook harness) ========"
    run "sgl-eval run gsm8k \
      --base-url http://127.0.0.1:${PORT}/v1 \
      --model ${SERVED} \
      --num-threads 64 --max-tokens 32768 \
      --temperature 1.0 --top-p 0.95 --thinking" 2>&1 \
      | tee "$D/gsm8k-full-sgleval.txt"
    echo "======== GSM8K full 1319 via in-tree run_eval (GLM-5.2 baseline口径) ========"
    run "python3 -m sglang.test.run_eval --port ${PORT} --eval-name gsm8k \
      --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319" 2>&1 \
      | tee "$D/gsm8k-full-runeval.txt"
    ;;
  aime25)
    run "sgl-eval run aime25 \
      --api-key EMPTY --base-url http://127.0.0.1:${PORT}/v1 \
      --model ${SERVED} \
      --n-repeats 16 --max-tokens 64000 \
      --temperature 1.0 --top-p 0.95 --thinking" 2>&1 \
      | tee "$D/aime25.txt"
    ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
