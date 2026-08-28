#!/usr/bin/env bash
# Reproduce the measured GLM-5.3-Flash MI355X cell against a running container.
#
#   bash bench.sh sanity <tag>  upstream 1K/1K anchors, concurrency 1 and 32
#   bash bench.sh main   <tag>  GLM-5.2 comparison shape, three repeats per point
#   bash bench.sh lat    <tag>  BS=1 latency, three repeats per input length
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: bench.sh <sanity|main|lat> <tag>" >&2
  exit 2
fi

MODE="$1"
TAG="$2"
NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
TOKENIZER="${TOKENIZER:-/hf-cache/hub/models--zai-org--GLM-5.3-Flash/snapshots/04c4e9e95c5da8862dced7e5056455116f83a7e0}"
SERVED="${SERVED:-glm-5.3-flash}"
HOST_OUT="${HOST_OUT:-$HOME/glm53-flash-results/bench/${TAG}}"
CONTAINER_OUT="/results/bench/${TAG}"

mkdir -p "$HOST_OUT"

run() {
  docker exec "$NAME" bash -lc "cd /sgl-workspace/sglang/python && $1"
}

serving_point() {
  local concurrency="$1"
  local prompts="$2"
  local isl="$3"
  local repeat="$4"
  local prefix="$5"
  local stem="${prefix}-c${concurrency}-r${repeat}"

  date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.started"
  run "python3 -m sglang.benchmark.serving \
    --backend sglang --dataset-name random \
    --random-input-len ${isl} --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts ${prompts} --max-concurrency ${concurrency} \
    --request-rate inf --temperature 0 --seed 42 \
    --flush-cache --warmup-requests ${concurrency} \
    --host 127.0.0.1 --port ${PORT} \
    --model '${SERVED}' --served-model-name '${SERVED}' --tokenizer '${TOKENIZER}' \
    --output-details --output-file '${CONTAINER_OUT}/${stem}.jsonl'" 2>&1 \
    | tee "${HOST_OUT}/${stem}.txt"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.finished"
}

case "$MODE" in
  sanity)
    serving_point 1 10 1024 1 sanity
    serving_point 32 320 1024 1 sanity
    ;;
  main)
    for concurrency in 1 8 16 32 64; do
      for repeat in 1 2 3; do
        serving_point \
          "$concurrency" "$((concurrency * 4))" 8192 "$repeat" perf
      done
    done
    ;;
  lat)
    for repeat in 1 2 3; do
      stem="latency-r${repeat}"
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.started"
      run "python3 -m sglang.benchmark.one_batch_server \
        --model-path '${TOKENIZER}' --base-url http://127.0.0.1:${PORT} \
        --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \
        --dataset-name random --skip-warmup \
        --result-filename '${CONTAINER_OUT}/${stem}.jsonl'" 2>&1 \
        | tee "${HOST_OUT}/${stem}.txt"
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "${HOST_OUT}/${stem}.finished"
    done
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
