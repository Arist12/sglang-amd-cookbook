#!/usr/bin/env bash
# Benchmark a running SGLang server inside the dev container.
#
#   bash bench.sh main   <tag>   ISL 8192 / OSL 1024, conc 1/16/64  (GLM-5.2 baseline protocol)
#   bash bench.sh cook   <tag>   ISL 1024 / OSL  256, conc 16/64/256 (cookbook GB300 protocol)
#   bash bench.sh long   <tag>   conc 1, OSL 512, ISL 8192->131072
#   bash bench.sh lat    <tag>   bench_one_batch_server, bs=1, ISL 1024/8192/16384
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: bench.sh <main|cook|long|lat> <tag>" >&2
  exit 2
fi
MODE="$1"
TAG="$2"
NAME="${NAME:-glm53-flash}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
OUT="/results/bench/${TAG}"

mkdir -p "$HOME/glm53-flash-results/bench/${TAG}"

run() { docker exec "$NAME" bash -lc "cd /sgl-workspace/sglang && $1"; }

case "$MODE" in
  main)
    for C in 1 16 64; do
      echo "======== main: conc=${C} ISL 8192 / OSL 1024 ========"
      run "python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
        --num-prompts \$(( ${C} * 2 )) --max-concurrency ${C} \
        --host 127.0.0.1 --port ${PORT} \
        --output-file ${OUT}/main-conc${C}.jsonl" 2>&1 \
        | tee "$HOME/glm53-flash-results/bench/${TAG}/main-conc${C}.txt"
    done
    ;;
  cook)
    # cookbook protocol: fixed num-prompts per concurrency, flush cache, greedy, seeded
    declare -A NP=( [16]=80 [64]=320 [256]=1280 )
    for C in 16 64 256; do
      echo "======== cook: conc=${C} ISL 1024 / OSL 256 n=${NP[$C]} ========"
      run "python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input-len 1024 --random-output-len 256 --random-range-ratio 1.0 \
        --num-prompts ${NP[$C]} --max-concurrency ${C} \
        --request-rate inf --temperature 0 --seed 42 --flush-cache \
        --host 127.0.0.1 --port ${PORT} \
        --output-file ${OUT}/cook-conc${C}.jsonl" 2>&1 \
        | tee "$HOME/glm53-flash-results/bench/${TAG}/cook-conc${C}.txt"
    done
    ;;
  long)
    for ISL in 8192 32768 131072; do
      echo "======== long: ISL=${ISL} OSL 512 conc 1 ========"
      run "python3 -m sglang.bench_serving --backend sglang --dataset-name random \
        --random-input-len ${ISL} --random-output-len 512 --random-range-ratio 1.0 \
        --num-prompts 2 --max-concurrency 1 \
        --host 127.0.0.1 --port ${PORT} \
        --output-file ${OUT}/long-isl${ISL}.jsonl" 2>&1 \
        | tee "$HOME/glm53-flash-results/bench/${TAG}/long-isl${ISL}.txt"
    done
    ;;
  lat)
    echo "======== latency: bs=1 ISL 1024/8192/16384 OSL 1024 ========"
    run "python3 -m sglang.bench_one_batch_server \
      --model-path '${MODEL}' --base-url http://127.0.0.1:${PORT} \
      --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \
      --dataset-name random --skip-warmup \
      --result-filename ${OUT}/latency.jsonl" 2>&1 \
      | tee "$HOME/glm53-flash-results/bench/${TAG}/latency.txt"
    ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
