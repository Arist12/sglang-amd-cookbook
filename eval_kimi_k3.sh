#!/usr/bin/env bash
# Wait for the running Kimi-K3 server, then run GSM8K + AIME25 against it.
# This is the script that produced the accuracy table in kimi_k3_playbook.md.
#
#   bash test_kimi_k3.sh &                 # or MODE=dspark
#   bash eval_kimi_k3.sh nospec            # or dspark; just names the output files
#
# GSM8K goes through the in-tree harness; AIME25 needs sgl-eval, whose
# answer extraction is the one that holds up (see the GLM-5.2 playbook).
# Serving DSpark here requires dspark_rocm_renorm.patch: AIME25 samples with
# top_p, which takes an unpatched ROCm scheduler down on the first decode batch.
set -uo pipefail

TAG="${1:-run}"
PORT="${PORT:-30000}"
W=/sgl-workspace/workspace

for _ in $(seq 1 180); do
  curl -sf -o /dev/null "http://127.0.0.1:${PORT}/health" && break
  sleep 10
done

cd /sgl-workspace/sglang
python3 -m sglang.test.run_eval \
  --port "${PORT}" --eval-name gsm8k \
  --num-examples 1319 --num-threads 32 \
  --max-tokens 8192 --temperature 0 \
  > "${W}/gsm8k-${TAG}.txt" 2>&1
echo "GSM8K DONE" >> "${W}/gsm8k-${TAG}.txt"

sgl-eval run aime25 \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model moonshotai/Kimi-K3 --api-key EMPTY \
  --n-repeats 8 --num-threads 48 \
  --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking \
  > "${W}/aime25-${TAG}.txt" 2>&1
echo "DONE" >> "${W}/aime25-${TAG}.txt"
