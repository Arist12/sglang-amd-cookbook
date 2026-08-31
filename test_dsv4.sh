#!/usr/bin/env bash
# Launch a pinned official DeepSeek-V4 target checkpoint on one 8x MI355X node.
#
# This intentionally does not enable DSpark. The published measurements cover
# target-only serving with the unified_kv_triton attention path.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "flash" && "$1" != "pro" ) ]]; then
  echo "usage: $0 flash|pro" >&2
  exit 2
fi

variant=$1
case "$variant" in
  flash)
    model_dir=${MODEL_DIR:-/data/DeepSeek-V4-Flash-0731}
    served_name=${SERVED_NAME:-deepseek-v4-flash-0731}
    expected_index_sha=98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
    expected_shards=48
    ;;
  pro)
    model_dir=${MODEL_DIR:-/data/DeepSeek-V4-Pro-0813}
    served_name=${SERVED_NAME:-deepseek-v4-pro-0813}
    expected_index_sha=2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a
    expected_shards=66
    ;;
esac

port=${PORT:-31000}
gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
results_dir=${RESULTS_DIR:-/tmp/dsv4-${variant}-$(date -u +%Y%m%dT%H%M%SZ)}
sglang_root=${SGLANG_ROOT:-}
aiter_root=${AITER_ROOT:-}
flydsl_site=${FLYDSL_SITE:-}

command -v amd-smi >/dev/null || { echo "amd-smi is required" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
[[ -f "$model_dir/config.json" ]] || { echo "missing checkpoint: $model_dir" >&2; exit 1; }
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "missing model index: $model_dir/model.safetensors.index.json" >&2
  exit 1
}

actual_index_sha=$(sha256sum "$model_dir/model.safetensors.index.json" | awk '{print $1}')
if [[ "$actual_index_sha" != "$expected_index_sha" ]]; then
  echo "checkpoint index SHA mismatch: $actual_index_sha" >&2
  exit 1
fi

shopt -s nullglob
shards=("$model_dir"/model-*.safetensors)
if [[ ${#shards[@]} -ne $expected_shards ]]; then
  echo "checkpoint has ${#shards[@]} shards; expected $expected_shards" >&2
  exit 1
fi

if [[ -n "$sglang_root" ]]; then
  expected_sglang_sha=71de97b264b04dcd514cf904003028aefe9775c8
  actual_sglang_sha=$(git -C "$sglang_root" rev-parse HEAD)
  [[ "$actual_sglang_sha" == "$expected_sglang_sha" ]] || {
    echo "SGLang SHA mismatch: $actual_sglang_sha" >&2
    exit 1
  }
fi
if [[ -n "$aiter_root" ]]; then
  expected_aiter_sha=d9e5ef7ce08ee7045d583aed768cff41aa9210fe
  actual_aiter_sha=$(git -C "$aiter_root" rev-parse HEAD)
  [[ "$actual_aiter_sha" == "$expected_aiter_sha" ]] || {
    echo "AITER SHA mismatch: $actual_aiter_sha" >&2
    exit 1
  }
fi

python_path_parts=()
[[ -n "$flydsl_site" ]] && python_path_parts+=("$flydsl_site")
[[ -n "$aiter_root" ]] && python_path_parts+=("$aiter_root")
[[ -n "$sglang_root" ]] && python_path_parts+=("$sglang_root/python")
if [[ ${#python_path_parts[@]} -gt 0 ]]; then
  joined_python_path=$(IFS=:; echo "${python_path_parts[*]}")
  export PYTHONPATH="$joined_python_path${PYTHONPATH:+:$PYTHONPATH}"
fi

IFS=, read -r -a gpu_array <<<"$gpu_ids"
if [[ ${#gpu_array[@]} -ne 8 ]]; then
  echo "this recipe requires exactly eight HIP indices; got $gpu_ids" >&2
  exit 1
fi

lock_fds=()
for gpu_id in "${gpu_array[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu_id" >&2; exit 1; }
  exec {lock_fd}>"/tmp/gpu-${gpu_id}.lock"
  flock -n "$lock_fd" || { echo "GPU lock busy: /tmp/gpu-${gpu_id}.lock" >&2; exit 1; }
  lock_fds+=("$lock_fd")
done

mkdir -p "$results_dir"
amd-smi list --json >"$results_dir/gpu-list-before.json"
amd-smi process --general --json >"$results_dir/gpu-process-before.json"
amd-smi metric --usage --mem-usage --power --temperature --json \
  >"$results_dir/gpu-metrics-before.json"
printf '%s\n' "$actual_index_sha" >"$results_dir/model-index.sha256"
date -u +%FT%TZ >"$results_dir/start-time.txt"

server_pid=""
cleanup() {
  status=$?
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  date -u +%FT%TZ >"$results_dir/end-time.txt"
  amd-smi process --general --json >"$results_dir/gpu-process-after.json" 2>/dev/null || true
  amd-smi metric --usage --mem-usage --power --temperature --json \
    >"$results_dir/gpu-metrics-after.json" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

export HIP_VISIBLE_DEVICES="$gpu_ids"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_DSV4_FP4_EXPERTS=true

setsid python3 -m sglang.launch_server \
  --model-path "$model_dir" \
  --served-model-name "$served_name" \
  --trust-remote-code \
  --tp 8 \
  --disable-radix-cache \
  --attention-backend dsv4 \
  --page-size 256 \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 \
  --disable-shared-experts-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --max-running-requests 256 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1200 \
  --host 127.0.0.1 \
  --port "$port" \
  >"$results_dir/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$results_dir/server.pid"
echo "starting $served_name on port $port; log: $results_dir/server.log"
wait "$server_pid"
