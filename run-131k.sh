#!/usr/bin/env bash
# Close the one row the v0.5.16 pass left on the old draft checkpoint: 131k
# input, single stream, both configs. OSL 1024 to match the rest of the context
# axis (16k/32k/64k) rather than the Day-0 rows' OSL 512, so the five points
# form one comparable series.
set -Eeuo pipefail

CONTAINER="${CONTAINER:-K3-v0516}"
OTHER_CONTAINER="${OTHER_CONTAINER:-K3-test}"
PORT="${PORT:-30000}"
WS_HOST=/data/jhinpan-tools/K3-v0516-workspace
RUN_DIR="${RUN_DIR:-${WS_HOST}/ctx131k}"
CTR_RUN_DIR="/sgl-workspace/workspace/${RUN_DIR#${WS_HOST}/}"
NEW_REV=56ce616ad7486f0e96cbb51ef23ed5a1bce1d92d
DRAFT="/sgl-workspace/models/hub/models--RadixArk--Kimi-K3-DSpark/snapshots/${NEW_REV}"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*" | tee -a "$LOG"; }

max_vram() { rocm-smi --showmemuse 2>/dev/null | sed -n 's/.*(VRAM%): \([0-9]\+\).*/\1/p' | sort -rn | head -1; }
other_busy() { docker exec "$OTHER_CONTAINER" pgrep -f 'ts-serve|smg_grpc_servicer|tokenspeed bench|tokenspeed::' >/dev/null 2>&1; }

wait_idle() {
  local w=0 s=0 v
  while (( w < 86400 )); do
    v="$(max_vram)"; v="${v:-100}"
    if (( v <= 10 )) && ! other_busy; then s=$(( s + 1 )); (( s >= 4 )) && return 0; else s=0; fi
    sleep 30; w=$(( w + 30 ))
  done
  return 1
}
stop_server() {
  docker exec "$CONTAINER" bash -c \
    "ps -eo pid,args | grep -E 'sglang serv[e]' | awk '{print \$1}' | xargs -r kill -TERM" >/dev/null 2>&1 || true
  local w=0
  while (( w < 180 )); do
    curl -sf --max-time 5 "http://127.0.0.1:${PORT}/get_model_info" >/dev/null 2>&1 || break
    sleep 5; w=$(( w + 5 ))
  done
  sleep 10; log "server stopped (max VRAM $(max_vram)%)"
}
start_server() {
  local mode="$1" extra=""
  [[ "$mode" == "dspark" ]] && extra="DRAFT_MODEL=${DRAFT}"
  log "starting ${mode}"
  docker exec -d "$CONTAINER" bash -lc \
    "MODE=${mode} RADIX=off PORT=${PORT} LOG_DIR=${CTR_RUN_DIR} ${extra} /sgl-workspace/workspace/serve-k3.sh"
  local w=0
  while (( w < 2400 )); do
    curl -sf --max-time 5 "http://127.0.0.1:${PORT}/get_model_info" >/dev/null 2>&1 \
      && { log "${mode} up after ${w}s"; return 0; }
    sleep 15; w=$(( w + 15 ))
  done
  log "ERROR: ${mode} never came up"; return 1
}

log "131k context row into $RUN_DIR"
wait_idle
for m in nospec dspark; do
  start_server "$m" || continue
  if [[ "$m" == "dspark" ]]; then
    curl -s "http://127.0.0.1:${PORT}/server_info" | python3 -c \
      'import json,sys; print("  draft:", json.load(sys.stdin).get("speculative_draft_model_path"))' \
      2>/dev/null | tee -a "$LOG" || true
  fi
  set +e
  docker exec "$CONTAINER" bash -lc "
    export PYTHONPATH=/sgl-workspace/sglang/python HF_HUB_OFFLINE=1
    curl -s -X POST http://127.0.0.1:${PORT}/set_internal_state -H 'Content-Type: application/json' -d '{\"server_args\": {}}' >/dev/null 2>&1
    python -m sglang.benchmark.serving --backend sglang \
      --base-url http://127.0.0.1:${PORT} --model moonshotai/Kimi-K3 \
      --dataset-name random --random-input-len 131072 --random-output-len 1024 \
      --random-range-ratio 1 --num-prompts 4 --max-concurrency 1 \
      --warmup-requests 1 --disable-tqdm \
      --output-file ${CTR_RUN_DIR}/${m}-isl131072-osl1024-c1.jsonl 2>&1 | grep -vE '^\[aiter\]'
  " 2>&1 | tee -a "$RUN_DIR/${m}-isl131072-osl1024-c1.log"
  set -e
  stop_server
done
log "131K ROW COMPLETE"
