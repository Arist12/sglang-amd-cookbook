#!/usr/bin/env bash
# Start the long-lived GLM-5.3-Flash dev container on 8x MI355X (gfx950).
# Kept as `sleep infinity` so the PR checkout and repeated server launches happen
# via `docker exec` without re-creating the container during bring-up debugging.
set -euo pipefail

TAG="${TAG:-rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260826}"
NAME="${NAME:-glm53-flash}"
HF_HOST="${HF_HOST:-/data/jhinpan-cache}"
GLM52_HOST="${GLM52_HOST:-/var/tmp/models/GLM-5.2-FP8}"
RESULTS="${RESULTS:-$HOME/glm53-flash-results}"

mkdir -p "$RESULTS/logs"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# GLM-5.2-FP8 baseline weights belong to another user; mounted read-only and never written.
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --shm-size=64g \
  -v "$HF_HOST:/hf-cache" \
  -v "$GLM52_HOST:/models/GLM-5.2-FP8:ro" \
  -v "$RESULTS:/results" \
  -e HF_HOME=/hf-cache \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  "$TAG" sleep infinity

echo "container: $NAME  image: $TAG"
docker exec "$NAME" bash -lc '
  echo "--- stack ---"
  python3 -c "import torch,triton; print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"triton\",triton.__version__)"
  python3 -c "import torch; print(\"gpus\",torch.cuda.device_count(),torch.cuda.get_device_name(0))"
  echo "--- sglang install ---"
  python3 -c "import sglang,os; print(\"sglang\",sglang.__version__,os.path.dirname(sglang.__file__))"
  echo "--- aiter / tilelang ---"
  python3 -c "import aiter; print(\"aiter ok\")" 2>&1 | tail -1
  python3 -c "import tilelang; print(\"tilelang\",tilelang.__version__)" 2>&1 | tail -1
'
