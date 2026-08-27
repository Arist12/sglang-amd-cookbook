#!/usr/bin/env bash
# Overlay SGLang PR #36507 (GLM-5.3-Flash support) onto the ROCm nightly image's
# editable sglang checkout at /sgl-workspace/sglang.
#
# The nightly image installs sglang with `pip install -e python[...]`, and the PR
# touches neither python/sglang/kernels/aot/** nor pyproject.toml, so a plain
# checkout takes effect without rebuilding the AOT kernels or reinstalling.
set -euo pipefail

NAME="${NAME:-glm53-flash}"
PR="${PR:-36507}"
EXPECT_HEAD="${EXPECT_HEAD:-fa8735a4ff2b2c047b464d2fb3286dfa0aab021f}"

docker exec "$NAME" bash -lc "
set -euo pipefail
cd /sgl-workspace/sglang
echo '--- before ---'
git log -1 --format='%H %ci %s'
git fetch --no-tags origin 'pull/${PR}/head:pr${PR}'
git checkout -q 'pr${PR}'
echo '--- after ---'
git log -1 --format='%H %ci %s'
HEAD_SHA=\$(git rev-parse HEAD)
if [ \"\$HEAD_SHA\" != '${EXPECT_HEAD}' ]; then
  echo \"WARNING: PR head is \$HEAD_SHA, expected ${EXPECT_HEAD} (PR was updated)\"
fi
echo '--- merge-base drift vs image build point ---'
git log -1 --format='%H %ci %s' \$(git merge-base HEAD origin/main) || true
echo '--- model registered? ---'
python3 -c \"
from sglang.srt.configs.glm5_next import Glm5NextConfig
print('Glm5NextConfig.model_type =', Glm5NextConfig.model_type)
import sglang.srt.models.glm5_next as m
print('glm5_next module ok:', m.__file__)
print('arch classes:', [n for n in dir(m) if n.startswith('Glm5Next') and n.endswith(('CausalLM','Generation'))])
\"
echo '--- kpool tail backend whitelist (the hard ROCm constraint) ---'
python3 -c \"
import inspect
from sglang.srt.layers.attention.dsa_backend import DSAAttnBackend as B
src = inspect.getsource(B._check_kpool_tail_backend)
print(src)
\"
"
