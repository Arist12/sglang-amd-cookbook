#!/usr/bin/env bash
# Smoke test for the grid harness: one boot, one cheap bench point, one CSV row.
# Validates launch/health/parse/scrape/teardown before the overnight run starts.
set -uo pipefail
W=/sgl-workspace/workspace
export GL_BUDGET_H=2
source "${W}/gridlib.sh"
gl_init "${W}/grid_results/smoke"
PHASE=smoke
gl_reset
export MEM_FRAC=0.85
GL_LABEL=smoke-nospec; GL_LANE=smoke; GL_PHASE=smoke
if gl_launch nospec; then
  gl_bench w1 1
else
  gl_row "${GL_BOOT_STATUS}"
fi
gl_teardown
echo "--- CSV ---"
cat "${GL_CSV}"
