#!/usr/bin/env bash
# test-grid-logic.sh -- exercise the whole phase driver with the GPU calls mocked.
#
# Replaces gl_launch/gl_bench/gl_teardown with instant fakes that synthesise
# plausible numbers, so the coordinate-descent flow, incumbent save/load, resume
# logic, best_of queries and budget guard can be validated in seconds instead of
# being debugged overnight. Run from /sgl-workspace/workspace.
set -uo pipefail
W=/sgl-workspace/workspace
TESTDIR="${W}/grid_results/_logictest"
rm -rf "${TESTDIR}"

export GL_BUDGET_MIN="${GL_BUDGET_MIN:-840}"
export MOCK=1

# The driver sources gridlib.sh itself; we re-override afterwards via a shim file.
cat > /tmp/mock-overrides.sh <<'MOCK'
# --- mocked GPU layer -------------------------------------------------------
MOCK_BOOT_FAIL="${MOCK_BOOT_FAIL:-}"
gl_teardown() { return 0; }
gl_alive() { return 0; }
gl_vram_max_mb() { echo 300; }

gl_launch() {
  GL_MODE="$1"
  GL_SRV_LOG="${GL_RUN_DIR}/logs/${GL_LABEL}.server.log"
  mkdir -p "${GL_RUN_DIR}/logs"; : > "${GL_SRV_LOG}"
  GL_BOOT_S=200

  # available_gpu_mem falls ~288 GB per unit mem-fraction (the card is 288 GiB);
  # anchor 0.85 -> 35.1 GB reproduces the real measurement.
  local mf="${MEM_FRAC:-0.85}"
  avail_gpu_mem_gb="$(awk -v m="${mf}" 'BEGIN{printf "%.2f", 35.10-(m-0.85)*288}')"
  max_total_num_tokens="$(awk -v m="${mf}" 'BEGIN{printf "%d", 833536+(m-0.85)*288*21500}')"
  max_running_requests=368; mamba_cap=368; mamba_slots=1
  if [[ "${GL_MODE}" == "dspark" ]]; then max_running_requests=48; mamba_cap=48; mamba_slots=1; fi
  if [[ "${EXTRA_ARGS:-}" == *max-running-requests* ]]; then
    max_running_requests="$(echo "${EXTRA_ARGS}" | rg -No 'max-running-requests (\d+)' -r '$1')"
  fi

  # Simulate the real failure boundary: past ~0.95 activations have no room.
  if awk -v a="${avail_gpu_mem_gb}" 'BEGIN{exit !(a<2.0)}'; then
    GL_BOOT_STATUS=OOM; GL_FATAL="mock: activations have no room at mem-fraction ${mf}"
    return 1
  fi
  if [[ -n "${MOCK_BOOT_FAIL}" && "${GL_LABEL}" == *"${MOCK_BOOT_FAIL}"* ]]; then
    GL_BOOT_STATUS=ARG_INVALID; GL_FATAL="mock: rejected arg"
    return 1
  fi
  GL_BOOT_STATUS=OK
  echo "  [mock] launched ${GL_LABEL} avail=${avail_gpu_mem_gb} mtnt=${max_total_num_tokens}"
  return 0
}

gl_bench() {
  local wl="$1" conc="$2"
  gl_workload_args "$wl" "$conc"
  # Throughput rises with KV capacity and saturates in concurrency; DSpark peaks
  # at a small batch. Enough structure to prove the descent picks a real argmax.
  local tps
  case "${wl}" in
    w2) tps="$(awk -v c="${conc}" -v k="${max_total_num_tokens:-833536}" -v d="${GL_MODE}" -v e="${EXTRA_ARGS:-}" -v cp="${CHUNKED_PREFILL:-16384}" '
          BEGIN{
            cap = k/9216;
            eff = (c<cap ? c : cap);
            if (d=="dspark") { base = 4000*eff/(eff+18); }
            else { base = 9000*eff/(eff+55); }
            if (cp==32768) base*=1.06; else if (cp==65536) base*=1.02; else if (cp==8192) base*=0.94;
            if (e ~ /extra_buffer_lazy/) base*=0.97;
            printf "%.2f", base;
          }')" ;;
    w1) tps="$(awk -v c="${conc}" 'BEGIN{printf "%.2f", 100*c/(c*0.1+1)}')" ;;
    w3) tps="$(awk -v e="${EXTRA_ARGS:-}" -v r="${RADIX:-0}" 'BEGIN{printf "%.2f", (r==1?7900:5000)}')" ;;
    stress) tps=3000 ;;
  esac
  local tpot; tpot="$(awk -v t="${tps}" -v c="${conc}" 'BEGIN{printf "%.2f", 1000*c/(t>0?t:1)}')"
  printf '%s\n' "$(date -Is),${GL_PHASE},${GL_LANE},${GL_LABEL},${GL_MODE},${MEM_FRAC:-0.85},${RADIX:-0},${CUDA_GRAPH_MAX_BS:-256},${CHUNKED_PREFILL:-default},\"${EXTRA_ARGS:-}\",${wl},${GL_ISL},${GL_OSL},${conc},${GL_NP},OK,\"\",${tps},${tps},0.2,${conc},3.0,500,400,900,${tpot},${tpot},50,90000,88000,1000,990,1.5,NA,${max_total_num_tokens:-NA},${max_running_requests:-NA},${avail_gpu_mem_gb:-NA},${mamba_cap:-NA},${mamba_slots:-NA},${conc},${conc},0.95,0.30,3.0,4,0,True,${GL_BOOT_S},120,mock" >> "${GL_CSV}"
  echo "  [mock] ${GL_LABEL} ${wl} c${conc} -> ${tps} tok/s"
  return 0
}
MOCK

# Inject the overrides right after the driver sources gridlib.sh.
sed 's#^source "${W}/gridlib.sh"$#source "${W}/gridlib.sh"\nsource /tmp/mock-overrides.sh#' \
  "${W}/grid-k3.sh" > /tmp/grid-k3-mocked.sh

echo "################ full run ################"
bash /tmp/grid-k3-mocked.sh all "${TESTDIR}" 2>&1 | rg -v '^\s*$'

echo
echo "################ chosen mem-fraction ################"
for f in "${TESTDIR}"/memfrac-*.txt; do echo "$(basename "$f"): $(cat "$f")"; done

echo
echo "################ incumbents ################"
for f in "${TESTDIR}"/incumbent-*.env; do echo "--- $(basename "$f")"; cat "$f"; done

echo
echo "################ row count / statuses ################"
awk -F',' 'NR>1{print $16}' "${TESTDIR}/results.csv" | sort | uniq -c
echo "rows: $(( $(wc -l < "${TESTDIR}/results.csv") - 1 ))"

echo
echo "################ resume check (re-run must skip everything) ################"
bash /tmp/grid-k3-mocked.sh all "${TESTDIR}" 2>&1 | rg -c "SKIP" | xargs -I{} echo "SKIP lines on re-run: {}"
echo "rows after re-run: $(( $(wc -l < "${TESTDIR}/results.csv") - 1 ))"

echo
echo "################ summary.md head ################"
head -40 "${TESTDIR}/summary.md"
