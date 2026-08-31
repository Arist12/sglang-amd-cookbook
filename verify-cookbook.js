// verify-cookbook.js — dependency-free checks for the buildless cookbook site.
//
//   node verify-cookbook.js [repo_dir]
//
// test_render.js is the real gate, but it needs jsdom, which needs a reachable npm
// registry. This covers the failure modes that actually matter when hand-editing
// models.js, using nothing but the node standard library, so an offline box is not
// left with zero verification:
//
//   1. models.js parses and still exposes window.HW / window.MODELS
//   2. every shipped launch command's flags have a glossary entry in app.js
//      (the check that caught eleven flags rendering as "—" originally)
//   3. benchmark rows carry the fields the roofline gauge and tables read
//
// The flag parser below is a port of parseFlags() in app.js. If that changes, this
// has to change with it — the point is to agree with the renderer, not to be clever.

const fs = require("fs");
const path = require("path");

const dir = process.argv[2] || __dirname;
let failures = 0;
const fail = (msg) => { console.error("FAIL  " + msg); failures++; };
const pass = (msg) => console.log("ok    " + msg);

// --- 1. models.js -----------------------------------------------------------
const modelsSrc = fs.readFileSync(path.join(dir, "models.js"), "utf8");
const win = {};
try {
  new Function("window", modelsSrc)(win);
} catch (e) {
  fail("models.js does not parse: " + e.message);
  process.exit(1);
}
const MODELS = win.MODELS;
if (!Array.isArray(MODELS) || !MODELS.length) fail("window.MODELS missing or empty");
if (!win.HW || !Array.isArray(win.HW.hardware)) fail("window.HW.hardware missing");
if (!failures) pass(`models.js parses: ${MODELS.length} models, ${win.HW.hardware.length} hardware entries`);

// --- 2. flag glossary coverage ---------------------------------------------
const appSrc = fs.readFileSync(path.join(dir, "app.js"), "utf8");
const flagsBlock = appSrc.match(/var FLAGS\s*=\s*\{([\s\S]*?)\n\s*\};/);
if (!flagsBlock) {
  fail("could not locate the FLAGS glossary in app.js");
} else {
  const known = new Set(
    [...flagsBlock[1].matchAll(/"(--[a-z0-9-]+)"\s*:/g)].map((m) => m[1])
  );
  pass(`app.js glossary has ${known.size} flags`);

  // Port of parseFlags() in app.js.
  function parseFlags(cmd) {
    const clean = cmd.replace(/\\\s*\n/g, " ").replace(/#[^\n]*/g, " ");
    const toks = clean.split(/\s+/).filter(Boolean);
    const out = [];
    let i = 0;
    while (i < toks.length && toks[i].indexOf("--") !== 0) i++;
    for (; i < toks.length; i++) {
      if (toks[i].indexOf("--") === 0) {
        const flag = toks[i];
        while (i + 1 < toks.length && toks[i + 1].indexOf("-") !== 0) i++;
        out.push(flag);
      }
    }
    return out;
  }

  let cmds = 0;
  const missing = new Map();
  for (const m of MODELS) {
    for (const c of m.configs || []) {
      if (!c.launch_python) continue;
      cmds++;
      for (const f of parseFlags(c.launch_python)) {
        if (!known.has(f)) {
          const key = `${f}`;
          if (!missing.has(key)) missing.set(key, []);
          missing.get(key).push(`${m.id}/${c.gfx}:${c.strategy}`);
        }
      }
    }
  }
  if (missing.size) {
    for (const [f, where] of missing) {
      fail(`flag ${f} has no glossary entry — renders as "—" in ${where.join(", ")}`);
    }
  } else {
    pass(`all flags in ${cmds} shipped launch commands have glossary entries`);
  }
}

// --- 3. benchmark row shape ------------------------------------------------
let rows = 0;
for (const m of MODELS) {
  for (const c of m.configs || []) {
    for (const b of c.benchmarks || []) {
      rows++;
      const label = `${m.id}/${c.strategy} isl=${b.isl} conc=${b.concurrency}`;
      for (const k of ["isl", "osl", "concurrency"]) {
        if (typeof b[k] !== "number") fail(`${label}: ${k} is not a number`);
      }
      const hasPerf = ["decode_tok_s", "total_tok_s", "tpot_ms", "prefill_tok_s"]
        .some((k) => typeof b[k] === "number");
      if (!hasPerf) fail(`${label}: no performance field`);
      if (!b.source) fail(`${label}: no source attribution`);
    }
  }
}
pass(`${rows} benchmark rows have shape + source`);

// --- 4. the entry this change touched --------------------------------------
const k3 = MODELS.find((m) => m.id === "kimi-k3");
if (!k3) {
  fail("kimi-k3 entry missing");
} else {
  const byStrat = {};
  for (const c of k3.configs) byStrat[c.strategy] = c;
  for (const s of ["low-latency", "high-throughput"]) {
    if (!byStrat[s]) { fail(`kimi-k3 is missing its ${s} cell`); continue; }
    const cmd = byStrat[s].launch_python || "";
    const mf = cmd.match(/--mem-fraction-static\s+([\d.]+)/);
    const want = s === "low-latency" ? "0.92" : "0.93";
    if (!mf) fail(`kimi-k3/${s}: no --mem-fraction-static in launch command`);
    else if (mf[1] !== want) fail(`kimi-k3/${s}: mem-fraction ${mf[1]}, expected the tuned ${want}`);
    else pass(`kimi-k3/${s}: mem-fraction ${mf[1]}`);
    if (s === "low-latency") {
      if (!/--speculative-dspark-block-size\s+3/.test(cmd)) {
        fail("kimi-k3/low-latency: missing --speculative-dspark-block-size 3");
      } else pass("kimi-k3/low-latency: dspark block size 3");
    }
    if (/--mamba-ssm-dtype/.test(cmd)) {
      fail(`kimi-k3/${s}: ships --mamba-ssm-dtype, which the accuracy gate says to keep opt-in`);
    }
  }
  const n = k3.configs.map((c) => (c.benchmarks || []).length).join("/");
  pass(`kimi-k3: ${k3.configs.length} cells, benchmark rows ${n}`);
}

const glm53 = MODELS.find((m) => m.id === "glm-5.3-flash");
if (!glm53) {
  fail("glm-5.3-flash entry missing");
} else {
  const cell = (glm53.configs || []).find(
    (c) => c.gfx === "gfx950" && c.strategy === "high-throughput");
  if (!cell || !cell.verified) {
    fail("glm-5.3-flash verified gfx950 high-throughput cell missing");
  } else {
    const cmd = cell.launch_python || "";
    for (const required of [
      "--revision 04c4e9e95c5da8862dced7e5056455116f83a7e0",
      "--kv-cache-dtype fp8_e4m3",
      "--moe-runner-backend aiter",
      "--cuda-graph-backend-decode full",
      "--cuda-graph-backend-prefill disabled",
      "--cuda-graph-bs-decode 1 32",
      "--disable-radix-cache",
    ]) {
      if (!cmd.includes(required)) fail(`glm-5.3-flash command missing ${required}`);
    }
    const perf = (cell.benchmarks || []).filter(
      (b) => b.isl === 8192 && b.osl === 1024 && b.total_tok_s != null);
    const byConcurrency = Object.fromEntries(perf.map((b) => [b.concurrency, b]));
    for (const concurrency of [1, 8, 16, 32, 64]) {
      if (!byConcurrency[concurrency]) {
        fail(`glm-5.3-flash missing concurrency ${concurrency} result`);
      }
    }
    if (byConcurrency[32] && byConcurrency[64] &&
        byConcurrency[64].total_tok_s >= byConcurrency[32].total_tok_s) {
      fail("glm-5.3-flash c64 eager-fallback result no longer matches its documented cliff");
    }
    for (const name of ["GSM8K", "AIME25"]) {
      if (!(cell.accuracy || []).some((a) => a.name === name)) {
        fail(`glm-5.3-flash missing ${name}`);
      }
    }
    if (!cell.provenance || !/9d208769/.test(cell.provenance.sglang || "") ||
        !/2026-08-27T23:31:10Z/.test(cell.provenance.date || "")) {
      fail("glm-5.3-flash frozen SHA/timestamp provenance missing");
    } else {
      pass(`glm-5.3-flash: verified cell, ${cell.benchmarks.length} benchmark rows`);
    }
  }
}

const glm53Full = MODELS.find((m) => m.id === "glm-5.3");
if (!glm53Full) {
  fail("full glm-5.3 entry missing");
} else {
  const cell = (glm53Full.configs || []).find(
    (c) => c.gfx === "gfx950" && c.strategy === "high-throughput");
  if (!cell) {
    fail("full glm-5.3 visible gfx950 cell missing");
  } else {
    if (glm53Full.status !== "not_benchmarked" || cell.verified) {
      fail("full glm-5.3 must distinguish serving verification from benchmark verification");
    }
    if ((cell.benchmarks || []).length) {
      fail("full glm-5.3 publishes non-standard single-run throughput as benchmark rows");
    }
    for (const required of [
      "--revision 935644c05e76fc198714f4cca449fd8b970ff6d7",
      "--kv-cache-dtype fp8_e4m3",
      "--dsa-prefill-backend tilelang",
      "--dsa-decode-backend tilelang",
    ]) {
      if (!(cell.launch_python || "").includes(required)) {
        fail(`full glm-5.3 command missing ${required}`);
      }
    }
    if (!(cell.gotchas || []).some((item) => /#36960/.test(item))) {
      fail("full glm-5.3 does not surface the required long-prefill fix");
    } else {
      pass("full glm-5.3: visible serving-verified cell, no benchmark rows");
    }
  }
}

if (MODELS.some((m) => m.id === "deepseek-v4-flash-fp8")) {
  fail("superseded DeepSeek-V4-Flash preview entry is still published");
}

for (const spec of [
  {
    id: "deepseek-v4-flash-0731",
    path: "deepseek-ai/DeepSeek-V4-Flash-0731",
    localPath: "/data/DeepSeek-V4-Flash-0731",
    accuracy: "91.964%",
    totals: { 1: 1020.12, 8: 6498.67, 32: 18800.22 },
    revision: "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    indexSha: "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
  },
  {
    id: "deepseek-v4-pro-0813",
    path: "deepseek-ai/DeepSeek-V4-Pro-0813",
    localPath: "/data/DeepSeek-V4-Pro-0813",
    accuracy: "94.617%",
    totals: { 1: 692.37, 8: 4057.7, 32: 10100.64 },
    revision: "72e1d3230f6c080a530b0a1d46f8eb4602340597",
    indexSha: "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a",
  },
]) {
  const model = MODELS.find((m) => m.id === spec.id);
  if (!model) {
    fail(`${spec.id} entry missing`);
    continue;
  }
  if (model.hf_path !== spec.path || model.status !== "verified") {
    fail(`${spec.id}: official path or verified status is wrong`);
  }
  const cell = (model.configs || []).find(
    (c) => c.gfx === "gfx950" && c.strategy === "low-latency");
  if (!cell || !cell.verified) {
    fail(`${spec.id}: verified gfx950 low-latency cell missing`);
    continue;
  }
  const cmd = cell.launch_python || "";
  for (const required of [
    `--model-path ${spec.localPath}`,
    "--tp 8",
    "--attention-backend dsv4",
    "--disable-radix-cache",
    "--page-size 256",
    "--mem-fraction-static 0.90",
    "--swa-full-tokens-ratio 0.1",
    "--kv-cache-dtype fp8_e4m3",
    "--chunked-prefill-size 8192",
    "--max-running-requests 256",
    "--disable-shared-experts-fusion",
    "--tool-call-parser deepseekv4",
    "--reasoning-parser deepseek-v4",
  ]) {
    if (!cmd.includes(required)) fail(`${spec.id}: launch command missing ${required}`);
  }
  if (cmd.includes("--speculative-algorithm")) {
    fail(`${spec.id}: unverified DSpark is present in the published launch command`);
  }
  const env = Object.fromEntries((cell.env || []).map((item) => [item.key, item.value]));
  if (env.SGLANG_HACK_FLASHMLA_BACKEND !== "unified_kv_triton" ||
      env.SGLANG_DSV4_FP4_EXPERTS !== "true") {
    fail(`${spec.id}: verified attention/MoE environment is missing`);
  }
  const rowsByConcurrency = Object.fromEntries(
    (cell.benchmarks || []).map((row) => [row.concurrency, row]));
  for (const concurrency of [1, 8, 32]) {
    const row = rowsByConcurrency[concurrency];
    if (!row || row.isl !== 8192 || row.osl !== 1024 ||
        row.total_tok_s !== spec.totals[concurrency]) {
      fail(`${spec.id}: bad or missing concurrency ${concurrency} benchmark row`);
    }
  }
  if ((cell.benchmarks || []).length !== 3) {
    fail(`${spec.id}: expected exactly three published benchmark rows`);
  }
  if (!(cell.accuracy || []).some(
      (item) => item.name === "GSM8K" && item.value === spec.accuracy && /invalid=0/.test(item.note))) {
    fail(`${spec.id}: three-round GSM8K evidence missing`);
  }
  const provenance = cell.provenance || {};
  if (!(provenance.weights || "").includes(spec.revision) ||
      !(provenance.weights || "").includes(spec.indexSha) ||
      !/71de97b264/.test(provenance.sglang || "") ||
      !/d9e5ef7ce/.test(provenance.aiter || "")) {
    fail(`${spec.id}: pinned checkpoint/runtime provenance missing`);
  } else {
    pass(`${spec.id}: official target-only cell, accuracy and 3 benchmark rows`);
  }
}

console.log();
if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all offline checks passed");
