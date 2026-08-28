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

console.log();
if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all offline checks passed");
