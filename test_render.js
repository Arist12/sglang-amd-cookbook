// Render the static site in jsdom and assert the parts that matter.
//
//   npm i jsdom && node test_render.js
//
// The site is buildless, so nothing else checks that models.js and app.js still
// agree. Two real bugs this caught on the way in: parseFlags anchored on
// "launch_server" and silently emptied the argument table for any recipe using
// the newer `sglang serve` CLI, and eleven flags in shipped launch commands had
// no glossary entry and rendered as "—".
const fs = require("fs");
const path = require("path");

function requireJsdom() {
  try {
    return require("jsdom");
  } catch (e) {
    console.error("jsdom is required: npm i jsdom");
    process.exit(2);
  }
}
const { JSDOM } = requireJsdom();

const dir = __dirname;
const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
let failures = 0;

function boot(hash) {
  const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "https://jhinpan.github.io/sglang-amd-cookbook/" + (hash || ""),
    pretendToBeVisual: true,
  });
  const { window } = dom;
  window.IntersectionObserver = class {
    observe(el) { el.style.width = el.dataset.w + "%"; }
    unobserve() {}
  };
  window.HTMLElement.prototype.scrollIntoView = function () {};  // not in jsdom
  const errors = [];
  window.addEventListener("error", (e) => errors.push(String(e.error || e.message)));
  window.eval(fs.readFileSync(path.join(dir, "models.js"), "utf8"));
  try {
    window.eval(fs.readFileSync(path.join(dir, "app.js"), "utf8"));
    if (window.document.readyState === "loading") {
      window.document.dispatchEvent(new window.Event("DOMContentLoaded", { bubbles: true }));
    }
  } catch (e) { errors.push("app.js threw: " + e.stack); }
  return { window, errors };
}

function check(name, cond, extra) {
  if (!cond) failures++;
  console.log((cond ? "  PASS  " : "  FAIL  ") + name + (extra ? "   [" + extra + "]" : ""));
}

// ---------------------------------------------------------------- 1. base render
console.log("\n=== base render ===");
{
  const { window, errors } = boot();
  const d = window.document;
  const qa = (s) => Array.from(d.querySelectorAll(s));
  check("no runtime errors", errors.length === 0, errors.join(" | ").slice(0, 300));
  check("hardware strip", qa("#hwstrip .hwcard").length === 2);
  check("model tabs", qa(".mtab").length === window.MODELS.length);
  check("roadmap cards", qa(".rm-card").length === window.MODELS.length);
  const verified = window.MODELS.reduce((n, m) => n + m.configs.filter((c) => c.verified).length, 0);
  check("compare row per verified cell", qa("#compare-body tbody tr").length === verified,
    qa("#compare-body tbody tr").length + " rows / " + verified + " verified cells");
  check("compare surfaces accuracy", /97\.64%/.test(d.querySelector("#compare-body").textContent));
  check("compare surfaces GLM-5.3 AIME25", /93\.75%/.test(d.querySelector("#compare-body").textContent));
  check("method section has 4 columns", qa("#method .refcol").length === 4);
  check("masthead rev is current", /2026\.08/.test(d.querySelector(".partline").textContent));
}

// ---------------------------------------------------------------- 2. deep links
console.log("\n=== deep links ===");
for (const [hash, wantModel, wantStrat] of [
  ["#m=glm-5.2-fp8", "glm-5.2-fp8", "low-latency"],                       // legacy form
  ["#m=glm-5.3-flash&c=gfx950:high-throughput", "glm-5.3-flash", "high-throughput"],
  ["#m=glm-5.3&c=gfx950:high-throughput", "glm-5.3", "high-throughput"],
  ["#m=deepseek-v4-flash-0731&c=gfx950:low-latency", "deepseek-v4-flash-0731", "low-latency"],
  ["#m=deepseek-v4-pro-0813&c=gfx950:low-latency", "deepseek-v4-pro-0813", "low-latency"],
  ["#m=kimi-k3", "kimi-k3", "low-latency"],                               // model only
  ["#m=kimi-k3&c=gfx950:high-throughput", "kimi-k3", "high-throughput"],  // cell
  ["#m=nope", null, null],                                                // garbage
]) {
  const { window, errors } = boot(hash);
  const d = window.document;
  const sel = Array.from(d.querySelectorAll(".mtab"))
    .filter((t) => t.getAttribute("aria-selected") === "true")
    .map((t) => t.dataset.model)[0];
  const chips = d.querySelector(".rc-chips") ? d.querySelector(".rc-chips").textContent : "";
  const ok = errors.length === 0 &&
    (wantModel ? sel === wantModel : !!sel) &&
    (wantStrat ? chips.indexOf(wantStrat) > -1 : true);
  check(hash + " -> " + sel, ok, "hash now " + window.location.hash);
}

// ---------------------------------------------------------------- 2b. documented-but-unbenchmarked recipe
console.log("\n=== documented unbenchmarked recipe ===");
{
  const { window, errors } = boot("#m=glm-5.3&c=gfx950:high-throughput");
  const text = window.document.querySelector("#recipe").textContent;
  check("full GLM-5.3 renders without runtime errors", errors.length === 0,
    errors.join(" | ").slice(0, 300));
  check("full GLM-5.3 keeps the recipe visible", text.includes("zai-org/GLM-5.3"));
  check("full GLM-5.3 is labeled not benchmarked", /not benchmarked/i.test(text));
  check("full GLM-5.3 does not claim missing serving validation",
    !text.includes("we have not measured it end-to-end yet"));
  check("full GLM-5.3 does not render benchmark rows",
    window.document.querySelectorAll("#recipe .dtable tbody tr").length === 0);
}

// ---------------------------------------------------------------- 2c. official DeepSeek-V4 recipes
console.log("\n=== official DeepSeek-V4 recipes ===");
for (const [id, path, accuracy] of [
  ["deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731", "91.964%"],
  ["deepseek-v4-pro-0813", "deepseek-ai/DeepSeek-V4-Pro-0813", "94.617%"],
]) {
  const { window, errors } = boot("#m=" + id + "&c=gfx950:low-latency");
  const recipe = window.document.querySelector("#recipe");
  const text = recipe.textContent;
  check(id + " renders without runtime errors", errors.length === 0,
    errors.join(" | ").slice(0, 300));
  check(id + " shows the official checkpoint", text.includes(path));
  check(id + " shows three benchmark rows",
    recipe.querySelectorAll(".dtable tbody tr").length === 3);
  check(id + " shows measured accuracy", text.includes(accuracy));
  check(id + " exposes the verified backend", text.includes("unified_kv_triton"));
}

// ---------------------------------------------------------------- 3. every verified cell
console.log("\n=== every verified cell: recipe + argument reference ===");
{
  const { window } = boot();
  const d = window.document;
  for (const m of window.MODELS) {
    for (const c of m.configs) {
      if (!c.verified) continue;
      const tab = d.querySelector('.mtab[data-model="' + m.id + '"]');
      tab.dispatchEvent(new window.Event("click", { bubbles: true }));
      const cell = d.querySelector(
        '.mcell.has[data-gfx="' + c.gfx + '"][data-strat="' + c.strategy + '"]');
      if (!cell) { check(m.id + " / " + c.strategy + " cell present", false); continue; }
      cell.dispatchEvent(new window.Event("click", { bubbles: true }));
      const rows = Array.from(d.querySelectorAll(".argtable tbody tr"));
      const undocumented = rows
        .filter((r) => r.querySelector(".why").textContent.trim() === "—")
        .map((r) => r.querySelector(".flag").textContent);
      const label = (m.id + " / " + c.strategy).padEnd(34);
      check(label + rows.length + " flags, all documented",
        rows.length > 0 && undocumented.length === 0,
        undocumented.length ? "undocumented: " + undocumented.join(" ") : "");
    }
  }
}

// ---------------------------------------------------------------- 4. compare row click
console.log("\n=== compare row navigates ===");
{
  const { window } = boot();
  const d = window.document;
  const row = d.querySelector('#compare-body tbody tr[data-model="kimi-k3"][data-strat="high-throughput"]');
  check("kimi-k3 high-throughput row exists", !!row);
  if (row) {
    row.dispatchEvent(new window.Event("click", { bubbles: true }));
    check("click selects that cell",
      window.location.hash === "#m=kimi-k3&c=gfx950:high-throughput",
      window.location.hash);
    check("recipe shows the non-spec command",
      d.querySelector("#recipe").textContent.indexOf("speculative-algorithm") === -1);
  }
}

console.log("\n" + (failures ? failures + " FAILURE(S)" : "all checks passed"));
process.exitCode = failures ? 1 : 0;
