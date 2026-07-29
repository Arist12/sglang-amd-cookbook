#!/usr/bin/env python3
"""Parsing/analysis helpers for the Kimi-K3 launch-parameter search.

Subcommands:
  parse-bench   sglang.benchmark.serving --output-file JSONL -> flat k=v pairs
  capacity      server log -> post-boot capacity line + mamba cap diagnosis
  scrape        server log line range -> scheduler telemetry aggregates
  summarize     results.csv -> summary.md ranking, per lane
  gibberish     --output-details JSONL -> repetition / degeneration report

Everything prints `key=value` on stdout (shell-eval friendly) except
summarize/gibberish, which write markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import sys
from collections import Counter

NA = "NA"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def emit(pairs: dict) -> None:
    """Print shell-eval-safe `key=value` lines (values may contain spaces)."""
    for k, v in pairs.items():
        if v is None:
            v = NA
        if isinstance(v, float):
            v = f"{v:.4f}".rstrip("0").rstrip(".")
        print(f"{k}={shlex.quote(str(v))}")


def rnd(v, n=2):
    return None if v is None else round(float(v), n)


def read_last_jsonl(path: str):
    if not path or not os.path.exists(path):
        return None
    rec = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
    return rec


# --------------------------------------------------------------------------- #
# parse-bench
# --------------------------------------------------------------------------- #
# Cache hit rate is printed but not stored in the JSONL record, so it is the one
# field that still has to come from stdout.
CACHE_HIT_RE = re.compile(r"^Cache hit rate:\s+([\d.]+)%", re.M)


def cmd_parse_bench(args) -> int:
    rec = read_last_jsonl(args.jsonl)
    if rec is None:
        emit({"status": "PARSE_FAILED"})
        return 1

    gen = rec.get("total_output_tokens")
    retok = rec.get("total_output_tokens_retokenized")
    div = None
    if gen and retok:
        div = abs(gen - retok) / gen * 100.0

    cache = None
    if args.text and os.path.exists(args.text):
        text = open(args.text, encoding="utf-8", errors="replace").read()
        hits = CACHE_HIT_RE.findall(text)
        if hits:
            cache = float(hits[-1])

    out = {
        "status": "OK",
        "conc": rec.get("max_concurrency"),
        "completed": rec.get("completed"),
        "duration_s": rnd(rec.get("duration")),
        "out_tps": rnd(rec.get("output_throughput")),
        "total_tps": rnd(rec.get("total_throughput")),
        "req_tps": rnd(rec.get("request_throughput"), 4),
        "conc_ach": rnd(rec.get("concurrency")),
        "accept_len": rnd(rec.get("accept_length"), 3),
        "mean_ttft_ms": rnd(rec.get("mean_ttft_ms")),
        "median_ttft_ms": rnd(rec.get("median_ttft_ms")),
        "p99_ttft_ms": rnd(rec.get("p99_ttft_ms")),
        "mean_tpot_ms": rnd(rec.get("mean_tpot_ms")),
        "median_tpot_ms": rnd(rec.get("median_tpot_ms")),
        "mean_itl_ms": rnd(rec.get("mean_itl_ms")),
        "mean_e2e_ms": rnd(rec.get("mean_e2e_latency_ms")),
        "median_e2e_ms": rnd(rec.get("median_e2e_latency_ms")),
        "gen_tok": gen,
        "retok_tok": retok,
        "retok_div_pct": rnd(div, 3),
        "cache_hit_pct": cache,
    }
    emit(out)
    return 0


# --------------------------------------------------------------------------- #
# capacity
# --------------------------------------------------------------------------- #
CAP_RE = re.compile(
    r"max_total_num_tokens=(\d+).*?chunked_prefill_size=(-?\d+).*?"
    r"max_prefill_tokens=(\d+).*?max_running_requests=(\d+).*?"
    r"context_len=(\d+).*?available_gpu_mem=([\d.]+)\s*GB"
)
MAMBA_CAP_RE = re.compile(
    r"max_running_requests is capped to (\d+) by the mamba state cache "
    r"\(max_mamba_cache_size=(\d+), (\d+) state slots per request"
)
READY_RE = re.compile(r"The server is fired up and ready to roll")
# Boot-time failure modes worth distinguishing in the CSV rather than lumping
# together as "did not come up".
FATAL_PATTERNS = [
    ("MEMFRAC_TOO_LOW", re.compile(r"Loaded weights leave no GPU memory for the KV cache")),
    (
        "OOM",
        re.compile(
            r"(?:HIP|CUDA) (?:error: )?out of memory|torch\.(?:cuda\.)?OutOfMemoryError"
            r"|hipErrorOutOfMemory|Not enough memory"
        ),
    ),
    (
        "ARG_INVALID",
        re.compile(
            r"(?:ValueError|AssertionError|argparse|error:).*?(?:must equal|must be|requires"
            r"|not supported|unrecognized|invalid choice|mutually exclusive|not yet compatible"
            r"|is not compatible|got an unexpected)"
        ),
    ),
    ("SCHED_EXCEPTION", re.compile(r"Scheduler hit an exception")),
]
# Only consulted once the process is known to be gone: a traceback logged during
# a boot that still succeeds must not abort the probe.
STRICT_PATTERNS = [
    (
        "BOOT_ERROR",
        re.compile(
            r"^(?:ValueError|AssertionError|TypeError|RuntimeError|KeyError|ImportError"
            r"|OSError|AttributeError|NotImplementedError):",
            re.M,
        ),
    ),
]
SSM_RE = re.compile(r"intermediate[_ ]ssm[^\d]*([\d.]+)\s*GB", re.I)


def cmd_capacity(args) -> int:
    # Deliberately not called "status": this output is eval'd into the same shell
    # scope as parse-bench's, and clobbering the bench verdict would silently
    # relabel a good result.
    if not os.path.exists(args.log):
        emit({"cap_status": "NO_LOG"})
        return 1
    text = open(args.log, encoding="utf-8", errors="replace").read()

    out = {}
    m = None
    for m in CAP_RE.finditer(text):
        pass
    if m:
        out.update(
            {
                "max_total_num_tokens": m.group(1),
                "chunked_prefill_size": m.group(2),
                "max_prefill_tokens": m.group(3),
                "max_running_requests": m.group(4),
                "context_len": m.group(5),
                "avail_gpu_mem_gb": m.group(6),
            }
        )

    mc = None
    for mc in MAMBA_CAP_RE.finditer(text):
        pass
    if mc:
        out["mamba_cap"] = mc.group(2)
        out["mamba_slots"] = mc.group(3)

    sm = None
    for sm in SSM_RE.finditer(text):
        pass
    if sm:
        out["intermediate_ssm_gb"] = sm.group(1)

    patterns = FATAL_PATTERNS + (STRICT_PATTERNS if args.strict else [])

    if READY_RE.search(text):
        out["boot_status"] = "READY"
    else:
        out["boot_status"] = "NOT_READY"
        for name, pat in patterns:
            if pat.search(text):
                out["boot_status"] = name
                break

    # The offending line verbatim, so a failed row still explains itself.
    for _, pat in patterns:
        fm = pat.search(text)
        if fm:
            line = text[text.rfind("\n", 0, fm.start()) + 1 : text.find("\n", fm.end())]
            out["fatal"] = re.sub(r"[,\n\r]+", " ", line.strip())[:220]
            break

    emit(out)
    return 0


# --------------------------------------------------------------------------- #
# scrape (scheduler telemetry)
# --------------------------------------------------------------------------- #
# Decode batch, #running-req: 14, #full token: 129378, full token usage: 0.16,
#   mamba num: 14, mamba usage: 0.04, [accept len: 2.83, accept rate: 0.26,]
#   cuda graph: True, gen throughput (token/s): 453.37, #queue-req: 0
DECODE_RE = re.compile(
    r"Decode batch.*?#running-req:\s*(\d+).*?full token usage:\s*([\d.]+)"
    r"(?:.*?mamba usage:\s*([\d.]+))?"
    r"(?:.*?accept len:\s*([\d.]+))?"
    r".*?cuda graph:\s*(\w+).*?gen throughput \(token/s\):\s*([\d.]+)"
    r".*?#queue-req:\s*(\d+)"
)
RETRACT_RE = re.compile(r"KV cache pool is full\. Retract requests")


def cmd_scrape(args) -> int:
    if not os.path.exists(args.log):
        emit({"scrape_status": "NO_LOG"})
        return 1

    start = max(0, args.start_line)
    end = args.end_line if args.end_line and args.end_line > 0 else None

    run, usage, mamba, acc, gen, queue = [], [], [], [], [], []
    cg = Counter()
    retract = 0
    # newline="\n" disables universal-newline splitting. The weight loader writes
    # hundreds of \r progress updates; treating those as line breaks makes Python's
    # line numbering disagree with `wc -l`, so the caller's line range would point
    # into the middle of the boot log and match nothing.
    with open(args.log, encoding="utf-8", errors="replace", newline="\n") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            if RETRACT_RE.search(line):
                retract += 1
            m = DECODE_RE.search(line)
            if not m:
                continue
            run.append(int(m.group(1)))
            usage.append(float(m.group(2)))
            if m.group(3) is not None:
                mamba.append(float(m.group(3)))
            if m.group(4) is not None:
                acc.append(float(m.group(4)))
            cg[m.group(5)] += 1
            gen.append(float(m.group(6)))
            queue.append(int(m.group(7)))

    def med(xs, n=2):
        return round(statistics.median(xs), n) if xs else None

    def mx(xs, n=2):
        return round(max(xs), n) if xs else None

    emit(
        {
            "sched_samples": len(run),
            "run_med": med(run, 1),
            "run_max": mx(run, 0),
            "tok_usage_max": mx(usage),
            "tok_usage_med": med(usage),
            "mamba_usage_max": mx(mamba),
            "mamba_usage_med": med(mamba),
            "sched_accept_med": med(acc, 3),
            "gen_tps_med": med(gen),
            "queue_max": mx(queue, 0),
            "retract_n": retract,
            "cudagraph": (cg.most_common(1)[0][0] if cg else None),
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def _load_csv(path):
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    v = (row.get(key) or "").strip()
    if not v or v == NA:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def cmd_summarize(args) -> int:
    rows = _load_csv(args.csv)
    ok = [r for r in rows if r.get("status") == "OK"]
    out = ["# Kimi-K3 launch-parameter search — results", ""]
    out.append(f"Rows: {len(rows)} total, {len(ok)} successful bench points.")
    out.append("")

    bad = [r for r in rows if r.get("status") not in ("OK", None, "")]
    if bad:
        out.append("## Failed / rejected configs")
        out.append("")
        for r in bad:
            out.append(
                f"- `{r.get('label')}` [{r.get('phase')}] status=**{r.get('status')}** "
                f"{('— ' + r['fatal']) if r.get('fatal') else ''}"
            )
        out.append("")

    def block(title, subset, key, reverse, cols):
        if not subset:
            return
        keyed = [r for r in subset if _f(r, key) is not None]
        keyed.sort(key=lambda r: _f(r, key), reverse=reverse)
        out.append(f"## {title}")
        out.append("")
        out.append("```")
        for r in keyed[: args.top]:
            parts = [f"{c}={r.get(c, NA)}" for c in cols]
            out.append(f"{r.get('label','?'):<52} " + "  ".join(parts))
        out.append("```")
        out.append("")

    tput_cols = [
        "conc",
        "total_tps",
        "out_tps",
        "median_ttft_ms",
        "mean_tpot_ms",
        "tok_usage_max",
        "mamba_usage_max",
        "queue_max",
        "max_total_num_tokens",
    ]
    lat_cols = ["conc", "mean_tpot_ms", "median_ttft_ms", "out_tps", "accept_len"]

    for lane in sorted({r.get("lane", "") for r in ok}):
        sub = [r for r in ok if r.get("lane") == lane]
        w2 = [r for r in sub if r.get("workload") == "w2"]
        w1 = [r for r in sub if r.get("workload") == "w1"]
        w3 = [r for r in sub if r.get("workload") == "w3"]
        block(f"lane `{lane}` — W2 8k/1k by total token throughput", w2, "total_tps", True, tput_cols)
        block(f"lane `{lane}` — W1 1k/1k by mean TPOT (lower is better)", w1, "mean_tpot_ms", False, lat_cols)
        block(
            f"lane `{lane}` — W3 shared prefix by total token throughput",
            w3,
            "total_tps",
            True,
            tput_cols + ["cache_hit_pct"],
        )

    flagged = [r for r in ok if (_f(r, "retok_div_pct") or 0) > args.retok_flag]
    out.append("## Gibberish screen (retokenized-token divergence)")
    out.append("")
    out.append(
        f"Threshold {args.retok_flag}% (about 1.9% is the observed healthy baseline). "
        f"{len(flagged)} of {len(ok)} points flagged."
    )
    if flagged:
        out.append("")
        out.append("```")
        for r in flagged:
            out.append(f"{r.get('label'):<52} retok_div={r.get('retok_div_pct')}%  conc={r.get('conc')}")
        out.append("```")
    out.append("")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# best (coordinate-descent incumbent lookup)
# --------------------------------------------------------------------------- #
KNOB_COLS = ["mem_frac", "radix", "cuda_graph_max_bs", "chunked_prefill", "extra_args"]


def cmd_best(args) -> int:
    rows = _load_csv(args.csv)
    sel = [r for r in rows if r.get("status") == "OK"]
    if args.lane:
        sel = [r for r in sel if r.get("lane") == args.lane]
    if args.workload:
        sel = [r for r in sel if r.get("workload") == args.workload]
    if args.phase:
        sel = [r for r in sel if r.get("phase") in args.phase.split(",")]
    if args.conc:
        sel = [r for r in sel if r.get("conc") == str(args.conc)]
    if args.label_prefix:
        sel = [r for r in sel if (r.get("label") or "").startswith(args.label_prefix)]
    sel = [r for r in sel if _f(r, args.metric) is not None]

    # A config that only "wins" by degenerating is not a win.
    if args.max_retok_div is not None:
        sel = [r for r in sel if (_f(r, "retok_div_pct") or 0) <= args.max_retok_div]

    if not sel:
        emit({"found": 0})
        return 1

    sel.sort(key=lambda r: _f(r, args.metric), reverse=(args.mode == "max"))
    best = sel[0]
    out = {"found": 1, "label": best.get("label"), "conc": best.get("conc"), args.metric: best.get(args.metric)}
    for c in KNOB_COLS:
        out[c] = best.get(c) or NA
    for c in ("max_total_num_tokens", "max_running_requests", "avail_gpu_mem_gb", "mode"):
        out[c] = best.get(c) or NA
    emit(out)
    return 0


# --------------------------------------------------------------------------- #
# report (playbook-ready markdown, generated from the CSV so numbers stay traced)
# --------------------------------------------------------------------------- #
def cmd_report(args) -> int:
    rows = [r for r in _load_csv(args.csv) if r.get("status") == "OK"]
    out = []

    def tbl(header, aligns, body):
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(aligns) + "|")
        out.extend(body)
        out.append("")

    # --- capacity per distinct launch recipe --------------------------------
    out.append("### Capacity by launch recipe")
    out.append("")
    seen, body = set(), []
    for r in rows:
        key = (r["lane"], r["mem_frac"], r["radix"], r["cuda_graph_max_bs"],
               r["chunked_prefill"], r["extra_args"])
        if key in seen or r.get("max_total_num_tokens") in ("", "NA"):
            continue
        seen.add(key)
        extra = (r["extra_args"] or "").strip() or "—"
        body.append(
            f"| `{r['label']}` | {r['lane']} | {r['mem_frac']} | {r['chunked_prefill']} | "
            f"{r['cuda_graph_max_bs']} | `{extra}` | {int(r['max_total_num_tokens']):,} | "
            f"{r['max_running_requests']} | {r['avail_gpu_mem_gb']} |"
        )
    tbl(["config", "lane", "mem-frac", "chunked-prefill", "cuda-graph-bs", "extra args",
         "max_total_num_tokens", "max_running_requests", "avail GB"],
        ["---", "---", "---:", "---:", "---:", "---", "---:", "---:", "---:"], body)

    # --- throughput per workload ------------------------------------------
    for wl, title, note in (
        ("w2", "Throughput sweep — random ISL 8192 / OSL 1024",
         "`num-prompts` = 2 x concurrency, `--random-range-ratio 1`, `--warmup-requests 4 --flush-cache`."),
        ("w1", "Latency sweep — random ISL 1024 / OSL 1024",
         "Same protocol as the published section 5 table."),
        ("w3", "Shared-prefix workload — 32 groups x 8 prompts, 4K system prompt",
         "`generated-shared-prefix` with `--cache-report`."),
    ):
        sub = [r for r in rows if r.get("workload") == wl]
        if not sub:
            continue
        out.append(f"### {title}")
        out.append("")
        out.append(note)
        out.append("")
        sub.sort(key=lambda r: (r["lane"], int(r["conc"]), -(_f(r, "total_tps") or 0)))
        body = []
        for r in sub:
            gpu = (_f(r, "total_tps") or 0) / 8.0
            cache = "" if r.get("cache_hit_pct") in ("", "NA") else f" {r['cache_hit_pct']}%"
            body.append(
                f"| `{r['label']}` | {r['lane']} | {r['conc']} | {r['median_ttft_ms']} | "
                f"{r['mean_tpot_ms']} | {r['out_tps']} | {r['total_tps']} | {gpu:.1f} | "
                f"{r['accept_len']} | {r['tok_usage_max']} | {r['mamba_usage_max']} | "
                f"{r['queue_max']} | {r['retok_div_pct']}{cache} |"
            )
        tbl(["config", "lane", "conc", "TTFT med ms", "TPOT ms", "out tok/s", "total tok/s",
             "tok/s/GPU", "accept", "KV use", "mamba use", "queue", "retok div"],
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            body)

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# gibberish
# --------------------------------------------------------------------------- #
def _ngram_repetition(text: str, n: int = 8) -> float:
    """Fraction of n-gram slots occupied by a repeat. 0 = no repeat."""
    toks = text.split()
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / len(grams)


def _longest_run(text: str) -> int:
    toks = text.split()
    best = cur = 1
    for a, b in zip(toks, toks[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best if toks else 0


def cmd_gibberish(args) -> int:
    texts = []
    with open(args.details, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("generated_texts", "generated_text", "outputs", "texts"):
                v = rec.get(key)
                if isinstance(v, list):
                    texts.extend(x for x in v if isinstance(x, str))
                elif isinstance(v, str):
                    texts.append(v)

    if not texts:
        print(f"no completion text found in {args.details}", file=sys.stderr)
        return 1

    reps = [_ngram_repetition(t, args.n) for t in texts]
    runs = [_longest_run(t) for t in texts]
    empt = sum(1 for t in texts if not t.strip())

    # Degeneration needs corroboration. A reasoning model legitimately repeats
    # phrasing while it thinks -- one measured sample summarising repetitive source
    # text hit 29% 8-gram repetition while being perfectly coherent, and with
    # longest_token_run = 1. A real loop shows extreme repetition *and* consecutive
    # token runs, so require either both together or one of them far out.
    flagged = [
        (i, r, u)
        for i, (r, u) in enumerate(zip(reps, runs))
        if (r > args.rep_threshold and u > args.run_threshold)
        or r > args.rep_hard
        or u > args.run_hard
    ]

    lines = [
        f"samples={len(texts)}",
        f"empty={empt}",
        f"rep{args.n}gram_mean={statistics.mean(reps):.4f}",
        f"rep{args.n}gram_max={max(reps):.4f}",
        f"rep{args.n}gram_p90={sorted(reps)[int(len(reps) * 0.9)]:.4f}",
        f"longest_token_run_max={max(runs)}",
        f"flagged={len(flagged)}",
        f"verdict={'FAIL' if (flagged or empt) else 'PASS'}",
    ]
    print("\n".join(lines))
    for i, r, u in flagged[:10]:
        print(f"# flagged sample {i}: rep={r:.3f} run={u}", file=sys.stderr)
        print("  " + texts[i][:300].replace("\n", " "), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("parse-bench")
    b.add_argument("--jsonl", required=True)
    b.add_argument("--text", default=None)
    b.set_defaults(fn=cmd_parse_bench)

    c = sub.add_parser("capacity")
    c.add_argument("--log", required=True)
    c.add_argument(
        "--strict",
        action="store_true",
        help="also treat any top-level exception as fatal (use only once the process is gone)",
    )
    c.set_defaults(fn=cmd_capacity)

    s = sub.add_parser("scrape")
    s.add_argument("--log", required=True)
    s.add_argument("--start-line", type=int, default=0)
    s.add_argument("--end-line", type=int, default=0)
    s.set_defaults(fn=cmd_scrape)

    m = sub.add_parser("summarize")
    m.add_argument("--csv", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--top", type=int, default=12)
    m.add_argument("--retok-flag", type=float, default=5.0)
    m.set_defaults(fn=cmd_summarize)

    rp = sub.add_parser("report")
    rp.add_argument("--csv", required=True)
    rp.add_argument("--out", default=None)
    rp.set_defaults(fn=cmd_report)

    bt = sub.add_parser("best")
    bt.add_argument("--csv", required=True)
    bt.add_argument("--metric", default="total_tps")
    bt.add_argument("--mode", choices=["max", "min"], default="max")
    bt.add_argument("--lane", default=None)
    bt.add_argument("--workload", default=None)
    bt.add_argument("--phase", default=None)
    bt.add_argument("--conc", type=int, default=None)
    bt.add_argument("--label-prefix", default=None)
    bt.add_argument("--max-retok-div", type=float, default=8.0)
    bt.set_defaults(fn=cmd_best)

    g = sub.add_parser("gibberish")
    g.add_argument("--details", required=True)
    g.add_argument("-n", type=int, default=8)
    # Corroborating pair: repetition that also shows consecutive token runs.
    g.add_argument("--rep-threshold", type=float, default=0.35)
    g.add_argument("--run-threshold", type=int, default=8)
    # Either signal on its own, but only when unambiguous.
    g.add_argument("--rep-hard", type=float, default=0.60)
    g.add_argument("--run-hard", type=int, default=20)
    g.set_defaults(fn=cmd_gibberish)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
