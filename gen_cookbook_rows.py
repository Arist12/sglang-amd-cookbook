#!/usr/bin/env python3
"""Emit cookbook benchmark rows straight from the bench_serving jsonl.

Hand-transcribing 20+ rows out of logs is how wrong numbers get published, so
the cookbook rows are generated from the same artefacts the analysis used.
"""

import argparse
import json
import re
from pathlib import Path

CELL = re.compile(r"^(?P<mode>nospec|dspark)-isl(?P<isl>\d+)-osl(?P<osl>\d+)-c(?P<c>\d+)$")


def load(run_dir: Path, mode: str) -> dict:
    out = {}
    for f in sorted(run_dir.glob(f"{mode}-isl*.jsonl")):
        m = CELL.match(f.stem)
        if not m:
            continue
        rec = None
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    pass
        if rec:
            out[(int(m["isl"]), int(m["osl"]), int(m["c"]))] = rec
    return out


def row(key, rec, source):
    isl, osl, c = key
    total = rec.get("total_throughput")
    tpot = rec.get("mean_tpot_ms")
    out = {
        "isl": isl,
        "osl": osl,
        "concurrency": c,
        "ttft_ms": round(rec.get("mean_ttft_ms", 0), 0),
        "tpot_ms": round(tpot, 2) if tpot else None,
    }
    # The site treats decode_tok_s as a per-stream rate (1000/TPOT) and blanks it
    # above concurrency 1, so only emit it where it means something.
    if c == 1 and tpot:
        out["decode_tok_s"] = round(1000 / tpot, 1)
    out.update({
        "output_tok_s": round(rec.get("output_throughput", 0), 2),
        "total_tok_s": round(total, 2) if total else None,
        "tok_s_per_gpu": round(total / 8, 1) if total else None,
        "source": source,
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase2", default="/data/jhinpan-tools/K3-v0516-workspace/dspark-study")
    p.add_argument("--yarn", default="/data/jhinpan-tools/K3-v0516-workspace/yarn-recheck")
    p.add_argument("--ctx131k", default="/data/jhinpan-tools/K3-v0516-workspace/ctx131k")
    args = p.parse_args()

    p2, yr, c131 = Path(args.phase2), Path(args.yarn), Path(args.ctx131k)
    nospec = load(p2, "nospec")
    dspark_yarn = load(yr, "dspark")
    # 131k was taken in a later pass, on the same image and the same YaRN draft.
    nospec.update(load(c131, "nospec"))
    dspark_yarn.update(load(c131, "dspark"))

    SRC_NS = ("kimi_k3_playbook.md section 8.1 (bench_serving, random, "
              "--random-range-ratio 1, radix cache off; "
              "rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260805, upstream sglang 4e7209caa)")
    SRC_DS = ("kimi_k3_playbook.md sections 5.4a and 8.1 (bench_serving, random, "
              "--random-range-ratio 1, radix cache off; draft revision 56ce616a "
              "with YaRN, the revision that removes the long-context collapse)")

    # Keep the published set close to the previous one in size: a concurrency
    # sweep at the reference shape, the saturation region, and the context axis.
    ns_keys = [
        (1024, 1024, 1), (1024, 1024, 8), (1024, 1024, 32), (1024, 1024, 64),
        (8192, 1024, 1), (8192, 1024, 32), (8192, 1024, 96), (8192, 1024, 128),
        (16384, 1024, 1), (32768, 1024, 1), (65536, 1024, 1), (131072, 1024, 1),
    ]
    ds_keys = sorted(dspark_yarn)

    print("// ---- high-throughput (no speculation) ----")
    print(json.dumps([row(k, nospec[k], SRC_NS) for k in ns_keys if k in nospec], indent=2))
    print()
    print("// ---- low-latency (DSpark, YaRN draft) ----")
    print(json.dumps([row(k, dspark_yarn[k], SRC_DS) for k in ds_keys], indent=2))


if __name__ == "__main__":
    main()
