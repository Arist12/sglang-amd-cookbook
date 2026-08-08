"""Download nvidia/SPEED-Bench and emit the JSONL shape sglang's loader wants.

sglang's SpeedBenchDataset reads one JSON object per line with:
    {"category": "low_entropy"|"mixed"|"high_entropy", "turns": ["<prompt>", ...]}
and takes turns[0] as the prompt. The repo points at NVIDIA's specdec_bench to
produce this file; this script goes straight to the HF dataset instead so the
node does not need the whole Model-Optimizer checkout.

SPEED-Bench matters here because its Throughput split is stratified by output
entropy (low = code/sorting, mixed = STEM, high = creative writing), which is
exactly the axis that decides speculative accept length -- and the axis a
random-token dataset cannot express at all.
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/sgl-workspace/workspace/data/speedbench")
    p.add_argument(
        "--buckets",
        default="1k,8k,32k",
        help="Throughput ISL buckets to fetch (of 1k,2k,8k,16k,32k).",
    )
    args = p.parse_args()

    os.environ.pop("HF_HUB_OFFLINE", None)
    from datasets import get_dataset_config_names, load_dataset

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = get_dataset_config_names("nvidia/SPEED-Bench")
    print("available configs:", configs)

    wanted = [b.strip() for b in args.buckets.split(",") if b.strip()]
    for bucket in wanted:
        # Config naming varies across releases; match on the bucket token.
        cands = [c for c in configs if "throughput" in c.lower() and bucket in c.lower()]
        if not cands:
            print(f"!! no config matched bucket {bucket}; skipping")
            continue
        cfg = cands[0]
        print(f"\n=== bucket {bucket} -> config {cfg} ===")
        ds = load_dataset("nvidia/SPEED-Bench", cfg)
        split = "train" if "train" in ds else list(ds.keys())[0]
        rows = ds[split]
        print(f"  split={split} n={len(rows)} cols={rows.column_names}")

        # Find the prompt column and the category column whatever they are named.
        prompt_col = next(
            (c for c in ("turns", "prompt", "question", "text", "input") if c in rows.column_names),
            None,
        )
        cat_col = next(
            (c for c in ("category", "difficulty", "entropy", "domain") if c in rows.column_names),
            None,
        )
        if prompt_col is None:
            print(f"  !! no prompt column in {rows.column_names}; skipping")
            continue
        print(f"  prompt column={prompt_col}  category column={cat_col}")

        path = out / f"throughput_{bucket}.jsonl"
        counts: dict = {}
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                val = r[prompt_col]
                turns = val if isinstance(val, list) else [val]
                turns = [t for t in turns if isinstance(t, str) and t]
                if not turns:
                    continue
                cat = str(r[cat_col]) if cat_col else "mixed"
                counts[cat] = counts.get(cat, 0) + 1
                f.write(json.dumps({"category": cat, "turns": turns}) + "\n")
        print(f"  wrote {path}  categories={counts}")

    print("\ndone")


if __name__ == "__main__":
    main()
