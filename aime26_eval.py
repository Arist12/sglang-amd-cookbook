"""AIME 2026 eval for a live SGLang server.

sglang 0.5.16 ships aime25 but not aime26, so this subclasses the in-tree
AIME25Eval and swaps only the dataset source. Prompt template, answer-extraction
regex, scorer and aggregation are inherited verbatim, which is what makes the
numbers comparable to the AIME25 cell in the cookbook.

Usage:
  python aime26_eval.py --base-url http://127.0.0.1:30000 \
      --data data/aime26.jsonl --repeat 4 --num-threads 32 --max-tokens 64000
"""

import argparse
import json
import statistics
import time
from typing import List

from sglang.test.simple_eval_aime25 import AIME25Eval
from sglang.test.simple_eval_common import ChatCompletionSampler, set_ulimit


class AIME26Eval(AIME25Eval):
    def __init__(self, data_path: str, num_examples, num_threads: int):
        examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    examples.append(
                        {"question": row["question"], "answer": str(row["answer"])}
                    )
        if num_examples:
            examples = examples[: min(num_examples, len(examples))]
        self.examples = examples
        self.num_threads = num_threads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--data", default="/sgl-workspace/workspace/data/aime26.jsonl")
    p.add_argument("--model", default=None)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    set_ulimit()

    # ChatCompletionSampler wraps an OpenAI client, so it needs the /v1 root;
    # run_eval appends it too. Passing the bare server URL makes the client's
    # models.list() probe 404 before the eval starts.
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    sampler = ChatCompletionSampler(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_url=base_url,
    )

    eval_obj = AIME26Eval(args.data, args.num_examples, args.num_threads)
    print(f"AIME26: {len(eval_obj.examples)} problems x {args.repeat} repeats "
          f"= {len(eval_obj.examples) * args.repeat} samples", flush=True)
    print(f"temperature={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens} num_threads={args.num_threads}", flush=True)

    scores: List[float] = []
    wall: List[float] = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        result = eval_obj(sampler)
        dt = time.perf_counter() - t0
        scores.append(result.score)
        wall.append(dt)
        print(f"  repeat {i + 1}/{args.repeat}: score={result.score:.4f} "
              f"wall={dt:.1f}s", flush=True)

    mean = statistics.mean(scores)
    sem = statistics.stdev(scores) / (len(scores) ** 0.5) if len(scores) > 1 else 0.0
    total_wall = sum(wall)

    print()
    print(f"AIME26 pass@1 (avg of {args.repeat}): {mean * 100:.2f}% +/- {sem * 100:.2f}% (SEM)")
    print(f"per-repeat scores: {[round(s * 100, 2) for s in scores]}")
    print(f"total wall clock: {total_wall:.1f}s  (mean {statistics.mean(wall):.1f}s per repeat)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "eval": "aime26",
                    "n_problems": len(eval_obj.examples),
                    "repeat": args.repeat,
                    "pass_at_1_mean": mean,
                    "pass_at_1_sem": sem,
                    "per_repeat_scores": scores,
                    "per_repeat_wall_s": wall,
                    "total_wall_s": total_wall,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "num_threads": args.num_threads,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
