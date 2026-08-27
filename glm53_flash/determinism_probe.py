#!/usr/bin/env python3
"""Quantify nondeterminism: repeat an identical greedy request and count distinct results.

Correctness checks are useless as long as the same input gives different outputs, so
this is the measurement to bisect against. Batch composition is held constant (one
request at a time, cache flushed between), and temperature is 0, so any variation is
genuine nondeterminism rather than a scheduling artifact.

Reports, per prompt:
  n_distinct over REPEATS runs, and the first token-divergence position.

Usage: determinism_probe.py [repeats] [max_new_tokens]
"""
import collections
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 8

PROMPTS = [
    "Q: What is 17 plus 25? A: The answer is",
    "1, 2, 3, 4, 5, 6,",
    "The capital of France is",
]


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def flush():
    try:
        urllib.request.urlopen(BASE + "/flush_cache", timeout=60).read()
    except Exception:
        pass


def gen(text, n):
    return post(
        "/generate",
        {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": n}},
    )["text"]


total_distinct = 0
for prompt in PROMPTS:
    outs = []
    for _ in range(REPEATS):
        flush()
        outs.append(gen(prompt, NTOK))
    counts = collections.Counter(outs)
    total_distinct += len(counts)
    print(f"PROMPT {prompt!r}")
    print(f"  distinct results: {len(counts)} / {REPEATS}")
    for text, c in counts.most_common():
        print(f"    x{c}  {text!r}")

print()
print("=" * 74)
if total_distinct == len(PROMPTS):
    print(">>> DETERMINISTIC across all prompts (every repeat identical).")
else:
    print(f">>> NONDETERMINISTIC: {total_distinct} distinct results across "
          f"{len(PROMPTS)} prompts x {REPEATS} repeats (deterministic would be "
          f"{len(PROMPTS)}).")
