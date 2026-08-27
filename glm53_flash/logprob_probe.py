#!/usr/bin/env python3
"""Flat logits from a systematic error, or genuinely wild nondeterminism?

SGLang inference is not bitwise deterministic by default (atomics in reductions,
split-K GEMMs), but that noise is normally last-bit and never flips a confident
argmax. Observed here: 24/24 distinct greedy continuations, with the very first
token varying. Two explanations with opposite implications:

  flat      - something upstream is systematically wrong, the logit distribution is
              near-uniform, and ordinary tiny noise decides the argmax
  confident - top-1 is genuinely confident yet changes between runs, i.e. large
              magnitude nondeterminism (a race or an uninitialized read)

A prompt with an overwhelmingly obvious continuation is used so that a healthy model
would be very confident.
"""
import json
import math
import urllib.request

BASE = "http://127.0.0.1:30000"
REPEATS = 6

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
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


for prompt in PROMPTS:
    print(f"PROMPT {prompt!r}")
    for r in range(REPEATS):
        flush()
        d = post(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "top_logprobs_num": 5,
            },
        )
        meta = d["meta_info"]
        top = meta.get("output_top_logprobs") or []
        if not top:
            print(f"  run {r}: no top_logprobs in meta_info; keys={list(meta)}")
            continue
        first = top[0]
        parts = []
        for entry in first[:5]:
            lp, tid = entry[0], entry[1]
            tok = entry[2] if len(entry) > 2 else str(tid)
            parts.append(f"{tok!r}:{lp:.2f}({math.exp(lp):.3f})")
        print(f"  run {r}: " + "  ".join(parts))
    print()

print("=" * 74)
print("Top-1 probability near 1.0 and stable  -> healthy, look elsewhere.")
print("Top-1 probability low / candidates shuffling -> flat logits, systematic error.")
print("Top-1 confident but a DIFFERENT token each run -> large-magnitude nondeterminism.")
