#!/usr/bin/env python3
"""Separate prefill correctness from decode correctness.

Autoregressive generation on this build produces a few correct tokens and then
degrades into fluent-shaped soup. That pattern fits state accumulated across decode
steps, not a uniformly wrong kernel. This compares:

  A) one normal 24-token autoregressive generation (prefill once, decode 23 times)
  B) 24 tokens produced one at a time, each from a full fresh prefill of the text
     so far, with the radix cache flushed in between so no decode step is involved

If B is coherent and A is not, the fault is in the decode path.
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"


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


PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "Q: What is 17 plus 25? A: The answer is",
]
N = 24

for prompt in PROMPTS:
    print("=" * 78)
    print(f"PROMPT: {prompt!r}")

    flush()
    auto = gen(prompt, N)
    print(f"  A) autoregressive (1 prefill + {N-1} decodes):")
    print(f"     {auto!r}")

    text = prompt
    pieces = []
    for _ in range(N):
        flush()  # force a full prefill, no prefix reuse, no decode
        tok = gen(text, 1)
        if not tok:
            break
        pieces.append(tok)
        text += tok
    print(f"  B) {N} x (full prefill -> 1 token), cache flushed each step:")
    print(f"     {''.join(pieces)!r}")
    sys.stdout.flush()

print("=" * 78)
print("If B reads sensibly and A does not, decode is the broken path.")
