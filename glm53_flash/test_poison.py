#!/usr/bin/env python3
"""Does running decode poison later prefills in the same process?

The previous version of this test ran the autoregressive case before the
prefill-only case for each prompt, so a decode-path out-of-bounds write could have
contaminated the prefill result. Order here is deliberate:

  B1  prefill-only, before any decode has ever run in this process
  A   autoregressive (exercises the decode path)
  B2  prefill-only again, identical to B1

B1 == B2 means decode does not leak into prefill state. B1 correct but B2 wrong
means the decode path corrupts shared state, which also explains why the fault
surfaces as an asynchronous HSA exception at varying points.

Run against a FRESHLY started server, or B1 is not actually decode-free.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:30000"
N = 20

PROMPTS = [
    ("Q: What is 17 plus 25? A: The answer is", "42"),
    ("1, 2, 3, 4, 5, 6,", "7, 8, 9"),
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


def prefill_only(prompt, n=N):
    text = prompt
    out = []
    for _ in range(n):
        flush()
        tok = gen(text, 1)
        if not tok:
            break
        out.append(tok)
        text += tok
    return "".join(out)


results = {}
print("### B1: prefill-only, no decode has run yet in this process")
for prompt, expect in PROMPTS:
    b1 = prefill_only(prompt)
    results[(prompt, "B1")] = b1
    print(f"  {prompt!r}\n    -> {b1!r}   [expect to contain {expect!r}: {expect in b1}]")

print("### A: autoregressive (runs the decode path)")
for prompt, expect in PROMPTS:
    flush()
    a = gen(prompt, N)
    results[(prompt, "A")] = a
    print(f"  {prompt!r}\n    -> {a!r}   [contains {expect!r}: {expect in a}]")

print("### B2: prefill-only again, after decode has run")
for prompt, expect in PROMPTS:
    b2 = prefill_only(prompt)
    results[(prompt, "B2")] = b2
    same = b2 == results[(prompt, "B1")]
    print(f"  {prompt!r}\n    -> {b2!r}   [contains {expect!r}: {expect in b2}] [same as B1: {same}]")

print()
print("=" * 74)
allsame = all(results[(p, "B1")] == results[(p, "B2")] for p, _ in PROMPTS)
b1ok = all(e in results[(p, "B1")] for p, e in PROMPTS)
aok = all(e in results[(p, "A")] for p, e in PROMPTS)
print(f"prefill correct before any decode : {b1ok}")
print(f"autoregressive correct            : {aok}")
print(f"prefill unchanged by decode       : {allsame}")
if b1ok and not aok and allsame:
    print(">>> decode path is wrong, but it does not corrupt shared state.")
elif b1ok and not allsame:
    print(">>> decode path CORRUPTS state that later prefills read.")
elif not b1ok:
    print(">>> prefill is already wrong on a clean process; the fault is not decode-only.")
