#!/usr/bin/env python3
"""Find the first module whose output differs between two identical forward passes.

--debug-tensor-dump-output-folder writes one .pt per forward pass containing every
leaf module's output, keyed by module path. Two passes over the same input should be
bit-identical; the first key that is not tells us where the nondeterminism enters, and
the module path says whether it is a linear_attention (KDA) or a
deepseek_sparse_attention (DSA/kpool) layer.

Usage: diff_dumps.py <PassA.pt> <PassB.pt>
"""
import sys

import torch


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def stat(a, b):
    """Return (bitwise_equal, max_abs_diff, rel) for two comparable tensors."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, float("inf"), float("inf")
    if torch.equal(a, b):
        return True, 0.0, 0.0
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    denom = af.abs().mean().clamp(min=1e-8)
    return False, d.max().item(), (d.mean() / denom).item()


def flatten(name, v, out):
    if isinstance(v, torch.Tensor):
        out[name] = v
    elif isinstance(v, (list, tuple)):
        for i, t in enumerate(v):
            flatten(f"{name}[{i}]", t, out)


A, B = load(sys.argv[1]), load(sys.argv[2])
fa, fb = {}, {}
for k, v in A.items():
    flatten(k, v, fa)
for k, v in B.items():
    flatten(k, v, fb)

only_a = sorted(set(fa) - set(fb))
only_b = sorted(set(fb) - set(fa))
if only_a or only_b:
    print(f"KEY SETS DIFFER: {len(only_a)} only in A, {len(only_b)} only in B")
    for k in only_a[:6]:
        print(f"  only A: {k}")
    for k in only_b[:6]:
        print(f"  only B: {k}")
    print()

shared = [k for k in fa if k in fb]
# order by module path so "first divergence" is meaningful for layers
def sort_key(k):
    parts = k.split(".")
    layer = -1
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer = int(parts[i + 1])
    return (layer, k)


shared.sort(key=sort_key)

diffs = []
for k in shared:
    eq, mx, rel = stat(fa[k], fb[k])
    if not eq:
        diffs.append((k, mx, rel, tuple(fa[k].shape), str(fa[k].dtype)))

print(f"compared {len(shared)} module outputs")
print(f"identical: {len(shared) - len(diffs)}   differing: {len(diffs)}")
print()
if not diffs:
    print(">>> the two passes are BITWISE IDENTICAL.")
    sys.exit(0)

print("first 25 divergences in layer order:")
for k, mx, rel, shape, dt in diffs[:25]:
    print(f"  {k:70s} shape={shape} {dt} max|d|={mx:.6g} rel={rel:.4g}")

print()
layers_hit = sorted({sort_key(k)[0] for k, *_ in diffs if sort_key(k)[0] >= 0})
print(f"layers containing a divergence: {layers_hit}")
