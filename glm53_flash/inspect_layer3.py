#!/usr/bin/env python3
"""Per-key verdict for the layers around the first divergence.

The point to establish: in the first MoE layer, are the attention and the router
outputs bit-identical while only the expert output differs? If so, identical inputs
and identical routing are producing different expert results, which localizes the
nondeterminism to the MoE expert computation rather than to attention or routing.

Usage: inspect_layer3.py <PassA.pt> <PassB.pt> [max_layer]
"""
import sys

import torch

A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
MAXL = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def flatten(name, v, out):
    if isinstance(v, torch.Tensor):
        out[name] = v
    elif isinstance(v, (list, tuple)):
        for i, t in enumerate(v):
            flatten(f"{name}[{i}]", t, out)


fa, fb = {}, {}
for k, v in A.items():
    flatten(k, v, fa)
for k, v in B.items():
    flatten(k, v, fb)


def layer_of(k):
    p = k.split(".")
    for i, x in enumerate(p):
        if x == "layers" and i + 1 < len(p) and p[i + 1].isdigit():
            return int(p[i + 1])
    return -1


for L in range(0, MAXL + 1):
    keys = sorted(k for k in fa if layer_of(k) == L and k in fb)
    if not keys:
        continue
    print(f"===== layer {L} =====")
    for k in keys:
        a, b = fa[k], fb[k]
        short = k.split(f"layers.{L}.", 1)[-1]
        if a.shape != b.shape:
            print(f"  SHAPE  {short:52s} {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if torch.equal(a, b):
            print(f"  same   {short:52s} {tuple(a.shape)}")
        else:
            d = (a.float() - b.float()).abs()
            n = int((a != b).sum())
            print(
                f"  DIFF   {short:52s} {tuple(a.shape)} "
                f"max|d|={d.max().item():.6g} elems_differing={n}/{a.numel()}"
            )
    print()

pre = sorted(k for k in fa if layer_of(k) == -1 and k in fb)
print("===== non-layer modules =====")
for k in pre:
    a, b = fa[k], fb[k]
    if a.shape != b.shape:
        print(f"  SHAPE  {k:52s} {tuple(a.shape)} vs {tuple(b.shape)}")
    elif torch.equal(a, b):
        print(f"  same   {k:52s} {tuple(a.shape)}")
    else:
        d = (a.float() - b.float()).abs()
        print(f"  DIFF   {k:52s} {tuple(a.shape)} max|d|={d.max().item():.6g}")
