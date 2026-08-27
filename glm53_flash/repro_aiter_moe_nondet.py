#!/usr/bin/env python3
"""Minimal reproducer: is aiter.fused_moe.fused_moe run-to-run deterministic on gfx950?

Context. On 8x MI355X, GLM-5.3-Flash produces a different greedy continuation on every
repeat of an identical request. Per-module tensor dumps localize the divergence to the
first MoE layer: identical hidden states, identical router logits and identical top-k
expert ids, but a different routed-expert output. GLM-5.2-FP8 on the same container,
the same aiter build and the same fused_moe entry point is fully deterministic.

The two models differ in MoE shape, not in code path -- bypassing the swiglu_limit /
GateMode branch, MoE padding and the clamp fusion all leave the nondeterminism intact.
So this calls fused_moe directly, twice, on identical synthetic inputs, at each model's
per-rank shapes, and reports whether the two results are bitwise equal.

`splitk` is swept because a split-K reduction accumulated with atomics would explain a
shape-dependent loss of determinism.
"""
import sys

import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

BLOCK = 128


def make_case(name, e_local, hidden, inter, topk=8, m=17, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    hs = (torch.randn(m, hidden, device=dev, dtype=torch.bfloat16) * 0.3)

    def q(shape):
        t = torch.randn(*shape, device=dev, dtype=torch.float32) * 0.05
        return t.to(torch.float8_e4m3fn)

    w1 = q((e_local, 2 * inter, hidden))            # gate+up
    w2 = q((e_local, hidden, inter))                # down
    w1_scale = (
        torch.rand(e_local, (2 * inter) // BLOCK, hidden // BLOCK, device=dev) * 0.02
        + 0.01
    )
    w2_scale = (
        torch.rand(e_local, hidden // BLOCK, inter // BLOCK, device=dev) * 0.02 + 0.01
    )

    logits = torch.randn(m, e_local, device=dev, dtype=torch.float32)
    tw, ti = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    tw = tw.to(torch.float32).contiguous()
    ti = ti.to(torch.int32).contiguous()
    return dict(
        name=name,
        hidden_states=hs,
        w1=w1,
        w2=w2,
        topk_weight=tw,
        topk_ids=ti,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def run(case, splitk, repeats=4):
    outs = []
    for _ in range(repeats):
        o = fused_moe(
            hidden_states=case["hidden_states"],
            w1=case["w1"],
            w2=case["w2"],
            topk_weight=case["topk_weight"],
            topk_ids=case["topk_ids"],
            quant_type=QuantType.per_128x128,
            activation=ActivationType.Silu,
            w1_scale=case["w1_scale"],
            w2_scale=case["w2_scale"],
            splitk=splitk,
        )
        outs.append(o.detach().clone())
    base = outs[0]
    n_diff = sum(0 if torch.equal(base, o) else 1 for o in outs[1:])
    if n_diff:
        worst = max((base.float() - o.float()).abs().max().item() for o in outs[1:])
        frac = max(float((base != o).sum()) / base.numel() for o in outs[1:])
    else:
        worst, frac = 0.0, 0.0
    return n_diff, worst, frac, base.shape


CASES = [
    # GLM-5.3-Flash: 288 routed experts / TP8 = 36 per rank, hidden 4096
    make_case("GLM-5.3-Flash (E=36, h=4096, i=2048)", 36, 4096, 2048),
    # GLM-5.2-FP8 control: 256 / TP8 = 32 per rank, hidden 6144
    make_case("GLM-5.2-FP8   (E=32, h=6144, i=2048)", 32, 6144, 2048),
]

print(f"quant_type=per_128x128  activation=Silu  repeats=4")
print()
for case in CASES:
    print(f"### {case['name']}")
    for splitk in (0, 1, 2):
        try:
            n_diff, worst, frac, shape = run(case, splitk)
        except Exception as e:
            print(f"  splitk={splitk}: FAILED {type(e).__name__}: {str(e)[:110]}")
            continue
        verdict = "DETERMINISTIC" if n_diff == 0 else "NONDETERMINISTIC"
        print(
            f"  splitk={splitk}: {verdict:16s} out{tuple(shape)} "
            f"differing_repeats={n_diff}/3 max|d|={worst:.6g} frac={frac:.3%}"
        )
    print()
