#!/usr/bin/env python3
"""Is the HIP tilelang sparse-MLA decode kernel numerically correct at tail_dim=0?

GLM-5.3-Flash has qk_rope_head_dim=0, so its DSA layers give tilelang_sparse_fwd
tail_dim = 0. Upstream's CUDA branch handles that by switching kernel factories;
the HIP branch had no equivalent and instead got `has_tail` guards added inside
sparse_mla_fwd_decode_partial. This checks whether those guards are equivalent.

tail_dim=64 is the control: it is the geometry GLM-5.1/5.2 and DeepSeek DSA already
run on this path, so a mismatch there would indict the reference, not the patch.
"""
import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

D_V = 512


def reference(q, kv, indices, sm_scale, d_v):
    """Gather the selected KV rows, softmax over them, weight the value part.

    q: [S, H, d_v+tail], kv: [S_kv, 1, d_v+tail], indices: [S, 1, topk] with -1 padding.
    """
    S, H, _ = q.shape
    out = torch.empty(S, H, d_v, dtype=torch.float32, device=q.device)
    kv2 = kv[:, 0, :].float()
    for s in range(S):
        idx = indices[s, 0, :]
        valid = idx >= 0
        gathered = kv2[idx.clamp(min=0).long()]              # [topk, d_v+tail]
        logits = (q[s].float() @ gathered.T) * sm_scale       # [H, topk]
        logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out[s] = probs @ gathered[:, :d_v]
    return out


def run_case(tail_dim, S, H, S_kv, topk, n_valid, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    dim = D_V + tail_dim
    q = (torch.randn(S, H, dim, device=dev, dtype=torch.bfloat16) * 0.5)
    kv = (torch.randn(S_kv, 1, dim, device=dev, dtype=torch.bfloat16) * 0.5)

    # -1-padded index rows, exactly like _forward_tilelang builds them
    indices = torch.full((S, 1, topk), -1, device=dev, dtype=torch.int32)
    for s in range(S):
        perm = torch.randperm(S_kv, device=dev)[:n_valid].to(torch.int32)
        indices[s, 0, :n_valid] = perm

    sm_scale = dim ** -0.5

    got = tilelang_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=D_V)
    ref = reference(q, kv, indices, sm_scale, D_V)

    g = got.float()
    absdiff = (g - ref).abs()
    denom = ref.abs().mean().clamp(min=1e-6)
    rel = absdiff.mean() / denom
    cos = torch.nn.functional.cosine_similarity(
        g.reshape(-1), ref.reshape(-1), dim=0
    )
    finite = torch.isfinite(g).all().item()
    print(
        f"  tail_dim={tail_dim:3d} S={S:3d} H={H:3d} topk={topk:5d} valid={n_valid:5d} | "
        f"finite={finite} max|d|={absdiff.max():.4f} rel={rel:.4f} cos={cos:.6f}"
    )
    return finite and cos > 0.99


if __name__ == "__main__":
    print("=== CONTROL: tail_dim=64 (the geometry GLM-5.1/5.2 already use here) ===")
    ctrl = []
    for S, H, S_kv, topk, nv in [(1, 16, 4096, 2112, 2051), (4, 16, 8192, 2112, 1000)]:
        ctrl.append(run_case(64, S, H, S_kv, topk, nv))

    print("=== SUBJECT: tail_dim=0 (GLM-5.3-Flash, qk_rope_head_dim=0) ===")
    subj = []
    for S, H, S_kv, topk, nv in [(1, 16, 4096, 2112, 2051), (4, 16, 8192, 2112, 1000)]:
        subj.append(run_case(0, S, H, S_kv, topk, nv))

    print()
    print(f"control (tail_dim=64) all pass: {all(ctrl)}")
    print(f"subject (tail_dim=0)  all pass: {all(subj)}")
    if all(ctrl) and not all(subj):
        print(">>> VERDICT: reference is sound; the tail_dim=0 HIP path is NUMERICALLY WRONG.")
    elif all(ctrl) and all(subj):
        print(">>> VERDICT: kernel is correct at tail_dim=0; the garbage comes from elsewhere.")
    else:
        print(">>> VERDICT: control failed too - the reference or the harness is wrong, not conclusive.")
