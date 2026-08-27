#!/usr/bin/env python3
"""Let the HIP TileLang sparse-MLA decode kernel handle tail_dim == 0.

GLM-5.3-Flash sets qk_rope_head_dim=0 (position handling lives in the KDA linear
layers), so its DSA layers reach tilelang_sparse_fwd with dim == d_v == 512 and
tail_dim == 0. The CUDA branch already switches kernels for that case
(sparse_attention_fwd_kernel_v1 if tail_dim == 0 else _v2), but the HIP branch calls
sparse_mla_fwd_decode_partial unconditionally, which allocates [.., 0] tail buffers
and emits `T.Parallel(BI, 0)`. TileLang's vectorizer then divides by the zero extent:
`Check failed: pb->value != 0 (0 vs. 0) : Divide by zero` during LayoutInference.

Every model previously on this HIP path (GLM-5.1/5.2, DeepSeek DSA) had tail_dim=64,
so this is the first tail_dim=0 model to use it.

Fix follows the `has_tail` pattern already used by sparse_attention_fwd_kernel_v1 in
this same file: skip the tail allocations, the tail load, and the tail GEMM at trace
time. With tail_dim == 0 the main GEMM already spans the full head dim, so the tail
GEMM's contribution is exactly zero and the result is unchanged.
"""
import sys
from pathlib import Path

F = Path(
    "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
)
src = F.read_text()
orig = src


def sub_once(old: str, new: str, label: str) -> None:
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAIL [{label}]: expected exactly 1 occurrence, found {n}")
    src = src.replace(old, new, 1)
    print(f"  ok  [{label}]")


# 1. has_tail flag, same name/semantics as sparse_attention_fwd_kernel_v1 above
sub_once(
    """    N_GROUPS = topk // (block_I * inner_iter)
    BI = block_I
    D = dim
    D_tail = tail_dim
""",
    """    N_GROUPS = topk // (block_I * inner_iter)
    BI = block_I
    D = dim
    D_tail = tail_dim
    has_tail = tail_dim > 0
""",
    "has_tail flag",
)

# 2. tail buffers
sub_once(
    """            if _q_in_shared:
                Q_buf = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)
                Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)
""",
    """            if _q_in_shared:
                Q_buf = T.alloc_shared([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)
""",
    "Q tail buffer",
)

sub_once(
    """            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
""",
    """            KV_shared = T.alloc_shared([BI, D], dtype)
            if has_tail:
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
""",
    "K tail buffer",
)

# 3. tail load of Q
sub_once(
    """            T.copy(Q[b_i, s_i, H0:H1, :D], Q_buf)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_buf)
""",
    """            T.copy(Q[b_i, s_i, H0:H1, :D], Q_buf)
            if has_tail:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_buf)
""",
    "Q tail copy",
)

# 4. tail gather of K -- T.Parallel(BI, 0) is the divide-by-zero site
sub_once(
    """                for bi_i, d_i in T.Parallel(BI, D_tail):
                    idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, T.if_then_else(idx >= 0, idx, 0), g_i, D + d_i
                    ]
""",
    """                if has_tail:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i, T.if_then_else(idx >= 0, idx, 0), g_i, D + d_i
                        ]
""",
    "K tail gather",
)

# 5. tail GEMM accumulating into acc_s
sub_once(
    """                T.gemm(
                    Q_tail_buf,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
""",
    """                if has_tail:
                    T.gemm(
                        Q_tail_buf,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
""",
    "tail GEMM",
)

assert src != orig
F.write_text(src)
print(f"patched {F}")
