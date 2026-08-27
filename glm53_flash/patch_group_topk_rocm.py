#!/usr/bin/env python3
"""Supply a portable group-selection primitive for the kpool top-k on HIP.

After patch 02 sends HIP down the non-fused branch, that branch calls
sgl_kernel.fast_topk_v2, which asserts topk == 2048 ("only optimized for deepseek
v3.2"). GLM-5.3-Flash needs group_topk=512, so the branch is unusable as-is.

Only the group-selection step is CUDA-specific. Everything kpool-specific after it
-- expand_pooled_groups_to_topk and append_kpool_tail_to_topk -- is already Triton
and runs on gfx950. So replace just that primitive with a torch.topk equivalent on
HIP and leave the rest of the branch untouched.

Semantics being reproduced, from the fast_topk_v2 docstring: per row i, select the
top `topk` positions of score[i] restricted to [row_starts[i], row_starts[i]+lengths[i]),
returned in descending-score order as int32 absolute column indices. Rows shorter
than topk are padded; the caller already masks those via its own `group_valid`.
"""
import sys
from pathlib import Path

F = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py")
src = F.read_text()
orig = src


def sub_once(old: str, new: str, label: str) -> None:
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAIL [{label}]: expected exactly 1 occurrence, found {n}")
    src = src.replace(old, new, 1)
    print(f"  ok  [{label}]")


# 1. the portable primitive
sub_once(
    """BLOCK_SIZE_K = 64
INDEX_HEAD_DIM = 128
""",
    """BLOCK_SIZE_K = 64
INDEX_HEAD_DIM = 128


def _group_topk_torch(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    \"\"\"Portable stand-in for sgl_kernel.fast_topk_v2, which hardcodes topk=2048.

    Selects the top `topk` columns of each row within
    [row_starts[i], row_starts[i] + lengths[i]), descending by score, as int32
    absolute column indices. Shapes are static, so this is CUDA-graph capturable.
    \"\"\"
    rows, cols = score.shape
    positions = torch.arange(cols, device=score.device).unsqueeze(0)
    lo = (
        torch.zeros_like(lengths)
        if row_starts is None
        else row_starts.to(lengths.dtype)
    )
    lo = lo.unsqueeze(1)
    hi = lo + lengths.unsqueeze(1)
    in_range = (positions >= lo) & (positions < hi)
    masked = score.masked_fill(~in_range, float("-inf"))

    k = min(topk, cols)
    selected = masked.topk(k, dim=-1, largest=True, sorted=True).indices.to(
        torch.int32
    )
    if k == topk:
        return selected
    padding = torch.full(
        (rows, topk - k), -1, dtype=torch.int32, device=score.device
    )
    return torch.cat([selected, padding], dim=-1)
""",
    "portable group topk",
)

# 2. use it on HIP
sub_once(
    """    from sgl_kernel import fast_topk_v2

    selected_groups = fast_topk_v2(
        logits,
        group_lengths.to(torch.int32),
        group_topk,
        row_starts=row_starts,
    )
""",
    """    if _is_hip:
        # fast_topk_v2 asserts topk==2048; group_topk here is 512 for GLM-5.3-Flash.
        selected_groups = _group_topk_torch(
            logits,
            group_lengths.to(torch.int32),
            group_topk,
            row_starts=row_starts,
        )
    else:
        from sgl_kernel import fast_topk_v2

        selected_groups = fast_topk_v2(
            logits,
            group_lengths.to(torch.int32),
            group_topk,
            row_starts=row_starts,
        )
""",
    "dispatch group topk",
)

assert src != orig
F.write_text(src)
print(f"patched {F}")
