#!/usr/bin/env python3
"""Route HIP away from the CUDA-only fused kpool top-k kernel.

`fast_kpool_topk_transform_fused` JIT-compiles jit/csrc/dsa/kpool_topk_transform.cuh,
which includes <cuda_fp16.h>; hipcc has no such header, so every ROCm run with
group_topk in (128,160,192,224,256,512) dies during decode CUDA graph capture.
GLM-5.3-Flash lands there exactly: index_topk=2048 / index_kpool=4 -> group_topk=512.

The same function already contains a portable fallback for group_topk=2048, built
from sgl_kernel.fast_topk_v2 plus the Triton expand/append kernels in this file --
all available on gfx950. It is a general implementation, not 2048-specific, so HIP
takes it. Porting the fused radix kernel to HIP (warpSize 64, HIP fp16 headers) is a
separate performance task, not a correctness requirement.

The fallback asserts page_table_row_index is None because the fused kernel does that
gather internally, so materialize it here.
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


# 1. platform probe (this file had no platform imports at all)
sub_once(
    """import torch
import triton
import triton.language as tl
""",
    """import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()
""",
    "is_hip import",
)

# 2. keep HIP off the fused CUDA JIT kernel
sub_once(
    """    if group_topk in (128, 160, 192, 224, 256, 512):
        from sglang.kernels.ops.moe.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )""",
    """    # The fused kernel is a CUDA JIT module (<cuda_fp16.h>); HIP falls through to
    # the portable fast_topk_v2 + Triton path below.
    if group_topk in (128, 160, 192, 224, 256, 512) and not _is_hip:
        from sglang.kernels.ops.moe.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )""",
    "fused path gate",
)

# 3. the portable path indexes page_table by row, so do the gather the fused
#    kernel would have done internally instead of asserting it away
sub_once(
    """    assert (
        page_table_row_index is None
    ), "page_table_row_index requires the fused fast_kpool group_topk path"
""",
    """    if page_table_row_index is not None:
        # Materialize what the fused kernel gathers internally; the portable path
        # below indexes page_table positionally.
        assert page_table is not None
        page_table = page_table.index_select(0, page_table_row_index.to(torch.long))
        page_table_row_index = None
""",
    "materialize page_table_row_index",
)

assert src != orig
F.write_text(src)
print(f"patched {F}")
