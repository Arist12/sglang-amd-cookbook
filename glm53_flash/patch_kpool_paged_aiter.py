#!/usr/bin/env python3
"""Make the kpool indexer honor dsa_paged_mqa_logits_backend, which resolves to
aiter on ROCm.

Symptom this addresses: prefill is numerically correct (verified by generating 24
tokens one at a time, each from a full fresh prefill -- coherent output, correct
arithmetic), while autoregressive generation degrades into fluent-shaped soup after
2-3 tokens. Prefill and decode differ in which indexer logits kernel they use.

`dsa_indexer.py` resolves the paged (decode) logits kernel through
DSAPagedMQALogitsBackend.resolve(), which returns AITER unconditionally on HIP.
The kpool indexer added by PR #36507 never wired that up; it has its own
_should_use_tilelang_paged_mqa_logits() that only knows tilelang vs DeepGEMM.
My earlier patch 01 made that helper return True on HIP, which put decode on
tilelang_fp8_paged_mqa_logits -- a kernel whose only CUDA callers are SM90 with
num_heads outside (32, 64), i.e. a narrow path, and never the ROCm one.

This routes HIP decode to aiter_paged_mqa_logits, the same kernel GLM-5.2 uses on
this node today, mirroring the sibling's tensor conventions: 3D q and the raw 2D
index-k read buffer, not the 4D reshapes the DeepGEMM/tilelang calls want.
"""
import sys
from pathlib import Path

F = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py")
src = F.read_text()
orig = src


def sub_once(old: str, new: str, label: str) -> None:
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAIL [{label}]: expected exactly 1 occurrence, found {n}")
    src = src.replace(old, new, 1)
    print(f"  ok  [{label}]")


# 1. resolve the configured backend once, like dsa_indexer.py does
sub_once(
    """from sglang.srt.runtime_context import get_parallel, get_server_args""",
    """from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args""",
    "get_exec import",
)

sub_once(
    """from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod""",
    """from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)""",
    "import backend enum",
)

sub_once(
    """        elif _is_hip:
            self.sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
""",
    """        elif _is_hip:
            self.sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
            get_exec().kernel.dsa_paged_mqa_logits_backend
        )
""",
    "resolve backend in init",
)

# 2. tilelang is no longer the HIP answer; the resolved backend is
sub_once(
    """        if _is_hip:
            return True
        if not is_cuda():
            return False""",
    """        if not is_cuda():
            return False""",
    "revert patch-01 tilelang-on-HIP",
)

# 3. keep the un-reshaped tensors the aiter kernel expects
sub_once(
    """        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64""",
    """        # aiter's paged MQA logits takes 3D q and the raw 2D index-k buffer and
        # does its own unsqueeze; DeepGEMM and tilelang take the 4D reshapes below.
        q_fp8_3d = q_fp8
        kv_cache_fp8_2d = kv_cache_fp8
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64""",
    "preserve 3D q / 2D kv",
)

# 4. dispatch
sub_once(
    """        use_tilelang_paged_mqa = self._should_use_tilelang_paged_mqa_logits(q_fp8)""",
    """        use_aiter_paged_mqa = self.paged_mqa_logits_backend.is_aiter()
        use_tilelang_paged_mqa = (
            not use_aiter_paged_mqa
            and self._should_use_tilelang_paged_mqa_logits(q_fp8)
        )""",
    "dispatch flags",
)

sub_once(
    """                build_schedule_metadata=not use_tilelang_paged_mqa,""",
    """                build_schedule_metadata=not (
                    use_tilelang_paged_mqa or use_aiter_paged_mqa
                ),""",
    "skip deepgemm schedule for aiter",
)

sub_once(
    """        if use_tilelang_paged_mqa:
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits,
            )
""",
    """        if use_aiter_paged_mqa:
            from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            logits = aiter_paged_mqa_logits(
                q_fp8_3d,
                kv_cache_fp8_2d,
                weights,
                pool_seqlens,
                pool_block_tables,
                pool_max_seq_len,
                preshuffle=aiter_can_use_preshuffle_paged_mqa(),
                kv_block_size=blocksize,
            )
        elif use_tilelang_paged_mqa:
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits,
            )
""",
    "aiter branch",
)

assert src != orig
F.write_text(src)
print(f"patched {F}")
