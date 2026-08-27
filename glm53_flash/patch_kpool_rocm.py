#!/usr/bin/env python3
"""Give dsa_indexer_kpool.py the ROCm path its sibling dsa_indexer.py already has.

PR #36507 adds the kpool DSA indexer as a parallel implementation next to the
existing dsa_indexer.py. The sibling already routes HIP to aiter's Triton
fp8_mqa_logits and to TileLang for paged MQA logits; the new file never got
those branches, so every ROCm run dies at
`NotImplementedError: kpool indexer is only supported on CUDA`.

Applies 5 edits, then `git diff` in the container yields the reviewable patch.
"""
import re
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


# 1. route the three ragged call sites through the dispatch helper.
#    Done before the helper is inserted, since the helper body contains the same
#    call text and would otherwise be rewritten into infinite recursion.
n = src.count("deep_gemm.fp8_mqa_logits(\n")
if n != 3:
    sys.exit(f"FAIL [ragged call sites]: expected 3, found {n}")
src = src.replace("deep_gemm.fp8_mqa_logits(\n", "_fp8_mqa_logits(\n")
print("  ok  [ragged call sites] 3 replaced")

# 2. module-level HIP constant + the ragged-logits dispatch helper
sub_once(
    """if is_npu():
    import custom_ops  # noqa: F401
""",
    """if is_npu():
    import custom_ops  # noqa: F401

_is_hip = is_hip()


def _fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    clean_logits: bool,
) -> torch.Tensor:
    # DeepGEMM has no ROCm build, so HIP uses aiter's Triton kernel. It takes the
    # (kv, scale) pair unpacked and needs no head padding, matching how
    # dsa_indexer.py routes the same call.
    if _is_hip:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

        kv, scale = kv_fp8
        return fp8_mqa_logits(
            q_fp8, kv, scale, weights, ks, ke, clean_logits=clean_logits
        )
    return deep_gemm.fp8_mqa_logits(
        q_fp8, kv_fp8, weights, ks, ke, clean_logits=clean_logits
    )
""",
    "helper + _is_hip",
)

# 2. sm_count must exist on HIP too: configure_deep_gemm_num_sms reads
#    half_device_sm_count while evaluating its argument, before any no-op kicks in.
sub_once(
    """        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
""",
    """        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
        elif _is_hip:
            self.sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
""",
    "sm_count on HIP",
)

# 3. TileLang is the only paged MQA logits kernel available on ROCm, and
#    tilelang_kernel.py carries gfx950 tile params for it. The CUDA condition
#    encodes "SM90 without a DeepGEMM instantiation for this head count"; on HIP
#    there is no DeepGEMM instantiation for any head count.
sub_once(
    """        if not is_cuda():
            return False
        arch_major, _ = torch.cuda.get_device_capability(q_fp8.device)""",
    """        if _is_hip:
            return True
        if not is_cuda():
            return False
        arch_major, _ = torch.cuda.get_device_capability(q_fp8.device)""",
    "tilelang paged mqa on HIP",
)

# 5. open the forward dispatch to HIP
sub_once(
    """        if is_cuda():
            if (
                forward_batch.forward_mode.is_decode_or_idle()""",
    """        if is_cuda() or _is_hip:
            if (
                forward_batch.forward_mode.is_decode_or_idle()""",
    "forward dispatch guard",
)
sub_once(
    '            raise NotImplementedError("kpool indexer is only supported on CUDA")',
    '            raise NotImplementedError(\n'
    '                "kpool indexer is only supported on CUDA and ROCm"\n'
    "            )",
    "guard message",
)

assert src != orig
F.write_text(src)
print(f"patched {F}")
