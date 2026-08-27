#!/usr/bin/env python3
"""aiter's deepgemm_fp8_paged_mqa_logits unpacks kv_cache.size() as 4D.

Observed: ValueError: not enough values to unpack (expected 4, got 2) at
aiter/ops/triton/attention/pa_mqa_logits.py:478. dsa_indexer.py reshapes the
index-k buffer to [n, page_size, 1, 132] before its aiter call and keeps only q 3D;
patch 05 wrongly handed aiter the raw 2D buffer.
"""
import sys
from pathlib import Path

F = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py")
s = F.read_text()

old_cap = (
    "        # aiter's paged MQA logits takes 3D q and the raw 2D index-k buffer and\n"
    "        # does its own unsqueeze; DeepGEMM and tilelang take the 4D reshapes below.\n"
    "        q_fp8_3d = q_fp8\n"
    "        kv_cache_fp8_2d = kv_cache_fp8\n"
)
new_cap = (
    "        # aiter's paged MQA logits does its own next_n unsqueeze so it needs q 3D;\n"
    "        # it takes the same 4D kv view as DeepGEMM and tilelang.\n"
    "        q_fp8_3d = q_fp8\n"
)
if s.count(old_cap) != 1:
    sys.exit(f"FAIL capture block: found {s.count(old_cap)}")
s = s.replace(old_cap, new_cap, 1)

old_call = "                q_fp8_3d,\n                kv_cache_fp8_2d,\n                weights,"
new_call = "                q_fp8_3d,\n                kv_cache_fp8,\n                weights,"
if s.count(old_call) != 1:
    sys.exit(f"FAIL call site: found {s.count(old_call)}")
s = s.replace(old_call, new_call, 1)

F.write_text(s)
print("fixed: aiter now receives the 4D kv view")
