#!/usr/bin/env python3
"""DIAGNOSTIC ONLY, NOT A FIX: force GLM-5.3-Flash down the same aiter fused_moe path
GLM-5.2-FP8 takes.

Per-module tensor dumps put the nondeterminism in model.layers.3.mlp.experts with
identical inputs, gate and top-k. The one aiter MoE code path GLM-5.3-Flash enters and
the deterministic GLM-5.2 control does not is `elif quant_info.swiglu_limit > 0`, whose
own comment says it exists for the gpt-oss MXFP4 layout, not FP8 block-quant. Neither
GateMode.INTERLEAVE nor GateMode.SEPARATED gives determinism.

Zeroing swiglu_limit skips that branch entirely. It drops the SwiGLU clamp, so results
are numerically incomplete by construction -- the only question being asked is whether
determinism returns, which would close the localization.
"""
import sys
from pathlib import Path

F = Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/aiter.py")
s = F.read_text()

old = """        elif quant_info.swiglu_limit > 0:"""
new = """        elif quant_info.swiglu_limit > 0 and not get_bool_env_var(
            "SGLANG_DEBUG_FORCE_NO_SWIGLU_BRANCH"
        ):"""

if s.count(old) != 1:
    sys.exit(f"FAIL: expected 1 occurrence of the swiglu branch guard, found {s.count(old)}")
s = s.replace(old, new, 1)

if "get_bool_env_var" not in s.split("elif quant_info.swiglu_limit")[0]:
    anchor = "from sglang.srt.utils import "
    i = s.index(anchor)
    line_end = s.index("\n", i)
    line = s[i:line_end]
    if "get_bool_env_var" not in line:
        s = s[:line_end] + s[line_end:]
        s = s.replace(line, line.rstrip() + ", get_bool_env_var", 1)

F.write_text(s)
print("patched: SGLANG_DEBUG_FORCE_NO_SWIGLU_BRANCH now bypasses the gpt-oss branch")
