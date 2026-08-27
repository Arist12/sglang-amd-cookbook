#!/usr/bin/env python3
"""Move the _fp8_mqa_logits helper below the import block.

Patch 01 inserted it right after the `if is_npu():` guard, which leaves module
imports on both sides of a function definition. Functionally irrelevant but it would
not survive upstream review.
"""
import re
import sys
from pathlib import Path

F = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py")
s = F.read_text()

start = s.index("_is_hip = is_hip()")
end = s.index("from sglang.srt.environ import envs")
block = s[start:end]
if "_fp8_mqa_logits" not in block:
    sys.exit("FAIL: helper is not where patch 01 put it; already moved?")

s = s[:start] + s[end:]

# reinsert after the last top-level import line
lines = s.split("\n")
last_import = max(
    i for i, ln in enumerate(lines)
    if re.match(r"^(from|import) \S", ln) or re.match(r"^\)$", ln)
)
insert_at = last_import + 1
lines[insert_at:insert_at] = ["", ""] + block.rstrip("\n").split("\n")
s = "\n".join(lines)

F.write_text(s)
print(f"moved helper below imports (line {insert_at})")
