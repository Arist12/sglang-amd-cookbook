#!/usr/bin/env python3
"""Which MoE config difference could make GLM-5.3-Flash nondeterministic where
GLM-5.2-FP8 on the same aiter MoE runner is not."""
import glob
import json

g53_path = glob.glob(
    "/hf-cache/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/config.json"
)[0]
g53 = json.load(open(g53_path))["text_config"]
g52 = json.load(open("/models/GLM-5.2-FP8/config.json"))

KEYS = [
    "swiglu_limit",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "first_k_dense_replace",
    "hidden_size",
    "intermediate_size",
    "scoring_func",
    "topk_method",
    "norm_topk_prob",
    "routed_scaling_factor",
    "n_group",
    "topk_group",
    "hidden_act",
]

print(f"{'key':26s} {'GLM-5.3-Flash':>22s}   {'GLM-5.2-FP8':>22s}")
print("-" * 76)
for k in KEYS:
    a, b = g53.get(k, "-"), g52.get(k, "-")
    mark = "   <<< differs" if str(a) != str(b) else ""
    print(f"{k:26s} {str(a):>22s}   {str(b):>22s}{mark}")

print()
qa = g53.get("quantization_config") or json.load(open(g53_path)).get(
    "quantization_config"
)
qb = g52.get("quantization_config")
print("GLM-5.3 quantization_config:", qa)
print("GLM-5.2 quantization_config:", qb)
