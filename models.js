/* =====================================================================
   models.js — GENERATED, verified-only data.
   Source: adversarial extraction + verify workflow over the cookbook
   repo (playbooks, test scripts, index.html, grid_results). Every
   benchmark number is traced to a source file; un-measured cells are
   marked not-benchmarked. HW specs from AMD Instinct datasheets.
   Regenerate via the extraction workflow — do not hand-edit numbers.
   ===================================================================== */
window.HW = {
  "hardware": [
    {
      "name": "MI300X",
      "gfx": "gfx942",
      "arch": "CDNA3",
      "hbm_gb": 192,
      "hbm_type": "HBM3",
      "mem_bw_tbps": 5.325,
      "fp8_tflops": 2614.9,
      "bf16_tflops": 1307.4,
      "fp4_tflops": null,
      "sparsity_note": "Dense (without sparsity) values, per AMD Performance Labs footnote dated Nov 11 2023. CDNA3 matrix cores do not implement 2:4 structured sparsity acceleration, so these dense figures are the hardware peak. (Some OEM tables, e.g. Lenovo, list a second 'with sparsity' column of 5,220/2,610 FP8/BF16 that is a 2x marketing figure, not a real hardware sparsity mode for MI300X.) No native FP4 support on CDNA3.",
      "source": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf (corroborated by https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)"
    },
    {
      "name": "MI355X",
      "gfx": "gfx950",
      "arch": "CDNA4",
      "hbm_gb": 288,
      "hbm_type": "HBM3E",
      "mem_bw_tbps": 8,
      "fp8_tflops": 5033,
      "bf16_tflops": 2517,
      "fp4_tflops": 10066,
      "sparsity_note": "Dense (without sparsity) values, taken from the left column of AMD's MI355X AI peak-performance table. AMD's datasheet quotes dense/with-sparsity pairs: FP4 (MXFP4) 10,066/20,133, FP8 (MXFP8 & OCP-FP8) 5,033/10,066, BF16 2,517/5,033 TFLOPS. The headline marketing figures (e.g. 20 PFLOPS FP4, 10 PFLOPS FP8 per GPU) are the WITH-sparsity numbers; the dense values reported here are exactly half of those. CDNA4 supports FP6/FP4 (MXFP) datatypes natively.",
      "source": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf (corroborated by https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html and verbatim AMD datasheet mirror)"
    }
  ]
};

window.MODELS = [
  {
    "id": "glm-5.2-fp8",
    "name": "GLM-5.2-FP8",
    "family": "GLM",
    "hf_path": "zai-org/GLM-5.2-FP8",
    "architecture": "MoE + MLA with DeepSeek Sparse Attention (DSA), model_type=glm_moe_dsa (DeepSeek-V3.2 / GLM-5.1 architecture). 1M context window.",
    "precision": "FP8 (block-FP8 weights), bf16 KV cache",
    "status": "verified",
    "params_active": "39B",
    "params_total": "743B",
    "active_params_billions": 39,
    "bytes_per_param": 1,
    "weights_gb": 704,
    "context_len": "1048576",
    "summary": "Single-node TP=8 deployment of zai-org/GLM-5.2-FP8 (MoE + MLA/DSA, glm_moe_dsa) with SGLang and the DSA tilelang prefill+decode backend, verified on BOTH 8x MI300X (gfx942) and 8x MI355X (gfx950). FP8 weights (704 GB -> 88 GB/GPU) fit single-node; BF16 (~1.4 TB -> ~175 GB/GPU) does not. bf16 KV cache, chunked-prefill 8192, no MTP/speculative decoding on AMD. The launch command is IDENTICAL on both GPUs; gfx950 additionally requires TWO mandatory source patches first (the _use_aiter_bpreshuffle_gfx95=False flag in BOTH fp8_utils.py and models/deepseek_common/utils.py) or GSM8K collapses to ~0.0 (#28685, ROCm 7.2 kernel miscompile) -- gfx942 needs none of them. (A third, SGLANG_FP8_PAGED_MQA_LOGITS_TORCH, is already the correct default for GLM-5.2's GlmMoeDsaForCausalLM path and only matters if you also serve DeepSeek-V4 from the same tree.) Accuracy at parity on both (GSM8K 97.2% / 97.7%; AIME25 via sgl-eval pass@1 avg-of-16: MI300X 90.6%, MI355X 91.5% -- both within noise, both beat the 87.7% ref). MI355X is ~1.4-1.9x faster than MI300X (single-stream 67 vs 48 tok/s; c64 1009 vs 528 tok/s). Long-context: LongBench-v2 59.5%, near-flat decode TPOT at long ctx.",
    "configs": [
      {
        "gfx": "gfx942",
        "hw_name": "MI300X",
        "gpus": 8,
        "quant": "FP8 (block-FP8 MoE weights), bf16 KV cache",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": "rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620",
        "launch_python": "export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True\npython3 -m sglang.launch_server \\\n  --model-path zai-org/GLM-5.2-FP8 \\\n  --served-model-name glm-5.2 \\\n  --trust-remote-code \\\n  --tp 8 \\\n  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \\\n  --kv-cache-dtype bfloat16 \\\n  --chunked-prefill-size 8192 \\\n  --mem-fraction-static 0.85 \\\n  --cuda-graph-max-bs 64 --max-running-requests 64 \\\n  --watchdog-timeout 1200 \\\n  --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "DSA tilelang (--dsa-prefill-backend tilelang --dsa-decode-backend tilelang)",
        "moe_backend": null,
        "aiter": {
          "enabled": true,
          "commit": "7d604afe",
          "kernels": [
            "GEMM (BF16 tuned GEMM via glm5_bf16_tuned_gemm.csv)"
          ],
          "tuned_artifacts": [
            "glm5_bf16_tuned_gemm.csv"
          ],
          "summary": "AITER is enabled (SGLANG_USE_AITER=1) and ships a tuned BF16 GEMM table (glm5_bf16_tuned_gemm.csv); aiter pinned at commit 7d604afe in the container image. DSA attention itself runs through the tilelang backend, not AITER."
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER kernels; ships the glm5_bf16_tuned_gemm.csv tuned GEMM table."
          },
          {
            "key": "SGLANG_USE_ROCM700A",
            "value": "1",
            "why": "ROCm 7.x build flag set for this verified run on the rocm720 image."
          },
          {
            "key": "SGLANG_MOE_PADDING",
            "value": "1",
            "why": "MoE padding enabled for this run."
          },
          {
            "key": "PYTORCH_HIP_ALLOC_CONF",
            "value": "expandable_segments:True",
            "why": "Reduce HIP allocator fragmentation for the large MoE weights."
          },
          {
            "key": "PYTORCH_ROCM_ARCH",
            "value": "gfx942;gfx950",
            "why": "Image targets both MI300X and MI35x; gfx942 used here."
          }
        ],
        "accuracy": [
          {
            "name": "GSM8K",
            "value": "97.2%",
            "note": "n=1319, chat+thinking; in-tree run_eval --eval-name gsm8k --thinking-mode glm-45 --max-tokens 8192 --temperature 0. Parity, FP8 numerics healthy on gfx942.",
            "ref": "98.2% (cookbook ref)"
          },
          {
            "name": "AIME25",
            "value": "90.6%",
            "note": "pass@1 avg-of-16 via sgl-eval (NV official harness), 95% CI 88.6-92.6; pass@16 100%, majority@16 96.7%, truncated 0%. CAVEAT: in-tree run_eval reports only 62.5% on the same model/server due to a strict Answer: first-match regex -- harness artifact, not the model. Always use sgl-eval for AIME-style answer-extraction evals. Run with --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking.",
            "ref": "87.7% (cookbook ref) -- near parity within noise"
          },
          {
            "name": "LongBench-v2",
            "value": "59.5%",
            "note": "subset, ~64k-tok cap; in-tree run_eval --eval-name longbench_v2 --thinking-mode glm-45 --num-examples 50 --max-context-length 256000. Beats human and o1-preview on the subset.",
            "ref": "human 53.7%, o1-preview 57.7%, best direct model 50.1%"
          },
          {
            "name": "KernelBench (Triton kernel-gen)",
            "value": "92/250 correct (37%)",
            "note": "Capability eval, NOT model QA: GLM-5.2 writes Triton GPU kernels for KernelBench PyTorch programs and they are compiled + correctness-checked on MI300X (gfx942), one-shot/greedy/1 sample. compiled 178/250 (71%). Per level (compiled / correct): L1 84/51, L2 78/35, L3 16/6. fast_1=0 -- no one-shot kernel beats PyTorch eager (rocBLAS/aten). Required a code-extraction fix (pick the ModelNew block, not the first code block); the buggy first pass scored only 50/250 (20%). Full report + scripts on branch jhinpan/KernelBench@MI300-GLM5.2.",
            "ref": "buggy-extraction baseline 20%; one-shot is a floor (no pass@k / self-repair)"
          }
        ],
        "benchmarks": [
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "prefill_tok_s": 6170,
            "decode_tok_s": 51,
            "tpot_ms": 19.6,
            "total_tok_s": 430.7,
            "source": "glm52_fp8_playbook.md (bench_one_batch, bs=1)",
            "tok_s_per_gpu": null,
            "ttft_ms": null
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "total_tok_s": 47.8,
            "tok_s_per_gpu": 6,
            "tpot_ms": 18.9,
            "ttft_ms": 1458,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 16,
            "total_tok_s": 311.5,
            "tok_s_per_gpu": 38.9,
            "tpot_ms": 41.4,
            "ttft_ms": 10226,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 64,
            "total_tok_s": 528.1,
            "tok_s_per_gpu": 66,
            "tpot_ms": 72.7,
            "ttft_ms": 15849,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 32768,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 19.5,
            "ttft_ms": 4600,
            "total_tok_s": 38.7,
            "source": "glm52_fp8_playbook.md (long-context)"
          },
          {
            "isl": 131072,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 21.5,
            "ttft_ms": 23600,
            "total_tok_s": 19,
            "source": "glm52_fp8_playbook.md (long-context)"
          },
          {
            "isl": 262144,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 24.2,
            "ttft_ms": 43800,
            "total_tok_s": 11.5,
            "source": "glm52_fp8_playbook.md (long-context)"
          }
        ],
        "vs_nvidia": [
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 1,
            "ttft_ms": 1458,
            "tpot_ms": 18.9,
            "tok_s_per_gpu": 6,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 16,
            "ttft_ms": 10226,
            "tpot_ms": 41.4,
            "tok_s_per_gpu": 38.9,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 64,
            "ttft_ms": 15849,
            "tpot_ms": 72.7,
            "tok_s_per_gpu": 66,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "H200",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 662,
            "tpot_ms": 3.03,
            "tok_s_per_gpu": 34,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "H200",
            "strategy": "low-latency",
            "concurrency": 16,
            "ttft_ms": 5080,
            "tpot_ms": 12.44,
            "tok_s_per_gpu": 113,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "H200",
            "strategy": "balanced",
            "concurrency": 64,
            "ttft_ms": 8013,
            "tpot_ms": 25.57,
            "tok_s_per_gpu": 219,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "B300",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 503,
            "tpot_ms": 3.24,
            "tok_s_per_gpu": 34,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "B300",
            "strategy": "balanced",
            "concurrency": 64,
            "ttft_ms": 6465,
            "tpot_ms": 23.36,
            "tok_s_per_gpu": 245,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "GB300",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 393,
            "tpot_ms": 2.78,
            "tok_s_per_gpu": 79,
            "speculative": "EAGLE MTP"
          }
        ],
        "gotchas": [
          "KV-cache dtype must be bfloat16 with the DSA tilelang backend -- FP8 KV is incompatible (SGLang auto-selects bf16 and warns if --kv-cache-dtype is omitted).",
          "Keep prefill chunked (--chunked-prefill-size 8192). An unchunked long-context prefill trips the tilelang DSA prefill tile limit: 'RuntimeError: tensor a (16384) must match b (131072)'. Raising the chunk (within the tile limit) improves TTFT.",
          "No MTP / speculative decoding on AMD -- omit all --speculative-* flags. The NV recipe's EAGLE MTP is not enabled on AMD, so AMD decode trails NV ~3-4x.",
          "gfx942 (MI300X) is NOT affected by the block-FP8 GEMM accuracy bug from PR #28471 -- that bug is gfx950 (MI350X/MI355X) ONLY. GSM8K 97.2% confirms healthy numerics on gfx942.",
          "AIME-style accuracy: ALWAYS use sgl-eval (NV official harness), not in-tree run_eval. In-tree run_eval reports only 62.5% vs sgl-eval's 90.6% on the same server, due to its strict ANSWER_PATTERN first-match regex grabbing intermediate 'Answer:' from the reasoning trace.",
          "GLM-5.2 is a thinking model defaulting to effective_reasoning_effort='max' -- budget tokens accordingly (use --max-tokens 64000 for AIME).",
          "bench_one_batch runs the whole batch x input_len as one unchunked forward, so for this DSA model it is bs=1-only at ISL 8192; higher bs hits the tilelang prefill tile limit. Batched throughput must be measured against the live server with chunked prefill.",
          "TTFT in the throughput table is pessimistic because chunked-prefill-size=8192 serializes prefills; decode metrics (TPOT, tok/s) are representative.",
          "BF16 (~1.4 TB -> ~175 GB/GPU) does not fit single-node on MI300X (only MI325X/MI355X). Use FP8 (704 GB -> 88 GB/GPU). No FP4 checkpoint exists.",
          "No DP-attention / DeepEP in this verified config (NV balanced/high-throughput recipes add --dp 8 --enable-dp-attention --moe-a2a-backend deepep); left out for a first-known-good MI300X config."
        ],
        "provenance": {
          "image": "rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620",
          "pr": "https://github.com/sgl-project/sglang/pull/28471",
          "sglang": "0.5.13.post1.dev20260621 (g3975ea5ac7); built from source @ a51d56d948",
          "aiter": "7d604afe",
          "rocm": "7.2.0",
          "date": "2026-06-20/2026-06-21",
          "node": "8x MI300X (gfx942), 192 GiB each; PyTorch 2.9.1+rocm7.2.0; tilelang 0.1.7.post3"
        }
      },
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "FP8 (block-FP8 MoE weights), bf16 KV cache",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": null,
        "launch_python": "# ============================================================================\n# STEP 1 (gfx950 ONLY) — apply TWO MANDATORY source patches before first launch.\n# Without them GSM8K collapses to ~0.0 with token-garbage. MI300X/gfx942 needs\n# NONE of these (see the MI300X tab: identical launch flags, no patches).\n# Editable install => patches take effect at next process start, no rebuild.\n# ============================================================================\nSRT=$(python3 -c 'import os,sglang.srt as m; print(os.path.dirname(m.__file__))')\n# Patch #1  block-FP8 bpreshuffle GEMM is miscompiled on ROCm 7.2 (sglang #28685).\nsed -i 's/^_use_aiter_bpreshuffle_gfx95 = .*/_use_aiter_bpreshuffle_gfx95 = False  # FORCED OFF (gfx950 #28685)/' \\\n  \"$SRT/layers/quantization/fp8_utils.py\"\n# Patch #1b SAME flag, SECOND definition. GLM-5.2 is GlmMoeDsaForCausalLM ->\n#          DeepseekV2ForCausalLM, and its MLA/activation-quant path reads the flag\n#          from deepseek_common/utils.py, NOT from fp8_utils.py. Patching only #1\n#          leaves GSM8K=0.0 -- THIS is the easy-to-miss one.\nsed -i 's/^_use_aiter_bpreshuffle_gfx95 = .*/_use_aiter_bpreshuffle_gfx95 = False  # FORCED OFF (gfx950 #28685)/' \\\n  \"$SRT/models/deepseek_common/utils.py\"\n# Verify BOTH say False (the fix is the value, not just the name):\ngrep -n '^_use_aiter_bpreshuffle_gfx95' \"$SRT/layers/quantization/fp8_utils.py\" \"$SRT/models/deepseek_common/utils.py\"\n\n# ----------------------------------------------------------------------------\n# OPTIONAL (only if you ALSO serve DeepSeek-V4 from this same tree):\n# SGLANG_FP8_PAGED_MQA_LOGITS_TORCH must be False for long-context correctness.\n# For GLM-5.2 this is ALREADY the default -- the .set(True) line lives in the\n# DeepseekV4ForCausalLM branch of server_args.py, and GLM-5.2's model_arch is\n# GlmMoeDsaForCausalLM, so it never runs and the global default (False) holds.\n# Long-context needle-recall verified PASS on the default, no patch. Apply the\n# env-aware patch below ONLY for DeepseekV4 (it is a no-op for GLM-5.2):\n#   python3 - \"$SRT/server_args.py\" <<'PY'\n#   import sys; p=sys.argv[1]; L=open(p).read().split(chr(10)); hip=False\n#   for i,l in enumerate(L):\n#       s=l.strip()\n#       if s=='elif is_hip():': hip=True; continue\n#       if hip and (s.startswith('elif ') or s.startswith('else:')): hip=False\n#       if hip and 'SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)' in l:\n#           ind=l[:len(l)-len(l.lstrip())]\n#           L[i]=ind+'if not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.is_set():'+chr(10)+ind+'    envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(False)'\n#           break\n#   open(p,'w').write(chr(10).join(L))\n#   PY\n\n# ============================================================================\n# STEP 2 — launch (TP=8, DSA tilelang). The launch FLAGS are IDENTICAL to the\n# MI300X tab; SGLANG_USE_AITER=1 is also set there (via the container env /\n# test_glm52_fp8.sh wrapper) -- set it explicitly here to be self-contained.\n# ============================================================================\nexport SGLANG_USE_AITER=1\nexport PYTORCH_HIP_ALLOC_CONF=expandable_segments:True\npython3 -m sglang.launch_server \\\n  --model-path zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 \\\n  --trust-remote-code --tp 8 \\\n  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \\\n  --kv-cache-dtype bfloat16 --chunked-prefill-size 8192 \\\n  --mem-fraction-static 0.85 --cuda-graph-max-bs 64 --max-running-requests 64 \\\n  --watchdog-timeout 1200 --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "DSA tilelang (prefill+decode)",
        "moe_backend": null,
        "aiter": {
          "enabled": true,
          "commit": "7d604afe5",
          "kernels": [
            "GEMM (glm5_bf16_tuned_gemm.csv)"
          ],
          "tuned_artifacts": [
            "glm5_bf16_tuned_gemm.csv"
          ],
          "summary": "AITER enabled (SGLANG_USE_AITER=1). The gfx950 bpreshuffle block-FP8 GEMM path is force-disabled in source in BOTH definition sites (miscompiled on ROCm 7.2, #28685); falls back to the correct ck_gemm_a8w8_blockscale (verified cosine 1.0 on gfx950). DSA attention runs via tilelang."
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "AITER kernels; ships glm5_bf16_tuned_gemm.csv"
          },
          {
            "key": "PYTORCH_HIP_ALLOC_CONF",
            "value": "expandable_segments:True",
            "why": "reduce HIP allocator fragmentation for large MoE weights"
          }
        ],
        "accuracy": [
          {
            "name": "GSM8K",
            "value": "97.7%",
            "note": "n=1319, chat+thinking; run_eval --eval-name gsm8k --thinking-mode glm-45 --max-tokens 8192 --temperature 0. REQUIRES the two mandatory gfx950 bpreshuffle patches (fp8_utils.py AND models/deepseek_common/utils.py). Without the SECOND one (deepseek_common/utils.py, which GLM-5.2's GlmMoeDsaForCausalLM->DeepseekV2ForCausalLM MLA/quant path actually reads), GSM8K collapses to 0.0 with token-garbage under batching. n=100 subset scored 98.0%.",
            "ref": "98.2% (cookbook ref); MI300X/gfx942 re-run 97.2% — gfx950 reaches parity once both bpreshuffle sites are fixed"
          },
          {
            "name": "AIME25",
            "value": "91.5%",
            "note": "pass@1 avg-of-16 via sgl-eval (NV official harness), n=30x16=480 samples; 95% CI 89.1-93.8 (pass@1 91.46% +/- 1.96*SEM, SEM 1.22%, std 4.86%). pass@16 100%, majority@16 93.3%, truncated 0.21%, error 0%, 9.46M completion tokens. Requires the two mandatory gfx950 bpreshuffle patches. Run with --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking. CAVEAT: always use sgl-eval, NOT in-tree run_eval (its strict Answer: first-match regex badly undercounts this thinking model). gfx950 reaches parity with gfx942 (90.6%) once the bpreshuffle sites are fixed.",
            "ref": "87.7% (cookbook ref) -- beats ref; MI300X/gfx942 re-run 90.6% -- gfx950 at parity within noise"
          }
        ],
        "benchmarks": [
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "total_tok_s": 66.92,
            "tok_s_per_gpu": 8.4,
            "tpot_ms": 14.43,
            "ttft_ms": 652.13,
            "source": "MI355X re-run (bench_serving, conc=1; output verified correct, GSM8K 97.7%)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 16,
            "total_tok_s": 535.66,
            "tok_s_per_gpu": 67.0,
            "tpot_ms": 25.22,
            "ttft_ms": 4790.27,
            "source": "MI355X re-run (bench_serving, conc=16; output verified correct)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 64,
            "total_tok_s": 1008.95,
            "tok_s_per_gpu": 126.1,
            "tpot_ms": 41.45,
            "ttft_ms": 13794.52,
            "source": "MI355X re-run (bench_serving, conc=64; output verified correct)"
          },
          {
            "isl": 8192,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 14.43,
            "ttft_ms": 410.76,
            "total_tok_s": 65.67,
            "source": "MI355X re-run (long-context, conc=1)"
          },
          {
            "isl": 32768,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 14.89,
            "ttft_ms": 1180.39,
            "total_tok_s": 58.19,
            "source": "MI355X re-run (long-context, conc=1)"
          },
          {
            "isl": 131072,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 16.63,
            "ttft_ms": 5457.26,
            "total_tok_s": 36.66,
            "source": "MI355X re-run (long-context, conc=1)"
          },
          {
            "isl": 262144,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 18.89,
            "ttft_ms": 9584.27,
            "total_tok_s": 26.59,
            "source": "MI355X re-run (long-context, conc=1)"
          }
        ],
        "vs_nvidia": [],
        "gotchas": [
          "CRITICAL gfx950 fix (TWO mandatory sites): _use_aiter_bpreshuffle_gfx95 must be forced False in BOTH (1) python/sglang/srt/layers/quantization/fp8_utils.py and (2) python/sglang/srt/models/deepseek_common/utils.py. ROCm 7.2 hipcc miscompiles aiter gemm_a8w8_blockscale_bpreshuffle (drops -mllvm -amdgpu-coerce-illegal-types), #28685. GLM-5.2 is GlmMoeDsaForCausalLM -> DeepseekV2ForCausalLM (models/glm4_moe.py), and its MLA/activation-quant path reads the flag from deepseek_common/utils.py, so patching ONLY fp8_utils.py leaves the model forward path broken -> GSM8K 0.0 and batched-decode token-garbage. With BOTH patched: GSM8K 97.7%, batched output correct. gfx942/MI300X is unaffected (never uses the bpreshuffle path).",
          "The miscompiled bpreshuffle kernel is M-tile-sensitive (wrong rows cluster at 16-row tile boundaries, shift between runs), so single-token-answer bs=1 prompts can look correct while batched/longer prompts corrupt -- it is ONE bug, not a separate batching bug.",
          "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH must be False for long-context correctness -- but for GLM-5.2 this is ALREADY the default and needs NO patch. The .set(True) line in server_args.py is inside the `model_arch in [DeepseekV4ForCausalLM]` branch (~line 3786); GLM-5.2's model_arch is GlmMoeDsaForCausalLM, so that branch never runs and the global EnvBool default (False, environ.py) holds. Verified: long-context needle-recall PASSED on the default with no patch. The env-aware server_args.py patch only matters if you ALSO serve DeepSeek-V4 from the same tree (where .set(True) would otherwise clobber a shell export). Do NOT list it as a GLM-5.2 requirement.",
          "KV-cache dtype must be bfloat16 with the DSA tilelang backend (FP8 KV incompatible).",
          "Keep --chunked-prefill-size 8192 (unchunked long prefill trips the tilelang DSA tile limit).",
          "No MTP / speculative decoding on AMD/gfx950.",
          "Upstream permanent fix is the CK kernel rewrite ROCm/rocm-libraries#8639 (scalar-FMA/VGPR accumulator), which supersedes the disable workaround; not in aiter 7d604afe5, so the source workaround is required for now.",
          "PERF: MI355X single-stream 67 tok/s and long-ctx TPOT near-flat 14.4->18.9ms (8k->256k) BEAT MI300X (48 tok/s, 18.9->24.2ms); c64 throughput 1009 tok/s vs MI300X 528."
        ],
        "provenance": {
          "image": null,
          "pr": "https://github.com/sgl-project/sglang/pull/28471",
          "sglang": "0.5.13.post1.dev20260622 (g4923bb93ae), editable + 2 mandatory gfx950 bpreshuffle disables (fp8_utils.py + deepseek_common/utils.py); MQA-logits env left at its correct GLM-5.2 default (False)",
          "aiter": "7d604afe5; SGLANG_USE_AITER=1; bpreshuffle gfx95 path disabled in source (both sites)",
          "rocm": "7.2.0",
          "date": "2026-06-24",
          "node": "mia1-p02-g45, 8x MI355X (gfx950), 288 GiB each; PyTorch 2.9.1+rocm7.2.0"
        }
      }
    ],
    "gaps": [
      {
        "title": "balanced / high-throughput",
        "kind": "strategy",
        "note": "Match the NV balanced recipe: add DP-attention + DeepEP, then sweep concurrency. Relaunch with --dp 8 --enable-dp-attention --moe-a2a-backend deepep, then:",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "MTP / speculative decode",
        "kind": "dependency",
        "note": "Not enabled on AMD for glm_moe_dsa yet — this is most of the ~3–4× decode gap vs NVIDIA. Tracking upstream; no AMD script until EAGLE/MTP lands on ROCm.",
        "cmd": null
      }
    ]
  },
  {
    "id": "glm-5-fp8",
    "name": "GLM-5-FP8",
    "family": "GLM",
    "hf_path": "zai-org/GLM-5-FP8",
    "architecture": "MoE + Native Sparse Attention (NSA); model_type glm_moe_dsa (described as DeepSeek-V2 architecture in the section). 744B total params, 40B active. NSA backend = tilelang (prefill + decode).",
    "precision": "FP8",
    "status": "not_benchmarked",
    "summary": "FP8 MoE + Native Sparse Attention model (744B total / 40B active), served on 8x MI355X (gfx950) via SGLang TP=8 with tilelang NSA prefill/decode backends. Cookbook gives copy-paste launch commands (TP=8 recommended, TP=4 alternative) plus a verification curl, but contains no measured benchmark, accuracy, or vs-NVIDIA numbers, so nothing is verified.",
    "params_total": "744B",
    "params_active": "40B",
    "active_params_billions": 40,
    "bytes_per_param": 1,
    "weights_gb": 705,
    "context_len": null,
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "FP8",
        "strategy": "balanced",
        "nodes": "single",
        "verified": false,
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path zai-org/GLM-5-FP8 \\\n    --served-model-name glm-5-fp8 \\\n    --tp 8 \\\n    --tool-call-parser glm47 \\\n    --reasoning-parser glm45 \\\n    --mem-fraction-static 0.80 \\\n    --nsa-prefill-backend tilelang \\\n    --nsa-decode-backend tilelang \\\n    --chunked-prefill-size 131072 \\\n    --watchdog-timeout 1200 \\\n    --port 30000",
        "aiter": {
          "enabled": false,
          "summary": "No AITER usage in the GLM-5-FP8 section. SGLANG_USE_AITER is not set, no aiter commit hash, no tuned GEMM/MoE artifacts. Attention is NSA via the tilelang backend (prefill + decode); no AITER MLA/GEMM/MoE/attention kernels are invoked for this config.",
          "commit": null,
          "kernels": [],
          "tuned_artifacts": []
        },
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "tilelang (NSA prefill + decode)",
        "moe_backend": null,
        "docker_image": null,
        "env": [],
        "benchmarks": [],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "GLM-5 uses a glm_moe_dsa model_type that stock HuggingFace Transformers does not recognize natively; it is registered in SGLang's config loader. Ensure your SGLang build includes the fix from PR #18911 (Day-0 PR). Also pip install --upgrade transformers for GLM-5 tokenizer support.",
          "TP=8 is strongly recommended. TP=4 (HIP_VISIBLE_DEVICES=0,1,2,3) is a tight fit -- 705 GB model in 1,152 GB -- so drop --mem-fraction-static to 0.60 to leave KV-cache room and add --disable-cuda-graph."
        ],
        "provenance": {
          "image": null,
          "pr": "#18911",
          "sglang": "v0.5.8-v0.5.10rc (cookbook-wide range; not pinned in-section)",
          "aiter": null,
          "rocm": "ROCm 7.0-7.2 (cookbook-wide range; not pinned in-section)",
          "date": null,
          "node": "8x MI355X"
        }
      }
    ],
    "gaps": [
      {
        "title": "Latency (BS=1)",
        "kind": "metric",
        "note": "Documented launch, zero measured numbers. Bring the server up (TP=8 command above) then:",
        "cmd": "# single-request latency (offline)\npython3 -m sglang.bench_one_batch_server \\\n  --model-path zai-org/GLM-5-FP8 --base-url http://127.0.0.1:30000 \\\n  --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \\\n  --dataset-name random --skip-warmup"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Concurrency 1/16/64 against the live server.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      }
    ]
  },
  {
    "id": "deepseek-v4-flash-fp8",
    "name": "DeepSeek-V4-Flash-FP8",
    "family": "DeepSeek",
    "hf_path": "sgl-project/DeepSeek-V4-Flash-FP8",
    "architecture": "MoE + Compressed MLA/MQA (num_key_value_heads=1), 256 routed experts, 43 layers, 1M context, native FP8 e4m3",
    "precision": "FP8 (e4m3)",
    "status": "verified",
    "params_total": null,
    "params_active": null,
    "active_params_billions": null,
    "bytes_per_param": 1,
    "weights_gb": 274,
    "context_len": "1M (max_model_len=1048576)",
    "summary": "DeepSeek-V4-Flash-FP8 (FP8 MoE, 256 routed experts, MQA/Compressed MLA with 1 KV head, 43 layers, 274 GiB weights) served on 8x MI355X (gfx950) via SGLang PR #23608. The \"correctness first\" config runs with every JIT fast-path disabled: torch-reference FlashMLA, Triton-forced MoE-FP8, CUDA graph off, radix cache off, 11 SGLANG_OPT_USE_*=false. Decode is flat at ~4 tok/s at BS=1 because attention is tiny (1 KV head) and the bottleneck is per-token MoE-FP8 matmul plus the pure-PyTorch FlashMLA reference; TP=8 does not help a BS=1 workload since per-expert MoE work is already small. Verified ISL/OSL latency sweep measured with bench_one_batch_server at TP=8 DP=8, BS=1.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "FP8 (e4m3) weights; MoE via Triton FP8; kv-cache silently fp8_e4m3",
        "strategy": "low-latency",
        "verified": true,
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": 8
        },
        "attention_backend": "compressed (torch FlashMLA reference, SGLANG_HACK_FLASHMLA_BACKEND=torch)",
        "moe_backend": "Triton FP8 (SGLANG_FORCE_TRITON_MOE_FP8=1)",
        "docker_image": "sglang-dsv4-mi355x:flash-r1",
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /hf-cache/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17 \\\n    --served-model-name dsv4-flash \\\n    --trust-remote-code \\\n    --tp 8 --dp 8 --enable-dp-attention \\\n    --disable-radix-cache --attention-backend compressed \\\n    --max-running-requests 256 --page-size 256 --chunked-prefill-size 8192 \\\n    --kv-cache-dtype auto \\\n    --host 0.0.0.0 --port 31000 \\\n    --disable-shared-experts-fusion --disable-cuda-graph \\\n    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4",
        "aiter": {
          "enabled": true,
          "commit": null,
          "kernels": [],
          "tuned_artifacts": [],
          "summary": "SGLANG_USE_AITER=1 is set, but in this correctness-first config AITER does not provide the hot kernels: MoE-FP8 is forced to Triton (SGLANG_FORCE_TRITON_MOE_FP8=1), FlashMLA is forced to a pure-PyTorch reference (SGLANG_HACK_FLASHMLA_BACKEND=torch), and 11 SGLANG_OPT_USE_*=false switches disable every JIT fast-path. No aiter commit hash or tuned GEMM/MoE artifacts are recorded in any source."
        },
        "env": [
          {
            "key": "CUDA_VISIBLE_DEVICES",
            "value": "0,1,2,3,4,5,6,7",
            "why": "All 8 MI355X GPUs for the TP=8 DP=8 benchmarked topology (script default is 0,1,2,3 for TP=4)"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_COMPRESS",
            "value": "false",
            "why": "Disable fused compress JIT fast-path (correctness-first, no CUDA toolchain port)"
          },
          {
            "key": "SGLANG_OPT_USE_OLD_COMPRESSOR",
            "value": "true",
            "why": "Use the old non-JIT compressor path"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_SWA_PREPARE",
            "value": "false",
            "why": "Disable TileLang SWA-prepare JIT kernel"
          },
          {
            "key": "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK",
            "value": "false",
            "why": "Disable JIT fused top-k kernel"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_HASH_TOPK",
            "value": "false",
            "why": "Disable fused hash top-k kernel"
          },
          {
            "key": "SGLANG_HACK_FLASHMLA_BACKEND",
            "value": "torch",
            "why": "Force pure-PyTorch FlashMLA reference instead of a JIT/CK kernel (correctness, slow)"
          },
          {
            "key": "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
            "value": "false",
            "why": "Disable DeepGEMM HC prenorm fast-path"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_MHC_PRE",
            "value": "false",
            "why": "Disable TileLang MHC pre JIT kernel"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_MHC_POST",
            "value": "false",
            "why": "Disable TileLang MHC post JIT kernel"
          },
          {
            "key": "SGLANG_ENABLE_THINKING",
            "value": "1",
            "why": "Thinking mode on; model emits <think>...</think>answer"
          },
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER (though MoE/MLA are forced to Triton/torch here)"
          },
          {
            "key": "SGLANG_USE_ROCM700A",
            "value": "1",
            "why": "ROCm 7.0.0-alpha path selection for the MI355X base image"
          },
          {
            "key": "SGLANG_TOPK_TRANSFORM_512_TORCH",
            "value": "1",
            "why": "Use torch path for the 512-wide top-k transform"
          },
          {
            "key": "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
            "value": "1",
            "why": "Use torch path for FP8 paged MQA logits"
          },
          {
            "key": "SGLANG_DSV4_FP4_EXPERTS",
            "value": "false",
            "why": "Do not use FP4 experts; keep FP8 experts"
          },
          {
            "key": "SGLANG_OPT_DPSK_V4_RADIX",
            "value": "0",
            "why": "Disable V4 radix optimization (compressed-attention radix not stable on HIP)"
          },
          {
            "key": "SGLANG_OPT_USE_OVERLAP_STORE_CACHE",
            "value": "false",
            "why": "Disable overlap store-cache fast-path"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_STORE_CACHE",
            "value": "false",
            "why": "Disable fused store-cache fast-path"
          },
          {
            "key": "SGLANG_FORCE_TRITON_MOE_FP8",
            "value": "1",
            "why": "Force Triton MoE-FP8 kernel (the AMD-available MoE path)"
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Use local HF cache mount, no network"
          }
        ],
        "benchmarks": [
          {
            "isl": 2048,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 5470,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 374.4,
            "total_tok_s": 11.71,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 2048,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 5360,
            "decode_tok_s": 3.97,
            "prefill_tok_s": 382.1,
            "total_tok_s": 7.86,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 4096,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 9530,
            "decode_tok_s": 4.02,
            "prefill_tok_s": 429.9,
            "total_tok_s": 19.37,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 4096,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 10210,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 401.2,
            "total_tok_s": 11.74,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 20030,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 409,
            "total_tok_s": 33.28,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 8192,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 19490,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 420.3,
            "total_tok_s": 19.2,
            "source": "dsv4_flash_playbook.md"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "provenance": {
          "image": "sglang-dsv4-mi355x:flash-r1 (base rocm/sgl-dev:deepseek-v4-mi35x, digest sha256:a5f71877...)",
          "pr": "#23608",
          "sglang": "0.5.8.dev20260129+gf959851eb + PR #23608 head 26fbc935300a3bfba34f3dfa8925310929f82680 overlay + 2 AMD patches (drop @dataclass from deepseek_v4.py, stub kernelkit/bench.py)",
          "aiter": null,
          "rocm": "ROCm 7.0 (SGLANG_USE_ROCM700A=1)",
          "date": "April 2026",
          "node": "mia1-p02-g45 (container dsv4-flash-tp8), 8x MI355X"
        },
        "gotchas": [
          "Decode is flat at ~4 tok/s across all ISL/OSL at BS=1 because V4-Flash is MQA (num_key_value_heads=1) so attention is tiny regardless of context; the real bottleneck is per-token MoE-FP8 matmul plus the pure-PyTorch torch-reference FlashMLA. TP=8 gives the same decode speed as TP=4 because per-expert MoE work is already small and all-to-all overhead eats any TP gain. This is the correctness-first config, not a tuned one.",
          "--enable-dp-attention is MANDATORY: V4 uses MQA (1 KV head) so attention cannot be TP-sharded; DP-attention replicates attention across all GPUs.",
          "DP-attention in this image silently auto-lowers --chunked-prefill-size from 8192 to 1024 to dodge an MoE-kernel sizing issue (warning at server_args.py:2057); every benchmark row actually ran at chunked_prefill_size=1024.",
          "--kv-cache-dtype auto is silently overridden to fp8_e4m3 for V4 (server_args.py:1193); the 'no scaling factors provided' warning is cosmetic for short contexts.",
          "The deepseek-v4 reasoning parser does not split <think>...</think> from the answer; both end up in choices[0].message.content and reasoning_content is always empty.",
          "Two AMD-side patches are REQUIRED on top of PR #23608 head 26fbc93 (baked into Dockerfile.dsv4): (1) drop @dataclass from python/sglang/srt/configs/deepseek_v4.py or import fails with TypeError because PretrainedConfig's metaclass strips field defaults; (2) create a stub for python/sglang/srt/flashmla_tests/kernelkit/bench.py (NotImplementedError bench_by_cuda_events / bench_kineto) because kernelkit/__init__.py unconditionally imports it but no file exists in the image or PR branch.",
          "Order-of-magnitude speed-ups require: re-enabling JIT kernels once their ROCm ports land in PR #23608, larger batch sizes (MoE amortizes across tokens, BS=1 is worst case), radix cache re-enabled once compressed-attention radix is stable on HIP, and CUDA-graph replay for decode (ROCm HIP-graph needs driver support).",
          "Served-model-name is 'dsv4-flash'; pass that as the model field in chat calls, not the HF repo id.",
          "Total and active parameter counts are NOT documented in any cookbook source; only 256 routed experts / 43 layers / 1 KV head / FP8 e4m3 / 274 GiB weights are stated. Any '~430B total, ~17B active' figure is unverified and was removed from this record."
        ]
      }
    ],
    "gaps": [
      {
        "title": "Tuned (not correctness-first) config",
        "kind": "perf",
        "note": "Current config disables every JIT fast-path (~4 tok/s decode). Re-enable kernels as their ROCm ports land in PR #23608, then re-bench latency + throughput.",
        "cmd": "# single-request latency (offline)\npython3 -m sglang.bench_one_batch_server \\\n  --model-path sgl-project/DeepSeek-V4-Flash-FP8 --base-url http://127.0.0.1:31000 \\\n  --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \\\n  --dataset-name random --skip-warmup"
      },
      {
        "title": "Batch > 1 throughput",
        "kind": "metric",
        "note": "BS=1 is the worst case for MoE; sweep concurrency where per-expert work amortizes.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 31000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy measured.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 31000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:31000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      }
    ]
  },
  {
    "id": "qwen35-397b-a17b",
    "name": "Qwen3.5-397B-A17B",
    "family": "Qwen",
    "hf_path": "Qwen/Qwen3.5-397B-A17B",
    "architecture": "Mixture-of-Experts with hybrid DeltaNet attention: 397B total params, 17B active per token. 60 layers total = 45 DeltaNet recurrent (linear-attention) layers + 15 GQA layers. Served BF16.",
    "precision": "BF16",
    "status": "verified",
    "params_total": "397B",
    "params_active": "17B",
    "active_params_billions": 17,
    "bytes_per_param": 2,
    "weights_gb": 752,
    "context_len": null,
    "summary": "Qwen3.5-397B-A17B (397B MoE, 17B active) served BF16 at TP=8 on 8x MI355X (gfx950) via SGLang. Hybrid DeltaNet architecture: 45 recurrent linear-attention layers + 15 GQA layers, run with the triton attention backend (NOT aiter — the verified image is a fixed build that disables the broken aiter stub and patches quark imports). --max-mamba-cache-size 128 is the load-bearing flag (+45% perf vs the default 64) because of the 45 DeltaNet recurrent layers. SGLANG_ROCM_FUSED_DECODE_MLA must be forced to 0 (the base image ships it as 1, which crashes the triton backend with a ForwardMetadata unpacking error). Verified decode ~52-59 tok/s at BS=1 across input lengths 1k-16k. Sub-note: a kernel-engineering fine-tune ships as the JinnP/Qwen3.5-397B-A17B-LoRA-SFT-v4 LoRA adapter (rank 32 / alpha 64, 128.5M trainable params = 0.032%, 13 target module types, 270-example AMD-GPU-kernel-engineering dataset, best eval loss 0.0547 at epoch 8). SGLang runtime LoRA cannot serve it (--lora-paths fails at init_lora_shapes() because the adapter targets 6 unsupported modules across the DeltaNet layers + MoE shared_expert_gate; see sglang#9897). Workaround: merge the adapter offline with LLaMA-Factory (llamafactory-cli export merge_qwen35_lora.yaml, ~25 min on CPU, ~800 GB RAM, output 743 GB / 122 shards) and serve the merged checkpoint with the identical base launch command (just swap --model-path and --served-model-name to Qwen3.5-397B-A17B-SFT-v4). This is NOT a separate verified config — no independent benchmark numbers were measured for the merged model.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "BF16",
        "strategy": "low-latency",
        "verified": true,
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton",
        "moe_backend": null,
        "docker_image": "sglang-test:v0.5.9-rocm700-mi35x-20260310",
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /sgl-workspace/models/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/98d1a504ba52e88924b3a3a008447cf2fdbd518c \\\n    --served-model-name Qwen3.5-397B-A17B \\\n    --tp 8 \\\n    --trust-remote-code \\\n    --attention-backend triton \\\n    --mem-fraction-static 0.80 \\\n    --max-mamba-cache-size 128 \\\n    --reasoning-parser qwen3 \\\n    --tool-call-parser qwen3_coder \\\n    --watchdog-timeout 1200 \\\n    --host 0.0.0.0 \\\n    --port 30000",
        "aiter": {
          "enabled": false,
          "summary": "AITER is intentionally disabled for this config. The verified image (sglang-test:v0.5.9-rocm700-mi35x-20260310) is a fixed build produced by Dockerfile.bisect that disables the broken aiter stub and patches quark imports; attention runs on the triton backend instead. No aiter commit, tuned GEMM CSV, or MoE bucket config applies.",
          "commit": null,
          "kernels": [],
          "tuned_artifacts": []
        },
        "env": [
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "Base image ships this as 1, which crashes the triton attention backend with a ForwardMetadata unpacking error; must be forced to 0."
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Serve from locally mounted weights without contacting the HF Hub."
          }
        ],
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 58.9,
            "ttft_ms": 110,
            "source": "index.html"
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 59.17,
            "ttft_ms": 110,
            "source": "index.html"
          },
          {
            "isl": 8192,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 55.71,
            "ttft_ms": 310,
            "source": "index.html"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 56.68,
            "ttft_ms": 250,
            "source": "index.html"
          },
          {
            "isl": 16384,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 52.32,
            "ttft_ms": 510,
            "source": "index.html"
          },
          {
            "isl": 16384,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 53.79,
            "ttft_ms": 450,
            "source": "index.html"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "--max-mamba-cache-size 128 is critical for DeltaNet's 45 recurrent layers (+45% perf vs the default 64).",
          "SGLANG_ROCM_FUSED_DECODE_MLA must be set to 0 — the base image ships it as 1, which crashes the triton backend with a ForwardMetadata unpacking error.",
          "The stock rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260310 base image has a broken aiter stub; use the fixed build (Dockerfile.bisect) that disables aiter and patches quark imports.",
          "DeltaNet's 45 recurrent layers leak memory over long runtime, causing NCCL process-group failures after ~10-12 hours (sglang issue #20010 / PR #20182). Workaround: periodic docker restart qwen35-serve.",
          "Use --ulimit core=0:0 to prevent GPU core dumps (~200 GB each) from filling the disk on crash.",
          "CUDA graph is enabled (default) — left on for this verified config.",
          "Use max_tokens >= 512 when testing inference; it is a reasoning model.",
          "LoRA-SFT-v4: SGLang runtime LoRA (--lora-paths) fails at init_lora_shapes() because the adapter targets DeltaNet modules (in_proj_a/b/z/qkv, out_proj) and MoE shared_expert_gate that SGLang LoRA does not support (sglang#9897). Merge offline with LLaMA-Factory and serve the merged checkpoint instead."
        ],
        "provenance": {
          "image": "sglang-test:v0.5.9-rocm700-mi35x-20260310",
          "pr": null,
          "sglang": "v0.5.9",
          "aiter": "disabled (Dockerfile.bisect stub)",
          "rocm": "7.0",
          "date": "March 2026",
          "node": "8x MI355X"
        }
      }
    ],
    "gaps": [
      {
        "title": "Throughput / concurrency sweep",
        "kind": "metric",
        "note": "Only BS=1 latency is measured; add the online sweep.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Merged LoRA-SFT-v4 numbers",
        "kind": "variant",
        "note": "The kernel-engineering adapter cannot serve via runtime LoRA; merge it offline, then benchmark the merged checkpoint with the identical launch (swap --model-path / --served-model-name).",
        "cmd": "llamafactory-cli export merge_qwen35_lora.yaml   # ~25 min CPU, ~800 GB RAM, 743 GB out\n# then relaunch with the merged path and re-run the latency + accuracy commands above"
      }
    ]
  },
  {
    "id": "kimi-k3",
    "name": "Kimi-K3",
    "family": "Moonshot",
    "hf_path": "moonshotai/Kimi-K3",
    "architecture": "Hybrid 93-layer MoE: 69 KDA linear-attention layers + 24 full MLA layers, 896 routed experts (top-16) + 2 shared, `situ` activation, 1M context, vision tower (KimiK3ForConditionalGeneration / text_config model_type kimi_linear)",
    "precision": "MXFP4 routed experts + bf16 everything else (effective 1.31 bytes/param on the active set)",
    "status": "verified",
    "params_active": "105.4B",
    "params_total": "2.78T",
    "active_params_billions": 105.4,
    "bytes_per_param": 1.31,
    "weights_gb": 1561,
    "context_len": "1048576 (measured to 131072)",
    "summary": "Day-0 bring-up of moonshotai/Kimi-K3 (2.78T total / 105.4B active, hybrid KDA + MLA, 896 routed experts) on 8x MI355X at TP=8, in both the plain and the DSpark speculative-decoding configuration from sgl-project/sglang#32548. Accuracy is at parity between the two and healthy in absolute terms: GSM8K 97.49% / 97.64%, AIME25 pass@1 avg-of-8 93.33% / 94.58% - speculative decoding is lossless, as it should be. Everything else about DSpark is conditional, and the condition is accept length. It doubles single-stream decode on short greedy prompts (51.40 -> 104.00 tok/s, accept 5.95 on GSM8K) and is 1.54x faster over a full GSM8K run; it is 3.45x slower on sampled AIME25 at 48 concurrent (accept ~2.9) and 10x slower at 131k context (accept 1.2), where the plain config holds TPOT almost flat at 22 ms because only 24 of 93 layers carry a growing KV cache. The 1.56 TB checkpoint lands at 194.38 GB/GPU under the aiter MXFP4 path, which is mandatory rather than a tuning knob. Serving DSpark with any non-greedy sampling needs dspark_rocm_renorm.patch, without which the first top_p batch takes the scheduler down on ROCm. A launch-parameter search on 2026-07-29 then found the Day-0 recipe was leaving a lot on the table, and two knobs recover almost all of it: --mem-fraction-static 0.93 instead of 0.85 lifts the non-spec KV pool 54% and takes peak throughput from 6198 to 7892 tok/s (+27%, 987 tok/s/GPU at concurrency 128), because 0.85 left 35 GB/GPU idle while the scheduler reported full token usage 0.99 with requests queued; and --speculative-dspark-block-size 3 is worth +68% to DSpark (2142 -> 3606 tok/s) while also cutting median TTFT from 11.2 s to 6.6 s and TPOT from 171 to 100 ms, since halving the verify window from 8 to 4 nearly halves the verify tax while accept length only falls from 3.00 to 2.55. Both are accuracy-neutral (GSM8K 97.489% / 97.641%, AIME25 91.67% / 95.42%). Most of the obvious knobs do nothing: chunked-prefill-size, cuda-graph-max-bs-decode and schedule-conservativeness all measured within noise, and the two mem-fraction values above 0.93 boot cleanly and then die on the first heavy prefill in the aiter MXFP4 fused-MoE stage-2 buffer.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "MXFP4 routed experts (compressed-tensors mxfp4-pack-quantized, group_size 32) + bf16 attention/shared-experts/lm_head, bf16 KV cache",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": "lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727",
        "launch_python": "export SGLANG_USE_AITER=1\nexport SGLANG_AITER_K3_OPT=1\nexport AITER_FLYDSL_FORCE=1\nexport AITER_SITUV2_A8W4=1\nsglang serve \\\n  --model-path moonshotai/Kimi-K3 \\\n  --trust-remote-code \\\n  --tp 8 \\\n  --attention-backend triton \\\n  --dtype bfloat16 \\\n  --mem-fraction-static 0.92 \\\n  --cuda-graph-max-bs-decode 256 \\\n  --disable-radix-cache \\\n  --reasoning-parser kimi_k3 \\\n  --tool-call-parser kimi_k3 \\\n  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \\\n  --speculative-algorithm DSPARK \\\n  --speculative-dspark-block-size 3 \\\n  --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton (full-MLA layers); KDA linear-attention layers default to the triton packed decode. DSpark draft attention overrides to triton on ROCm.",
        "moe_backend": "aiter MXFP4 (Mxfp4MoEMethod), 896 routed experts top-16 + 2 shared",
        "aiter": {
          "enabled": true,
          "commit": "68e42f5f",
          "kernels": [
            "MXFP4 MoE (mxfp4-pack-quantized routed experts)",
            "SITU activation / A8W4 GEMM",
            "FlyDSL codegen kernels",
            "fused RoPE, fused qk_norm_mrope_3d"
          ],
          "tuned_artifacts": [
            "aiter k3-for-amd branch @ 68e42f5f (carries the #17 xinyi/k3-opt merge)"
          ],
          "summary": "AITER is not optional here. SGLANG_USE_AITER=1 + SGLANG_AITER_K3_OPT=1 select the MXFP4 path that keeps the 896 routed experts packed at 194.38 GB/GPU; without it they unpack to 249.29 GB/GPU and no --mem-fraction-static leaves room for a KV pool on a 288 GiB card. AITER_SITUV2_A8W4 backs the model's `situ` activation, AITER_FLYDSL_FORCE the FlyDSL MoE kernels."
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Mandatory. Enables the aiter kernels, including the MXFP4 MoE path that keeps routed-expert weights packed."
          },
          {
            "key": "SGLANG_AITER_K3_OPT",
            "value": "1",
            "why": "Mandatory. K3-specific opt paths in models/kimi_k3.py and layers/quantization/mxfp4.py."
          },
          {
            "key": "AITER_FLYDSL_FORCE",
            "value": "1",
            "why": "Force the FlyDSL-generated MoE kernels on gfx950."
          },
          {
            "key": "AITER_SITUV2_A8W4",
            "value": "1",
            "why": "8-bit-activation / 4-bit-weight GEMM for the model's `situ` activation (hidden_act=situ)."
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Serve the 1.56 TB checkpoint from the local HF cache; avoids a gated-repo revalidation stalling boot."
          }
        ],
        "accuracy": [
          {
            "name": "GSM8K",
            "value": "97.64%",
            "note": "n=1314, in-tree run_eval --eval-name gsm8k --max-tokens 8192 --temperature 0 --num-threads 32. Wall clock 393.7 s at 711.2 tok/s. Mean speculative accept length over the run 5.95 (min 3.72, max 7.67).",
            "ref": "97.49% on the same server without DSpark - parity, i.e. speculative decoding is lossless here"
          },
          {
            "name": "AIME25",
            "value": "94.58%",
            "note": "pass@1 avg-of-8 via sgl-eval, +/-3.05% (SEM 1.08%), 240 samples; stop_rate 100%, truncated 0%, no_answer 0%, error 0%. Run with --n-repeats 8 --num-threads 48 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking. Wall clock 6779.7 s at 188 tok/s. Requires dspark_rocm_renorm.patch - without it the top_p path takes the server down.",
            "ref": "93.33% +/-4.36% without DSpark - the 1.25 pp gap is 0.7 sigma of the difference, i.e. parity"
          },
          {
            "name": "GSM8K (tuned, block size 3)",
            "value": "97.641%",
            "note": "Re-run of the command this cell now ships (--speculative-dspark-block-size 3), identical protocol: n=1319, run_eval --max-tokens 8192 --temperature 0 --num-threads 32. Wall clock 393.1 s at 714.5 tok/s. Gate: grid_results/20260729_091009/accuracy_gate.md.",
            "ref": "97.64% at the Day-0 default block size - identical to three decimal places"
          },
          {
            "name": "AIME25 (tuned, block size 3)",
            "value": "95.42%",
            "note": "pass@1 avg-of-8 via sgl-eval, +/-3.54% (SEM 1.25%), 240 samples; stop_rate 100%, truncated 0%, no_answer 0%, error 0%. Shortening the verify window from 8 to 4 does not cost accuracy - the target still verifies every token - and here it came out slightly above baseline.",
            "ref": "94.58% +/-3.05% at the Day-0 default block size - the +0.84 pp is well inside 1 sigma"
          }
        ],
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 178,
            "tpot_ms": 8.78,
            "decode_tok_s": 111.58,
            "total_tok_s": 223.16,
            "tok_s_per_gpu": 27.9,
            "source": "kimi_k3_playbook.md section 5.1 (benchmark.serving, random 1024/1024, --speculative-dspark-block-size 3, accept length 2.76; grid_results/20260729_091009 p3-lat-win). Day-0 default block size measured 9.43 ms / 104.0 tok/s here."
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 4,
            "ttft_ms": 423,
            "tpot_ms": 11.74,
            "decode_tok_s": 289.14,
            "total_tok_s": 578.28,
            "tok_s_per_gpu": 72.3,
            "source": "kimi_k3_playbook.md section 5.1 (block size 3, accept length 2.68). Day-0 default: 14.55 ms / 253.84 tok/s."
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 8,
            "ttft_ms": 638,
            "tpot_ms": 15.25,
            "decode_tok_s": 470.72,
            "total_tok_s": 941.44,
            "tok_s_per_gpu": 117.7,
            "source": "kimi_k3_playbook.md section 5.1 (block size 3, accept length 2.61). Day-0 default: 18.37 ms / 382.87 tok/s."
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 48,
            "ttft_ms": 6569,
            "tpot_ms": 100.05,
            "total_tok_s": 3606.12,
            "tok_s_per_gpu": 450.8,
            "source": "kimi_k3_playbook.md section 5.1 (benchmark.serving, random 8192/1024, mem-fraction 0.92 + block size 3, accept length 2.55; grid_results/20260729_091009 p3-g3). Day-0 default block size on the identical workload: 2142.48 tok/s at 11249 ms TTFT."
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 32,
            "ttft_ms": 1073,
            "tpot_ms": 40.09,
            "total_tok_s": 1338.66,
            "tok_s_per_gpu": 167.3,
            "source": "kimi_k3_playbook.md (bench_serving, random 1024/1024, accept length 3.26) - measured at the Day-0 default block size, not the block size 3 this cell now ships"
          },
          {
            "isl": 8192,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 644,
            "tpot_ms": 15.2,
            "decode_tok_s": 65.8,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, DSpark)"
          },
          {
            "isl": 32768,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 3189,
            "tpot_ms": 48.93,
            "decode_tok_s": 20.4,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, DSpark)"
          },
          {
            "isl": 131072,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 23956,
            "tpot_ms": 221.49,
            "decode_tok_s": 4.5,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, DSpark)"
          }
        ],
        "vs_nvidia": [],
        "gotchas": [
          "Turn DSpark OFF for long context - this is the sharpest knob on the page. Single-stream TPOT runs 9.43 / 15.20 / 48.93 / 221.49 ms at 1k / 8k / 32k / 131k input with DSpark, against 19.28 / 19.54 / 20.41 / 22.13 ms without it. The plain config is almost flat because only 24 of 93 layers carry a growing KV cache - the other 69 are constant-state KDA - so a 128x longer prompt costs 15% more per token. DSpark inverts from 2x faster to 10x slower over the same range because accept length collapses to 1.18-1.32 at 131k (accept rate 0.03): the 5-layer dense draft model cannot track that much context, so every step pays 8 target token-slots to land ~1.2 tokens. TTFT is unaffected either way (23.96 s vs 24.03 s at 131k) - speculative decoding does not touch prefill.",
          "Whether DSpark helps or hurts flips with accept length x concurrency, and the eval runs make the size of it concrete. GSM8K (greedy, 32 threads, accept 5.95): 393.7 s / 711 tok/s with DSpark vs 605.4 s / 469 tok/s without - 1.54x faster. AIME25 (temperature 1.0, 48 threads, accept ~2.9): 6779.7 s / 188 tok/s with DSpark vs 1964.1 s / 692 tok/s without - 3.45x SLOWER. At 8 draft tokens per step and accept 2.9, each accepted token costs 2.76 target token-slots; with a full batch the target is compute-bound, so that overhead lands directly on throughput.",
          "DSpark + any non-greedy sampling killed the scheduler on ROCm before dspark_rocm_renorm.patch: build_dflash_verify_target_probs calls top_k_renorm_prob / top_p_renorm_prob, which sglang imports from sgl_kernel only under is_cuda() or is_musa() and leaves as None elsewhere, so the first decode batch carrying top_p or top_k dies with \"TypeError: 'NoneType' object is not callable\". Greedy traffic (GSM8K) never touches it, so this hides until an AIME-style run. DFLASH escapes because its worker gates non-greedy verify on is_dflash_sampling_verify_available() and degrades to greedy argmax; DSPARK has no such gate. The patch routes the renorm to the torch implementations instead of degrading, which keeps sampling semantics intact.",
          "DSpark accept length is a property of the workload, not of the platform. Measured on this node: 5.95 mean over 1314 GSM8K requests (greedy, structured math; min 3.72 max 7.67), 3.26-3.32 on bench_serving random 1024/1024, 3.28 on ShareGPT, and 2.9-3.0 during AIME25 (temperature 1.0, top_p 0.95, long open-ended reasoning). The 5.29-5.93 quoted in #32548 sits at the GSM8K-like end of that range and reproduces here - an earlier revision of this page wrongly filed the low random/ShareGPT numbers as an unexplained platform gap.",
          "Accuracy is unchanged by speculative decoding, as it should be - the target verifies every token. GSM8K 97.64% vs 97.49%, AIME25 94.58% vs 93.33%, both within noise of the non-spec cell. Use that as the regression check when changing anything in the DSpark path: a real accept-length or verify bug shows up as lost throughput, not as lost accuracy.",
          "The draft-worker verify CUDA graph captures num_tokens_per_req=7 while the runner reports verify_num_draft_tokens=8. That is by design - SpeculativeAlgorithm.get_num_tokens_per_req_for_target_verify returns num_draft_tokens - 1 for the DSpark draft worker - not a mis-sized window.",
          "ROCm backend fallbacks are otherwise correct and silent: is_sm100_supported() is false so the trtllm_mha draft default never applies (it overrides to triton), and the nv_cutedsl verify backend that kimi_k3_hook.py pins unconditionally resolves to the triton KDA kernel off CUDA, with the fused DSpark CuTe MTP path gated behind is_cuda().",
          "DSpark cuts max_running_requests to 48 (from 368) and the KV pool to 14.02 GB / 544,533 tokens, paying for draft weights, a second CUDA-graph set and the 8-wide verify window.",
          "--speculative-dspark-block-size 3 is the single biggest win on this page and it is not a trade: +68% throughput (2142 -> 3606 tok/s at concurrency 48, ISL 8192/OSL 1024) while median TTFT drops from 11249 to 6569 ms and TPOT from 170.98 to 100.05 ms, and it wins the latency probe at every concurrency too (TPOT 8.78 / 11.74 / 15.25 ms at 1 / 4 / 8 against 9.84 / 14.55 / 18.37 for the default). The mechanism is the verify tax: at accept length a, each accepted token costs window/a target token-slots, and halving the window from 8 to 4 nearly halves the tax while accept length only falls from 3.00 to 2.55. It also frees memory, since the intermediate SSM verify scratch shrinks - the KV pool goes from 1,174,618 to 1,504,168 tokens at the same mem-fraction.",
          "The block-size curve is a step, not a slope, so 3 is a genuine interior optimum rather than 'smaller is better'. At concurrency 48: block size 2 gives 3305 tok/s, 3 gives 3606, 5 gives 2118 and the default 7 gives 2142. Sizes 5 and 7 are indistinguishable; below 3 the window is too short to amortise a step. Never set --speculative-num-draft-tokens directly, it is asserted to equal block size + 1 and setting it independently fails at boot.",
          "--mem-fraction-static 0.92 for this cell, one step below the non-spec cell's 0.93, because the draft weights and verify window add activation pressure - 0.93 boots and then dies under a heavy prefill. 0.92 lifts the KV pool from 551,629 to 1,174,618 tokens (+113%). Note that DSpark barely uses it: KV usage runs 0.24-0.36 at concurrency 48 while mamba usage sits at 0.98, so this lane is bounded by the KDA state pool, not by KV. That is also why raising --max-running-requests does not help.",
          "Two DSpark knobs that look like wins and are not. --max-running-requests 16-40 appears to buy up to +52% throughput (3265 tok/s at 24 vs 2142 at 48), but holding everything else fixed shows the mechanism: at 24 with 48 clients offered, 27 requests sit in the queue and median TTFT is 70 s. The server is not faster, it is serving fewer people. --enable-linear-replayssm-spec genuinely cuts median TTFT from 11249 to 3168 ms on its own with throughput flat, but once block size 3 is set it adds nothing (3584 vs 3606 tok/s) and slightly worsens TTFT, so it stays out of the recipe.",
          "The four AITER env vars are load-bearing, not tuning knobs. Without them the routed experts unpack from MXFP4 and target weights go 194.38 -> 249.29 GB/GPU; the server then dies in _profile_available_bytes with 'Loaded weights leave no GPU memory for the KV cache' at --mem-fraction-static 0.85 AND 0.93 alike. No mem-fraction rescues it on a 288 GiB card.",
          "First weight load is disk-bound: ~16 min cold (96 shards, ~25 s/shard). With the page cache warm the same load takes 105 s and the whole boot is ~3 min - budget the first launch, then stop worrying about restarts.",
          "--reasoning-parser kimi_k3 --tool-call-parser kimi_k3 split the reasoning trace into reasoning_content. Without them a short --max-tokens looks like it returns an empty answer, because the budget is spent inside the reasoning block.",
          "Effective weight precision for the roofline is 1.31 bytes/param, not 0.5: MXFP4 covers only the routed-expert Linears, and the ignore list keeps self_attn, shared_experts, the dense MLP, lm_head, vision_tower and mm_projector in bf16. Those bf16 tensors dominate the *active* set (114.4 of 137.8 GB per decode step) even though MXFP4 dominates the *total*.",
          "AIME25 needs sgl-eval, not in-tree run_eval, for the same answer-extraction reason as GLM-5.2. Both K3 configs answered every one of the 240 samples with a proper stop (stop_rate 100%, truncated 0%, no_answer 0%) at --max-tokens 64000.",
          "The vision path works in both cells, but only as a smoke test. A 420x160 PNG carrying rendered text came back correctly transcribed with image_tokens=90 in the usage block, under the plain config and under DSpark alike - notable for DSpark, whose draft model is text-only and might have been expected to choke on an image-bearing prefill. No multimodal benchmark or eval has been run, so treat this as \"the path is wired up\", not as a quality claim."
        ],
        "provenance": {
          "image": "source build in a ROCm 7.2.0 container - NOT the published Day-0 image lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727 (that image takes the identical command and env, but was not what these numbers were measured on)",
          "pr": "sgl-project/sglang#32541 (Kimi-K3 support); #32548 (AMD Day-0 recipe)",
          "sglang": "DarkSharpness/sglang-kimi @ amd/kimi-k3 533bff471 + dspark_rocm_renorm.patch, reports 0.5.15.post1.dev20260723+g6c9fd0adc5; the parameter-search rows are from 3d35b45f7 on the same branch",
          "aiter": "k3-for-amd 68e42f5f",
          "rocm": "7.2.0 (torch 2.9.1+rocm7.2.0)",
          "date": "2026-07-28 (Day-0 bring-up) and 2026-07-29 (launch-parameter search: mem-fraction, chunked prefill, cuda-graph batch, radix strategy, DSpark block size; raw rows in grid_results/20260729_091009/results.csv)",
          "node": "8x AMD Instinct MI355X (gfx950), 288 GiB each, single node"
        }
      },
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "MXFP4 routed experts (compressed-tensors mxfp4-pack-quantized, group_size 32) + bf16 attention/shared-experts/lm_head, bf16 KV cache",
        "strategy": "high-throughput",
        "nodes": "single",
        "verified": true,
        "docker_image": "lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727",
        "launch_python": "export SGLANG_USE_AITER=1\nexport SGLANG_AITER_K3_OPT=1\nexport AITER_FLYDSL_FORCE=1\nexport AITER_SITUV2_A8W4=1\nsglang serve \\\n  --model-path moonshotai/Kimi-K3 \\\n  --trust-remote-code \\\n  --tp 8 \\\n  --attention-backend triton \\\n  --dtype bfloat16 \\\n  --mem-fraction-static 0.93 \\\n  --cuda-graph-max-bs-decode 256 \\\n  --disable-radix-cache \\\n  --reasoning-parser kimi_k3 \\\n  --tool-call-parser kimi_k3 \\\n  --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton (full-MLA layers); KDA linear-attention layers default to the triton packed decode",
        "moe_backend": "aiter MXFP4 (Mxfp4MoEMethod), 896 routed experts top-16 + 2 shared",
        "aiter": {
          "enabled": true,
          "commit": "68e42f5f",
          "kernels": [
            "MXFP4 MoE (mxfp4-pack-quantized routed experts)",
            "SITU activation / A8W4 GEMM",
            "FlyDSL codegen kernels",
            "fused RoPE, fused qk_norm_mrope_3d"
          ],
          "tuned_artifacts": [
            "aiter k3-for-amd branch @ 68e42f5f (carries the #17 xinyi/k3-opt merge)"
          ],
          "summary": "AITER is not optional here. SGLANG_USE_AITER=1 + SGLANG_AITER_K3_OPT=1 select the MXFP4 path that keeps the 896 routed experts packed at 194.38 GB/GPU; without it they unpack to 249.29 GB/GPU and no --mem-fraction-static leaves room for a KV pool on a 288 GiB card. AITER_SITUV2_A8W4 backs the model's `situ` activation, AITER_FLYDSL_FORCE the FlyDSL MoE kernels."
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Mandatory. Enables the aiter kernels, including the MXFP4 MoE path that keeps routed-expert weights packed."
          },
          {
            "key": "SGLANG_AITER_K3_OPT",
            "value": "1",
            "why": "Mandatory. K3-specific opt paths in models/kimi_k3.py and layers/quantization/mxfp4.py."
          },
          {
            "key": "AITER_FLYDSL_FORCE",
            "value": "1",
            "why": "Force the FlyDSL-generated MoE kernels on gfx950."
          },
          {
            "key": "AITER_SITUV2_A8W4",
            "value": "1",
            "why": "8-bit-activation / 4-bit-weight GEMM for the model's `situ` activation (hidden_act=situ)."
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Serve the 1.56 TB checkpoint from the local HF cache; avoids a gated-repo revalidation stalling boot."
          }
        ],
        "accuracy": [
          {
            "name": "GSM8K",
            "value": "97.49%",
            "note": "n=1314, in-tree run_eval --eval-name gsm8k --max-tokens 8192 --temperature 0 --num-threads 32. Wall clock 605.4 s at 468.8 tok/s.",
            "ref": "97.64% with DSpark on the same server - parity"
          },
          {
            "name": "AIME25",
            "value": "93.33%",
            "note": "pass@1 avg-of-8 via sgl-eval, +/-4.36% (SEM 1.54%), 240 samples; stop_rate 100%, truncated 0%, no_answer 0%, error 0%. Same flags as the DSpark cell. Wall clock 1964.1 s at 692 tok/s - 3.45x faster than the DSpark cell on this workload.",
            "ref": "94.58% +/-3.05% with DSpark - parity within 0.7 sigma"
          },
          {
            "name": "GSM8K (tuned, mem-fraction 0.93)",
            "value": "97.489%",
            "note": "Re-run at mem-fraction 0.93 plus the opt-in --mamba-ssm-dtype bfloat16, identical protocol: n=1319, run_eval --max-tokens 8192 --temperature 0 --num-threads 32. Wall clock 580.7 s at 480.7 tok/s. Gate: grid_results/20260729_091009/accuracy_gate.md.",
            "ref": "97.49% at Day-0 settings - identical to three decimal places"
          },
          {
            "name": "AIME25 (tuned, mem-fraction 0.93)",
            "value": "91.67%",
            "note": "pass@1 avg-of-8 via sgl-eval, +/-3.09% (SEM 1.09%), 240 samples; stop_rate 99.17%, truncated 0.83%, no_answer 0.83%, error 0%. Measured with the opt-in --mamba-ssm-dtype bfloat16 in place. The -1.66 pp delta is 0.88 sigma of the pooled SEM, so not a detectable regression, but it is the largest delta in the gate and this is the only config that lost samples to truncation - which is why the shipped command in this cell leaves the bfloat16 SSM knob out and keeps only the memory change.",
            "ref": "93.33% +/-4.36% at Day-0 settings - inside 1 sigma"
          }
        ],
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 178,
            "tpot_ms": 19.28,
            "decode_tok_s": 51.4,
            "total_tok_s": 102.81,
            "tok_s_per_gpu": 12.9,
            "source": "kimi_k3_playbook.md (bench_serving, random 1024/1024)"
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 8,
            "ttft_ms": 993,
            "tpot_ms": 24.72,
            "total_tok_s": 623.57,
            "tok_s_per_gpu": 78.0,
            "source": "kimi_k3_playbook.md (bench_serving, random 1024/1024)"
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 32,
            "ttft_ms": 2397,
            "tpot_ms": 35.58,
            "total_tok_s": 1695.74,
            "tok_s_per_gpu": 212.0,
            "source": "kimi_k3_playbook.md (bench_serving, random 1024/1024)"
          },
          {
            "isl": 8192,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 623,
            "tpot_ms": 19.54,
            "decode_tok_s": 51.2,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, no speculative decoding)"
          },
          {
            "isl": 32768,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 3135,
            "tpot_ms": 20.41,
            "decode_tok_s": 49.0,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, no speculative decoding)"
          },
          {
            "isl": 131072,
            "osl": 512,
            "concurrency": 1,
            "ttft_ms": 24026,
            "tpot_ms": 22.13,
            "decode_tok_s": 45.2,
            "source": "kimi_k3_playbook.md (bench_serving, long context, single stream, no speculative decoding)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 96,
            "ttft_ms": 29530,
            "tpot_ms": 92.87,
            "total_tok_s": 7094.1,
            "tok_s_per_gpu": 886.8,
            "source": "kimi_k3_playbook.md section 5.1 (benchmark.serving, random 8192/1024, mem-fraction 0.93; grid_results/20260729_091009 p2-base-mf0.93)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 128,
            "ttft_ms": 39005,
            "tpot_ms": 108.05,
            "total_tok_s": 7891.8,
            "tok_s_per_gpu": 986.5,
            "source": "kimi_k3_playbook.md section 5.1 (benchmark.serving, random 8192/1024, mem-fraction 0.93; three-run mean of 7875.74/7896.33/7903.34, +/-0.35%). Saturation point: KV use 0.92 with zero queueing."
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 160,
            "ttft_ms": 48425,
            "tpot_ms": 116.97,
            "total_tok_s": 7044.08,
            "tok_s_per_gpu": 880.5,
            "source": "kimi_k3_playbook.md section 5.1 (past saturation: KV use 0.99, 22 queued, running pinned at 138)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 192,
            "ttft_ms": 60714,
            "tpot_ms": 119.6,
            "total_tok_s": 7352.29,
            "tok_s_per_gpu": 919.0,
            "source": "kimi_k3_playbook.md section 5.1 (past saturation: KV use 0.99, 54 queued, running pinned at 138)"
          }
        ],
        "vs_nvidia": [],
        "gotchas": [
          "Turn DSpark OFF for long context - this is the sharpest knob on the page. Single-stream TPOT runs 9.43 / 15.20 / 48.93 / 221.49 ms at 1k / 8k / 32k / 131k input with DSpark, against 19.28 / 19.54 / 20.41 / 22.13 ms without it. The plain config is almost flat because only 24 of 93 layers carry a growing KV cache - the other 69 are constant-state KDA - so a 128x longer prompt costs 15% more per token. DSpark inverts from 2x faster to 10x slower over the same range because accept length collapses to 1.18-1.32 at 131k (accept rate 0.03): the 5-layer dense draft model cannot track that much context, so every step pays 8 target token-slots to land ~1.2 tokens. TTFT is unaffected either way (23.96 s vs 24.03 s at 131k) - speculative decoding does not touch prefill.",
          "This is the throughput answer whenever the batch is full or the traffic is sampled rather than greedy: 1695.74 vs DSpark's 1338.66 tok/s at concurrency 32 on the synthetic sweep, and 3.45x faster wall clock on the real AIME25 run (1964.1 s / 692 tok/s vs 6779.7 s / 188 tok/s). It also keeps max_running_requests at 368 with a 21.35 GB / 829,332-token KV pool. Below ~concurrency 8, or on greedy structured workloads like GSM8K, DSpark wins instead - see the low-latency cell.",
          "--mem-fraction-static 0.93, not the Day-0 0.85, and operate at concurrency 128. The Day-0 setting left 35.13 GB/GPU free after graph capture while the scheduler simultaneously reported full token usage 0.99 with requests queued - the KV pool was full and under-allocated at the same time. 0.93 lifts it from 838,048 to 1,292,032 tokens (+54%) and moves peak throughput from 6198 tok/s at concurrency 96 to 7892 tok/s at 128, i.e. +27% and 987 tok/s/GPU, reproduced across three runs at +/-0.35%. Read running/queued together to see the new ceiling: 128 is the largest batch the pool holds outright, and past it running pins at 138 while the queue grows, so extra clients buy latency rather than throughput.",
          "Do NOT raise --mem-fraction-static past 0.93 on this model, and do not validate a memory change with a boot check. 0.94 (9.35 GB free) and 0.95 (6.45 GB free) both start cleanly, serve /health and print a plausible available_gpu_mem, then die on the first heavy prefill. The allocation that fails is not attention but the aiter MXFP4 fused-MoE stage-2 output buffer: 'torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.75 GiB' from flydsl_moe_stage2. That buffer scales with tokens per forward pass, so it peaks at --chunked-prefill-size rather than at batch size, and K3's safe floor here is ~12 GB free after capture - well above the 5-8 GB the generic SGLang tuning guide suggests. Any mem-fraction or chunked-prefill change needs a real long-prefill load test.",
          "The knobs that did nothing, measured at concurrency 128 against a 7876 tok/s baseline, so you can skip them: --chunked-prefill-size 32768 and 65536 gave +0.2% (noise) and 8192 gave -6.8%, so the 16384 default is already right and TTFT at saturation is queueing rather than chunk size; --cuda-graph-max-bs-decode 384 gave exactly 0% (the running ceiling is 138, so 256 already covers every replayed batch) and 512 crashed by dropping headroom under the floor above; --schedule-conservativeness 0.6 gave 0% because the tuned config never retracts and never queues at 128. --mamba-ssm-dtype bfloat16 is the one real extra, worth +1.24% (7892 -> 7990, three runs each with non-overlapping ranges) and it doubles max_running_requests from 570 to 1104, but it changes SSM state precision and is deliberately left out of the shipped command - 1% is not worth a numerics knob unless you re-run your own accuracy gate. Ours passed: GSM8K 97.489% against the 97.49% baseline.",
          "If your traffic has shared prefixes, drop --disable-radix-cache and add --mamba-radix-cache-strategy extra_buffer_lazy. On generated-shared-prefix (32 groups x 8 prompts, 4K shared system prompt, concurrency 32) that is 12,561 vs 8,276 tok/s - 1.52x, 68.2% cache hit - and it costs only 0.8% on traffic with no reuse. The default radix strategy reaches the same 1.5x but costs 14.4% on no-reuse traffic, and the reason is a KDA state-slot budget: prefix caching on a hybrid model charges 5 state slots per request under the default strategy versus 4 under extra_buffer_lazy, which drops max_running_requests from 570 to 114 and 142. 114 is below the throughput-optimal batch of 128, so the default strategy starts queueing and loses the 14%, while extra_buffer_lazy still clears 128.",
          "Accuracy matches the DSpark cell (GSM8K 97.49% vs 97.64%, AIME25 93.33% vs 94.58%, both within noise), so the choice between the two cells is purely a throughput/latency one.",
          "The four AITER env vars are load-bearing, not tuning knobs. Without them the routed experts unpack from MXFP4 and target weights go 194.38 -> 249.29 GB/GPU; the server then dies in _profile_available_bytes with 'Loaded weights leave no GPU memory for the KV cache' at --mem-fraction-static 0.85 AND 0.93 alike. No mem-fraction rescues it on a 288 GiB card.",
          "First weight load is disk-bound: ~16 min cold (96 shards, ~25 s/shard). With the page cache warm the same load takes 105 s and the whole boot is ~3 min - budget the first launch, then stop worrying about restarts.",
          "--reasoning-parser kimi_k3 --tool-call-parser kimi_k3 split the reasoning trace into reasoning_content. Without them a short --max-tokens looks like it returns an empty answer, because the budget is spent inside the reasoning block.",
          "Effective weight precision for the roofline is 1.31 bytes/param, not 0.5: MXFP4 covers only the routed-expert Linears, and the ignore list keeps self_attn, shared_experts, the dense MLP, lm_head, vision_tower and mm_projector in bf16. Those bf16 tensors dominate the *active* set (114.4 of 137.8 GB per decode step) even though MXFP4 dominates the *total*.",
          "AIME25 needs sgl-eval, not in-tree run_eval, for the same answer-extraction reason as GLM-5.2. Both K3 configs answered every one of the 240 samples with a proper stop (stop_rate 100%, truncated 0%, no_answer 0%) at --max-tokens 64000.",
          "The vision path works in both cells, but only as a smoke test. A 420x160 PNG carrying rendered text came back correctly transcribed with image_tokens=90 in the usage block, under the plain config and under DSpark alike - notable for DSpark, whose draft model is text-only and might have been expected to choke on an image-bearing prefill. No multimodal benchmark or eval has been run, so treat this as \"the path is wired up\", not as a quality claim."
        ],
        "provenance": {
          "image": "source build in a ROCm 7.2.0 container - NOT the published Day-0 image lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727 (that image takes the identical command and env, but was not what these numbers were measured on)",
          "pr": "sgl-project/sglang#32541 (Kimi-K3 support); #32548 (AMD Day-0 recipe)",
          "sglang": "DarkSharpness/sglang-kimi @ amd/kimi-k3 533bff471 + dspark_rocm_renorm.patch, reports 0.5.15.post1.dev20260723+g6c9fd0adc5; the parameter-search rows are from 3d35b45f7 on the same branch",
          "aiter": "k3-for-amd 68e42f5f",
          "rocm": "7.2.0 (torch 2.9.1+rocm7.2.0)",
          "date": "2026-07-28 (Day-0 bring-up) and 2026-07-29 (launch-parameter search: mem-fraction, chunked prefill, cuda-graph batch, radix strategy, DSpark block size; raw rows in grid_results/20260729_091009/results.csv)",
          "node": "8x AMD Instinct MI355X (gfx950), 288 GiB each, single node"
        }
      }
    ],
    "gaps": [
      {
        "title": "GPQA-diamond / HLE vs the model card",
        "kind": "metric",
        "note": "Blocked, not skipped: the checkpoint ships .eval_results/ claiming GPQA-diamond 93.5 and HLE 56.0, but Idavidrein/gpqa is a gated HF dataset and the run account has not been granted access. Request access on the dataset page, then the command below works as-is.",
        "cmd": "# 1. request access at https://huggingface.co/datasets/Idavidrein/gpqa\n# 2. export HF_TOKEN=<your token>   (never write it to a file)\nsgl-eval run gpqa --base-url http://127.0.0.1:30000/v1 \\\n  --model moonshotai/Kimi-K3 --api-key EMPTY \\\n  --n-repeats 4 --num-threads 48 --max-tokens 64000 \\\n  --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Multimodal (vision tower) \u2014 benchmark, not just smoke test",
        "kind": "metric",
        "note": "A single rendered-text PNG round-trips correctly in both cells (image_tokens=90, answer transcribed, server stable), so the vision path is wired up. Nothing beyond that: no image benchmark, no MMMU-Pro, no throughput with image prefill.",
        "cmd": "sgl-eval run mmmu_pro --base-url http://127.0.0.1:30000/v1 \\\n  --model moonshotai/Kimi-K3 --api-key EMPTY \\\n  --num-threads 32 --max-tokens 32000 --temperature 1.0 --top-p 0.95"
      },
      {
        "title": "Day-0 image cross-check",
        "kind": "strategy",
        "note": "All numbers come from a source build. Re-running inside lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727 would confirm the published image behaves identically - and whether it already carries the DSpark ROCm renorm fix or still dies on the first top_p batch.",
        "cmd": "docker run -d --name k3-day0 \\\n  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \\\n  --group-add video --cap-add SYS_PTRACE --shm-size=32g \\\n  -v $HOME/hf-cache:/hf-cache -e HF_HOME=/hf-cache \\\n  lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727 sleep infinity\n\n# does the image still crash on non-greedy DSpark?\ncurl -s localhost:30000/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"moonshotai/Kimi-K3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":1.0,\"top_p\":0.95}'"
      },
      {
        "title": "MI300X (gfx942)",
        "kind": "hardware",
        "note": "Not attempted. At 194.38 GB/GPU of weights the MXFP4 path does not fit 8x192 GiB, so gfx942 needs more GPUs or a different sharding - and #32548 already reports MI350 running with untuned AITER kernels.",
        "cmd": "# would need >8 GPUs or multi-node; no verified recipe yet"
      }
    ]
  },
  {
    "id": "kimi-k2.6",
    "name": "Kimi-K2.6",
    "family": "Moonshot",
    "hf_path": "moonshotai/Kimi-K2.6",
    "architecture": "Mixture-of-Experts (MLA attention, 384 routed experts), 1T total params / 32B active per token",
    "precision": "W4A16",
    "status": "verified",
    "summary": "Kimi-K2.6 (1T MoE, 32B active, W4A16, 384 routed experts) verified on 8x MI355X (gfx950) with the prebuilt jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420 image. Default TP=8 EP=1 hits the pre-tuned 13-bucket E=384,N=128 int4_w4a16 MoE configs; triton MLA for decode + aiter for prefill. Measured BS=1 single-request decode 34-45 tok/s across 1k-32k context. EP8/tp2ep4/tp4ep2 mori-a2a variants exist but lack tuned N=2048 configs and are slower at BS=1.",
    "params_total": "1T",
    "params_active": "32B",
    "active_params_billions": 32,
    "bytes_per_param": 0.5,
    "weights_gb": 555,
    "context_len": "32768 (max tested context in benchmark; model supports longer)",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "W4A16 (int4_w4a16 MoE)",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": "jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420",
        "attention_backend": "triton (decode) / aiter (prefill)",
        "moe_backend": "TP-sharded MoE (ep-size 1), pre-tuned int4_w4a16 E=384,N=128 configs",
        "parallelism": {
          "tp": 8,
          "ep": 1,
          "dp": null
        },
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /hf-cache/models--moonshotai--Kimi-K2.6/snapshots/<rev> \\\n    --served-model-name kimi-k2.6 \\\n    --tensor-parallel-size 8 \\\n    --ep-size 1 \\\n    --trust-remote-code \\\n    --reasoning-parser kimi_k2 \\\n    --tool-call-parser kimi_k2 \\\n    --decode-attention-backend triton \\\n    --prefill-attention-backend aiter \\\n    --host 0.0.0.0 \\\n    --port 30000",
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER kernels (aiter MLA TP=8 fix + tuned int4_w4a16 MoE configs ship in this image)"
          },
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "Avoids the triton MLA tuple-unpack crash during fused decode MLA on ROCm"
          },
          {
            "key": "SGLANG_DEEPSEEK_LOAD_MAX_WORKERS",
            "value": "4",
            "why": "Keeps weight-load RAM pressure bounded while loading the 1T MoE checkpoint"
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Use the local HF cache snapshot, no network fetch"
          }
        ],
        "aiter": {
          "enabled": true,
          "commit": "3125d3b01",
          "kernels": [
            "MLA",
            "MoE",
            "attention (prefill)"
          ],
          "tuned_artifacts": [
            "13-bucket E=384,N=128,...MI355X,int4_w4a16 MoE configs",
            "aiter 3125d3b01 MLA TP=8 fix"
          ],
          "summary": "Prebuilt image ships aiter commit 3125d3b01 with the MLA TP=8 fix and 13-bucket E=384,N=128 int4_w4a16 MoE configs tuned for MI355X; aiter provides the heavy prefill attention and the TP-sharded MoE GEMMs, while decode MLA runs on triton."
        },
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 300,
            "decode_tok_s": 45.23,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 2048,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 350,
            "decode_tok_s": 44.69,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 4096,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 430,
            "decode_tok_s": 43.62,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 650,
            "decode_tok_s": 41.88,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 16384,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 1100,
            "decode_tok_s": 38.93,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 32768,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 2230,
            "decode_tok_s": 34,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "SGLANG_ROCM_FUSED_DECODE_MLA=0 is required - fused decode MLA triton path hits a tuple-unpack crash on ROCm.",
          "Keep --ep-size 1 for BS=1: it keeps MoE TP-sharded across 8 ranks and hits the pre-tuned E=384,N=128 int4_w4a16 config.",
          "EP variants (ep8 / tp2ep4 / tp4ep2) all need --moe-a2a-backend mori plus the mori env group (SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384, MORI_SHMEM_MODE=vmm, MORI_SHMEM_HEAP_SIZE=34359738368, TORCH_NCCL_BLOCKING_WAIT=0, NCCL_ASYNC_ERROR_HANDLING=0) and use --disable-cuda-graph --skip-server-warmup --watchdog-timeout 1800 --dist-timeout 3600. tp2ep4 adds --moe-dp-size 2 (--ep-size 4); tp4ep2 adds --moe-dp-size 4 (--ep-size 2).",
          "EP variants lack tuned MoE configs for N=2048 (only the default TP=8 N=128 config ships), so they fall back to generic kernels and are slower at BS=1.",
          "--reasoning-parser kimi_k2 --tool-call-parser kimi_k2 are needed to split <think> blocks and DSML-style tool calls out of choices[0].message.content.",
          "Decode runs on triton MLA; prefill on aiter - this split is the MI355X sweet spot for best TTFT.",
          "The separate Dockerfile.kimi-opt build (base v0.5.9-rocm700-mi35x) stubs aiter and sets SGLANG_USE_AITER=0; it is NOT the verified serving image - use the prebuilt v0.5.10rc0-rocm720 image with SGLANG_USE_AITER=1 instead."
        ],
        "provenance": {
          "image": "jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420",
          "pr": "#19552 (Kimi-K2/K2.5 tool-call parser fixes; applied in Dockerfile.kimi-opt build, not the prebuilt serving image)",
          "sglang": "v0.5.10rc0",
          "aiter": "3125d3b01",
          "rocm": "7.2.0",
          "date": "April 2026 (image tag 20260420)",
          "node": "8x AMD Instinct MI355X (gfx950), single node"
        }
      }
    ],
    "gaps": [
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Only BS=1 latency measured.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "EP variants (ep8 / tp2ep4 / tp4ep2)",
        "kind": "strategy",
        "note": "Need N=2048 tuned MoE configs; today they fall back to generic kernels and lose at BS=1. Launch a variant, then sweep:",
        "cmd": "TAG=ep8 bash test_kimi_k26.sh    # or tp2ep4 / tp4ep2\n\n# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      }
    ]
  },
  {
    "id": "kimi-k2.5",
    "name": "Kimi-K2.5",
    "family": "Moonshot",
    "hf_path": "moonshotai/Kimi-K2.5",
    "architecture": "Mixture-of-Experts (MoE) with MLA attention; 384 experts (E=384, N=128), W4A16 INT4 quantized weights",
    "precision": "W4A16 (INT4 weight, A16 activation)",
    "params_total": "1T",
    "params_active": "32B",
    "active_params_billions": 32,
    "bytes_per_param": 0.5,
    "weights_gb": 555,
    "context_len": "131072",
    "status": "verified",
    "summary": "Kimi-K2.5 (1T total / 32B active W4A16 INT4 MoE + MLA) served on 8x MI355X (gfx950) at TP=8. Optimized config hits 23.5ms decode median (42.6 tok/s) at BS=1 vs 38.3ms baseline (+38.6%) using hybrid attention (triton decode + aiter prefill), GEMM A16W16 small-M tuning, and MoE Triton config tuning. Optional Eagle3 speculative decoding adds ~1.8x on short-context coding/math.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "W4A16 INT4",
        "strategy": "low-latency",
        "verified": true,
        "docker_image": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317",
        "launch_python": "/opt/venv/bin/python3 -m sglang.launch_server \\\n    --model-path moonshotai/Kimi-K2.5 \\\n    --tp 8 \\\n    --trust-remote-code \\\n    --decode-attention-backend triton \\\n    --prefill-attention-backend aiter \\\n    --mem-fraction-static 0.85 \\\n    --reasoning-parser kimi_k2 \\\n    --tool-call-parser kimi_k2 \\\n    --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton decode + aiter prefill (hybrid)",
        "moe_backend": "triton (E=384, N=128 tuned configs)",
        "aiter": {
          "enabled": true,
          "commit": null,
          "summary": "AITER provides the prefill attention path (ASM kernels) plus optimized GEMM A16W16 small-M configs for M=1 decode; built from the Arist12/aiter:kimi-k25-optimize-v2 branch with non-editable pip install.",
          "kernels": [
            "prefill attention (aiter ASM)",
            "GEMM A16W16 small-M"
          ],
          "tuned_artifacts": [
            "GEMM A16W16 small-M configs (M_LEQ_4/8/16/32/64 with BLOCK_SIZE_M=16-64, default was 256)",
            "Arist12/aiter:kimi-k25-optimize-v2 branch"
          ]
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "enables the aiter prefill attention path"
          },
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "required; the image default is 1 and must be disabled for this hybrid attention config"
          },
          {
            "key": "GPU_COREDUMP_ENABLE",
            "value": "0",
            "why": "set on docker run to disable GPU coredumps"
          }
        ],
        "benchmarks": [
          {
            "isl": 8192,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 42.6,
            "tpot_ms": 23.5,
            "prefill_tok_s": 12847,
            "ttft_ms": 637,
            "source": "index.html",
            "total_tok_s": null,
            "tok_s_per_gpu": null
          },
          {
            "isl": 1024,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 45.19,
            "ttft_ms": 270,
            "source": "index.html",
            "tpot_ms": null,
            "prefill_tok_s": null,
            "total_tok_s": null,
            "tok_s_per_gpu": null
          },
          {
            "isl": 2048,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 44.67,
            "ttft_ms": 330,
            "source": "index.html",
            "tpot_ms": null,
            "prefill_tok_s": null,
            "total_tok_s": null,
            "tok_s_per_gpu": null
          }
        ],
        "accuracy": [],
        "gotchas": [
          "aiter must be installed with 'pip install .' (non-editable). 'pip install -e .' creates a broken namespace package that fails to resolve compiled C extensions. Verify with: python3 -c \"from aiter import dynamic_per_tensor_quant; print('OK')\".",
          "SGLANG_ROCM_FUSED_DECODE_MLA=0 is required; the image default is 1 and the hybrid attention config breaks otherwise.",
          "MoE Triton config tuning is the final optimization rung in the ladder: baseline triton attn 38.3ms -> +aiter prefill 34.4ms (10.2%) -> +GEMM A16W16 small-M tuning 24.3ms (36.6%) -> +MoE Triton config tuning 23.5ms / 42.6 tok/s (38.6%).",
          "Eagle3 must use --speculative-algorithm EAGLE3 (not EAGLE); EAGLE silently degrades accept_length to 1.0. Eagle3 also needs --mem-fraction-static 0.75 (down from 0.85) for the draft model, and accept_length degrades to 1.6-2.0 at 8K+ input tokens making it slower than baseline on long context.",
          "Eagle3 non-greedy (temp>0) on ROCm needs PyTorch fallback kernels for 3 missing sgl_kernel C++ ops via patch_eagle_rocm.py (or PR #21275); without it temp>0 silently falls back to greedy."
        ],
        "provenance": {
          "image": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317",
          "pr": null,
          "sglang": "Arist12/sglang:kimi-k25-optimize-v2 (MoE Triton configs E=384,N=128, BLOCK_SIZE_M=16 for batch=1 decode)",
          "aiter": "Arist12/aiter:kimi-k25-optimize-v2 (GEMM A16W16 small-M configs)",
          "rocm": "7.2",
          "date": "2026-03 (March 2026)",
          "node": "8x MI355X"
        },
        "vs_nvidia": []
      }
    ],
    "gaps": [
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Only single-stream decode measured.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Eagle3 speculative decode",
        "kind": "perf",
        "note": "Relaunch with --speculative-algorithm EAGLE3 (not EAGLE) + --mem-fraction-static 0.75; measure accept_length and speedup on short vs long context.",
        "cmd": "# add to launch: --speculative-algorithm EAGLE3 --mem-fraction-static 0.75\n# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      }
    ]
  }
];
