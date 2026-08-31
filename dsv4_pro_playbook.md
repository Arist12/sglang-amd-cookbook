# DeepSeek-V4-Pro-0813 on MI355X

This is the verified ROCm recipe for the official
`deepseek-ai/DeepSeek-V4-Pro-0813` checkpoint.

**Decision:** the target-only configuration below is ready to publish for one
8x MI355X node. It uses the `unified_kv_triton` DeepSeek-V4 attention path and
AITER FP4-expert kernels. DSpark was not validated on ROCm and is not part of
the merge claim.

## 1. Checkpoint identity

| item | verified value |
|---|---|
| Hugging Face repo | `deepseek-ai/DeepSeek-V4-Pro-0813` |
| revision | `72e1d3230f6c080a530b0a1d46f8eb4602340597` |
| shards | 66 / 66 |
| index SHA-256 | `2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a` |
| indexed tensor bytes | 892,727,580,904 bytes (831.42 GiB) |
| architecture | 1.6T total / 49B active, 61 layers, 384 routed experts, top-6, 1M context |
| target active bytes | 41,538,578,380 bytes (38.69 GiB), or 0.8477 physical bytes per official active parameter |

The target uses FP4 routed experts and block-FP8 dense weights. The checkpoint
also contains an MTP/DSpark module, but this validation deliberately did not load
or benchmark it. All four checkpoint-provided encoding and parsing fixtures pass.

Download and pin the exact snapshot before launch:

```bash
hf download deepseek-ai/DeepSeek-V4-Pro-0813 \
  --revision 72e1d3230f6c080a530b0a1d46f8eb4602340597 \
  --local-dir /data/DeepSeek-V4-Pro-0813

sha256sum /data/DeepSeek-V4-Pro-0813/model.safetensors.index.json
# 2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a
```

## 2. Validated software and hardware

- Host: `mia1-p02-g23`
- GPU: 8x AMD Instinct MI355X (`gfx950`), single node
- GPU BDFs: `05:00.0`, `15:00.0`, `65:00.0`, `75:00.0`, `85:00.0`,
  `95:00.0`, `e5:00.0`, `f5:00.0`
- ROCm: 7.2.0 (`HIP 7.2.26015`)
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- Triton: 3.6.0
- FlyDSL: 0.2.4
- SGLang source: `71de97b264b04dcd514cf904003028aefe9775c8`
- AITER source: `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`

This was a source/runtime overlay, not a frozen official image. Python executed
the SGLang checkout above through `PYTHONPATH`; installed wheel metadata still
reported `0.5.16.dev20260728+g32c30c0f96`. Reproduce from the source and AITER
SHAs, not from that package-version string alone.

Before and after the formal performance sweep, every GPU had 30 consecutive
one-second samples at 0% GFX and 0% UMC activity. The server held all eight GPU
locks for the run, and shutdown returned every card to about 284 MiB used VRAM.

## 3. Launch

[`test_dsv4_pro.sh`](test_dsv4_pro.sh) wraps the checked launch and validates the
checkpoint index before starting. The effective command was:

```bash
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_DSV4_FP4_EXPERTS=true

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Pro-0813 \
  --served-model-name deepseek-v4-pro-0813 \
  --trust-remote-code \
  --tp 8 \
  --disable-radix-cache \
  --attention-backend dsv4 \
  --page-size 256 \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 \
  --disable-shared-experts-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --max-running-requests 256 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1200 \
  --host 127.0.0.1 \
  --port 31000
```

The observed MoE route was AITER's two-stage
`flydsl_moe1_afp8_wfp4_bf16` family with FP8 activations, packed FP4 expert
weights, BF16 output, 384 experts and top-6 routing. This proves that the run did
not silently fall back to the old FP8 preview path.

## 4. Correctness

The public API smoke test passed `/v1/models`, native `/generate` (`9 * 7 = 63`),
and a structured `get_weather({"city":"Paris, France"})` tool call.

GSM8K used the SGLang eight-shot evaluator with all 1,319 questions,
`temperature=0`, `max_new_tokens=512`, and the dataset SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

| round | correct | accuracy | invalid |
|---:|---:|---:|---:|
| 1 | 1,249 / 1,319 | 94.693% | 0 |
| 2 | 1,246 / 1,319 | 94.466% | 0 |
| 3 | 1,248 / 1,319 | 94.617% | 0 |

All three runs clear the upstream `accuracy > 0.92` gate. The median is 94.617%.

## 5. Serving performance

Method: random fixed-length requests, ISL=8192, OSL=1024,
`random_range_ratio=1.0`, `4 * concurrency` measured requests, one warmup burst,
cache flush per point, and three full rounds in rotated order (`1/8/32`,
`32/8/1`, `8/1/32`). Values below are medians across the three complete runs.
The ShareGPT seed file SHA-256 was
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

| concurrency | TTFT ms | TPOT ms | decode tok/s | output tok/s | total tok/s | tok/s/GPU | total-throughput spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 314.58 | 12.69 | 78.8 | 76.93 | 692.37 | 86.5 | 0.126% |
| 8 | 1,628.72 | 16.18 | — | 450.86 | 4,057.70 | 507.2 | 0.032% |
| 32 | 5,109.56 | 23.51 | — | 1,122.29 | 10,100.64 | 1,262.6 | 0.155% |

`decode tok/s` is `1000 / median TPOT` and is shown only for one stream.
`output tok/s` is aggregate generation throughput; `total tok/s` includes prompt
tokens. The rows in `models.js` are generated from the raw JSONL records by
[`gen_dsv4_mi355x_rows.py`](gen_dsv4_mi355x_rows.py), not hand-transcribed.

## 6. Startup and API behavior

Process start to ready took about 19 minutes 54 seconds with a warm local
checkpoint. Per-rank weight loading was highly uneven: most ranks finished in
roughly 241-260 seconds, TP0 took 912 seconds, and TP7 took 1,043 seconds. Do not
treat a quiet log during this interval as a hang without checking process and
device activity.

The `deepseekv4` tool parser correctly emits OpenAI-compatible tool calls. The
current `deepseek-v4` reasoning-parser integration does not split the model's
reasoning into `reasoning_content`: it remains `null`, and reasoning text plus a
stray `</think>` remain in `message.content`. Two identical temperature-zero
smoke requests also produced different reasoning text while preserving the
correct final answer; the three full GSM8K rounds above are the stability gate.

## 7. Boundaries and known limitations

- This result is **target-only**. The checkpoint's DSpark module was not loaded,
  and no speculative-decoding result is claimed on ROCm.
- Only `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` is covered. The separate
  plain `triton` regression is currently documented and blocked on upstream
  SGLang PR #36388 for Flash-0731; it was not substituted into this Pro recipe.
- Radix cache and shared-expert fusion were disabled in the measured contract.
  Enabling either changes the configuration and requires a fresh correctness
  and performance run.
- The 1M context limit is a model maximum, not a statement that one 8-GPU node
  can host arbitrary concurrency at that length.

## 8. Reproduce the published rows

Keep raw benchmark output outside the repository, then run:

```bash
python3 gen_dsv4_mi355x_rows.py \
  --variant pro \
  --results-root /path/to/dsv4-runs \
  --check-models models.js
```

The generator rejects missing repeats, token-count mismatches, wrong model or
runtime provenance, non-quiescent before/after samples, and more than 5%
total-throughput spread.
