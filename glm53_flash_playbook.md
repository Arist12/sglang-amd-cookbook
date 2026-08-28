# GLM-5.3-Flash on MI355X (gfx950)

**Status: verified for the high-throughput cell.** This report measures
`zai-org/GLM-5.3-Flash` on one 8x MI355X node, compares it with
GLM-5.2-FP8 on the same software stack and workload, and records every
unmerged dependency by immutable commit.

Measurement window: **2026-08-27T23:31:10Z through 2026-08-28 UTC**.
Node: `mia1-p02-g46`, 8x MI355X (gfx950, 288 GiB each).

## 1. Frozen environment

| Component | Frozen value |
|---|---|
| Image | `rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260826` |
| torch / HIP | `2.11.0+rocm7.2` / `7.2.26015` |
| SGLang | `0.5.18.dev20260826+g937af8538b`, source at `9d208769398882e20220cb97722bf610397e66d8` plus `hybrid_fp8_metadata.patch` |
| AITER | image commit `c16d44b93a528b2a4bfd6d8d3409116d465872a9` plus the tuning CSV from `95565e33c8287a8c56bc31a84edf2de3ecc97662` |
| GLM-5.3 weights | `zai-org/GLM-5.3-Flash` revision `04c4e9e95c5da8862dced7e5056455116f83a7e0` |
| GLM-5.2 control | `/models/GLM-5.2-FP8`, index SHA-256 `e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf` |

The GLM-5.3 revision is newer than the `3f1971b7b5f7` revision used by the
upstream performance PR. The two revisions have the same 76,108 tensor keys,
the same shard mapping and the same 328,326,771,576-byte total. The revision is
still pinned because branch names are not provenance.

## 2. GLM-5.3-Flash versus GLM-5.2 on paper

The GLM-5.3 figures below come from the
[model card](https://huggingface.co/zai-org/GLM-5.3-Flash) and
[Z.ai launch report](https://z.ai/blog/glm-5.3-flash). GLM-5.2 geometry comes
from its pinned checkpoint.

| | GLM-5.3-Flash | GLM-5.2-FP8 |
|---|---:|---:|
| Total parameters | 320B | 743B |
| Advertised active parameters | 18B | 39B |
| Checkpoint size | 305.8 GiB | 703.7 GiB |
| Text layers | 45 | 78 |
| Attention | 34 KDA linear + 11 NoPE sparse-MLA/DSA | MLA/DSA, no linear-attention layers |
| Routed experts / selected | 288 / 8 | 256 / 8 |
| Context | 1,048,576 | 1,048,576 |
| Multimodal | text, image, video | text |

The smaller active set is not the full memory argument. Reading only
safetensors headers and weighting eight routed experts per token gives:

| Decode weight stream | GLM-5.3-Flash | GLM-5.2-FP8 |
|---|---:|---:|
| Always-active text + shared experts + active routed experts | 23.68 GB | 43.29 GB |
| Less the one-row embedding lookup | **22.42 GB** | **41.38 GB** |
| Effective bytes / advertised active parameter | **1.25** | **1.06** |

The lm head remains included because every decode step evaluates the full
vocabulary. The embedding matrix is excluded because only one row is read. This
predicts a 1.85x memory-bound decode advantage for GLM-5.3; the measured
ISL-8192 single-stream gain is 1.78x.

The vendor's model-quality comparison is separate from this serving study:
Terminal-Bench 2.1 is 84.3 vs 81.0, DeepSWE v1.1 is 63.4 vs 46.2,
Toolathlon Verified is 78.4 vs 59.9, and AutomationBench is 48.8 vs 26.2
for GLM-5.3 versus GLM-5.2. Those are reported results under their own
harnesses, not numbers reproduced here. Section 7 is the controlled comparison
run on this node.

## 3. PR stack and timestamp

These states were captured at `2026-08-27T23:31:10Z` and then frozen for the
whole run:

| Role | PR | Frozen head | State |
|---|---|---|---|
| Model implementation | [sglang#36507](https://github.com/sgl-project/sglang/pull/36507) | `c4d5d45e506d` | open |
| AMD enablement and optimized kernels | [sglang#36607](https://github.com/sgl-project/sglang/pull/36607) | `9d2087693988` | open |
| gfx950 BF16 GEMM tuning | [ROCm/aiter#5060](https://github.com/ROCm/aiter/pull/5060) | `95565e33c828` | open |
| Upstream AMD recipe reference | [sglang#36732](https://github.com/sgl-project/sglang/pull/36732) | `8c0d81f9cf30` | open |

PR #36607 is stacked on #36507 and therefore contains both runtime changes.
PR #36732 is documentation only and was not applied to the runtime.

The stack moved while the measurements were running. At the final status
snapshot, `2026-08-28T06:29:56Z`, #36607 had been merged into the #36507
feature branch at `c821c425c31b` (not into `main`), #36507 remained open at
`aa8c950a3df6`, AITER #5060 remained open at `95565e33c828`, and #36732
remained open at `8c0d81f9cf30`. Those newer SGLang commits are not silently
mixed into this dataset; the launch helper checks that the measured
`9d2087693988` commit remains an ancestor and checks it out directly.

One additional local patch is required:
[`glm53_flash/hybrid_fp8_metadata.patch`](glm53_flash/hybrid_fp8_metadata.patch).
Under concurrent variable-prefix requests, the FP8-KV MHA fallback asked
`get_attn_backend().forward_metadata`, but a hybrid KDA model returns
`HybridLinearAttnBackend`; the DSA metadata lives on its `full_attn_backend`
child. Without the unwrap, GSM8K aborts with:

```text
AttributeError: 'HybridLinearAttnBackend' object has no attribute
'forward_metadata'
```

The patch routes all model-side MHA helper lookups through the existing
`resolve_attn_backend()` boundary and was validated by the full performance and
accuracy runs below.

## 4. Verified launch

Run [`glm53_flash/setup_pr.sh`](glm53_flash/setup_pr.sh) first. It hard-checks
the SGLang and AITER heads, applies the local metadata patch idempotently, and
overlays only AITER #5060's tuning CSV so the image's compiled AITER source is
not replaced.

```bash
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.3-Flash \
  --revision 04c4e9e95c5da8862dced7e5056455116f83a7e0 \
  --served-model-name glm-5.3-flash \
  --tp-size 8 --ep-size 1 \
  --trust-remote-code \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --quantization fp8 \
  --moe-runner-backend aiter \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 32 \
  --disable-radix-cache \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port 30000
```

Startup took 86.1 seconds with warm filesystem cache. Per rank:

- weights: 38.02 GB;
- FP8 KV pool: 211.16 GB / 32,006,720 tokens;
- KDA state pool for 64 requests: 0.04 GB conv + 1.08 GB SSM;
- decode graphs: 1.15 GB;
- free after capture: 23.16 GB.

The log confirms all intended paths on all eight ranks:
`Shared experts fusion optimization enabled`,
`Using AITER gfx950 mHC pre/post kernels`, and
`Using fused AITER mHC attention-to-FFN boundary`. The AITER config merge also
names `glm53_bf16_tuned_gemm.csv`.

## 5. Online throughput

`sglang.benchmark.serving`, random token IDs, ISL 8192 / OSL 1024,
`--random-range-ratio 1.0`, temperature 0, seed 42, infinite request rate,
cache flush and one concurrency-wide warmup per point. Each point uses `4*C`
measured prompts and is the median of three complete runs.

| conc | GLM-5.3 TTFT ms | TPOT ms | output tok/s | total tok/s | tok/s/GPU | GLM-5.2 total | 5.3 / 5.2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 199.49 | 6.95 | 139.93 | 1,259.36 | 157.4 | 703.83 | **1.79x** |
| 8 | 955.25 | 9.81 | 744.60 | 6,701.38 | 837.7 | 3,657.26 | **1.83x** |
| 16 | 1,624.32 | 10.80 | 1,292.46 | 11,632.14 | 1,454.0 | 5,940.07 | **1.96x** |
| 32 | 2,945.24 | 12.70 | 2,054.79 | **18,493.13** | **2,311.6** | 8,434.38 | **2.19x** |
| 64 | 5,555.98 | 69.45 | 828.73 | 7,458.58 | 932.3 | 10,557.68 | **0.71x** |

Request and token accounting passed for all 30 model/point/repeat records. The
largest total-throughput spread was 0.87% for GLM-5.3 and 0.13% for GLM-5.2.

Concurrency 64 is an execution-boundary result, not a saturation result.
GLM-5.3 captures full decode graphs only at batch sizes 1 and 32; batches wider
than 32 run eager, so throughput falls 60% from the c32 peak. Add 64 to
`--cuda-graph-bs-decode` and revalidate before operating that wide.

The same-stack GLM-5.2 control uses an 8192-token prefill chunk. Its published
0.5.17 high-throughput recipe uses 32768, but on this 0.5.18 head that setting
aborted on the first c8 warmup while lazily compiling aiter
`fp8_mqa_logits`, inside Triton/LLVM:

```text
llvm::iota_range<unsigned int>::iota_range:
Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
```

Reducing only the chunk to 8192 made c8 and the complete sweep stable. The
control therefore removes node, engine and workload differences, while this
one recorded flag difference from the published GLM-5.2 cell remains explicit.

## 6. Single-stream latency

`bench_one_batch_server`, BS=1, OSL 1024, three runs per input length:

| ISL | GLM-5.3 E2E s | prefill tok/s | decode tok/s | GLM-5.2 E2E s | prefill tok/s | decode tok/s | decode gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 7.12 | 9,986.94 | 145.88 | 11.74 | 8,085.47 | 88.15 | **1.66x** |
| 8,192 | 7.30 | 43,329.00 | 144.03 | 13.06 | 19,383.98 | 81.03 | **1.78x** |
| 16,384 | 7.49 | 46,039.27 | 143.64 | 13.64 | 19,380.22 | 80.02 | **1.80x** |

The ISL-8192 decode gain, 1.78x, is close to the 1.85x ratio predicted from the
two checkpoints' active weight streams.

## 7. Accuracy

GSM8K uses the in-tree `run_eval` path shared with the GLM-5.2 cookbook:
thinking enabled, temperature 0, 8192 output-token cap and 32 threads. Although
`--num-examples 1319` was requested, this evaluator revision materialized 1,314
examples; both models scored the identical set.

| model | correct / n | score | wall time |
|---|---:|---:|---:|
| GLM-5.3-Flash | 1,281 / 1,314 | **97.49%** | 219.0 s |
| GLM-5.2-FP8 | 1,281 / 1,314 | **97.49%** | 773.2 s |

AIME25 uses `sgl-eval`, never in-tree `run_eval`: 30 problems x 16 repeats,
temperature 1.0, top-p 0.95, thinking enabled, 64K output cap and 32 threads.

| model | pass@1 | SEM | pass@16 | majority@16 | stop | truncated | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | **93.75%** | 0.42 pp | 100% | 100% | 96.04% | 3.96% (19/480) | 0% |
| GLM-5.2-FP8 | **90.83%** | 0.89 pp | 100% | 95% | 99.79% | 0.21% (1/480) | 0% |

GLM-5.3's 19 capped outputs make 93.75% a lower bound under this fixed
comparison protocol. Raising the cap would answer a different question and
must be reported as a separate row, not silently mixed with GLM-5.2's 64K
baseline. GLM-5.3 is +2.92 percentage points, about 3.0 combined standard
errors, despite the higher truncation rate. GLM-5.2 also produced three
no-answer samples, one of which was the truncated output.

## 8. Correctness and determinism boundary

The old bring-up state generated incorrect, degenerate text and produced 24
different continuations from 24 identical greedy requests. PR #36607 fixes the
deterministic error: the final AITER recipe scores 99/100 on the gate and
97.49% on full GSM8K.

It does not make AITER MoE bitwise deterministic. Three prompts x eight serial
greedy repeats, with a cache flush between calls, produced six exact strings.
Changing only `--moe-runner-backend aiter` to `triton` produced one exact
string per prompt across all 24 requests. All AITER variants remained
semantically correct. The residual is the last-bit reduction noise already
isolated by `repro_aiter_moe_nondet.py`; it is no longer catastrophically
amplified into wrong answers.

SGLang's `--enable-deterministic-inference` cannot substitute here: this head
rejects the DSA attention backend before launch. Exact replay therefore
requires the slower Triton-MoE validation configuration; the published
high-throughput cell is explicitly numerically, not bitwise, stable.

## 9. Reproducing and evidence

The canonical scripts are:

```bash
bash glm53_flash/start_container.sh
bash glm53_flash/setup_pr.sh
bash glm53_flash/serve_glm53.sh
bash glm53_flash/eval.sh smoke 20260827T233110Z
bash glm53_flash/bench.sh sanity 20260827T233110Z
bash glm53_flash/bench.sh main 20260827T233110Z
bash glm53_flash/bench.sh lat 20260827T233110Z
bash glm53_flash/eval.sh gsm8k 20260827T233110Z
bash glm53_flash/eval.sh aime25 20260827T233110Z
```

[`gen_glm53_mi355x_rows.py`](gen_glm53_mi355x_rows.py) reads the 15 serving
JSON records and nine latency records, validates the frozen server config and
request/token accounting, enforces the three-repeat set and 5% spread limit,
then emits the `models.js` rows.

Compact evidence for the frozen run is in
[`glm53_flash/results/20260827T233110Z/`](glm53_flash/results/20260827T233110Z/):

- `manifest.json` -- node, image, SGLang/AITER commits, model revision, both
  launch configurations, the correctness gates, the two observed failures, and
  the four upstream PRs at both freeze time and the final status snapshot.
- `performance.json` -- all 15 serving points and nine latency points per model,
  three repeats each, with the observed spread.
- `accuracy.json` -- GSM8K and AIME25 aggregates for both models, the AIME25
  difference in combined standard errors, and the determinism probes.

Large prediction files and server logs stay off-repo. `manifest.json` carries the
sha256 of the two compact artifacts and of each model's AIME prediction set, so
the published numbers can be tied back to the raw run.

## 10. Unmeasured

- MTP/NEXTN on ROCm: the checkpoint has one draft layer, but k-pool target
  verification was not validated in this study.
- Decode graph batch size 64: the current row measures the eager fallback.
- Long-context speed and accuracy past ISL 16,384.
- Image and video quality through the 24-layer vision encoder.
