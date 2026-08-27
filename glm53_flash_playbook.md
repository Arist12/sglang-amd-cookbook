# GLM-5.3-Flash on MI355X (gfx950) — ROCm enablement report

**Status: serves, but output is not trustworthy.** Five patches take GLM-5.3-Flash from
"cannot start" to "starts and answers requests" on 8x MI355X, closing six distinct
ROCm gaps. A seventh problem remains and blocks all scoring: generation is
**nondeterministic at large magnitude**, so no GSM8K, AIME, or throughput number for
this model is meaningful yet. GLM-5.2-FP8 run in the identical container is fully
deterministic and correct, which localizes the fault to GLM-5.3-specific code.

Date: 2026-08-27. Node: `mia1-p02-g46`, 8x MI355X (gfx950, 288 GiB each), ROCm 7.2.4.

## 1. Why the cookbook recipe does not apply

The upstream [GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)
publishes NVIDIA recipes only (H100/H200/B200/B300/GB200/GB300) and directs you to
`docker pull lmsysorg/sglang:glm-5.3-flash`, which is CUDA-only. Three facts change
the approach on ROCm:

- **The model is not in SGLang main.** `Glm5NextForConditionalGeneration` exists only
  in the unmerged PR [#36507](https://github.com/sgl-project/sglang/pull/36507).
- **No ROCm image contains it.** The newest gfx950 nightly,
  `rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260826`, is a build of main and therefore
  lacks the model.
- **Overlaying the PR works.** `docker/rocm.Dockerfile` installs sglang with
  `pip install -e`, and the PR touches neither `python/sglang/kernels/aot/**` nor
  `pyproject.toml`, so a plain `git checkout` takes effect with no rebuild.

Fortunate detail: the PR's merge-base with `origin/main` is `dfc40e0efe10`, the exact
commit the nightly was built from, so there is zero base drift against the image's
compiled aiter / tilelang / sgl-kernel artifacts.

## 2. Environment

| Component | Value |
|---|---|
| Base image | `rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260826` |
| Patched image | `glm53-flash-rocm:pr36507-patched-20260826` (local) |
| torch / HIP / Triton | 2.11.0+rocm7.2 / 7.2.26015 / 3.7.0 |
| aiter / tilelang | `c16d44b9` / `a55a8230` (0.1.7.post3) |
| SGLang | PR #36507 head `fa8735a4ff`, base `dfc40e0efe10` |
| Weights | `zai-org/GLM-5.3-Flash` rev `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`, 305.8 GiB, 72 files |

The published 4x GB300 numbers were measured on revision `c5b82b63e37b`, a different
checkpoint, so any comparison against them is indicative only.

### Model geometry that drives the ROCm behaviour

From the checkpoint's `text_config`: 45 text layers cycling
`[linear_attention x3, deepseek_sparse_attention x1]`, `index_kpool=4`,
`index_topk=2048`, `index_n_heads=32`, `kv_lora_rank=512`, **`qk_rope_head_dim=0`**,
`qk_nope_head_dim=256`, `v_head_dim=256`, `hc_mult=4`, `hc_sinkhorn_iters=20`, 288
routed experts. Two of these decide almost everything below: `index_kpool=4` puts the
model on a brand-new indexer, and `qk_rope_head_dim=0` gives the DSA layers no RoPE
tail, a geometry no prior model on the HIP path had.

Note also that gfx950 reports `torch.cuda.get_device_capability() == (9, 5)`, so
SGLang's `device_sm_major` is 9 and gfx950 takes Hopper-shaped branches throughout the
DSA code.

## 3. Backend selection is forced, not chosen

`dsa_backend.py::_check_kpool_tail_backend` raises unless the DSA backend is one of
`fa3`, `tilelang`, `trtllm` when `index_kpool > 1`. `trtllm` is CUDA-only, `fa3` is
CUDA SM90, and the AMD-native `aiter` DSA backend is **not on that whitelist**, so
TileLang is the only possibility.

`--kv-cache-dtype bfloat16` was chosen because the CUDA-side cookbook states FP8 KV
only pairs with the TRT-LLM DSA backend. **That constraint does not hold on ROCm**:
[`glm52_fp8_mi355x_playbook.md`](glm52_fp8_mi355x_playbook.md) measures `fp8_e4m3` KV
as legal on the tilelang DSA path on ROCm and only on ROCm, taking GLM-5.2's pool from
1,645,440 to 3,194,368 tokens at +0.0 pp GSM8K. FP8 KV was therefore never tried for
GLM-5.3-Flash here and is an open option, worth revisiting once section 5 is resolved.

For linear attention, every KDA kernel except Triton raises on non-CUDA, and Triton is
also the only one implementing the `lower_bound` safe gate this checkpoint's
`linear_lower_bound` requires. For MoE, `deep_gemm` is gated to CUDA/MUSA in its
configurer, leaving `aiter`.

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.3-Flash --served-model-name glm-5.3-flash --tp 8 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype bfloat16 --quantization fp8 \
  --moe-runner-backend aiter \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --max-running-requests 64 --mem-fraction-static 0.85 \
  --watchdog-timeout 1200 --host 0.0.0.0 --port 30000
```

No `--speculative-*`, but for a narrower reason than the older GLM-5.2 playbook gave.
MTP/NEXTN **does** now run on ROCm for `glm_moe_dsa` as of SGLang 0.5.17 (accept length
3.56 of 4, single-stream decode 81 to 200 tok/s at ISL 8192, per
[`glm52_fp8_mi355x_playbook.md`](glm52_fp8_mi355x_playbook.md)), so "the DSA nextn
draft path is CUDA-only" has expired for that architecture. It still holds for this
one: `dsa_indexer_kpool.py` asserts `is_cuda()` outright for kpool target-verify, so
GLM-5.3-Flash cannot speculate on ROCm until that path is ported.
`--disable-shared-experts-fusion` is redundant, since
`shared_experts_fusion_disable_reason()` already disables it on non-CUDA.

Startup on this configuration reports KV cache 8,761,216 tokens (103.76 GB per rank,
bf16), `max_mamba_cache_size=5431` (conv_state 3.17 GB + ssm_state 90.18 GB),
`max_total_num_tokens=8753664`, `context_len=1048576`, ~40 GB free per GPU after the
pools, and a 166 s decode CUDA-graph capture. The KDA state pool is far from limiting
at `--max-running-requests 64`, so the cookbook's `--mamba-full-memory-ratio` /
`--max-mamba-cache-size` tuning advice is not needed here.

## 4. The six ROCm gaps and their fixes

Full diff: [`glm53_flash_rocm.diff`](glm53_flash_rocm.diff) (354 lines, 3 files).
Patch scripts are idempotent, assert their anchor occurrence counts, and abort without
writing on a mismatch.

### 4.1 The kpool indexer never inherited its sibling's ROCm support

`dsa_indexer_kpool.py` wraps its whole forward dispatch in `if is_cuda():` and raises
`NotImplementedError: kpool indexer is only supported on CUDA` otherwise. The block it
guards is pure routing; what is genuinely CUDA-specific is one call underneath,
`deep_gemm.fp8_mqa_logits`.

The sibling indexer `dsa_indexer.py` — the one GLM-5.2 uses on this node today —
already solved this at lines 1161-1162: `if _is_hip: from
aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits`. The kpool indexer added by
PR #36507 is a parallel implementation that never picked it up.

Fix: a `_fp8_mqa_logits` dispatch helper (aiter takes the `(kv, scale)` pair unpacked
and needs no head padding), the three ragged call sites routed through it, the outer
guard opened to HIP, and `sm_count` / `half_device_sm_count` also initialized on HIP
since they were only set under `if is_cuda():`.

### 4.2 The fused kpool top-k kernel is a CUDA JIT module

`fast_kpool_topk_transform_fused` JIT-compiles
`kernels/jit/csrc/dsa/kpool_topk_transform.cuh`, which `#include <cuda_fp16.h>`.
hipcc has no such header, so decode graph capture dies with
`Failed to build JIT module sgl_kernel_jit_kpool_topk_transform_512`. GLM-5.3-Flash
lands there exactly: `group_topk = index_topk / index_kpool = 512`.

Fix: HIP falls through to the portable branch already present in the same function.
Porting the 465-line CUDA radix kernel (warp intrinsics assuming warpSize 32; gfx950
is 64) is a performance task, not a correctness requirement.

### 4.3 The portable branch's group-selection primitive hardcodes topk=2048

That branch calls `sgl_kernel.fast_topk_v2`, which asserts
`topk == 2048` ("only optimized for deepseek v3.2 model"). Everything kpool-specific
after it — `expand_pooled_groups_to_topk`, `append_kpool_tail_to_topk` — is already
Triton and runs on gfx950.

Fix: only the primitive is replaced on HIP, by a `torch.topk`-based equivalent
validated against a brute-force per-row reference over `(B,L,K)` in
{(4,32,8), (8,4096,512), (64,8192,512), (3,10,512)} x `row_starts` in {None, random},
all eight cases set-matching. **This is a known performance regression** and should be
treated as the first thing to optimize, not as representative of the hardware.

The same branch asserted `page_table_row_index is None`; it is now materialized with
`page_table.index_select(0, page_table_row_index)`, the gather the fused kernel did
internally.

### 4.4 `qk_rope_head_dim=0` divides by zero in the HIP TileLang decode kernel

`tilelang_sparse_fwd` computes `tail_dim = dim - d_v`. With `kv_lora_rank=512` and
`qk_rope_head_dim=0`, the MLA-absorb q dim is 512 and `tail_dim = 0`.
`sparse_mla_fwd_decode_partial` then emits `T.Parallel(BI, 0)`, and TileLang's
vectorize planner divides by the loop extent:
`Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`, on all 8 ranks at graph
capture. GLM-5.1/5.2 and DeepSeek DSA all have `tail_dim = 64`, so this had never
fired.

**The CUDA branch of the same function already handles it five lines below**, by
switching kernel factories on `tail_dim == 0`; the HIP branch has no equivalent.

Fix: `has_tail = tail_dim > 0` guards on the six tail-only sites, following the
`has_tail` pattern already used by `sparse_attention_fwd_kernel_v1` in the same file.
Verified numerically equivalent: against a gather/softmax/weighted-sum reference the
patched kernel reaches cosine similarity 0.999996 at `tail_dim=0`, identical to the
`tail_dim=64` control, so the reference is sound and the guards are correct.

### 4.5 The kpool indexer ignores `--dsa-paged-mqa-logits-backend`

SGLang resolves the paged (decode) indexer logits kernel through
`DSAPagedMQALogitsBackend.resolve()`, which returns `AITER` unconditionally on HIP, and
`dsa_indexer.py` stores and honours it. The kpool indexer never wired that up: it has
its own `_should_use_tilelang_paged_mqa_logits()` that only knows tilelang versus
DeepGEMM, so the user-facing flag has no effect and ROCm cannot reach aiter's kernel at
all. Patch 4.1's first cut worked around this by making that helper return `True` on
HIP, which is the wrong layer.

Fix: resolve the backend in `__init__` like the sibling does, add an aiter branch, and
drop the HIP special-case from the tilelang helper. aiter's
`deepgemm_fp8_paged_mqa_logits` wants q 3D (it does its own `next_n` unsqueeze) and the
same 4D `[n, page_size, 1, 132]` kv view the other two take.

This did **not** fix the nondeterminism in section 5 — both tilelang and aiter are
equally affected — but it is the correct wiring and makes the documented flag work.

### 4.6 `--moe-runner-backend triton` cannot start on HIP

`AssertionError: fused silu_and_mul_clamp kernel is CUDA/XPU only; HIP must disable
SWIGLU_CLAMP_FUSION` (`moe_runner/triton_utils/fused_moe.py:696`). Setting
`SGLANG_OPT_SWIGLU_CLAMP_FUSION=0` does not rescue it either; the triton MoE runner
remains unreachable on ROCm for this model. Not patched — `aiter` is the working
default — but it removes an option and blocked one bisection arm.

## 5. The remaining blocker: large-magnitude nondeterminism

With all four patches the server starts, serves, and generates fluent-looking text
that is wrong. The failure is **not** a systematically incorrect kernel:

- Identical greedy request, `temperature=0`, one request at a time, cache flushed
  between calls, so batch composition is constant: **24 of 24 repeats produced
  distinct continuations**, and the very first token varies.
- For `"1, 2, 3, 4, 5, 6,"` the top-1 probability across six repeats was
  0.948 / 0.968 / 0.869 / 0.768 / 0.551 / 0.822. A swing from 0.55 to 0.97 on the same
  input is orders of magnitude above last-bit reduction noise.
- Under `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 --disable-cuda-graph` the server
  stops crashing but output stays wrong; without those it aborts with
  `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` at a position that moves with wherever
  a synchronization happens to land.

### The control that localizes it

**GLM-5.2-FP8, in the same container, same patched SGLang, same node, same
tilelang-DSA + aiter + bf16-KV stack, is fully deterministic and correct**: 1 of 6
distinct per prompt, answering `"The capital of France is Paris. Distance from Paris to
Lyon is"` and `"1, 2, 3, 4, 5, 6, 7, 8, 9"`.

That clears the ROCm stack, the image, the node hardware, aiter, the tilelang DSA
kernels, the MoE runner, and any shared-path side effect of these patches. The fault
is in code only GLM-5.3-Flash exercises: the kpool DSA indexer at `index_kpool=4`, KDA
as configured here, or `glm5_next` itself.

### Ruled out by measurement, not by argument

Do not re-tread these:

| Hypothesis | How it was excluded |
|---|---|
| mHC TileLang kernels | `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0` (torch reference): still 22/24 distinct |
| aiter fused RoPE | `USE_ROCM_AITER_ROPE_BACKEND=0`: still 24/24 distinct |
| aiter as a whole | `SGLANG_USE_AITER=0`: still nondeterministic |
| Prefix / mamba state reuse | `--disable-radix-cache`: still 18/18 distinct |
| CUDA graph capture | every measurement above ran `--disable-cuda-graph` |
| TileLang sparse-MLA decode kernel | synthetic-data check vs reference, cos 0.999996, `tail_dim=0` matching the `tail_dim=64` control |
| The replacement group-topk primitive | set-matches a brute-force reference on 8 shape/`row_starts` combinations |
| Paged MQA logits backend choice | nondeterministic on both tilelang and aiter |
| KDA kernels in isolation | upstream `test/registered/attention/test_kda_kernels.py`, 14 passed on two consecutive runs |
| aiter FP8 bpreshuffle on gfx950 | the documented GLM-5.2 workaround changes nothing (and the earlier single-sample test of this was invalid, see below) |
| Index width 2051 vs 2048 | `_forward_tilelang` already `-1`-pads 2051 to 2112 via `new_full`, upstream code, correct |

Two methodological notes worth carrying forward. First, **single-sample correctness
tests are invalid on a nondeterministic build**; an early conclusion of "prefill is
correct, decode is broken" came from one lucky sample and did not survive a controlled
re-run. Second, **the measurement to bisect against is repeat-count of distinct
outputs, not whether one output looks right** (`determinism_probe.py`).

### The nondeterminism is benign noise plus catastrophic amplification

A later round of per-module tensor dumps and an isolated kernel reproducer changed this
diagnosis, so read the preceding subsection as symptoms and this one as the finding.

**Localization.** `--debug-tensor-dump-output-folder` was used to capture every leaf
module's output on two identical single-token prefills, then diffed
([`glm53_flash/diff_dumps.py`](glm53_flash/diff_dumps.py),
[`glm53_flash/inspect_layer3.py`](glm53_flash/inspect_layer3.py)). Result:

- layers 0, 1, 2 — every module bitwise identical. These are the `first_k_dense_replace`
  dense-MLP layers, and their KDA attention internals agree exactly.
- layer 3, the first MoE layer and the first DSA layer — all eight `self_attn` modules
  bitwise identical, including `attn_mha`, `indexer.wk`, `indexer.k_norm` and
  `kv_b_proj`; `mlp.gate` identical; `mlp.topk[0..2]` identical, so routing agrees
  exactly; `mlp.shared_experts.*` identical; and **`mlp.experts` differs**, 13,672 of
  69,632 elements.

Identical inputs, identical routing, different routed-expert output. That exonerates
KDA, DSA/kpool including the indexer, mHC and the router as the *source*.

**But the magnitude is one bit.** 73.9% of those divergences are exactly one bf16 ULP,
median exactly 1.00 ULP, largest absolute difference 0.015625 on an output whose mean
magnitude is 0.20. And an isolated reproducer
([`glm53_flash/repro_aiter_moe_nondet.py`](glm53_flash/repro_aiter_moe_nondet.py))
calling `aiter.fused_moe.fused_moe` twice on identical synthetic inputs finds it
nondeterministic at ~1e-7 (fp32 last-bit) for **both** models' per-rank shapes —
GLM-5.3-Flash (E=36, h=4096) and the deterministic GLM-5.2 control (E=32, h=6144) —
at `splitk` 0, 1 and 2 alike.

So the MoE reduction noise is ordinary, expected, and shared. **The defect is that
GLM-5.3-Flash amplifies one ULP into a different token while GLM-5.2 does not.** With
top-1 probability sitting at 0.08-0.22 on `"The capital of France is"`, the model is
plausibly in a degenerate regime where a separate *deterministic* numerical error has
flattened the logits enough for last-bit noise to decide the argmax. That also fits the
output being fluent-shaped rather than uniformly broken.

Consequence for the exclusion table above: those toggles establish that mHC, KDA, aiter
as a whole, prefix/state reuse, CUDA graph, `gate_mode`, `splitk`, MoE padding and the
clamp fusion are not the *noise source*. They do **not** exonerate any of them as the
amplifier or as the site of the deterministic error.

Also ruled out along the way, each by measurement: the aiter MoE `swiglu_limit`
branch — GLM-5.3-Flash enters `elif quant_info.swiglu_limit > 0` (10.0) while GLM-5.2
does not, and that branch's own comment says it exists for the gpt-oss MXFP4 layout
rather than FP8 block-quant, yet `GateMode.INTERLEAVE`, `GateMode.SEPARATED` and a
diagnostic patch bypassing the branch entirely all leave the behaviour unchanged.
Note also that `SGLANG_USE_AITER=0` never touched the MoE runner, because
`--moe-runner-backend aiter` was passed explicitly; the earlier "aiter as a whole"
exclusion is weaker than it looks.

### Where to look next

Chasing the nondeterministic component was the wrong frame; the source is benign. The
question to answer is where a **deterministic** numerical error flattens the logits, and
determinism probes cannot find that. The next step is a correctness comparison, not
another bisection: run the same weights through a reference implementation and diff
layer by layer to find the first module that is systematically wrong rather than merely
last-bit unstable. The tensor-dump harness already in place does the diffing; what it
needs is a trustworthy reference for the same inputs.

Candidate sites for that deterministic error, in descending suspicion:

1. mHC. `hc_mult=4` with `hc_sinkhorn_iters=20` is a normalization fixed point applied
   at every layer, and Sinkhorn iteration is exactly the kind of construct that turns a
   small input perturbation into a large output one. It is also GLM-5.3-only, so it fits
   the GLM-5.2 contrast. Switching to the torch reference did not restore determinism,
   but that only rules mHC out as the noise source, not as the amplifier — and note the
   standalone probe of `_mhc_pre_torch` could not be completed because the kernel
   requires an initialized TP group, so the torch path itself is unverified.
2. The kpool compress-and-write path (`kpool_softmax_rotate_write_cache`): four tokens
   folded into one pooled cache entry. Layer 3's indexer outputs were bitwise identical
   on the pair examined, which weakens this, but identical is not the same as correct.
3. KDA at this checkpoint's head geometry with `lower_bound` active — upstream kernel
   tests pass but do not obviously cover this configuration, and KDA has no AMD CI
   coverage ([#19324](https://github.com/sgl-project/sglang/pull/19324) still open) plus
   an open ROCm hang report
   ([#33846](https://github.com/sgl-project/sglang/issues/33846)).
4. `glm5_next.py` itself — note that its one gfx950-specific path, the
   `rocm_linear_utils` zero-allocator, is gated on `n_routed_experts == 256` while this
   model has 288, so it never activates.

A separate genuine gap, found while testing `SGLANG_DSA_FUSE_TOPK=0`: the non-fused
decode transform asserts `topk_indices.shape[1] == 2048` at
`kernels/ops/attention/dsa/transform_index.py:194`, but kpool produces 2051
(`index_topk + index_kpool - 1`). That hardcoded 2048 predates kpool and is not
ROCm-specific.

## 6. GLM-5.2-FP8 same-build baseline

Because GLM-5.2-FP8 is correct and deterministic on this build, it is a usable
baseline, and running it here removes both the node and the SGLang-version confound
from the playbook's June numbers. Weights were mounted read-only from
`/var/tmp/models/GLM-5.2-FP8`.

Throughput, `bench_serving --dataset-name random`, ISL 8192 / OSL 1024,
`--random-range-ratio 1.0`, `--num-prompts 2*C`:

| conc | output tok/s | median TTFT (ms) | median TPOT (ms) | playbook (g45, v0.5.13.post1) | delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 74.56 | 471 | 12.95 | 66.92 / 652 / 14.43 | +11.4% |
| 16 | 576.08 | 4389 | 23.50 | 535.66 / 4790 / 25.22 | +7.5% |
| 64 | 927.21 | 17833 | 44.21 | 1008.95 / 13795 / 41.45 | -8.1% |

The nearer comparison point is the 0.5.17 re-measurement in
[`glm52_fp8_mi355x_playbook.md`](glm52_fp8_mi355x_playbook.md), whose continuity arm
carried the same no-speculation recipe onto a newer image and reproduced the June
concurrency-16 row to 0.7% (531.7 output tok/s, TPOT 25.01). My 576.08 on
0.5.18 + PR #36507 is 8.3% above that, on a config differing only in SGLang version.
I cannot attribute that gap without repeats, so read the deltas above as directional.
Their configurations also differ from mine in ways worth knowing before comparing
further: `--chunked-prefill-size 16384`, `--cuda-graph-max-bs 32`,
`--max-running-requests 32`, and MTP enabled.

Latency, `bench_one_batch_server`, bs=1, OSL 1024 — **new data**, the playbook records
that MI355X `bench_one_batch` had never been captured:

| ISL | end-to-end (s) | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 1024 | 12.33 | 7594.22 | 83.97 |
| 8192 | 14.92 | 4805.96 | 77.51 |
| 16384 | 14.53 | 14080.76 | 76.62 |

For reference the playbook's MI300X figures at ISL 8192 are 6170 tok/s prefill and
51.0 tok/s decode, so MI355X decode is roughly 52% higher.

Two caveats. Each cell is a single measurement, so the low-concurrency gains and the
concurrency-64 regression are directional, not tight. And concurrency 64 hung once
before succeeding on retry: a forward pass exceeded the 1200 s watchdog
(`watchdog.py:147`, in `deepseek_v2.py:2510 forward`) and aborted the server, then the
identical run completed at 927.21 tok/s. Treat that as an observed one-off, not a
reproducible regression; the 8% throughput drop and the TTFT moving 13.8 s to 17.8 s
do suggest this build is tighter under heavy prefill queueing.

These patches are inert for GLM-5.2: the two kpool files only act when
`index_kpool > 1` (GLM-5.2 has 1), and the `has_tail` guards all evaluate true at
`tail_dim = 64`, executing exactly the pre-patch code.

## 7. Not obtainable today

GSM8K, AIME25, and any GLM-5.3-Flash throughput or latency figure. Scoring a
nondeterministic server produces a number that describes the noise, not the model. The
comparison against GLM-5.2-FP8 and against the published 4x GB300 results has to wait
on section 5.

Also untested for GLM-5.3-Flash: the multimodal path (the 24-layer vision encoder;
`--mm-feature-transport` does correctly auto-resolve to `cpu`), video (needs
`torchcodec`, absent), and MTP / `--speculative-algorithm NEXTN`, which is pointless to
probe before basic correctness holds.

## 8. Reproducing

```bash
# 1. base image + weights (306 GiB; the HF xet CAS path returns 416 for this repo,
#    so HF_HUB_DISABLE_XET=1 is required)
docker pull rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260826
HF_HOME=/data/jhinpan-cache HF_HUB_DISABLE_XET=1 hf download zai-org/GLM-5.3-Flash

# 2. container, then overlay the PR
bash start_container.sh
bash setup_pr.sh                       # git fetch pull/36507/head, checkout, verify

# 3. the patches, in order (the last two are fixups to patch_kpool_paged_aiter)
docker exec glm53-flash bash -lc 'cd /sgl-workspace/sglang && \
  python3 /results/patch_kpool_rocm.py && \
  python3 /results/patch_kpool_topk_rocm.py && \
  python3 /results/patch_group_topk_rocm.py && \
  python3 /results/patch_tilelang_tail_dim0.py && \
  python3 /results/patch_kpool_paged_aiter.py && \
  python3 /results/fix_aiter_kv_rank.py && \
  python3 /results/fix_helper_placement.py'

# 4. serve, then measure determinism BEFORE anything else
bash serve_glm53.sh
python3 determinism_probe.py 8 8       # expect NONDETERMINISTIC 24/24 today
```

Harness scripts live alongside this file under `bench/` and in
`/home/jinpan12@amd.com/glm53-flash-results/` on `mia1-p02-g46`; the patched container
is saved as `glm53-flash-rocm:pr36507-patched-20260826`.
