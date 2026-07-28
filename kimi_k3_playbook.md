# Kimi-K3 on MI355X — playbook

Day-0 bring-up of `moonshotai/Kimi-K3` on 8× AMD Instinct MI355X (gfx950), TP=8,
in both the plain and the DSpark speculative-decoding configuration. Both were
launched, served real traffic, swept for throughput, and evaluated on GSM8K and
AIME25; every number below comes from the run recorded in section 7.

Upstream: [sgl-project/sglang#32541](https://github.com/sgl-project/sglang/pull/32541)
(model support) and [#32548](https://github.com/sgl-project/sglang/issues/32548)
(the AMD Day-0 recipe this reproduces).

**Headline:** the two configurations are accuracy-identical and speed-opposite.
DSpark is 1.54× faster on greedy GSM8K, 3.45× *slower* on sampled AIME25, and
10× slower at 131k context. Pick per workload, not once.

## 1. The model

| | |
|---|---|
| HF path | `moonshotai/Kimi-K3` |
| Checkpoint | 1.56 TB, 96 safetensors shards, 118 files |
| Total params | 2.78 T |
| Active per token | 105.4 B (16 of 896 routed experts) |
| Layers | 93 hybrid — 69 KDA linear-attention + 24 full MLA |
| MoE | 896 routed experts (top-16) + 2 shared, `moe_intermediate_size` 3072 |
| Activation | `situ` (hence `AITER_SITUV2_A8W4`) |
| Context | 1,048,576 |
| Modality | text + vision (`KimiK3ForConditionalGeneration`) |

Quantization is `compressed-tensors` **mxfp4-pack-quantized**, group size 32, and
it covers only the routed-expert `Linear`s. The ignore list keeps `self_attn`,
`shared_experts`, the dense MLP `gate/up/down`, `lm_head`, `vision_tower` and
`mm_projector` in bf16. That split is what the weight footprint follows:

| Component | Params | Bytes |
|---|---:|---:|
| routed experts (MXFP4) | 2722.7 B | 1361.4 GB |
| attention (bf16) | 36.2 B | 72.4 GB |
| shared experts (bf16) | 12.2 B | 24.3 GB |
| other (bf16) | 6.0 B | 12.1 GB |
| embed + lm_head (bf16) | 2.3 B | 4.7 GB |
| vision + projector (bf16) | 0.4 B | 0.9 GB |

So a text decode step touches 137.8 GB for 105.4 B active params — an effective
**1.31 bytes/param**, not the 0.5 the "MXFP4" label suggests, because the bf16
attention and shared experts dominate the *active* set even though the MXFP4
experts dominate the *total*. The roofline entry in `models.js` uses 1.31.

## 2. Launch

```bash
bash test_kimi_k3.sh                # non speculative-decoding
MODE=dspark bash test_kimi_k3.sh    # + DSpark speculative decoding
```

The two configs differ only by two flags:

```
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark
  --speculative-algorithm DSPARK
```

Two things are mandatory rather than optional:

- The four AITER environment variables — see the footprint gotcha in section 6.
- [`dspark_rocm_renorm.patch`](dspark_rocm_renorm.patch) if you serve DSpark with
  any non-greedy sampling — see section 5.

A turnkey alternative is the published Day-0 image
`lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727`, which takes the identical
command and env. The numbers here were *not* measured on it; they come from the
source build in section 7.

## 3. Memory on a 288 GiB card

Target weights land at **194.38 GB/GPU** (1.56 TB ÷ 8) under the aiter MXFP4
path, leaving ~93 GB before pools and graphs. At `--mem-fraction-static 0.85`:

| | non-spec | DSpark |
|---|---:|---:|
| KV cache | 21.35 GB | 14.02 GB |
| `max_total_num_tokens` | 829,332 | 544,533 |
| `max_running_requests` | 368 | 48 |
| free after capture | ~35 GB | ~30 GB |

DSpark pays for the draft weights, a second CUDA-graph set, and an 8-wide verify
window, which is where the KV pool and `max_running_requests` go.

## 4. Accuracy

Speculative decoding is supposed to be lossless — the target verifies every
token — so the point of running both configs is to confirm that, and to have a
regression check that distinguishes a broken verify path (which costs accuracy)
from a merely inefficient one (which costs throughput).

| Eval | non-spec | DSpark | verdict |
|---|---:|---:|---|
| GSM8K, n=1314, greedy | **97.49%** | **97.64%** | parity (2 problems apart) |
| AIME25 pass@1 avg-of-8 | **93.33%** ±4.36% | **94.58%** ±3.05% | parity (0.7 σ) |

Both AIME25 runs were clean: 240 samples each, `stop_rate` 100%, `truncated` 0%,
`no_answer` 0%, `error` 0% at `--max-tokens 64000`.

```bash
# GSM8K — in-tree harness is fine here
python3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \
  --num-examples 1319 --num-threads 32 --max-tokens 8192 --temperature 0

# AIME25 — sgl-eval, NOT in-tree run_eval (same answer-extraction reason as GLM-5.2)
pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 --base-url http://127.0.0.1:30000/v1 \
  --model moonshotai/Kimi-K3 --api-key EMPTY \
  --n-repeats 8 --num-threads 48 --max-tokens 64000 \
  --temperature 1.0 --top-p 0.95 --thinking
```

Both are driven by [`eval_kimi_k3.sh`](eval_kimi_k3.sh). No `--thinking-mode` is
needed: `--reasoning-parser kimi_k3` already puts the trace in
`reasoning_content` and leaves `content` clean for the answer extractor.

## 5. Throughput — and when DSpark is the wrong answer

Synthetic sweep: `sglang.bench_serving`, `--dataset-name random`, ISL 1024 /
OSL 1024, `--random-range-ratio 1`, `--num-prompts` = 2 × concurrency.

**Non speculative-decoding**

| conc | TTFT ms | TPOT ms | decode tok/s | total tok/s | tok/s/GPU |
|-----:|--------:|--------:|-------------:|------------:|----------:|
| 1    | 178     | 19.28   | 51.40        | 102.81      | 12.9      |
| 8    | 993     | 24.72   | —            | 623.57      | 78.0      |
| 32   | 2397    | 35.58   | —            | 1695.74     | 212.0     |

**DSpark**

| conc | TTFT ms | TPOT ms | decode tok/s | total tok/s | tok/s/GPU | accept len |
|-----:|--------:|--------:|-------------:|------------:|----------:|-----------:|
| 1    | 182     | 9.43    | 104.00       | 208.00      | 26.0      | 3.32       |
| 8    | 669     | 18.27   | —            | 758.03      | 94.8      | 3.26       |
| 32   | 1073    | 40.09   | —            | 1338.66     | 167.3     | 3.26       |

The two eval runs are the same story on real traffic, and they bracket the
effect far more sharply than the synthetic sweep does:

| Workload | non-spec | DSpark | |
|---|---:|---:|---|
| GSM8K, 32 threads, greedy, accept 5.95 | 605.4 s / 469 tok/s | 393.7 s / 711 tok/s | DSpark **1.54× faster** |
| AIME25, 48 threads, temp 1.0, accept ~2.9 | 1964.1 s / 692 tok/s | 6779.7 s / 188 tok/s | DSpark **3.45× slower** |

The mechanism is arithmetic, not a bug. DSpark proposes 7 draft tokens and
verifies an 8-wide window every step. At accept length `a`, each accepted token
costs `8/a` target token-slots: 1.34 at `a=5.95`, 2.76 at `a=2.9`. When the batch
is small the target is memory-bound and those extra slots are nearly free, so the
step-count saving wins. When the batch is full the target is compute-bound and
the extra slots come straight off throughput. Concurrency and accept length
therefore have to be read together — which is why this cookbook files DSpark
under *low-latency* and the plain config under *high-throughput*.

### Long context: the plain config is flat, DSpark falls off a cliff

Single stream, OSL 512, `--dataset-name random`:

| ISL | non-spec TTFT | non-spec TPOT | DSpark TTFT | DSpark TPOT |
|------:|-------:|-------:|-------:|--------:|
| 1,024 | 178 ms | 19.28 ms | 182 ms | **9.43 ms** |
| 8,192 | 623 ms | 19.54 ms | 644 ms | **15.20 ms** |
| 32,768 | 3135 ms | **20.41 ms** | 3189 ms | 48.93 ms |
| 131,072 | 24.03 s | **22.13 ms** | 23.96 s | 221.49 ms |

The plain config barely notices: a 128× longer prompt costs 15% more per token.
That is the hybrid architecture doing its job — only 24 of the 93 layers are
full MLA and carry a growing KV cache, the other 69 are constant-state KDA.

DSpark goes the other way, from 2× faster at 1k to 10× slower at 131k, and the
server log says why: accept length collapses to **1.18–1.32** at 131k, an accept
rate of 0.03. The draft model is 5 dense MQA layers; it cannot track that much
context, so every step pays 8 target token-slots to land ~1.2 tokens. TTFT is
identical between the two at every length, which is the expected control —
speculative decoding does not touch prefill.

**So: turn DSpark off for long-context serving.** It is the sharpest
configuration knob on this page.

### Accept length is a workload property, not a platform defect

An earlier revision of this page filed our low accept-length readings as an
unexplained gap against the 5.29–5.93 that #32548 reports. That was wrong, and
the GSM8K run settles it. Measured on this node:

| Workload | Accept length |
|---|---:|
| GSM8K (greedy, structured math, n=1314) | **5.95** (min 3.72, max 7.67) |
| ShareGPT (open-ended chat) | 3.28 |
| `bench_serving` random 1024/1024 | 3.26–3.32 |
| AIME25 (temp 1.0, top_p 0.95, long reasoning) | 2.9–3.0 |
| 131k-token context, single stream | **1.18–1.32** |

The upstream figure sits at the GSM8K-like end of that range and reproduces here
exactly. Both content and sampling move it: predictable structured text lets the
5-layer draft model run ahead, while open-ended prose and a temperature-1.0
target flatten the distribution the draft is trying to guess.

### DSpark + non-greedy sampling crashes on ROCm without the patch

`build_dflash_verify_target_probs` calls `top_k_renorm_prob` /
`top_p_renorm_prob`, which sglang imports from `sgl_kernel` only under
`is_cuda()` or `is_musa()` and leaves as `None` everywhere else. DSpark's triton
accept kernel reaches that helper on every device, so the first decode batch
carrying `top_p` or `top_k` kills the scheduler:

```
File ".../speculative/dflash_utils.py", line 789, in build_dflash_verify_target_probs
    target_probs = top_p_renorm_prob(
TypeError: 'NoneType' object is not callable
```

Greedy traffic never touches the path, so this stays hidden until the first
AIME-style run — GSM8K at `--temperature 0` passes happily, and then the server
dies 30 seconds into AIME25.

DFLASH escapes it because its worker gates the whole non-greedy verify on
`is_dflash_sampling_verify_available()` and silently degrades to greedy argmax
verification. DSPARK has no such gate, and degrading would be the wrong fix
anyway: verifying greedily against a sampled target changes the output
distribution. [`dspark_rocm_renorm.patch`](dspark_rocm_renorm.patch) routes the
renorm to the torch implementations instead — `top_p_normalize_probs_torch`
already existed in `layers/sampler.py`, and the top-k counterpart mirrors the
keep-mask convention of `top_k_top_p_min_p_sampling_from_probs_torch` beside it.
Both match a reference renorm to ~1e-8 and are identity at `top_p=1.0` /
`top_k=vocab_size`.

```bash
git apply dspark_rocm_renorm.patch   # in your sglang checkout
```

## 6. Gotchas

- **The AITER env group is load-bearing.** Drop it and the routed experts are
  unpacked from MXFP4: target weights go 194.38 → 249.29 GB/GPU and the server
  dies in `_profile_available_bytes` with *"Loaded weights leave no GPU memory
  for the KV cache"*, at `--mem-fraction-static` 0.85 **and** 0.93 alike. There
  is no mem-fraction that rescues it on a 288 GiB card.
- **First load is disk-bound, not GPU-bound.** Cold, the 96 shards take ~16 min
  (~25 s/shard); with the page cache warm the same load is 105 s and the whole
  boot is ~3 min. On a 3 TB-RAM node, budget the first launch and then stop
  worrying about restarts.
- **`--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`** are needed to split
  the reasoning trace out of `choices[0].message.content` into
  `reasoning_content`. Without them a short `max_tokens` returns what looks like
  an empty answer, because the budget is spent inside the reasoning block.
- **`--attention-backend triton`** is the recipe's choice for the 24 full-MLA
  layers; the 69 KDA layers pick their own kernels and default to the triton
  packed decode on this path.
- **The draft verify window is not mis-sized.** The draft-worker graph captures
  `num_tokens_per_req=7` while the runner reports `verify_num_draft_tokens=8`;
  `get_num_tokens_per_req_for_target_verify` returns `num_draft_tokens - 1` for
  the DSpark draft worker by design.
- **ROCm backend fallbacks are otherwise correct and silent.**
  `is_sm100_supported()` is false, so the `trtllm_mha` draft default never
  applies (it overrides to triton), and the `nv_cutedsl` verify backend that
  `kimi_k3_hook.py` pins unconditionally resolves to the triton KDA kernel off
  CUDA, with the fused DSpark CuTe MTP path gated behind `is_cuda()`.
- **DSpark cuts `max_running_requests` to 48** (from 368). If you need more
  concurrent streams than that, serve the non-spec config — which, per section 5,
  is also the faster one under exactly those conditions.
- **The vision path works, but only smoke-tested.** A 420×160 PNG with rendered
  text came back correctly transcribed (`image_tokens: 90` in the usage block)
  under both cells — including DSpark, whose draft model is text-only and might
  have been expected to choke on an image-bearing prefill. Every *number* in
  this playbook is still text-only; no image benchmark or eval has been run.
- **GPQA is blocked, not skipped.** The checkpoint ships `.eval_results/`
  claiming GPQA-diamond 93.5 and HLE 56.0. Reproducing GPQA needs
  `Idavidrein/gpqa`, which is a gated HF dataset — an `HF_TOKEN` alone is not
  enough, the account has to be granted access on the dataset page first.
  `sgl-eval` fails with `DatasetNotFoundError` until then.

## 7. Provenance

| | |
|---|---|
| node | 8× AMD Instinct MI355X (gfx950), 288 GiB each, single node |
| sglang | `DarkSharpness/sglang-kimi` @ `amd/kimi-k3` `533bff471` (= #32541 `kimi-k3` + HIP multi-stream disable) + [`dspark_rocm_renorm.patch`](dspark_rocm_renorm.patch), reporting `0.5.15.post1.dev20260723+g6c9fd0adc5` |
| aiter | `k3-for-amd` `68e42f5f` |
| ROCm | 7.2.0 |
| torch | 2.9.1+rocm7.2.0 |
| date | 2026-07-28 |
| script | [`test_kimi_k3.sh`](test_kimi_k3.sh) |

Not the published Day-0 image — see section 2. The GSM8K numbers predate the
patch (greedy, so unaffected); the AIME25 numbers require it.

## 8. Teardown

```bash
pkill -f 'sglang serve'
rocm-smi --showmeminfo vram | grep "Total Used"   # expect ~300 MB per GPU
```
