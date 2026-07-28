# Kimi-K3 on MI355X — playbook

Day-0 bring-up of `moonshotai/Kimi-K3` on 8× AMD Instinct MI355X (gfx950), TP=8,
in both the plain and the DSpark speculative-decoding configuration. Both were
launched, served real traffic, and swept; every number below comes from the run
recorded in section 5.

Upstream: [sgl-project/sglang#32541](https://github.com/sgl-project/sglang/pull/32541)
(model support) and [#32548](https://github.com/sgl-project/sglang/issues/32548)
(the AMD Day-0 recipe this reproduces).

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

The four AITER environment variables are **mandatory on AMD, not tuning knobs**
— see the footprint gotcha in section 4.

A turnkey alternative is the published Day-0 image
`lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727`, which takes the identical
command and env. The numbers here were *not* measured on it; they come from the
source build in section 5.

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

## 4. Benchmarks

`sglang.bench_serving`, `--dataset-name random`, ISL 1024 / OSL 1024,
`--random-range-ratio 1`, `--num-prompts` = 2 × concurrency.

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

DSpark roughly doubles single-stream decode (51.4 → 104.0 tok/s) and cuts TTFT
and TPOT at low and medium concurrency. It **loses** on aggregate throughput at
concurrency 32 (1338.66 vs 1695.74 tok/s): once the batch is full the target is
already compute-saturated and the rejected draft tokens are wasted work. That
crossover is why this cookbook files DSpark under *low-latency* and the plain
config under *high-throughput* — the same crossover appears in the reference
table in #32548 (4898 vs 3715 tok/s at concurrency 32).

### Open: accept length is ~3.3, upstream reports 5.75

#32548 reports accept length 5.29–5.93 across concurrency 2–32 on MI355X. We
measure 3.26–3.32, i.e. an accept rate of 0.32 rather than 0.68 against the same
`gamma=7` / 8-wide verify window. Ruled out so far:

- **Workload** — ShareGPT gives 3.28, random 1024/1024 gives 3.26. Same.
- **Sampling** — `temperature=0` greedy gives 2.3–3.7. Same.
- **`AITER_SITUV2_A8W4`** — disabling it gives 2.5–3.6. Same.
- **Draft window mis-sizing** — the draft-worker verify graph captures
  `num_tokens_per_req=7`, which is `num_draft_tokens - 1` by design in
  `SpeculativeAlgorithm.get_num_tokens_per_req_for_target_verify`, not a bug.
- **ROCm backend fallbacks** — all correct: draft attention overrides to triton,
  and the `nv_cutedsl` verify backend that `kimi_k3_hook.py` pins resolves to the
  triton KDA kernel off CUDA, with the fused DSpark CuTe path behind `is_cuda()`.

Output quality is unaffected (the target verifies every token), so this costs
speed, not correctness. The most likely remaining cause is version skew against
whatever sglang+aiter pair the Day-0 image pins; the next step is to run the same
sweep inside `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` and compare.

## 5. Provenance

| | |
|---|---|
| node | 8× AMD Instinct MI355X (gfx950), 288 GiB each, single node |
| sglang | `DarkSharpness/sglang-kimi` @ `amd/kimi-k3` `533bff471` (= #32541 `kimi-k3` + HIP multi-stream disable), reporting `0.5.15.post1.dev20260723+g6c9fd0adc5` |
| aiter | `k3-for-amd` `68e42f5f` |
| ROCm | 7.2.0 |
| torch | 2.9.1+rocm7.2.0 |
| date | 2026-07-28 |
| script | [`test_kimi_k3.sh`](test_kimi_k3.sh) |

Not the published Day-0 image — see section 2.

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
- **DSpark cuts `max_running_requests` to 48** (from 368). If you need more
  concurrent streams than that, serve the non-spec config — which, per the table
  above, is also the faster one in aggregate past ~concurrency 32.
- **Multimodal is present but untested here.** `KimiK3ForConditionalGeneration`
  carries a vision tower; every number in this playbook is text-only.

## 7. Teardown

```bash
pkill -f 'sglang serve'
rocm-smi --showmeminfo vram | grep "Total Used"   # expect ~300 MB per GPU
```
