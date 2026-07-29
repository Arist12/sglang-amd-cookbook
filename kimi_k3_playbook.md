# Kimi-K3 on MI355X — playbook

Day-0 bring-up of `moonshotai/Kimi-K3` on 8× AMD Instinct MI355X (gfx950), TP=8,
in both the plain and the DSpark speculative-decoding configuration. Both were
launched, served real traffic, swept for throughput, and evaluated on GSM8K and
AIME25; every number below comes from a run recorded in section 8.

Upstream: [sgl-project/sglang#32541](https://github.com/sgl-project/sglang/pull/32541)
(model support) and [#32548](https://github.com/sgl-project/sglang/issues/32548)
(the AMD Day-0 recipe this reproduces).

**Headline:** the two configurations are accuracy-identical and speed-opposite.
DSpark is 1.54× faster on greedy GSM8K, 3.45× *slower* on sampled AIME25, and
10× slower at 131k context. Pick per workload, not once.

**Second headline, from the parameter search in section 7:** the Day-0 recipe
leaves a lot on the table, and two knobs recover almost all of it.
`--mem-fraction-static 0.93` (up from 0.85) lifts the no-spec KV pool 54% and moves
peak throughput from 6,198 to 7,892 tok/s — **+27%, 987 tok/s/GPU** — because the
Day-0 setting left 35 GB/GPU unused while the scheduler sat at `full token usage
0.99`. For DSpark, `--speculative-dspark-block-size 3` is worth **+68%** (2,142 →
3,606 tok/s) *and* cuts median TTFT from 11.2 s to 6.6 s and TPOT from 171 to
100 ms — a shorter verify window is better on every axis, not a trade. Sections 3
and 5 carry the numbers; section 7 records what did *not* work, which is most of it.

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
source build in section 8.

### 2.1 Tuned recipes

The Day-0 command above is the reproduction target. These are what the parameter
search in section 7 actually recommends; both are one flag away from Day-0.

**Throughput lane** — batch serving, offline jobs, anything that cares about
aggregate tokens per second. Operate it at **concurrency 128**:

```shell Command
export SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1
sglang serve \
  --model-path moonshotai/Kimi-K3 \
  --trust-remote-code \
  --tp 8 \
  --attention-backend triton \
  --dtype bfloat16 \
  --mem-fraction-static 0.93 \
  --cuda-graph-max-bs-decode 256 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --host 0.0.0.0 --port 30000
```

**Interactive lane** — chat, agents, anything that cares about time per token.
Good from concurrency 1 up to about 48:

```shell Command
export SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1
sglang serve \
  --model-path moonshotai/Kimi-K3 \
  --trust-remote-code \
  --tp 8 \
  --attention-backend triton \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --cuda-graph-max-bs-decode 256 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 3 \
  --host 0.0.0.0 --port 30000
```

Two deliberate omissions, both measured rather than assumed:

- **Do not raise `--mem-fraction-static` past these values.** 0.94 and 0.95 both
  *boot cleanly*, report a healthy-looking 9.35 GB / 6.45 GB free, and then die
  under load. Section 3 has the failing allocation.
- **`--mamba-ssm-dtype bfloat16` is left out on purpose.** It is a real +1.24%
  (7,892 → 7,990 tok/s, reproduced three times each, non-overlapping), but it
  changes SSM state precision, and a numerics knob is not worth 1% unless you have
  re-run your own accuracy gate. If you want it, section 4 has ours.

If your traffic has shared prefixes — multi-turn chat, agent loops, a common system
prompt — add `--mamba-radix-cache-strategy extra_buffer_lazy` and drop
`--disable-radix-cache`. That is worth **1.52×** on reused prefixes for 0.8% on
traffic with none; see section 5.3.

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

### 3.1 0.85 was leaving 35 GB/GPU idle

"Free after capture ~35 GB" in the table above is not headroom, it is waste.
SGLang's own [hyperparameter guide](https://docs.sglang.ai/advanced_features/hyperparameter_tuning)
calls 5–8 GB the healthy target and says to raise `--mem-fraction-static` until you
hit OOM. Nobody had. And it mattered, because the scheduler was reporting
`full token usage 0.99` with requests queued at the same time — the pool was
simultaneously full and under-allocated.

`available_gpu_mem` is very nearly linear in mem-fraction (slope −288 GB per unit,
the card's capacity), so one measurement predicts the target instead of needing a
blind sweep. Probing from 0.85 and stepping down from the prediction:

| mem-fraction | `max_total_num_tokens` | `max_running_requests` | free after capture | 16k-prefill load test |
|---:|---:|---:|---:|---|
| 0.85 (Day-0) | 838,048 | 370 | 35.13 GB | pass |
| 0.93 | **1,292,032** | 570 | 12.21 GB | **pass ← no-spec** |
| 0.94 | 1,348,780 | 595 | 9.35 GB | **boots, then dies** |
| 0.95 | 1,405,528 | 620 | 6.45 GB | **boots, then dies** |

DSpark has its own boundary one step lower, since the draft weights and verify
window add activation pressure:

| mem-fraction | `max_total_num_tokens` | free after capture | 16k-prefill load test |
|---:|---:|---:|---|
| 0.85 (Day-0) | 551,629 | 30.38 GB | pass |
| 0.92 | **1,174,618** | 12.81 GB | **pass ← DSpark** |
| 0.93 | 1,263,617 | 10.30 GB | **boots, then dies** |

So no-spec gains **+54%** KV capacity and DSpark **+113%**, for free.

**The part worth remembering: a clean boot proves nothing here.** 0.94 and 0.95
start, serve `/health`, and print a plausible `available_gpu_mem`; they fall over on
the first heavy prefill. The allocation that fails is not attention, it is the aiter
MXFP4 fused-MoE stage-2 output buffer:

```text Output
File "/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py", line 1782, in flydsl_moe_stage2
  target = torch.empty(
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.75 GiB.
GPU 3 has a total capacity of 287.98 GiB of which 212.00 MiB is free.
```

That buffer scales with tokens per forward pass, so it peaks at
`chunked-prefill-size`, not at your batch size. Practical consequence: **K3's safe
floor on this platform is ~12 GB free after capture, not the 5–8 GB the generic
guide suggests**, and any mem-fraction change has to be validated with a real
long-prefill load test rather than a boot and a health check. If you change
`--chunked-prefill-size` upward, re-validate — you have moved the peak.

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

### 4.1 The tuned recipes are output-neutral

Both section 2.1 recipes were re-run through the identical protocol. Full report:
[`accuracy_gate.md`](grid_results/20260729_091009/accuracy_gate.md).

| Eval | tuned no-spec | Day-0 | tuned DSpark | Day-0 |
|---|---:|---:|---:|---:|
| GSM8K n=1319 greedy | **97.489%** | 97.49% | **97.641%** | 97.64% |
| AIME25 pass@1 avg-of-8 | **91.67%** ±3.09% | 93.33% ±4.36% | **95.42%** ±3.54% | 94.58% ±3.05% |
| AIME25 stop rate | 99.17% | 100% | **100%** | 100% |

Worth being explicit about which knobs this gate is actually for, because most of them
cannot fail it. `--mem-fraction-static`, `--chunked-prefill-size`,
`--cuda-graph-max-bs-decode`, `--max-running-requests`, `--schedule-conservativeness`
and `--mamba-radix-cache-strategy` change memory and scheduling only; they cannot move
the token distribution. Two can, and they are the reason for the GPU hours:

- **`--speculative-dspark-block-size 3` is clean.** AIME25 landed 0.84 pp *above* the
  Day-0 DSpark baseline with a perfect stop rate. Speculative decoding is lossless
  because the target verifies every token, and shortening the verify window does not
  change that — this is the evidence, and the knob ships.
- **`--mamba-ssm-dtype bfloat16` passes but is not clean.** It is the only config that
  lost samples to truncation (0.83% truncated, 0.83% no-answer, stop rate 99.17%) and
  it carries the −1.66 pp AIME25 delta. That delta is 0.88 σ of the pooled SEM, so it
  is not a detectable regression — but paired with a measured gain of only +1.24% it is
  not a trade worth making blind, which is why section 2.1 leaves it out.

The GSM8K numbers are the sharper instrument here despite looking boring: greedy
decoding means a broken verify path or a mis-quantised state shows up immediately, and
both configs reproduce the baseline to three decimal places.

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

### 5.1 High concurrency, and where the tuned recipes land

The table above is ISL 1024 / OSL 1024, which is a latency probe. For throughput the
workload that matters is ISL 8192 / OSL 1024 at saturation — the protocol #32548
reports. `num-prompts` = 2 × concurrency, `--random-range-ratio 1`,
`--warmup-requests 4 --flush-cache`.

**No-spec, `--mem-fraction-static 0.93`.** The Day-0 0.85 setting saturated near
concurrency 90; with a 54% bigger KV pool the knee moves to **128**:

| conc | TTFT med | TPOT | total tok/s | tok/s/GPU | KV use | queued | running |
|-----:|---------:|-----:|------------:|----------:|-------:|-------:|--------:|
| 96   | 29.5 s   | 92.9 ms  | 7,094 | 887 | 0.69 | 0  | 96  |
| **128** | **39.0 s** | **108.1 ms** | **7,892** | **987** | 0.92 | 0  | 128 |
| 160  | 48.4 s   | 117.0 ms | 7,044 | 881 | 0.99 | 22 | 138 |
| 192  | 60.7 s   | 119.6 ms | 7,352 | 919 | 0.99 | 54 | 138 |

Day-0 for comparison: 6,198 tok/s (775/GPU) at concurrency 96, with KV at 0.99 and
7 requests already queued. The tuned number is a three-run mean (7,876 / 7,896 /
7,903), so **+27%** is reproducible, not a lucky sample.

Read the last three columns together and the ceiling explains itself: 128 is the
largest batch the pool holds outright, and past it `running` pins at 138 while the
queue grows — so the extra clients buy latency, not throughput. Concurrency 128 is
therefore the number to operate at, not just the number we measured.

**DSpark, `--mem-fraction-static 0.92`, at concurrency 48.** The lever here is the
draft window, and it has a genuine interior optimum:

| `--speculative-dspark-block-size` | verify window | total tok/s | TTFT med | TPOT | accept |
|---:|---:|------------:|---------:|-----:|-------:|
| 2 | 3 | 3,305 | 6.7 s | 109.1 ms | 2.22 |
| **3** | **4** | **3,606** | **6.6 s** | **100.1 ms** | 2.55 |
| 5 | 6 | 2,118 | 10.1 s | 174.7 ms | 2.89 |
| 7 (default) | 8 | 2,142 | 11.3 s | 171.0 ms | 3.00 |

That is a step, not a slope: 5 and 7 are indistinguishable, 3 is 68% better, 2 gives
some of it back. The arithmetic in the section below explains why — at accept length
`a` each accepted token costs `window/a` target slots, so cutting the window from 8
to 4 nearly halves the tax while accept length only falls from 3.00 to 2.55. Below 3
the window is too short to amortise a step at all.

And it is not a throughput-for-latency trade. Block size 3 wins the latency probe
too, at every concurrency:

| conc | TPOT, block 3 | TPOT, default | decode tok/s, block 3 | decode tok/s, default |
|-----:|--------------:|--------------:|----------------------:|----------------------:|
| 1 | **8.78 ms**  | 9.84 ms  | 111.6 | 99.7  |
| 4 | **11.74 ms** | 14.55 ms | 289.1 | 253.8 |
| 8 | **15.25 ms** | 18.37 ms | 470.7 | 382.9 |

### 5.2 What did not work

Worth recording, because these are the knobs everyone reaches for first. All at
no-spec, mem-fraction 0.93, concurrency 128, against a 7,876 tok/s baseline:

| knob | result | why |
|---|---|---|
| `--chunked-prefill-size 32768` / `65536` | 7,893 / 7,890 (+0.2%) | noise; TTFT at saturation is queueing, not chunk size |
| `--chunked-prefill-size 8192` | 7,342 (−6.8%) | smaller chunks just cost prefill efficiency |
| `--cuda-graph-max-bs-decode 384` | 7,874 (±0%) | the running ceiling is 138, so 256 already covers every replayed batch |
| `--cuda-graph-max-bs-decode 512` | **crash** | costs 4 GB of headroom, which puts it under the floor in section 3.1 |
| `--schedule-conservativeness 0.6` | 7,867 (±0%) | nothing to fix — the tuned config never retracts and never queues at 128 |
| `--max-running-requests` 16–40 (DSpark) | 2,107–3,265 | "gains" are an artifact: capping the server below the offered load pushes median TTFT to 25–96 s |
| `--enable-linear-replayssm-spec` (DSpark) | 2,143 (±0%) | cuts median TTFT 11.2 s → 3.2 s on its own, but adds nothing once block size 3 is set |

The `--max-running-requests` row is the one to be careful with. Measured naively it
looks like a 52% throughput win (3,265 vs 2,142 at mrr=24), and an earlier round of
confounded runs read it exactly that way. Holding every other knob fixed shows the
mechanism: at mrr=24 with 48 clients offered, 27 requests sit in the queue and median
TTFT is 70 s. The server is not faster, it is serving fewer people.

### 5.3 Shared prefixes: prefix caching is worth 1.52×

The Day-0 recipe ships `--disable-radix-cache`, which is right for synthetic
benchmarks and wrong for agentic or multi-turn traffic. On `generated-shared-prefix`
(32 groups × 8 prompts, 4K shared system prompt, concurrency 32), no-spec at
mem-fraction 0.93:

| config | total tok/s | TTFT med | cache hit | cost on no-reuse traffic |
|---|------------:|---------:|----------:|---:|
| `--disable-radix-cache` | 8,276 | 5.9 s | 0% | — |
| radix on, default strategy | 12,471 | 1.7 s | 68.2% | **−14.4%** |
| radix on, `extra_buffer_lazy` | **12,561** | **1.5 s** | 68.2% | **−0.8%** |

Both cache strategies deliver the same 1.5× on reuse; the difference is what they
cost when there is none, and it is entirely a KDA state-slot story. Prefix caching on
a hybrid model has to keep linear-attention state per cached prefix, which the
allocator charges per request: **5 slots/request** under the default strategy versus
**4** under `extra_buffer_lazy`. That drops `max_running_requests` from 570 to 114 and
142 respectively — and 114 is below the throughput-optimal batch of 128, so the
default strategy starts queueing (15 requests, KV use 0.81) and loses 14%, while
`extra_buffer_lazy` still clears 128 and loses under 1%.

So `extra_buffer_lazy` is close to a free option: take it whenever the traffic *might*
have reuse. The default strategy is only worth it if you know every request shares a
prefix.

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

### 5.4 Long context: the plain config is flat, DSpark falls off a cliff

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

### 5.5 Accept length is a workload property, not a platform defect

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

### 5.6 DSpark + non-greedy sampling crashes on ROCm without the patch

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

## 7. How the optimum was found

The method is reusable for the next model, so it is worth writing down. Harness:
[`grid_k3/grid-k3.sh`](grid_k3/grid-k3.sh) plus
[`grid_k3/gridlib.sh`](grid_k3/gridlib.sh); raw rows in
[`grid_results/20260729_091009/results.csv`](grid_results/20260729_091009/results.csv).

**A full grid was never on the table.** Eight knobs at three values each is 6,561
configurations, and a configuration here costs a 3-minute boot plus a 5-minute
benchmark — about seven months. So the search attacks the constraint instead of the
cross product, in three stages.

**Stage 1: read the binding constraint off the scheduler, don't guess it.** Every
decode line reports what is full:

```text Output
Decode batch, #running-req: 89, #full token: 825k, full token usage: 0.99,
  mamba num: 89, mamba usage: 0.24, cuda graph: True, #queue-req: 7
```

`full token usage 0.99` with `mamba usage 0.24` says the no-spec ceiling is the MLA
KV pool, full stop. DSpark inverts it — `0.66` KV against `0.98` mamba — so the two
lanes need different knobs, and one number in the log settles which.

**Stage 2: bisect the monotone scalar, don't sweep it.** `--mem-fraction-static` is
continuous and sits directly on the no-spec constraint, and `available_gpu_mem` is
linear in it with a known slope (−288 GB per unit, the card's capacity). One anchor
measurement at 0.85 therefore *predicts* the target rather than needing a blind
sweep: 0.85 + (35.13 − 7)/288 = 0.95. Probing 0.95, then stepping down as it and
0.94 failed the load test, found the boundary in **three probes instead of five**,
and every probe was informative because the prediction put them next to the edge.

**Stage 3: coordinate-descend the discrete knobs, one at a time, at a fixed
operating point.** Each candidate is measured at the same concurrency as the
incumbent, then the winner gets a fresh concurrency curve because tuning moves where
saturation sits. Coordinate descent cannot see interactions, so the top two knobs per
lane were finally combined and re-measured — for no-spec that produced 8,012 tok/s
against 7,990 for the better knob alone, i.e. nothing, which is the useful answer.

Four practices did most of the work, and all four came from getting burned:

- **A boot is not a validation.** Each memory probe runs a 16k-token prefill at the
  lane's real concurrency. Both mem-fraction values that a boot check would have
  approved die on that test (section 3.1).
- **Measure the noise floor before believing a delta.** The winning no-spec knob is
  +1.24%, which is meaningless without a spread. Three runs of each config gave
  ±0.15% and ±0.35% with non-overlapping ranges, so it is real — and small enough
  that we still do not ship it by default.
- **Never compare configs at different operating points.** An early version of this
  search picked ReplaySSM as the DSpark winner purely because it happened to be
  given a concurrency-96 point while its rivals only had 48.
- **Capture the telemetry with every row.** `results.csv` carries running batch, KV
  usage, mamba usage, queue depth and retraction count per measurement. That is what
  turns "128 is fastest" into "128 is the largest batch the pool holds, and past it
  `running` pins at 138" — a result you can reason about instead of memorise.

Cost: 44 configurations, 66 measurements, about 6.5 h of the 8 h budgeted, plus the
accuracy gate in section 4.

**One caveat on the ceiling.** This search finds the best point inside the
AMD-reachable envelope, which is not the same as closing the gap to published NVIDIA
figures. Three of the levers those recipes use are unavailable on gfx950 today: DCP
(`--dcp-size`, which would multiply exactly the MLA KV pool that binds here) asserts
a Blackwell MLA backend, verify-budget trimming needs a ROCm MLA backend that sets
`supports_ragged_verify_graph`, and a PP prefill lane is incompatible with DSpark's
`pp_size == 1`. Those are the next real wins, and none of them is a launch flag.

## 8. Provenance

| | |
|---|---|
| node | 8× AMD Instinct MI355X (gfx950), 288 GiB each, single node |
| sglang | `DarkSharpness/sglang-kimi` @ `amd/kimi-k3` `533bff471` (= #32541 `kimi-k3` + HIP multi-stream disable) + [`dspark_rocm_renorm.patch`](dspark_rocm_renorm.patch), reporting `0.5.15.post1.dev20260723+g6c9fd0adc5` |
| aiter | `k3-for-amd` `68e42f5f` |
| ROCm | 7.2.0 |
| torch | 2.9.1+rocm7.2.0 |
| date | 2026-07-28 (Day-0 bring-up), 2026-07-29 (parameter search, sglang `3d35b45f7`) |
| scripts | [`test_kimi_k3.sh`](test_kimi_k3.sh) for the Day-0 numbers, [`grid_k3/grid-k3.sh`](grid_k3/grid-k3.sh) for sections 3.1, 5.1–5.3 and 7 |

Not the published Day-0 image — see section 2. The GSM8K numbers predate the
patch (greedy, so unaffected); the AIME25 numbers require it.

## 9. Teardown

```bash
pkill -f 'sglang serve'
rocm-smi --showmeminfo vram | grep "Total Used"   # expect ~300 MB per GPU
```
