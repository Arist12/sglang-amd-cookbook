# GLM-5.3 on MI355X (gfx950), SGLang 0.5.18 — same weights shape, one new failure

Single-node **TP=8** deployment of `zai-org/GLM-5.3` on **8× AMD Instinct MI355X** with SGLang and the **DSA tilelang** backend.

> **This is the full GLM-5.3, not [`glm53_flash_playbook.md`](glm53_flash_playbook.md).** Those are different weights and a different architecture — `glm5_next` at 328 GB against `glm_moe_dsa` at 756 GB — and nothing below transfers between them.

GLM-5.3 is a post-training release on GLM-5.2's base, and on this hardware it is **the same serving problem to four kilobytes**:

| | `zai-org/GLM-5.2-FP8` | `zai-org/GLM-5.3` |
|---|---|---|
| `model_type` / architecture | `glm_moe_dsa` | `glm_moe_dsa` |
| `quantization_config` | `fp8` / `e4m3`, block `[128,128]` | `fp8` / `e4m3`, block `[128,128]` |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |
| safetensors shards | 141 | 141 |
| total bytes | 755,663,692,616 | 755,663,688,511 |

There is **no `zai-org/GLM-5.3-FP8`** — the published repository is already fp8, and asking for the `-FP8` name answers **401**, which reads as a permissions problem and is not one.

So the [`glm52_fp8_mi355x_playbook.md`](glm52_fp8_mi355x_playbook.md) recipes transfer, and the interesting content of this playbook is the one thing that does not: **a long cold prefill aborts the server on the `high-throughput` recipe.**

## 0. Environment (verified)

| Item | Value |
|------|-------|
| GPUs | 8× MI355X (gfx950), 288 GiB each, single node |
| Image | `rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260827` — stock, pulled and run unmodified |
| SGLang | `0.5.18.dev20260827+g20a491d1d3` — no fork, no patch |
| ROCm | 7.2.4 (image); host 7.1.1 / amdgpu 6.16.6 |
| aiter | `SGLANG_USE_AITER=1` |
| Weights | `zai-org/GLM-5.3` — 703.8 GiB, 141 shards, downloaded with `huggingface_hub` 1.28 at ~385 MB/s |
| Cold start | **~6 min** to `/health` on the `high-throughput` recipe |
| Parsers | `--reasoning-parser glm45 --tool-call-parser glm47`, the same pair GLM-5.2 uses — confirmed against `/get_server_info`, not assumed |

Six minutes to `/health` is worth noting against GLM-5.2's ~19 min on 0.5.17 in the playbook above. Same shard count, same bytes, same filesystem.

## 1. The finding: a cold bulk prefill aborts the tuned recipe

Deployed on the `high-throughput` recipe — fp8 KV, `--chunked-prefill-size 32768`, `--mem-fraction-static 0.92` — this pool served **76 real agentic calls** and a full 1→16 concurrency sweep without a fault. Then a single chat completion with a long prompt that was **not already in the prefix cache** took every TP rank down:

```
/__w/triton/triton/llvm-project/llvm/include/llvm/ADT/Sequence.h:275:
llvm::iota_range<unsigned int>::iota_range(T, T, bool) [T = unsigned int]:
Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
Fatal Python error: Aborted
  File ".../sglang/srt/utils/watchdog.py", line 147 in _watchdog_once
```

followed by sglang's watchdog calling `kill_process_tree`. The container exits 0 and costs a full weight load to come back. Reproduced twice with the identical assertion. The client sees `HTTP 500` in 7.9 s.

It is an **assertion on an inverted computed range**, not an out-of-memory: `Begin > End` in a `iota_range` inside a compiled kernel. Nothing about the failure is gradual and nothing about it is a resource limit.

### Why a well-exercised pool never finds it

The traffic that had been running on this pool is **prefix replay, not prompting**. Over 6,426 recorded served calls the p50 turn carries **76,549 prompt tokens of which 504 are new** — a long agentic turn is a long cache *hit*, and the cold-prefill path it never touches is where this lives. The pool answered `/health`, served a concurrency sweep at 66.97 tok/s, and completed 26 of 27 real agent calls before the first caller ever sent a long prompt in one piece.

If you serve agents, your monitoring will not find this. If you serve documents, it is your first request.

### The three flags it is suspected of

The serving command SGLang publishes for these weights differs from `high-throughput` in exactly three places, all on the prefill path:

| flag | published | `high-throughput` |
|---|---|---|
| `--chunked-prefill-size` | 131072 | 32768 |
| `--mem-fraction-static` | 0.80 | 0.92 |
| `--kv-cache-dtype` | *(absent — bf16)* | `fp8_e4m3` |

The third is the leading suspect, and the GLM-5.2 playbook in this repository already contains the reason without knowing it: §1.3 records that `fp8_e4m3` KV is legal on the tilelang DSA path **on ROCm only**, and §2.2's gotcha records that the fp8 DSA indexer kernel is **loaded lazily on the first long-context prefill**. That is the code path the assertion fires on, and it is the one thing in the recipe that upstream does not have.

**This has not yet been isolated.** A run of the published command on the same image, same weights, same node — moving only those three flags — is in progress; when it answers, this section gets a verdict rather than a hypothesis. Anyone with a gfx950 node and 704 GiB can run it sooner: serve both recipes and send one 200k-token novel prompt to each.

### What to do until it is isolated

- **Do not advertise a context window you have not sent a cold prompt to.** `max_req_input_len` reads 1,048,570 on this pool and the server accepts the request; the abort is downstream of admission. A front door limit does not protect you, because it cannot tell a cold prompt from a cached one.
- If you serve long documents on `high-throughput`, **assume this is live** until you have ladder evidence on your own build.
- The assertion string is a stable signature to grep a log for: ``Assertion `Begin <= End``.

## 2. Launch

The `high-throughput` recipe, verbatim from the GLM-5.2 playbook with the model path and served name changed. **Read §1 before using it for long prompts.**

```bash
docker run -d --name glm53-serve-ht \
  --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=64g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v /data/GLM-5.3:/models/GLM-5.3:ro \
  -e SGLANG_USE_AITER=1 -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260827 \
  python3 -m sglang.launch_server \
    --model-path /models/GLM-5.3 \
    --served-model-name glm-5.3 \
    --trust-remote-code \
    --tp 8 \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --watchdog-timeout 1200 \
    --host 0.0.0.0 --port 30000 \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.92 \
    --cuda-graph-max-bs 64 \
    --max-running-requests 64 \
    --schedule-policy lpm \
    --num-continuous-decode-steps 2
```

Two notes carried from the GLM-5.2 playbook because they cost time to rediscover. The `--group-add` values must be **numeric**: passing `video` by name resolves against the container's `/etc/group` and grants nothing, `device_count()` still answers 8, and only a real HIP context fails. And on this fleet's four nodes the render gid is **110 on three of them and 109 on the fourth** — read it, do not copy it.

`/get_server_info` on the running pool reports, and these are worth checking rather than assuming:

```
max_total_num_tokens  3735744        max_req_input_len  1048570
tp_size 8             kv_cache_dtype fp8_e4m3           page_size 64
reasoning_parser glm45                tool_call_parser glm47
```

`max_total_num_tokens` 3,735,744 is within 0.5% of GLM-5.2-FP8's 3,717,888 on the same recipe — as it should be, for weights of the same shape and size.

## 3. Sizing

§3 of the GLM-5.2 playbook applies unchanged, and its arithmetic is the one number to carry away:

```
capacity = floor(KV_pool_tokens / peak_context_tokens)
```

With `max_total_num_tokens` 3,735,744 and this fleet's measured p50 agentic prompt of 76,549 tokens, capacity is **48 concurrent conversations**, and `--max-running-requests 64` is the looser of the two bounds. That ordering is the GLM-5.2 finding and it holds here for the same reason.

A corollary worth stating for GLM-5.3 specifically, because its `max_position_embeddings` invites it: **a 1M-token conversation is roughly a quarter of the entire KV pool.** Serving four of them at once fills it. `--max-running-requests 64` and a million-token context are not simultaneously satisfiable, and the flag will not be what tells you.

## 4. Throughput — indicative only, and not to the standard of this repository's tables

**These are not `sglang.bench_serving` numbers and they are not median-of-3.** They were taken through a serving front door on this fleet with a fixed 256-output-token prompt, single run per point, and they are published here as an order-of-magnitude check that GLM-5.3 costs what GLM-5.2 costs — not as datasheet rows. They are deliberately **not** added to `models.js`.

| concurrency | aggregate tok/s | per-stream tok/s |
|---|---|---|
| 1 | 64.6 | 64.6 |
| 2 | 119.7 | 59.9 |
| 4 | 210.9 | 52.8 |
| 8 | 400.5 | 50.1 |

Against the same instrument on GLM-5.2-FP8 (64.9 / 120.4 / 226.7 / 408.6 aggregate), GLM-5.3 is **within ~2% up to concurrency 8**. Which is the expected answer, and is why no recipe tuning was attempted: identical weights shape, identical argv, identical hardware.

A 16-wide point was taken and is **withheld** — a real workload was calling the same pool at the time and the number measures the contention, not the pool. A proper `bench_serving` ladder on an idle pool, three repeats, at the shapes §4 of the GLM-5.2 playbook uses, is what belongs in the tables, and it is not what this is.

## 5. Open

- **Isolate the abort.** The published command against `high-throughput`, three flags apart, one cold prompt each. Then, if the flags are exonerated, the same pair on `v0.5.18-rocm724-mi35x-20260828` — since the assertion is inside a compiled kernel, a fix arrives as an image.
- **Find the boundary.** An ascending cold-prefill ladder — 50k, 100k, 200k, 400k — stopping at the first abort. Ascending because each abort costs a full weight load, and one crash is the price of the whole curve.
- **A verified `bench_serving` table** at the GLM-5.2 shapes, so GLM-5.3 can join the datasheet properly.
