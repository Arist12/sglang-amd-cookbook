# grid_k3 — Kimi-K3 launch-parameter search harness

The harness behind sections 3.1, 5.1–5.3 and 7 of
[`kimi_k3_playbook.md`](../kimi_k3_playbook.md). Runs **inside** the serving
container (there is no `docker` binary in there), unlike the older
[`grid_search.sh`](../grid_search.sh), which builds a container per config.

```bash
GL_BUDGET_MIN=450 ./grid-k3.sh all          # phases 1-4
GL_BUDGET_MIN=55  ./grid-k3-supp.sh         # follow-ups the main phases implied
GL_BUDGET_MIN=330 ./accuracy-k3.sh          # GSM8K + AIME25 gate on the winners
```

| file | role |
|---|---|
| `grid-k3.sh` | phase driver: bisect mem-fraction, then coordinate-descend per lane, then confirm |
| `gridlib.sh` | one config = launch, health/OOM/crash detect, bench, one CSV row, teardown |
| `gridtools.py` | parse bench JSONL, scrape scheduler telemetry, rank, report, degeneration check |
| `accuracy-k3.sh` + `gate-k3.py` | re-run the winners through the published eval protocol and score them |
| `serve-k3-ext.sh` | the launch recipe, fully env-parameterised |
| `test-grid-logic.sh` | exercises the whole phase flow with the GPU calls mocked, in seconds |

Design notes worth carrying to the next model:

- **A boot is not a validation.** Every memory probe runs a real 16k-token prefill at
  the lane's concurrency. Two mem-fraction values that a boot check approves die under
  load.
- **Resumable and deadline-aware.** Any point already in `results.csv` is skipped, and
  the budget guard drops work that will not fit rather than being cut off mid-eval.
- **Telemetry per row.** Running batch, KV usage, mamba usage, queue depth and
  retraction count land in the CSV next to every measurement, which is what turns a
  ranking into an explanation.
- **Compare at one operating point.** Candidates are measured at the incumbent's
  concurrency; only the winner gets a fresh curve. Skipping this once handed the win
  to whichever config happened to get a higher-concurrency point.
- **Measure the noise floor first.** The winning no-spec knob is +1.24%, meaningless
  without the ±0.15% / ±0.35% spreads from repeat runs.
