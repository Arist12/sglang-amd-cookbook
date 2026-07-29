### Capacity by launch recipe

| config | lane | mem-frac | chunked-prefill | cuda-graph-bs | extra args | max_total_num_tokens | max_running_requests | avail GB |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `p1-nospec-mf0.85` | mem-nospec | 0.85 | default | 256 | `—` | 838,048 | 370 | 35.13 |
| `p1-nospec-mf0.93` | mem-nospec | 0.93 | default | 256 | `—` | 1,292,032 | 570 | 12.21 |
| `p1-dspark-mf0.85` | mem-dspark | 0.85 | default | 256 | `—` | 551,629 | 48 | 30.38 |
| `p1-dspark-mf0.92` | mem-dspark | 0.92 | default | 256 | `—` | 1,174,618 | 48 | 12.81 |
| `p2-base-mf0.93` | nospec | 0.93 | default | 256 | `—` | 1,292,032 | 570 | 12.21 |
| `p2-cp8192` | nospec | 0.93 | 8192 | 256 | `—` | 1,292,032 | 570 | 12.21 |
| `p2-cp32768` | nospec | 0.93 | 32768 | 256 | `—` | 1,292,032 | 570 | 12.21 |
| `p2-cp65536` | nospec | 0.93 | 65536 | 256 | `—` | 1,292,032 | 570 | 12.21 |
| `p2-cg384` | nospec | 0.93 | default | 384 | `—` | 1,292,032 | 570 | 10.08 |
| `p2-radix-default` | nospec | 0.93 | default | 256 | `—` | 1,292,032 | 114 | 16.36 |
| `p2-radix-lazy` | nospec | 0.93 | default | 256 | `--mamba-radix-cache-strategy extra_buffer_lazy` | 1,292,032 | 142 | 15.82 |
| `p2-ssmbf16` | nospec | 0.93 | default | 256 | `--mamba-ssm-dtype bfloat16` | 1,291,496 | 1104 | 10.11 |
| `p2-sched0.6` | nospec | 0.93 | default | 256 | `--schedule-conservativeness 0.6` | 1,292,032 | 570 | 12.21 |
| `p3-base-mf0.92` | dspark | 0.92 | default | 256 | `—` | 1,174,618 | 48 | 12.81 |
| `p3-mrr16` | dspark | 0.92 | default | 256 | `--max-running-requests 16` | 1,658,855 | 16 | 20.81 |
| `p3-mrr24` | dspark | 0.92 | default | 256 | `--max-running-requests 24` | 1,537,796 | 24 | 18.88 |
| `p3-mrr32` | dspark | 0.92 | default | 256 | `--max-running-requests 32` | 1,416,737 | 32 | 16.83 |
| `p3-mrr40` | dspark | 0.92 | default | 256 | `--max-running-requests 40` | 1,295,677 | 40 | 14.98 |
| `p3-g3` | dspark | 0.92 | default | 256 | `--speculative-dspark-block-size 3` | 1,504,168 | 48 | 19.94 |
| `p3-g5` | dspark | 0.92 | default | 256 | `--speculative-dspark-block-size 5` | 1,339,393 | 48 | 16.01 |
| `p3-replayssm` | dspark | 0.85 | default | 256 | `--enable-linear-replayssm-spec --max-running-requests 128` | 997,441 | 128 | 13.59 |
| `p4-nospec-combo` | nospec | 0.93 | 32768 | 256 | `--mamba-ssm-dtype bfloat16` | 1,291,496 | 1104 | 10.11 |
| `p3b-g2` | dspark | 0.92 | default | 256 | `--speculative-dspark-block-size 2` | 1,586,556 | 48 | 21.51 |
| `p3b-g3-replayssm` | dspark | 0.92 | default | 256 | `--speculative-dspark-block-size 3 --enable-linear-replayssm-spec` | 1,803,795 | 48 | 20.51 |

### Throughput sweep — random ISL 8192 / OSL 1024

`num-prompts` = 2 x concurrency, `--random-range-ratio 1`, `--warmup-requests 4 --flush-cache`.

| config | lane | conc | TTFT med ms | TPOT ms | out tok/s | total tok/s | tok/s/GPU | accept | KV use | mamba use | queue | retok div |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p3-g3` | dspark | 48 | 6568.96 | 100.05 | 400.68 | 3606.12 | 450.8 | 2.552 | 0.29 | 0.98 | 3 | 1.586 |
| `p3b-g3-replayssm` | dspark | 48 | 7711.01 | 100.06 | 398.24 | 3584.12 | 448.0 | 2.513 | 0.24 | 0.98 | 3 | 1.879 |
| `p3b-g2` | dspark | 48 | 6710.85 | 109.06 | 367.27 | 3305.43 | 413.2 | 2.222 | 0.27 | 0.98 | 2 | 1.917 |
| `p3-mrr24` | dspark | 48 | 70099.54 | 54.85 | 362.72 | 3264.52 | 408.1 | 3.024 | 0.13 | 0.96 | 27 | 1.094 |
| `p3-mrr16` | dspark | 48 | 96194.75 | 40.45 | 334.11 | 3007.01 | 375.9 | 3.064 | 0.08 | 1 | 34 | 1.156 |
| `p3-replayssm` | dspark | 48 | 3168.47 | 175.13 | 238.06 | 2142.53 | 267.8 | 3.016 | 0.44 | 0.38 | 0 | 1.79 |
| `p3-base-mf0.92` | dspark | 48 | 11249.6 | 170.98 | 238.05 | 2142.48 | 267.8 | 2.999 | 0.36 | 0.98 | 3 | 1.757 |
| `p3-g5` | dspark | 48 | 10067.27 | 174.73 | 235.3 | 2117.66 | 264.7 | 2.893 | 0.31 | 0.98 | 3 | 1.537 |
| `p3-mrr32` | dspark | 48 | 56086.75 | 117.38 | 234.38 | 2109.45 | 263.7 | 3.041 | 0.2 | 0.97 | 19 | 1.167 |
| `p3-mrr40` | dspark | 48 | 24589.04 | 144.08 | 234.07 | 2106.59 | 263.3 | 2.976 | 0.27 | 0.97 | 11 | 1.959 |
| `p3-curve-p3-g3` | dspark | 64 | 25721.65 | 101.74 | 397.65 | 3578.82 | 447.4 | 2.527 | 0.29 | 0.98 | 19 | 2.264 |
| `p4-dspark-rep2` | dspark | 96 | 118724.77 | 103.09 | 408.53 | 3676.77 | 459.6 | 2.601 | 0.28 | 0.98 | 51 | 1.45 |
| `p3b-g3-replayssm` | dspark | 96 | 121220.48 | 103.53 | 408.05 | 3672.41 | 459.1 | 2.553 | 0.24 | 0.98 | 51 | 2.827 |
| `p3-curve-p3-g3` | dspark | 96 | 121423.96 | 103.79 | 406.42 | 3657.8 | 457.2 | 2.574 | 0.28 | 0.98 | 51 | 2.047 |
| `p4-dspark-rep3` | dspark | 96 | 122322.3 | 104.45 | 404.26 | 3638.32 | 454.8 | 2.551 | 0.28 | 0.98 | 51 | 2.179 |
| `p4-dspark-combo` | dspark | 96 | 120235.15 | 104.03 | 403.55 | 3631.99 | 454.0 | 2.554 | 0.28 | 0.98 | 51 | 2.298 |
| `p3-replayssm` | dspark | 96 | 4251.14 | 387.91 | 219.51 | 1975.6 | 246.9 | 2.672 | 0.87 | 0.75 | 0 | 4.028 |
| `p2-base-mf0.93` | nospec | 96 | 29530.24 | 92.87 | 788.23 | 7094.1 | 886.8 | NA | 0.69 | 0.17 | 0 | 2.25 |
| `p4-nospec-combo` | nospec | 128 | 39340.13 | 105.54 | 890.25 | 8012.22 | 1001.5 | NA | 0.92 | 0.12 | 0 | 2.253 |
| `p4-nospec-rep3` | nospec | 128 | 38996.61 | 105.6 | 889.04 | 8001.37 | 1000.2 | NA | 0.92 | 0.12 | 0 | 2.674 |
| `p4-nospec-rep2` | nospec | 128 | 39220.96 | 105.76 | 887.9 | 7991.12 | 998.9 | NA | 0.92 | 0.12 | 0 | 2.422 |
| `p2-ssmbf16` | nospec | 128 | 39039.13 | 106.04 | 886.35 | 7977.11 | 997.1 | NA | 0.92 | 0.12 | 0 | 2.372 |
| `p2b-base-rep3` | nospec | 128 | 38953.9 | 107.6 | 878.15 | 7903.34 | 987.9 | NA | 0.92 | 0.22 | 0 | 1.788 |
| `p2b-base-rep2` | nospec | 128 | 38975.31 | 107.74 | 877.37 | 7896.33 | 987.0 | NA | 0.92 | 0.22 | 0 | 2.135 |
| `p2-cp32768` | nospec | 128 | 39388.98 | 107.93 | 876.96 | 7892.63 | 986.6 | NA | 0.92 | 0.22 | 0 | 2.555 |
| `p2-cp65536` | nospec | 128 | 39142.88 | 107.95 | 876.68 | 7890.12 | 986.3 | NA | 0.92 | 0.22 | 0 | 2.163 |
| `p2-base-mf0.93` | nospec | 128 | 39005.09 | 108.05 | 875.08 | 7875.74 | 984.5 | NA | 0.92 | 0.22 | 0 | 2.204 |
| `p2-cg384` | nospec | 128 | 39099.59 | 107.9 | 874.85 | 7873.68 | 984.2 | NA | 0.92 | 0.22 | 0 | 1.997 |
| `p2-sched0.6` | nospec | 128 | 39065.36 | 108.19 | 874.1 | 7866.86 | 983.4 | NA | 0.92 | 0.22 | 0 | 2.039 |
| `p2-radix-lazy` | nospec | 128 | 39380.22 | 109.08 | 868.11 | 7812.98 | 976.6 | NA | 0.92 | 0.67 | 0 | 1.76 |
| `p2-cp8192` | nospec | 128 | 43965 | 113.6 | 815.79 | 7342.14 | 917.8 | NA | 0.92 | 0.22 | 0 | 1.885 |
| `p2-radix-default` | nospec | 128 | 39503.43 | 100.79 | 748.61 | 6737.52 | 842.2 | NA | 0.81 | 0.79 | 15 | 2.538 |
| `p2-curve-p2-ssmbf16` | nospec | 160 | 48457.14 | 114.86 | 793.24 | 7139.17 | 892.4 | NA | 0.99 | 0.12 | 22 | 1.723 |
| `p2-base-mf0.93` | nospec | 160 | 48425.12 | 116.97 | 782.68 | 7044.08 | 880.5 | NA | 0.99 | 0.24 | 22 | 2.057 |
| `p2-curve-p2-ssmbf16` | nospec | 192 | 60831.17 | 117.41 | 828.06 | 7452.52 | 931.6 | NA | 0.99 | 0.12 | 54 | 1.906 |
| `p2-base-mf0.93` | nospec | 192 | 60714.27 | 119.6 | 816.92 | 7352.29 | 919.0 | NA | 0.99 | 0.24 | 54 | 2.132 |

### Latency sweep — random ISL 1024 / OSL 1024

Same protocol as the published section 5 table.

| config | lane | conc | TTFT med ms | TPOT ms | out tok/s | total tok/s | tok/s/GPU | accept | KV use | mamba use | queue | retok div |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p3-lat-win` | dspark | 1 | 178.48 | 8.78 | 111.58 | 223.16 | 27.9 | 2.757 | 0 | 0.02 | 0 | 6.787 |
| `p3-lat-base` | dspark | 1 | 181.42 | 9.84 | 99.7 | 199.4 | 24.9 | 3.141 | 0 | 0.02 | 0 | 2.832 |
| `p3-lat-win` | dspark | 4 | 423.29 | 11.74 | 289.14 | 578.28 | 72.3 | 2.678 | 0.01 | 0.08 | 0 | 0.537 |
| `p3-lat-base` | dspark | 4 | 433.98 | 14.55 | 253.84 | 507.68 | 63.5 | 3.042 | 0.01 | 0.08 | 0 | 0.684 |
| `p3-lat-win` | dspark | 8 | 638.17 | 15.25 | 470.72 | 941.44 | 117.7 | 2.61 | 0.01 | 0.17 | 0 | 1.196 |
| `p3-lat-base` | dspark | 8 | 657.75 | 18.37 | 382.87 | 765.73 | 95.7 | 3.273 | 0.01 | 0.17 | 0 | 0.5 |

### Shared-prefix workload — 32 groups x 8 prompts, 4K system prompt

`generated-shared-prefix` with `--cache-report`.

| config | lane | conc | TTFT med ms | TPOT ms | out tok/s | total tok/s | tok/s/GPU | accept | KV use | mamba use | queue | retok div |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p2-radix-lazy` | nospec | 32 | 1497.04 | 36.16 | 702.03 | 12560.81 | 1570.1 | NA | 0.09 | 0.17 | 0 | 2.963 68.2% |
| `p2-radix-default` | nospec | 32 | 1741.83 | 36.25 | 696.99 | 12470.67 | 1558.8 | NA | 0.09 | 0.22 | 0 | 3.033 68.2% |
| `p2b-noradix-gsp` | nospec | 32 | 5927.67 | 47.33 | 462.52 | 8275.57 | 1034.4 | NA | 0.12 | 0.06 | 0 | 2.96 0% |

