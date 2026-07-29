# Kimi-K3 launch-parameter search — results

Rows: 53 total, 49 successful bench points.

## Failed / rejected configs

- `p1-nospec-mf0.95` [p1] status=**CRASH** — torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.75 GiB. GPU 1 has a total capacity of 287.98 GiB of which 1.01 GiB is free. Of the allocated memory 273.38 GiB is allocated by PyTorch  with 238.00 MiB alloc
- `p1-nospec-mf0.94` [p1] status=**CRASH** — torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.75 GiB. GPU 0 has a total capacity of 287.98 GiB of which 1.23 GiB is free. Of the allocated memory 270.64 GiB is allocated by PyTorch  with 238.00 MiB alloc
- `p1-dspark-mf0.93` [p1] status=**CRASH** 
- `p2-cg512` [p2] status=**CRASH** — torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.75 GiB. GPU 0 has a total capacity of 287.98 GiB of which 1006.00 MiB is free. Of the allocated memory 271.92 GiB is allocated by PyTorch  with 478.00 MiB al

## lane `dspark` — W2 8k/1k by total token throughput

```
p4-dspark-rep2                                       conc=96  total_tps=3676.77  out_tps=408.53  median_ttft_ms=118724.77  mean_tpot_ms=103.09  tok_usage_max=0.28  mamba_usage_max=0.98  queue_max=51  max_total_num_tokens=1504168
p3b-g3-replayssm                                     conc=96  total_tps=3672.41  out_tps=408.05  median_ttft_ms=121220.48  mean_tpot_ms=103.53  tok_usage_max=0.24  mamba_usage_max=0.98  queue_max=51  max_total_num_tokens=1803795
p3-curve-p3-g3                                       conc=96  total_tps=3657.8  out_tps=406.42  median_ttft_ms=121423.96  mean_tpot_ms=103.79  tok_usage_max=0.28  mamba_usage_max=0.98  queue_max=51  max_total_num_tokens=1504168
p4-dspark-rep3                                       conc=96  total_tps=3638.32  out_tps=404.26  median_ttft_ms=122322.3  mean_tpot_ms=104.45  tok_usage_max=0.28  mamba_usage_max=0.98  queue_max=51  max_total_num_tokens=1504168
p4-dspark-combo                                      conc=96  total_tps=3631.99  out_tps=403.55  median_ttft_ms=120235.15  mean_tpot_ms=104.03  tok_usage_max=0.28  mamba_usage_max=0.98  queue_max=51  max_total_num_tokens=1504168
p3-g3                                                conc=48  total_tps=3606.12  out_tps=400.68  median_ttft_ms=6568.96  mean_tpot_ms=100.05  tok_usage_max=0.29  mamba_usage_max=0.98  queue_max=3  max_total_num_tokens=1504168
p3b-g3-replayssm                                     conc=48  total_tps=3584.12  out_tps=398.24  median_ttft_ms=7711.01  mean_tpot_ms=100.06  tok_usage_max=0.24  mamba_usage_max=0.98  queue_max=3  max_total_num_tokens=1803795
p3-curve-p3-g3                                       conc=64  total_tps=3578.82  out_tps=397.65  median_ttft_ms=25721.65  mean_tpot_ms=101.74  tok_usage_max=0.29  mamba_usage_max=0.98  queue_max=19  max_total_num_tokens=1504168
p3b-g2                                               conc=48  total_tps=3305.43  out_tps=367.27  median_ttft_ms=6710.85  mean_tpot_ms=109.06  tok_usage_max=0.27  mamba_usage_max=0.98  queue_max=2  max_total_num_tokens=1586556
p3-mrr24                                             conc=48  total_tps=3264.52  out_tps=362.72  median_ttft_ms=70099.54  mean_tpot_ms=54.85  tok_usage_max=0.13  mamba_usage_max=0.96  queue_max=27  max_total_num_tokens=1537796
p3-mrr16                                             conc=48  total_tps=3007.01  out_tps=334.11  median_ttft_ms=96194.75  mean_tpot_ms=40.45  tok_usage_max=0.08  mamba_usage_max=1  queue_max=34  max_total_num_tokens=1658855
p3-replayssm                                         conc=48  total_tps=2142.53  out_tps=238.06  median_ttft_ms=3168.47  mean_tpot_ms=175.13  tok_usage_max=0.44  mamba_usage_max=0.38  queue_max=0  max_total_num_tokens=997441
```

## lane `dspark` — W1 1k/1k by mean TPOT (lower is better)

```
p3-lat-win                                           conc=1  mean_tpot_ms=8.78  median_ttft_ms=178.48  out_tps=111.58  accept_len=2.757
p3-lat-base                                          conc=1  mean_tpot_ms=9.84  median_ttft_ms=181.42  out_tps=99.7  accept_len=3.141
p3-lat-win                                           conc=4  mean_tpot_ms=11.74  median_ttft_ms=423.29  out_tps=289.14  accept_len=2.678
p3-lat-base                                          conc=4  mean_tpot_ms=14.55  median_ttft_ms=433.98  out_tps=253.84  accept_len=3.042
p3-lat-win                                           conc=8  mean_tpot_ms=15.25  median_ttft_ms=638.17  out_tps=470.72  accept_len=2.61
p3-lat-base                                          conc=8  mean_tpot_ms=18.37  median_ttft_ms=657.75  out_tps=382.87  accept_len=3.273
```

## lane `nospec` — W2 8k/1k by total token throughput

```
p4-nospec-combo                                      conc=128  total_tps=8012.22  out_tps=890.25  median_ttft_ms=39340.13  mean_tpot_ms=105.54  tok_usage_max=0.92  mamba_usage_max=0.12  queue_max=0  max_total_num_tokens=1291496
p4-nospec-rep3                                       conc=128  total_tps=8001.37  out_tps=889.04  median_ttft_ms=38996.61  mean_tpot_ms=105.6  tok_usage_max=0.92  mamba_usage_max=0.12  queue_max=0  max_total_num_tokens=1291496
p4-nospec-rep2                                       conc=128  total_tps=7991.12  out_tps=887.9  median_ttft_ms=39220.96  mean_tpot_ms=105.76  tok_usage_max=0.92  mamba_usage_max=0.12  queue_max=0  max_total_num_tokens=1291496
p2-ssmbf16                                           conc=128  total_tps=7977.11  out_tps=886.35  median_ttft_ms=39039.13  mean_tpot_ms=106.04  tok_usage_max=0.92  mamba_usage_max=0.12  queue_max=0  max_total_num_tokens=1291496
p2b-base-rep3                                        conc=128  total_tps=7903.34  out_tps=878.15  median_ttft_ms=38953.9  mean_tpot_ms=107.6  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2b-base-rep2                                        conc=128  total_tps=7896.33  out_tps=877.37  median_ttft_ms=38975.31  mean_tpot_ms=107.74  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-cp32768                                           conc=128  total_tps=7892.63  out_tps=876.96  median_ttft_ms=39388.98  mean_tpot_ms=107.93  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-cp65536                                           conc=128  total_tps=7890.12  out_tps=876.68  median_ttft_ms=39142.88  mean_tpot_ms=107.95  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-base-mf0.93                                       conc=128  total_tps=7875.74  out_tps=875.08  median_ttft_ms=39005.09  mean_tpot_ms=108.05  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-cg384                                             conc=128  total_tps=7873.68  out_tps=874.85  median_ttft_ms=39099.59  mean_tpot_ms=107.9  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-sched0.6                                          conc=128  total_tps=7866.86  out_tps=874.1  median_ttft_ms=39065.36  mean_tpot_ms=108.19  tok_usage_max=0.92  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032
p2-radix-lazy                                        conc=128  total_tps=7812.98  out_tps=868.11  median_ttft_ms=39380.22  mean_tpot_ms=109.08  tok_usage_max=0.92  mamba_usage_max=0.67  queue_max=0  max_total_num_tokens=1292032
```

## lane `nospec` — W3 shared prefix by total token throughput

```
p2-radix-lazy                                        conc=32  total_tps=12560.81  out_tps=702.03  median_ttft_ms=1497.04  mean_tpot_ms=36.16  tok_usage_max=0.09  mamba_usage_max=0.17  queue_max=0  max_total_num_tokens=1292032  cache_hit_pct=68.2
p2-radix-default                                     conc=32  total_tps=12470.67  out_tps=696.99  median_ttft_ms=1741.83  mean_tpot_ms=36.25  tok_usage_max=0.09  mamba_usage_max=0.22  queue_max=0  max_total_num_tokens=1292032  cache_hit_pct=68.2
p2b-noradix-gsp                                      conc=32  total_tps=8275.57  out_tps=462.52  median_ttft_ms=5927.67  mean_tpot_ms=47.33  tok_usage_max=0.12  mamba_usage_max=0.06  queue_max=0  max_total_num_tokens=1292032  cache_hit_pct=0
```

## Gibberish screen (retokenized-token divergence)

Threshold 5.0% (about 1.9% is the observed healthy baseline). 1 of 49 points flagged.

```
p3-lat-win                                           retok_div=6.787%  conc=1
```

