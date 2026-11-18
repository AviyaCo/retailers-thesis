# Updated matched retailer comparison

New evaluation only; previous curves cached unchanged. No fitting or OOS selection.
Mean-rule recovery uses aligned training/AP means on the existing Moreau path, not the entire historical SCS algorithm.
31 positive-radius prefixes are unavailable: no eligible post-initial AP. The .03 cohort has only 9/10 and is not drawn.
Prefix phase sums exclude uninstrumented overhead and are NOT newly measured short-run totals or sweep elapsed.

## Cost versus number of MPs

Ten seeds per displayed row; same fixed 10,000 scenarios. No radius is chosen
using these scores. The shorter rule also changes returned-checkpoint selection
to the minimum-AP incumbent, so this is not a stopping-only ablation.

| Radius | Full-rule mean MPs | Recovered mean MPs | Full-rule OOS mean | Recovered OOS mean |
|---:|---:|---:|---:|---:|
| 0.2 | 6.6 | 2.2 | 20.476167 | 20.637740 |
| 0.4 | 10.2 | 4.1 | 20.470175 | 20.529497 |
| 0.7 | 20.8 | 5.6 | 20.666352 | 20.733442 |
| 1.5 | 19.4 | 7.4 | 21.375432 | 21.484426 |

For seed9/radius0.7, the first eligible mean stop and the minimum-AP return are
both candidate8 (8 MPs versus 40 actually executed). Its persisted prefix-phase
sum is962.424854 seconds (16.040414 minutes); actual full CCG total is
11411.629000 seconds (190.193817 minutes). These totals have different overhead
coverage and must not be reported as an exact measured speedup. The recovered
every-center residual is not certified below1%.

All149 evaluations completed; 1,490,000 fresh projections accepted as OPTIMAL.
Every frozen input hash, including previous figure and training checkpoints,
was reverified after evaluation. Six recovery tests and the existing five-QP
strict parity fixture passed before launch. Extraction/freezing took30.448857
seconds (0.507481 minutes), separate from the evaluation/plot invocation below.
Presentation-only redraw trims unused negative-radius padding without changing
data; its timing is stored separately in render_timing.json.

| Radius | Recovered n | Mean MPs | Mean AP s | Mean MP s | Prefix phase sum mean / median / min / max s |
|---:|---:|---:|---:|---:|---|
| 0 | 10 | 0.00 | 1.903 | 0.000 | 1.929 / 1.845 / 1.628 / 2.468 |
| 0.03 | 9 | 1.00 | 15.504 | 29.732 | 45.279 / 51.579 / 8.579 / 68.288 |
| 0.2 | 10 | 2.20 | 63.460 | 97.299 | 160.829 / 146.615 / 134.019 / 243.538 |
| 0.4 | 10 | 4.10 | 186.652 | 205.853 | 392.621 / 302.411 / 206.875 / 858.838 |
| 0.7 | 10 | 5.60 | 318.085 | 286.646 | 604.884 / 487.596 / 329.856 / 1339.570 |
| 1.5 | 10 | 7.40 | 525.870 | 311.654 | 837.722 / 489.950 / 321.757 / 2924.835 |

| Evaluation method | Radius | n | Mean / median / min / max wall s |
|---|---:|---:|---|
| terminal | 0 | 10 | 5.044505 / 5.034327 / 4.837062 / 5.223559 |
| terminal | 0.001 | 10 | 4.896795 / 4.900835 / 4.674373 / 5.164667 |
| terminal | 0.003 | 10 | 4.922167 / 4.879542 / 4.818928 / 5.169730 |
| terminal | 0.01 | 10 | 4.972427 / 4.967748 / 4.840788 / 5.182527 |
| terminal | 0.03 | 10 | 4.947257 / 4.928052 / 4.817691 / 5.152905 |
| terminal | 0.2 | 10 | 4.976670 / 4.981344 / 4.753182 / 5.193031 |
| terminal | 0.4 | 10 | 4.943463 / 4.914159 / 4.803091 / 5.161293 |
| terminal | 0.7 | 10 | 4.967452 / 4.933434 / 4.824115 / 5.265261 |
| terminal | 1.5 | 10 | 4.834598 / 4.808038 / 4.666740 / 5.077993 |
| mean_gap | 0 | 10 | 4.950250 / 4.916411 / 4.803183 / 5.182075 |
| mean_gap | 0.03 | 9 | 4.909740 / 4.900127 / 4.820112 / 5.006826 |
| mean_gap | 0.2 | 10 | 4.907051 / 4.904892 / 4.781459 / 5.094693 |
| mean_gap | 0.4 | 10 | 4.879101 / 4.876981 / 4.771529 / 4.963597 |
| mean_gap | 0.7 | 10 | 4.986345 / 4.993649 / 4.823834 / 5.193938 |
| mean_gap | 1.5 | 10 | 4.809166 / 4.792197 / 4.639488 / 5.083641 |

Current evaluation/plot invocation: 106.856984 s / 1.780950 min / 0.029682 h; completed.
Returned failed/incomplete evaluations: 0; missing/unreturned: 0. All excluded from completed aggregates.
| Category | Timing / scope |
|---|---|
| Prior nominal LR-search phase | 121.796206 s / 2.029937 min / 0.033832 h; not rerun |
| Prior robust CCG sweep | 14942.580206 s / 249.043003 min / 4.150717 h; per-iteration/model details in ../RESULTS.md and source summaries |
| Recovered shorter CCG | Persisted phase sums above and manifest; fresh end-to-end and sweep times unavailable, rerun required |
| SRO-FA | Cached result; not rerun, construction/solver timing not imported |
| CV / new hyperparameter selection | Not run |
| OOS evaluation | 149/149 completed; measured separately, no cache substitution for new curves |
Environment: {'host': 'gold-srv2.iem.technion.ac.il', 'platform': 'Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28', 'cpu_count': 72}; 8 workers, one numerical/solver thread each.
No summed concurrent model duration is represented as sweep elapsed. Full run has 89 separated / 1 capped; reconstruction does not certify every-center separation.
[Frozen selections/timings](frozen/manifest.json)
[Updated plot](matched_oos_updated.png) · [Raw plotted values](plot_data.json)
