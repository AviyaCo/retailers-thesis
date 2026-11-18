# Shipping: blue versus green historical runtimes

Read-only analysis of the frozen October 5 STOP comparison. No retraining, solver run, test-driven selection, or source/result modification. Seeds 0–9. Stored units are seconds.

## Scope and limitations

- Blue = historical Clarabel neural training; green = Moreau. AP uses Gurobi, not Clarabel.
- Training samples differ: blue 50/200/800, green 40/160/640. Blue N800 uses corrected minibatch MP (batch 200). Stopping/return rules, optimizers and execution contention differ.
- Green CCG job wall includes loading and native own-pool assessments, but excludes cached nominal training and held-out OOS replay. Green uses monotonic wall timers.
- Blue totals are reported historical timings, not certified complete end-to-end runtimes. Old code times cache loading as work and can retain rounded zeros. Absence of a cache flag does not prove a cache-free history. Historical clock choice is not certified for every run.
- Resume-only times, partial/cache-timed histories and non-converged histories are separate cohorts below. No missing work is estimated. Reruns would be needed for missing timings.
- Small green radii through 0.03 perform zero MP updates and reuse the nominal policy. Those times are stopping checks, not fresh neural training.
- Host recorded for green/replay: gold-srv2.iem.technion.ac.il, Linux. Green CCG: 62 single-thread workers; continuation: five. Nominal fits: 52 workers. Replay: 16 workers, one Torch/Gurobi thread each. Historical blue concurrency is not uniform or fully established; the N800 launch records include a 24-worker parallel continuation. Exact hardware equivalence and CPU model are not established. Do not interpret these differences as solver-only speedups.

## Main comparison: mean reported CCG minutes (number of runs)

Blue is shown only when all ten records converged and have no explicit partial/cache timing flag or resume flag. This is a conservative display rule, not a timing-completeness certificate. Green shows uninterrupted completed runs; five continuations are detailed separately. Dashes are deliberately not replaced by misleading mixed-cohort means.

| Radius | N50 blue | N50 green | N200 blue | N200 green | N800 blue | N800 green |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 0.06 (10) | — | 0.16 (10) | — | 0.49 (10) |
| 0.001 | — | 0.06 (10) | 61.13 (10) | 0.19 (10) | 120.11 (10) | 0.51 (10) |
| 0.003 | — | 0.07 (10) | 66.05 (10) | 0.17 (10) | 100.34 (10) | 0.47 (10) |
| 0.01 | 17.21 (10) | 0.05 (10) | 47.41 (10) | 0.13 (10) | 126.06 (10) | 0.44 (10) |
| 0.03 | — | 0.06 (10) | — | 0.14 (10) | — | 0.54 (10) |
| 0.2 | 81.67 (10) | 3.56 (10) | 133.48 (10) | 5.01 (10) | 407.42 (10) | 6.33 (9) |
| 0.4 | 171.50 (10) | 7.33 (10) | 428.96 (10) | 9.88 (10) | 586.58 (10) | 10.19 (9) |
| 0.7 | — | 8.79 (10) | 571.73 (10) | 11.76 (10) | 842.88 (10) | 14.31 (9) |
| 1.5 | — | 5.62 (9) | 434.27 (10) | 9.42 (9) | 658.01 (10) | 14.38 (10) |

Radius zero green is the nominal-policy CCG check, not nominal training. Blue has no CCG job at zero and no matching 0.03 radius.

## Nominal timing (different scopes; do not form speedup ratios)

| N | Method | Scope | Runs | Mean s (min) | Median s | Min s | Max s |
|---:|---|---|---:|---:|---:|---:|---:|
| 50 | legacy | SAA + CV + final fit + hard evaluation | 0 | unavailable | — | — | — |
| 50 | green | Selected fit only | 10 | 91.402 (1.523) | 97.810 | 54.810 | 105.290 |
| 200 | legacy | SAA + CV + final fit + hard evaluation | 10 | 24217.078 (403.618) | 23842.084 | 22845.354 | 26030.372 |
| 200 | green | Selected fit only | 10 | 109.254 (1.821) | 109.106 | 75.080 | 129.525 |
| 800 | legacy | SAA + CV + final fit + hard evaluation | 10 | 83568.354 (1392.806) | 83536.645 | 81729.858 | 85901.876 |
| 800 | green | Selected fit only | 10 | 131.555 (2.193) | 119.575 | 82.102 | 175.521 |

## Five green continuations (not in uninterrupted-run means)

| N | Radius | Seed | Prior failed job s | Continuation s | Sum of active invocation times s |
|---:|---:|---:|---:|---:|---:|
| 50 | 1.5 | 7 | 592.895 | 35.891 | 628.787 |
| 200 | 1.5 | 5 | 166.076 | 176.184 | 342.260 |
| 800 | 0.2 | 7 | 108.063 | 20.837 | 128.900 |
| 800 | 0.4 | 9 | 98.224 | 22.945 | 121.170 |
| 800 | 0.7 | 6 | 670.238 | 31.641 | 701.878 |

## Runtime coverage and parallel elapsed time

| Component | Coverage |
|---|---|
| Nominal | Green selected-fit times above; full initialization and 90-fit campaign below. Blue N50 unavailable; N200/N800 bundled SAA/CV/final-fit time, not isolated fit time. |
| Robust CCG | All saved timing references and AP/MP iteration fields retained in [comparison.json](comparison.json); cohorts below. |
| SRO-FA partition / optimization / evaluation / total | Not run in this comparison. No comparable complete phase/total measurements established here; cached reference curve is not a measured zero-cost run. |
| CV / selection | Blue separate CV duration unavailable; included in N200/N800 seed totals. Green nominal LR validation stage below, not five-fold CV. No new CV/radius selection. |
| Final OOS / IS | Green Oct5, blue cached Sept28 measurements. Same Gurobi replay, not native Moreau-vs-Clarabel inference. Separate phase statistics below. |

Parallel elapsed times below are observed campaign/phase durations, not sums of worker times. The original green CCG campaign includes five failed jobs; the continuation is separate. The September replay includes both colors, so its elapsed time is not a blue-only runtime.

| Campaign | Runner seconds (minutes) | Phase elapsed seconds |
|---|---:|---|
| shipping_nominal_R3_N50_N200_N800_20260924 | 5984.883 (99.748) | initialization: 4490.221; nominal: 1494.636 |
| shipping_live_green_20260926 | 1760.299 (29.338) | cached_grid_verification: 5.615; nominal_validation: 12.317; ccg: 1740.416 |
| shipping_advisory_continue_20260928 | 211.764 (3.529) |  |
| shipping_comparison_20260928 | 662.730 (11.046) | freeze: 142.282; tests: 24.660; training_only_preflight: 15.203; test_generation: 9.209; evaluation: 283.268; journal_verification: 168.555; final_integrity: 13.237; render: 6.097 |
| shipping_stopping_20261005 | 342.332 (5.706) | tests: 26.064; selection_and_freeze: 96.254; training_only_preflight: 15.791; test_binding: 9.418; green_evaluation: 108.726; raw_verification: 65.751; final_integrity: 14.374; render: 5.793 |

## All CCG cohorts (seconds; failed/partial/resumed kept separate)

| Method | N | Radius | Cohort | Runs | Mean | Median | Min | Max |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| green | 50 | 0 | completed_single_invocation | 10 | 3.400 | 3.401 | 3.061 | 3.562 |
| green | 50 | 0.001 | completed_single_invocation | 10 | 3.767 | 3.763 | 3.229 | 4.283 |
| green | 50 | 0.003 | completed_single_invocation | 10 | 4.444 | 4.377 | 3.229 | 5.673 |
| green | 50 | 0.01 | completed_single_invocation | 10 | 3.280 | 3.261 | 2.974 | 3.576 |
| green | 50 | 0.03 | completed_single_invocation | 10 | 3.317 | 3.191 | 2.836 | 3.951 |
| green | 50 | 0.2 | completed_single_invocation | 10 | 213.353 | 246.338 | 55.400 | 422.313 |
| green | 50 | 0.4 | completed_single_invocation | 10 | 439.909 | 470.122 | 90.247 | 647.967 |
| green | 50 | 0.7 | completed_single_invocation | 10 | 527.567 | 580.939 | 155.447 | 870.455 |
| green | 50 | 1.5 | completed_continuation_current_only | 1 | 35.891 | 35.891 | 35.891 | 35.891 |
| green | 50 | 1.5 | completed_single_invocation | 9 | 337.307 | 371.037 | 139.813 | 542.335 |
| green | 200 | 0 | completed_single_invocation | 10 | 9.336 | 9.214 | 8.126 | 11.060 |
| green | 200 | 0.001 | completed_single_invocation | 10 | 11.586 | 11.604 | 10.710 | 12.209 |
| green | 200 | 0.003 | completed_single_invocation | 10 | 10.388 | 10.885 | 7.689 | 11.749 |
| green | 200 | 0.01 | completed_single_invocation | 10 | 7.722 | 7.650 | 7.139 | 8.346 |
| green | 200 | 0.03 | completed_single_invocation | 10 | 8.188 | 8.231 | 7.177 | 8.833 |
| green | 200 | 0.2 | completed_single_invocation | 10 | 300.845 | 288.434 | 148.462 | 568.069 |
| green | 200 | 0.4 | completed_single_invocation | 10 | 593.072 | 499.178 | 352.092 | 1144.795 |
| green | 200 | 0.7 | completed_single_invocation | 10 | 705.668 | 711.046 | 535.452 | 904.350 |
| green | 200 | 1.5 | completed_continuation_current_only | 1 | 176.184 | 176.184 | 176.184 | 176.184 |
| green | 200 | 1.5 | completed_single_invocation | 9 | 565.051 | 614.877 | 318.065 | 880.774 |
| green | 800 | 0 | completed_single_invocation | 10 | 29.663 | 30.279 | 27.759 | 31.373 |
| green | 800 | 0.001 | completed_single_invocation | 10 | 30.788 | 30.864 | 29.582 | 31.723 |
| green | 800 | 0.003 | completed_single_invocation | 10 | 28.292 | 28.635 | 25.360 | 30.403 |
| green | 800 | 0.01 | completed_single_invocation | 10 | 26.651 | 26.766 | 25.083 | 27.922 |
| green | 800 | 0.03 | completed_single_invocation | 10 | 32.181 | 31.503 | 27.025 | 38.934 |
| green | 800 | 0.2 | completed_continuation_current_only | 1 | 20.837 | 20.837 | 20.837 | 20.837 |
| green | 800 | 0.2 | completed_single_invocation | 9 | 379.937 | 329.969 | 244.725 | 640.523 |
| green | 800 | 0.4 | completed_continuation_current_only | 1 | 22.945 | 22.945 | 22.945 | 22.945 |
| green | 800 | 0.4 | completed_single_invocation | 9 | 611.665 | 626.999 | 117.254 | 899.913 |
| green | 800 | 0.7 | completed_continuation_current_only | 1 | 31.641 | 31.641 | 31.641 | 31.641 |
| green | 800 | 0.7 | completed_single_invocation | 9 | 858.895 | 868.428 | 735.384 | 936.613 |
| green | 800 | 1.5 | completed_single_invocation | 10 | 862.547 | 829.699 | 761.506 | 1035.914 |
| legacy | 50 | 0.001 | converged_partial_or_cache_timed | 10 | 1010.896 | 1072.565 | 423.570 | 1322.000 |
| legacy | 50 | 0.003 | converged_partial_or_cache_timed | 10 | 1172.293 | 1124.905 | 340.680 | 2025.890 |
| legacy | 50 | 0.005 | converged_no_cache_flag | 10 | 1173.378 | 1181.110 | 363.630 | 1990.820 |
| legacy | 50 | 0.01 | converged_no_cache_flag | 10 | 1032.815 | 1074.700 | 377.920 | 1669.660 |
| legacy | 50 | 0.02 | converged_no_cache_flag | 10 | 1032.608 | 1018.870 | 636.860 | 1287.430 |
| legacy | 50 | 0.05 | converged_no_cache_flag | 10 | 1101.754 | 1173.925 | 543.800 | 1622.040 |
| legacy | 50 | 0.1 | converged_no_cache_flag | 10 | 2812.920 | 2903.515 | 741.800 | 5543.770 |
| legacy | 50 | 0.15 | converged_no_cache_flag | 9 | 3694.830 | 3324.430 | 1710.760 | 5330.500 |
| legacy | 50 | 0.15 | not_converged_partial_or_cache_timed | 1 | 3399.330 | 3399.330 | 3399.330 | 3399.330 |
| legacy | 50 | 0.2 | converged_no_cache_flag | 10 | 4900.065 | 4523.605 | 2708.930 | 10173.440 |
| legacy | 50 | 0.3 | converged_no_cache_flag | 10 | 9899.535 | 7619.180 | 2752.930 | 29126.600 |
| legacy | 50 | 0.4 | converged_no_cache_flag | 10 | 10289.742 | 10377.135 | 4327.920 | 17734.080 |
| legacy | 50 | 0.5 | converged_no_cache_flag | 7 | 12685.640 | 13975.620 | 5207.910 | 20488.690 |
| legacy | 50 | 0.5 | not_converged_partial_or_cache_timed | 3 | 60728.257 | 10568.660 | 6160.710 | 165455.400 |
| legacy | 50 | 0.7 | converged_no_cache_flag | 3 | 35262.987 | 14192.750 | 5277.420 | 86318.790 |
| legacy | 50 | 0.7 | converged_no_cache_flag_resumed | 1 | 81813.730 | 81813.730 | 81813.730 | 81813.730 |
| legacy | 50 | 0.7 | converged_partial_or_cache_timed | 2 | 32219.715 | 32219.715 | 8297.960 | 56141.470 |
| legacy | 50 | 0.7 | not_converged_partial_or_cache_timed | 4 | 11943.730 | 10996.115 | 10656.180 | 15126.510 |
| legacy | 50 | 1 | converged_no_cache_flag | 2 | 14785.460 | 14785.460 | 13947.180 | 15623.740 |
| legacy | 50 | 1 | converged_partial_or_cache_timed | 2 | 32891.395 | 32891.395 | 18765.430 | 47017.360 |
| legacy | 50 | 1 | not_converged_partial_or_cache_timed | 6 | 8309.852 | 8279.560 | 5871.430 | 10307.920 |
| legacy | 50 | 1.5 | converged_no_cache_flag | 6 | 7848.417 | 3912.545 | 3044.020 | 23054.820 |
| legacy | 50 | 1.5 | converged_partial_or_cache_timed | 1 | 84303.340 | 84303.340 | 84303.340 | 84303.340 |
| legacy | 50 | 1.5 | not_converged_partial_or_cache_timed | 3 | 3889.540 | 3513.950 | 3253.040 | 4901.630 |
| legacy | 200 | 0.001 | converged_no_cache_flag | 10 | 3667.532 | 3321.750 | 1624.750 | 5391.330 |
| legacy | 200 | 0.003 | converged_no_cache_flag | 10 | 3962.881 | 4036.460 | 2401.730 | 5089.520 |
| legacy | 200 | 0.005 | converged_no_cache_flag | 10 | 3041.129 | 2566.560 | 2045.230 | 4573.920 |
| legacy | 200 | 0.01 | converged_no_cache_flag | 10 | 2844.869 | 2715.305 | 1997.630 | 4505.040 |
| legacy | 200 | 0.02 | converged_no_cache_flag | 10 | 2846.755 | 2646.675 | 1553.800 | 4605.170 |
| legacy | 200 | 0.05 | converged_no_cache_flag | 10 | 2818.476 | 2771.605 | 1585.740 | 4648.570 |
| legacy | 200 | 0.1 | converged_no_cache_flag | 10 | 4446.125 | 2913.000 | 1643.020 | 10037.330 |
| legacy | 200 | 0.15 | converged_no_cache_flag | 10 | 7165.317 | 6834.810 | 2050.540 | 10101.280 |
| legacy | 200 | 0.2 | converged_no_cache_flag | 10 | 8008.716 | 6884.840 | 2339.520 | 12529.530 |
| legacy | 200 | 0.3 | converged_no_cache_flag | 10 | 14835.214 | 13617.805 | 2504.240 | 33409.180 |
| legacy | 200 | 0.4 | converged_no_cache_flag | 10 | 25737.531 | 24979.275 | 8638.500 | 56058.200 |
| legacy | 200 | 0.5 | converged_no_cache_flag | 9 | 26391.370 | 23474.310 | 13810.410 | 44149.680 |
| legacy | 200 | 0.5 | converged_no_cache_flag_resumed | 1 | 84731.050 | 84731.050 | 84731.050 | 84731.050 |
| legacy | 200 | 0.7 | converged_no_cache_flag | 10 | 34303.980 | 38023.325 | 18625.600 | 52178.330 |
| legacy | 200 | 1 | converged_no_cache_flag | 8 | 29950.232 | 32156.645 | 13942.510 | 44957.340 |
| legacy | 200 | 1 | converged_no_cache_flag_resumed | 2 | 67479.200 | 67479.200 | 54464.960 | 80493.440 |
| legacy | 200 | 1.5 | converged_no_cache_flag | 10 | 26056.035 | 25379.450 | 15453.160 | 37706.080 |
| legacy | 800 | 0.001 | converged_no_cache_flag | 10 | 7206.879 | 7067.060 | 6283.410 | 9531.670 |
| legacy | 800 | 0.003 | converged_no_cache_flag | 10 | 6020.662 | 6584.520 | 1591.860 | 7483.880 |
| legacy | 800 | 0.005 | converged_no_cache_flag | 10 | 6828.839 | 7199.645 | 4394.680 | 8377.660 |
| legacy | 800 | 0.01 | converged_no_cache_flag | 10 | 7563.497 | 8361.500 | 4056.710 | 8472.720 |
| legacy | 800 | 0.02 | converged_no_cache_flag | 10 | 7077.204 | 7611.480 | 4561.170 | 8453.810 |
| legacy | 800 | 0.05 | converged_no_cache_flag | 10 | 8196.987 | 8412.740 | 7018.150 | 8705.250 |
| legacy | 800 | 0.1 | converged_no_cache_flag | 10 | 12346.121 | 8596.980 | 7085.470 | 20105.090 |
| legacy | 800 | 0.15 | converged_no_cache_flag | 10 | 22445.509 | 20198.445 | 17238.820 | 36242.280 |
| legacy | 800 | 0.2 | converged_no_cache_flag | 10 | 24445.149 | 20726.065 | 15086.860 | 52600.230 |
| legacy | 800 | 0.3 | converged_no_cache_flag | 10 | 28401.705 | 33682.465 | 14829.170 | 36752.160 |
| legacy | 800 | 0.4 | converged_no_cache_flag | 10 | 35194.732 | 35072.565 | 17720.200 | 52942.870 |
| legacy | 800 | 0.5 | converged_no_cache_flag | 10 | 37721.901 | 33739.525 | 23336.310 | 53543.400 |
| legacy | 800 | 0.7 | converged_no_cache_flag | 10 | 50572.976 | 50490.745 | 26923.870 | 73979.300 |
| legacy | 800 | 1 | converged_no_cache_flag | 10 | 42726.187 | 43467.940 | 27496.570 | 69540.690 |
| legacy | 800 | 1.5 | converged_no_cache_flag | 10 | 39480.492 | 38094.950 | 19368.340 | 60248.490 |

## Replay evaluation by method, sample size and radius (seconds)

| Method | N | Radius | Runs | Total mean | Median | Min | Max | OOS mean | Own-pool mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| green | 50 | 0 | 10 | 5.684 | 5.637 | 5.361 | 6.120 | 5.376 | 0.048 |
| green | 50 | 0.001 | 10 | 5.690 | 5.540 | 5.330 | 6.336 | 5.377 | 0.050 |
| green | 50 | 0.003 | 10 | 5.738 | 5.668 | 5.296 | 6.301 | 5.440 | 0.041 |
| green | 50 | 0.01 | 10 | 5.651 | 5.407 | 5.260 | 7.023 | 5.344 | 0.049 |
| green | 50 | 0.03 | 10 | 5.569 | 5.471 | 5.248 | 6.070 | 5.259 | 0.053 |
| green | 50 | 0.2 | 10 | 5.499 | 5.393 | 5.320 | 6.123 | 5.129 | 0.077 |
| green | 50 | 0.4 | 10 | 5.721 | 5.600 | 5.318 | 6.667 | 5.295 | 0.104 |
| green | 50 | 0.7 | 10 | 5.812 | 5.673 | 5.371 | 6.443 | 5.350 | 0.127 |
| green | 50 | 1.5 | 10 | 5.596 | 5.434 | 4.916 | 6.648 | 5.131 | 0.140 |
| green | 200 | 0 | 10 | 5.365 | 5.359 | 5.229 | 5.614 | 5.007 | 0.106 |
| green | 200 | 0.001 | 10 | 5.400 | 5.385 | 5.258 | 5.674 | 5.043 | 0.108 |
| green | 200 | 0.003 | 10 | 5.375 | 5.357 | 5.249 | 5.573 | 5.013 | 0.108 |
| green | 200 | 0.01 | 10 | 5.398 | 5.395 | 5.178 | 5.712 | 5.038 | 0.107 |
| green | 200 | 0.03 | 10 | 5.368 | 5.340 | 5.267 | 5.508 | 5.017 | 0.101 |
| green | 200 | 0.2 | 10 | 5.506 | 5.463 | 5.370 | 5.766 | 5.003 | 0.217 |
| green | 200 | 0.4 | 10 | 5.612 | 5.563 | 5.370 | 6.012 | 4.982 | 0.305 |
| green | 200 | 0.7 | 10 | 5.734 | 5.706 | 5.504 | 5.997 | 5.017 | 0.377 |
| green | 200 | 1.5 | 10 | 5.828 | 5.811 | 5.425 | 6.241 | 5.088 | 0.407 |
| green | 800 | 0 | 10 | 5.672 | 5.652 | 5.523 | 5.963 | 5.048 | 0.345 |
| green | 800 | 0.001 | 10 | 5.696 | 5.655 | 5.571 | 6.038 | 5.064 | 0.360 |
| green | 800 | 0.003 | 10 | 5.756 | 5.697 | 5.516 | 6.306 | 5.155 | 0.338 |
| green | 800 | 0.01 | 10 | 5.632 | 5.618 | 5.522 | 5.784 | 5.021 | 0.344 |
| green | 800 | 0.03 | 10 | 5.654 | 5.657 | 5.553 | 5.824 | 5.035 | 0.349 |
| green | 800 | 0.2 | 10 | 6.022 | 5.967 | 5.797 | 6.486 | 5.009 | 0.703 |
| green | 800 | 0.4 | 10 | 6.220 | 6.198 | 5.886 | 6.721 | 5.021 | 0.865 |
| green | 800 | 0.7 | 10 | 6.621 | 6.631 | 6.292 | 7.039 | 5.053 | 1.179 |
| green | 800 | 1.5 | 10 | 6.946 | 6.997 | 6.684 | 7.255 | 5.051 | 1.474 |
| legacy | 50 | 0 | 10 | 5.416 | 5.400 | 5.260 | 5.555 | 5.131 | 0.039 |
| legacy | 50 | 0.001 | 10 | 5.398 | 5.379 | 5.166 | 5.782 | 5.082 | 0.065 |
| legacy | 50 | 0.003 | 10 | 5.381 | 5.345 | 5.197 | 5.653 | 5.069 | 0.067 |
| legacy | 50 | 0.005 | 10 | 5.289 | 5.278 | 5.186 | 5.389 | 4.981 | 0.062 |
| legacy | 50 | 0.01 | 10 | 5.288 | 5.289 | 5.184 | 5.435 | 4.984 | 0.063 |
| legacy | 50 | 0.02 | 10 | 5.318 | 5.284 | 5.178 | 5.511 | 5.014 | 0.060 |
| legacy | 50 | 0.05 | 10 | 5.355 | 5.319 | 5.241 | 5.559 | 5.045 | 0.066 |
| legacy | 50 | 0.1 | 10 | 5.342 | 5.315 | 5.198 | 5.605 | 5.002 | 0.093 |
| legacy | 50 | 0.15 | 10 | 5.367 | 5.353 | 5.281 | 5.508 | 5.016 | 0.106 |
| legacy | 50 | 0.2 | 10 | 5.334 | 5.305 | 5.279 | 5.493 | 4.976 | 0.112 |
| legacy | 50 | 0.3 | 10 | 5.402 | 5.404 | 5.260 | 5.554 | 4.986 | 0.167 |
| legacy | 50 | 0.4 | 10 | 5.340 | 5.313 | 5.196 | 5.474 | 4.938 | 0.158 |
| legacy | 50 | 0.5 | 10 | 5.612 | 5.533 | 5.327 | 6.601 | 5.135 | 0.230 |
| legacy | 50 | 0.7 | 10 | 5.587 | 5.539 | 5.372 | 6.103 | 5.126 | 0.214 |
| legacy | 50 | 1 | 10 | 5.632 | 5.540 | 5.329 | 6.401 | 5.137 | 0.234 |
| legacy | 50 | 1.5 | 10 | 5.335 | 5.307 | 4.553 | 6.391 | 4.887 | 0.202 |
| legacy | 200 | 0 | 10 | 5.398 | 5.303 | 5.269 | 5.885 | 5.036 | 0.112 |
| legacy | 200 | 0.001 | 10 | 5.556 | 5.553 | 5.340 | 5.862 | 5.050 | 0.254 |
| legacy | 200 | 0.003 | 10 | 5.557 | 5.502 | 5.373 | 5.855 | 5.046 | 0.262 |
| legacy | 200 | 0.005 | 10 | 5.589 | 5.519 | 5.330 | 6.045 | 5.080 | 0.258 |
| legacy | 200 | 0.01 | 10 | 5.542 | 5.506 | 5.322 | 5.927 | 5.069 | 0.227 |
| legacy | 200 | 0.02 | 10 | 5.559 | 5.493 | 5.407 | 6.093 | 5.078 | 0.232 |
| legacy | 200 | 0.05 | 10 | 5.582 | 5.532 | 5.332 | 5.933 | 5.113 | 0.224 |
| legacy | 200 | 0.1 | 10 | 5.565 | 5.529 | 5.370 | 5.963 | 5.051 | 0.268 |
| legacy | 200 | 0.15 | 10 | 5.550 | 5.478 | 5.322 | 6.106 | 4.983 | 0.319 |
| legacy | 200 | 0.2 | 10 | 5.622 | 5.566 | 5.370 | 6.119 | 5.024 | 0.353 |
| legacy | 200 | 0.3 | 10 | 5.640 | 5.577 | 5.393 | 6.232 | 4.979 | 0.416 |
| legacy | 200 | 0.4 | 10 | 5.653 | 5.617 | 5.220 | 6.229 | 4.945 | 0.463 |
| legacy | 200 | 0.5 | 10 | 5.843 | 5.814 | 5.604 | 6.320 | 5.003 | 0.593 |
| legacy | 200 | 0.7 | 10 | 5.844 | 5.879 | 5.404 | 6.053 | 4.994 | 0.600 |
| legacy | 200 | 1 | 10 | 5.972 | 5.847 | 5.607 | 6.803 | 4.980 | 0.739 |
| legacy | 200 | 1.5 | 10 | 5.804 | 5.837 | 5.575 | 5.933 | 4.909 | 0.648 |
| legacy | 800 | 0 | 10 | 5.693 | 5.695 | 5.552 | 5.842 | 5.027 | 0.419 |
| legacy | 800 | 0.001 | 10 | 5.890 | 5.927 | 5.625 | 6.134 | 4.982 | 0.656 |
| legacy | 800 | 0.003 | 10 | 5.854 | 5.806 | 5.606 | 6.194 | 4.989 | 0.617 |
| legacy | 800 | 0.005 | 10 | 5.889 | 5.952 | 5.512 | 6.086 | 4.938 | 0.705 |
| legacy | 800 | 0.01 | 10 | 6.092 | 6.075 | 5.672 | 6.822 | 5.058 | 0.785 |
| legacy | 800 | 0.02 | 10 | 6.041 | 6.025 | 5.718 | 6.313 | 5.016 | 0.733 |
| legacy | 800 | 0.05 | 10 | 6.050 | 6.015 | 5.735 | 6.534 | 5.036 | 0.768 |
| legacy | 800 | 0.1 | 10 | 6.164 | 6.149 | 5.669 | 6.677 | 5.064 | 0.847 |
| legacy | 800 | 0.15 | 10 | 6.421 | 6.371 | 5.683 | 7.192 | 4.973 | 1.201 |
| legacy | 800 | 0.2 | 10 | 6.464 | 6.384 | 5.544 | 7.366 | 4.971 | 1.240 |
| legacy | 800 | 0.3 | 10 | 6.608 | 6.712 | 5.623 | 6.858 | 4.987 | 1.363 |
| legacy | 800 | 0.4 | 10 | 7.006 | 7.037 | 6.299 | 7.549 | 5.051 | 1.701 |
| legacy | 800 | 0.5 | 10 | 7.095 | 7.113 | 6.677 | 7.696 | 5.098 | 1.747 |
| legacy | 800 | 0.7 | 10 | 7.251 | 7.249 | 6.808 | 7.672 | 4.984 | 2.017 |
| legacy | 800 | 1 | 10 | 7.206 | 7.220 | 6.929 | 7.579 | 5.014 | 1.920 |
| legacy | 800 | 1.5 | 10 | 7.260 | 7.324 | 6.444 | 7.761 | 4.968 | 2.042 |

Raw per-seed timing references, per-iteration AP/MP fields, available timestamps, workers, statuses, nominal records, campaign receipts and input SHA-256 bindings: [comparison.json](comparison.json). Reproduction: run [compare.py](compare.py) with the existing ap_env Python. No external package or solver is required.
