# Shipping: validation-selected radius, raw positive shipping fees

Both colors use the unchanged strict common Gurobi projection and charge 0.1 per raw q>0 shipment. All 60 choices were sealed before test-cost lookup. No retraining, clipping, fallback or test-driven selection.

Green: saved STOP policies, existing 10/40/160-row holdout (already used for LR selection), nine radii including zero. Blue: saved historical policies, independent 13/50/200-row validation sets with seed 88880+s, original 15 positive radii. Seeds 0–9; exact ties use smaller radius; no refit.

The inspected historical 10,000-row seed-9999 test and test-informed historical grids are descriptive, not fresh confirmatory evidence. Green retains the mixed 265-original/5-advisory cohort. Blue retains historical non-converged/capped policies where present; no post-hoc filtering. Training sizes and validation protocols differ, so this is not a matched-protocol solver comparison.

## Selected test costs (lower is better)

| Method | Ntotal | Seeds | Mean | Sample SD | Selected radius counts |
|---|---:|---:|---:|---:|---|
| green | 50 | 10 | 20.817281 | 0.444580 | {0.0: 6, 0.2: 1, 0.4: 1, 0.7: 2} |
| green | 200 | 10 | 20.342430 | 0.247809 | {0.0: 3, 0.2: 3, 0.4: 3, 0.7: 1} |
| green | 800 | 10 | 19.967017 | 0.117240 | {0.0: 1, 0.2: 4, 0.4: 3, 0.7: 2} |
| legacy | 50 | 10 | 20.604174 | 0.181066 | {0.01: 1, 0.15: 1, 0.3: 2, 0.4: 4, 0.5: 2} |
| legacy | 200 | 10 | 20.091235 | 0.204745 | {0.1: 1, 0.2: 1, 0.3: 3, 0.4: 3, 0.5: 1, 0.7: 1} |
| legacy | 800 | 10 | 19.857999 | 0.023775 | {0.003: 1, 0.005: 1, 0.01: 1, 0.02: 1, 0.05: 1, 0.1: 2, 0.15: 1, 0.3: 1, 0.4: 1} |

## Per-seed choices

| Method | N | Seed | Radius | Validation cost | Test cost |
|---|---:|---:|---:|---:|---:|
| green | 50 | 0 | 0.4 | 19.074611 | 20.397530 |
| green | 50 | 1 | 0 | 18.587061 | 21.598501 |
| green | 50 | 2 | 0 | 22.041174 | 20.272913 |
| green | 50 | 3 | 0.7 | 19.256773 | 21.060428 |
| green | 50 | 4 | 0 | 18.166581 | 20.986390 |
| green | 50 | 5 | 0.2 | 17.109989 | 20.991289 |
| green | 50 | 6 | 0 | 21.702588 | 20.437394 |
| green | 50 | 7 | 0.7 | 14.196608 | 21.343787 |
| green | 50 | 8 | 0 | 20.207844 | 20.593589 |
| green | 50 | 9 | 0 | 18.994847 | 20.490985 |
| green | 200 | 0 | 0.4 | 23.060076 | 20.135360 |
| green | 200 | 1 | 0 | 22.530628 | 20.261377 |
| green | 200 | 2 | 0.2 | 23.943348 | 20.158759 |
| green | 200 | 3 | 0.4 | 16.943260 | 20.480687 |
| green | 200 | 4 | 0.7 | 21.350558 | 20.233462 |
| green | 200 | 5 | 0 | 19.885950 | 20.246324 |
| green | 200 | 6 | 0 | 20.369166 | 20.639064 |
| green | 200 | 7 | 0.2 | 20.809050 | 20.820746 |
| green | 200 | 8 | 0.4 | 20.589462 | 20.429895 |
| green | 200 | 9 | 0.2 | 19.329253 | 20.018627 |
| green | 800 | 0 | 0.7 | 21.467408 | 20.027231 |
| green | 800 | 1 | 0.4 | 20.428498 | 19.940943 |
| green | 800 | 2 | 0.4 | 18.829303 | 19.896191 |
| green | 800 | 3 | 0.2 | 18.821926 | 20.247602 |
| green | 800 | 4 | 0.2 | 19.339453 | 19.920673 |
| green | 800 | 5 | 0 | 19.723967 | 19.881709 |
| green | 800 | 6 | 0.2 | 20.507071 | 19.885259 |
| green | 800 | 7 | 0.4 | 21.269470 | 19.922637 |
| green | 800 | 8 | 0.2 | 19.129255 | 19.880471 |
| green | 800 | 9 | 0.7 | 20.341841 | 20.067451 |
| legacy | 50 | 0 | 0.3 | 17.654416 | 20.644693 |
| legacy | 50 | 1 | 0.3 | 18.708234 | 20.756184 |
| legacy | 50 | 2 | 0.5 | 28.149298 | 20.245766 |
| legacy | 50 | 3 | 0.5 | 22.118801 | 20.597635 |
| legacy | 50 | 4 | 0.4 | 17.667016 | 20.424172 |
| legacy | 50 | 5 | 0.15 | 21.324985 | 20.630473 |
| legacy | 50 | 6 | 0.01 | 16.260362 | 20.449322 |
| legacy | 50 | 7 | 0.4 | 24.061053 | 20.745652 |
| legacy | 50 | 8 | 0.4 | 19.009131 | 20.714345 |
| legacy | 50 | 9 | 0.4 | 16.032744 | 20.833502 |
| legacy | 200 | 0 | 0.7 | 17.973982 | 19.950320 |
| legacy | 200 | 1 | 0.4 | 20.151076 | 20.028273 |
| legacy | 200 | 2 | 0.3 | 23.048273 | 19.923222 |
| legacy | 200 | 3 | 0.4 | 19.400548 | 19.967614 |
| legacy | 200 | 4 | 0.1 | 19.891669 | 20.032003 |
| legacy | 200 | 5 | 0.2 | 18.533867 | 19.992531 |
| legacy | 200 | 6 | 0.3 | 20.518137 | 20.320979 |
| legacy | 200 | 7 | 0.3 | 22.968265 | 20.581828 |
| legacy | 200 | 8 | 0.4 | 21.408389 | 20.035402 |
| legacy | 200 | 9 | 0.5 | 17.933790 | 20.080179 |
| legacy | 800 | 0 | 0.005 | 18.156527 | 19.870657 |
| legacy | 800 | 1 | 0.3 | 19.508659 | 19.821917 |
| legacy | 800 | 2 | 0.02 | 20.711173 | 19.850271 |
| legacy | 800 | 3 | 0.1 | 19.829803 | 19.851695 |
| legacy | 800 | 4 | 0.003 | 19.462402 | 19.816203 |
| legacy | 800 | 5 | 0.1 | 20.168811 | 19.873796 |
| legacy | 800 | 6 | 0.01 | 20.424928 | 19.869300 |
| legacy | 800 | 7 | 0.05 | 19.609929 | 19.865972 |
| legacy | 800 | 8 | 0.15 | 21.118417 | 19.866630 |
| legacy | 800 | 9 | 0.4 | 20.050184 | 19.893547 |

## Runtime coverage

| Component | Status / scope |
|---|---|
| Nominal neural training | Not rerun. Persisted selected-model timings retained; missing historical isolated fit time remains unavailable. |
| Robust neural / CCG AP and MP | Not rerun. Historical per-iteration timing references retained in selected_test_results.json. |
| SRO-FA partition / optimization / evaluation / total | Not run; no new comparable timing. |
| Validation / selection | Newly measured, perf_counter seconds, full grid per seed and each candidate below. No cross-validation or refit. |
| Final test | Cached strict-q>0 replay, with original OOS time retained. Current lookup/raw verification is NOT test evaluation time. |

Environment: {'host': 'gold-srv2.iem.technion.ac.il', 'platform': 'Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28', 'affinity_cpus': 72, 'versions': {'python': '3.11.13 (main, Jan 28 2026, 00:22:30) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)]', 'numpy': '2.4.2', 'scipy': '1.17.1', 'torch': '2.10.0+cu128', 'gurobi': [13, 0, 1]}}. Eight validation workers; one Torch/Gurobi thread each. Concurrent job times are not summed as sweep elapsed. Full child time, command, log and cwd: ../production_v1.invocation.json.

| Method | N | Runs | Grid mean s (min) | Median s | Min s | Max s | Historical selected OOS mean s |
|---|---:|---:|---:|---:|---:|---:|---:|
| green | 50 | 10 | 0.248 (0.004) | 0.264 | 0.150 | 0.310 | 5.301 |
| green | 200 | 10 | 0.342 (0.006) | 0.340 | 0.336 | 0.353 | 5.017 |
| green | 800 | 10 | 0.888 (0.015) | 0.887 | 0.838 | 0.948 | 4.986 |
| legacy | 50 | 10 | 0.272 (0.005) | 0.268 | 0.252 | 0.293 | 4.964 |
| legacy | 200 | 10 | 0.575 (0.010) | 0.573 | 0.563 | 0.598 | 4.926 |
| legacy | 800 | 10 | 1.799 (0.030) | 1.788 | 1.746 | 1.959 | 5.104 |

| Method | N | Radius | Runs | Arm mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
| green | 50 | 0 | 10 | 0.074 | 0.086 | 0.015 | 0.104 |
| green | 50 | 0.001 | 10 | 0.019 | 0.016 | 0.016 | 0.028 |
| green | 50 | 0.003 | 10 | 0.025 | 0.022 | 0.015 | 0.051 |
| green | 50 | 0.01 | 10 | 0.020 | 0.016 | 0.015 | 0.040 |
| green | 50 | 0.03 | 10 | 0.023 | 0.016 | 0.015 | 0.062 |
| green | 50 | 0.2 | 10 | 0.022 | 0.021 | 0.015 | 0.029 |
| green | 50 | 0.4 | 10 | 0.018 | 0.016 | 0.015 | 0.028 |
| green | 50 | 0.7 | 10 | 0.018 | 0.016 | 0.015 | 0.027 |
| green | 50 | 1.5 | 10 | 0.018 | 0.016 | 0.015 | 0.028 |
| green | 200 | 0 | 10 | 0.037 | 0.036 | 0.036 | 0.047 |
| green | 200 | 0.001 | 10 | 0.036 | 0.036 | 0.036 | 0.040 |
| green | 200 | 0.003 | 10 | 0.036 | 0.036 | 0.035 | 0.038 |
| green | 200 | 0.01 | 10 | 0.037 | 0.036 | 0.036 | 0.040 |
| green | 200 | 0.03 | 10 | 0.036 | 0.036 | 0.035 | 0.037 |
| green | 200 | 0.2 | 10 | 0.036 | 0.036 | 0.036 | 0.037 |
| green | 200 | 0.4 | 10 | 0.037 | 0.036 | 0.036 | 0.038 |
| green | 200 | 0.7 | 10 | 0.038 | 0.036 | 0.036 | 0.049 |
| green | 200 | 1.5 | 10 | 0.037 | 0.036 | 0.036 | 0.041 |
| green | 800 | 0 | 10 | 0.098 | 0.097 | 0.096 | 0.107 |
| green | 800 | 0.001 | 10 | 0.096 | 0.097 | 0.086 | 0.097 |
| green | 800 | 0.003 | 10 | 0.096 | 0.097 | 0.086 | 0.097 |
| green | 800 | 0.01 | 10 | 0.098 | 0.097 | 0.086 | 0.120 |
| green | 800 | 0.03 | 10 | 0.097 | 0.097 | 0.086 | 0.108 |
| green | 800 | 0.2 | 10 | 0.098 | 0.097 | 0.086 | 0.108 |
| green | 800 | 0.4 | 10 | 0.099 | 0.097 | 0.096 | 0.109 |
| green | 800 | 0.7 | 10 | 0.099 | 0.097 | 0.097 | 0.109 |
| green | 800 | 1.5 | 10 | 0.098 | 0.097 | 0.086 | 0.109 |
| legacy | 50 | 0.001 | 10 | 0.019 | 0.016 | 0.016 | 0.030 |
| legacy | 50 | 0.003 | 10 | 0.016 | 0.016 | 0.016 | 0.017 |
| legacy | 50 | 0.005 | 10 | 0.017 | 0.016 | 0.016 | 0.028 |
| legacy | 50 | 0.01 | 10 | 0.017 | 0.016 | 0.016 | 0.028 |
| legacy | 50 | 0.02 | 10 | 0.017 | 0.016 | 0.016 | 0.021 |
| legacy | 50 | 0.05 | 10 | 0.016 | 0.016 | 0.016 | 0.018 |
| legacy | 50 | 0.1 | 10 | 0.017 | 0.016 | 0.016 | 0.021 |
| legacy | 50 | 0.15 | 10 | 0.018 | 0.016 | 0.016 | 0.038 |
| legacy | 50 | 0.2 | 10 | 0.016 | 0.016 | 0.016 | 0.017 |
| legacy | 50 | 0.3 | 10 | 0.016 | 0.016 | 0.016 | 0.017 |
| legacy | 50 | 0.4 | 10 | 0.018 | 0.016 | 0.016 | 0.028 |
| legacy | 50 | 0.5 | 10 | 0.017 | 0.016 | 0.016 | 0.028 |
| legacy | 50 | 0.7 | 10 | 0.016 | 0.016 | 0.016 | 0.017 |
| legacy | 50 | 1 | 10 | 0.018 | 0.016 | 0.016 | 0.028 |
| legacy | 50 | 1.5 | 10 | 0.017 | 0.016 | 0.016 | 0.020 |
| legacy | 200 | 0.001 | 10 | 0.038 | 0.037 | 0.036 | 0.048 |
| legacy | 200 | 0.003 | 10 | 0.038 | 0.036 | 0.036 | 0.047 |
| legacy | 200 | 0.005 | 10 | 0.039 | 0.037 | 0.036 | 0.046 |
| legacy | 200 | 0.01 | 10 | 0.037 | 0.036 | 0.036 | 0.041 |
| legacy | 200 | 0.02 | 10 | 0.037 | 0.036 | 0.036 | 0.040 |
| legacy | 200 | 0.05 | 10 | 0.036 | 0.036 | 0.036 | 0.037 |
| legacy | 200 | 0.1 | 10 | 0.038 | 0.036 | 0.036 | 0.047 |
| legacy | 200 | 0.15 | 10 | 0.038 | 0.036 | 0.036 | 0.050 |
| legacy | 200 | 0.2 | 10 | 0.037 | 0.036 | 0.036 | 0.041 |
| legacy | 200 | 0.3 | 10 | 0.036 | 0.036 | 0.036 | 0.037 |
| legacy | 200 | 0.4 | 10 | 0.036 | 0.036 | 0.036 | 0.037 |
| legacy | 200 | 0.5 | 10 | 0.036 | 0.036 | 0.036 | 0.037 |
| legacy | 200 | 0.7 | 10 | 0.038 | 0.036 | 0.036 | 0.046 |
| legacy | 200 | 1 | 10 | 0.038 | 0.037 | 0.036 | 0.051 |
| legacy | 200 | 1.5 | 10 | 0.037 | 0.037 | 0.036 | 0.040 |
| legacy | 800 | 0.001 | 10 | 0.118 | 0.117 | 0.107 | 0.129 |
| legacy | 800 | 0.003 | 10 | 0.117 | 0.117 | 0.107 | 0.128 |
| legacy | 800 | 0.005 | 10 | 0.119 | 0.117 | 0.117 | 0.128 |
| legacy | 800 | 0.01 | 10 | 0.118 | 0.117 | 0.106 | 0.129 |
| legacy | 800 | 0.02 | 10 | 0.120 | 0.117 | 0.117 | 0.129 |
| legacy | 800 | 0.05 | 10 | 0.119 | 0.117 | 0.117 | 0.128 |
| legacy | 800 | 0.1 | 10 | 0.121 | 0.117 | 0.117 | 0.140 |
| legacy | 800 | 0.15 | 10 | 0.117 | 0.117 | 0.107 | 0.128 |
| legacy | 800 | 0.2 | 10 | 0.120 | 0.117 | 0.117 | 0.129 |
| legacy | 800 | 0.3 | 10 | 0.120 | 0.117 | 0.117 | 0.130 |
| legacy | 800 | 0.4 | 10 | 0.119 | 0.117 | 0.107 | 0.129 |
| legacy | 800 | 0.5 | 10 | 0.121 | 0.117 | 0.117 | 0.140 |
| legacy | 800 | 0.7 | 10 | 0.116 | 0.117 | 0.107 | 0.128 |
| legacy | 800 | 1 | 10 | 0.117 | 0.117 | 0.106 | 0.129 |
| legacy | 800 | 1.5 | 10 | 0.120 | 0.117 | 0.106 | 0.142 |

## Current sweep phases (nested; not summed worker walls)

- regression_tests: 2.586 s (0.043 min), completed.
- input_freeze_and_split_verification: 15.809 s (0.263 min), completed.
- training_only_preflight: 0.666 s (0.011 min), completed.
- parallel_validation: 9.087 s (0.151 min), completed.
- raw_validation_verification_and_selection: 1.516 s (0.025 min), completed.
- cached_test_lookup_and_raw_verification: 10.580 s (0.176 min), completed.
- final_integrity: 5.739 s (0.096 min), completed.

Raw per-arm arrays, solve journals, status/iteration counts and timing are under validation/. Input hashes and split identities: inputs.json. Pre-test choices: selections.json and its SHA256 seal. Selected test identities, raw verification time and prior training/evaluation times: selected_test_results.json.

Reporting: 0.018 s (0.000 min); completed, excludes final timing snapshot writes.
