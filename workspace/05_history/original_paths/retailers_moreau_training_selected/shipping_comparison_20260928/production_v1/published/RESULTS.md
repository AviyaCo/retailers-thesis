# Shipping comparison — completed fixed-policy replay

750 policies: 270 green (265 original + five advisory continuations), 480 historical Clarabel.
Same already-inspected 10,000 scenarios, seed 9999; descriptive, not a fresh confirmatory test.
Both neural families: fresh Gurobi projection, raw q>0 fee, no clipping or cutoff. Physical tolerance 1e-6; all projection statuses must be OPTIMAL.
IS uses each returned checkpoint own saved finite pool; not AP mean or full robust supremum.
Green membership remains strict. Historical coordinates are unchanged with explicit residual disclosure.
No black shipping method was trained. No test-driven choice, radius validation, refit, CV, or new benchmark fitting.
Historical kNN and SRO-FA retain thresholds 1e-6 and 1e-10 respectively; their cached evaluations are descriptive, not certified common-fee comparisons.
The gray conditional finite 1M-scenario MIP estimate is not an exact population lower bound.

Host: gold-srv2.iem.technion.ac.il; 16 workers, one solver/Torch thread each.
Bands: pointwise 20–80% across ten training seeds, not confidence intervals.

## Current evaluation runtimes (seconds)

| Method | N | Radius | Runs | Mean | Median | Min | Max | Mean OOS | Mean IS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| green | 50 | 0 | 10 | 5.450 | 5.416 | 5.238 | 5.625 | 5.146 | 0.041 |
| green | 50 | 0.001 | 10 | 5.383 | 5.347 | 5.226 | 5.595 | 5.090 | 0.040 |
| green | 50 | 0.003 | 10 | 5.407 | 5.414 | 5.191 | 5.676 | 5.121 | 0.037 |
| green | 50 | 0.01 | 10 | 5.343 | 5.338 | 5.219 | 5.497 | 5.053 | 0.041 |
| green | 50 | 0.03 | 10 | 5.359 | 5.360 | 5.263 | 5.449 | 5.067 | 0.043 |
| green | 50 | 0.2 | 10 | 5.357 | 5.343 | 5.244 | 5.593 | 5.005 | 0.075 |
| green | 50 | 0.4 | 10 | 5.465 | 5.471 | 5.335 | 5.665 | 5.061 | 0.096 |
| green | 50 | 0.7 | 10 | 5.480 | 5.482 | 5.365 | 5.624 | 5.041 | 0.114 |
| green | 50 | 1.5 | 10 | 5.303 | 5.346 | 4.761 | 5.707 | 4.875 | 0.119 |
| green | 200 | 0 | 10 | 5.480 | 5.483 | 5.227 | 5.788 | 5.125 | 0.102 |
| green | 200 | 0.001 | 10 | 5.612 | 5.510 | 5.268 | 6.402 | 5.239 | 0.111 |
| green | 200 | 0.003 | 10 | 5.433 | 5.445 | 5.210 | 5.750 | 5.081 | 0.100 |
| green | 200 | 0.01 | 10 | 5.484 | 5.494 | 5.257 | 5.632 | 5.130 | 0.098 |
| green | 200 | 0.03 | 10 | 5.542 | 5.548 | 5.300 | 5.850 | 5.179 | 0.102 |
| green | 200 | 0.2 | 10 | 5.620 | 5.651 | 5.386 | 5.774 | 5.119 | 0.212 |
| green | 200 | 0.4 | 10 | 5.785 | 5.814 | 5.537 | 6.050 | 5.155 | 0.296 |
| green | 200 | 0.7 | 10 | 5.786 | 5.785 | 5.587 | 5.977 | 5.085 | 0.361 |
| green | 200 | 1.5 | 10 | 5.707 | 5.645 | 5.383 | 6.135 | 4.969 | 0.400 |
| green | 800 | 0 | 10 | 5.572 | 5.567 | 5.487 | 5.658 | 4.972 | 0.332 |
| green | 800 | 0.001 | 10 | 5.663 | 5.601 | 5.479 | 6.076 | 5.052 | 0.338 |
| green | 800 | 0.003 | 10 | 5.577 | 5.576 | 5.501 | 5.690 | 4.972 | 0.337 |
| green | 800 | 0.01 | 10 | 5.585 | 5.585 | 5.458 | 5.769 | 4.977 | 0.335 |
| green | 800 | 0.03 | 10 | 5.601 | 5.605 | 5.489 | 5.699 | 4.992 | 0.335 |
| green | 800 | 0.2 | 10 | 5.877 | 5.909 | 5.540 | 6.201 | 4.995 | 0.566 |
| green | 800 | 0.4 | 10 | 6.168 | 6.164 | 5.770 | 6.535 | 5.040 | 0.793 |
| green | 800 | 0.7 | 10 | 6.490 | 6.466 | 6.220 | 6.682 | 4.990 | 1.118 |
| green | 800 | 1.5 | 10 | 6.783 | 6.761 | 6.417 | 7.231 | 4.973 | 1.394 |
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

## Model-runtime coverage

| Category | Current work / historical scope |
|---|---|
| Nominal neural training | Not rerun; persisted per-seed nominal records in runtime report |
| Robust neural/CCG | Not rerun; persisted AP/MP/current/cached timing distinctions retained |
| SRO-FA partition/optimization/evaluation/total | Not rerun; original per-seed time_partition/time_lp/time_eval/time_total retained in plot-data provenance |
| Hyperparameter selection/CV/radius validation | Not run |
| Common neural OOS and own-pool evaluation | Fresh, separately timed above |
Missing historical fields remain unavailable; a rerun is required to measure them. Cache-loading time is never substituted for training time.

## Phase runtimes

- freeze: 142.282068 s / 2.371368 min / 0.039523 h (completed).
- tests: 24.660168 s / 0.411003 min / 0.006850 h (completed).
- training_only_preflight: 15.203220 s / 0.253387 min / 0.004223 h (completed).
- test_generation: 9.209429 s / 0.153490 min / 0.002558 h (completed).
- evaluation: 283.267780 s / 4.721130 min / 0.078685 h (completed).
- journal_verification: 168.555040 s / 2.809251 min / 0.046821 h (completed).
- final_integrity: 13.236601 s / 0.220610 min / 0.003677 h (completed).
- render: 6.097350 s / 0.101623 min / 0.001694 h (completed).

Whole-child runtime and UTC boundaries are recorded in the adjacent invocation JSON. Phases are nested; concurrent job times are not elapsed campaign time.

Raw per-row projection journals and cost/target/shipment arrays were verified before publication. All original and frozen source/input hashes were checked again; no failed models are silently omitted.
