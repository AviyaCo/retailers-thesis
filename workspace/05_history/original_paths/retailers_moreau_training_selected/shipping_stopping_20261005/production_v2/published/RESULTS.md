# Shipping — stopping-checkpoint comparison

All 270 green policies freshly replayed at their first passing stopping assessment, with its own pool.
Changed selections: 12; changed learned states: 7; changed pools: 12.
The 480 historical Clarabel policies, kNN, SRO-FA and conditional reference are unchanged cached parent curves.
Same 10,000 historical test scenarios, seed 9999, all ten training seeds and original nine radii.
Common strict Gurobi replay; raw q>0 fees; OPTIMAL and physical residual <=1e-6 required. No clipping or retry.
Training40/160/640; green mixed cohort265 original +5 advisory; no exact/global/every-center robustness claim.
IS is mean per-center maximum on the stopping policy own pre-insertion pool, not a full robust supremum.
This known test and historically test-informed design are descriptive, not fresh confirmatory evidence. No radius selection or tuning. Earlier minimum-AP artifacts remain unchanged.

## Runtime coverage

| Component | Current work |
|---|---|
| Nominal neural training | Not rerun; original per-seed timings retained |
| Robust CCG / AP / MP | Not rerun; original per-iteration timings retained |
| SRO-FA partition / optimization / evaluation / total | Not rerun; parent plot provenance retained |
| CV / radius validation / refitting | Not run |
| Green OOS and own-pool evaluation | Fresh; per-configuration table below |
| Legacy neural evaluation | Cached September28 results; not current replay time |

Host: gold-srv2.iem.technion.ac.il; 16 workers, one Torch/Gurobi thread each.
Missing historical times remain unavailable; a rerun is required to measure them.
Whole-child elapsed and UTC start/end: ../../production_v2.invocation.json (written at process exit).

## Completed evaluation runtimes (seconds)

| N | Radius | Runs | Mean | Median | Min | Max | Mean OOS seconds | Mean IS seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 10 | 5.684 | 5.637 | 5.361 | 6.120 | 5.376 | 0.048 |
| 50 | 0.001 | 10 | 5.690 | 5.540 | 5.330 | 6.336 | 5.377 | 0.050 |
| 50 | 0.003 | 10 | 5.738 | 5.668 | 5.296 | 6.301 | 5.440 | 0.041 |
| 50 | 0.01 | 10 | 5.651 | 5.407 | 5.260 | 7.023 | 5.344 | 0.049 |
| 50 | 0.03 | 10 | 5.569 | 5.471 | 5.248 | 6.070 | 5.259 | 0.053 |
| 50 | 0.2 | 10 | 5.499 | 5.393 | 5.320 | 6.123 | 5.129 | 0.077 |
| 50 | 0.4 | 10 | 5.721 | 5.600 | 5.318 | 6.667 | 5.295 | 0.104 |
| 50 | 0.7 | 10 | 5.812 | 5.673 | 5.371 | 6.443 | 5.350 | 0.127 |
| 50 | 1.5 | 10 | 5.596 | 5.434 | 4.916 | 6.648 | 5.131 | 0.140 |
| 200 | 0 | 10 | 5.365 | 5.359 | 5.229 | 5.614 | 5.007 | 0.106 |
| 200 | 0.001 | 10 | 5.400 | 5.385 | 5.258 | 5.674 | 5.043 | 0.108 |
| 200 | 0.003 | 10 | 5.375 | 5.357 | 5.249 | 5.573 | 5.013 | 0.108 |
| 200 | 0.01 | 10 | 5.398 | 5.395 | 5.178 | 5.712 | 5.038 | 0.107 |
| 200 | 0.03 | 10 | 5.368 | 5.340 | 5.267 | 5.508 | 5.017 | 0.101 |
| 200 | 0.2 | 10 | 5.506 | 5.463 | 5.370 | 5.766 | 5.003 | 0.217 |
| 200 | 0.4 | 10 | 5.612 | 5.563 | 5.370 | 6.012 | 4.982 | 0.305 |
| 200 | 0.7 | 10 | 5.734 | 5.706 | 5.504 | 5.997 | 5.017 | 0.377 |
| 200 | 1.5 | 10 | 5.828 | 5.811 | 5.425 | 6.241 | 5.088 | 0.407 |
| 800 | 0 | 10 | 5.672 | 5.652 | 5.523 | 5.963 | 5.048 | 0.345 |
| 800 | 0.001 | 10 | 5.696 | 5.655 | 5.571 | 6.038 | 5.064 | 0.360 |
| 800 | 0.003 | 10 | 5.756 | 5.697 | 5.516 | 6.306 | 5.155 | 0.338 |
| 800 | 0.01 | 10 | 5.632 | 5.618 | 5.522 | 5.784 | 5.021 | 0.344 |
| 800 | 0.03 | 10 | 5.654 | 5.657 | 5.553 | 5.824 | 5.035 | 0.349 |
| 800 | 0.2 | 10 | 6.022 | 5.967 | 5.797 | 6.486 | 5.009 | 0.703 |
| 800 | 0.4 | 10 | 6.220 | 6.198 | 5.886 | 6.721 | 5.021 | 0.865 |
| 800 | 0.7 | 10 | 6.621 | 6.631 | 6.292 | 7.039 | 5.053 | 1.179 |
| 800 | 1.5 | 10 | 6.946 | 6.997 | 6.684 | 7.255 | 5.051 | 1.474 |

## Current phases (nested; not additive concurrent job times)

- tests: 26.064499 s / 0.434408 min / 0.007240 h; completed.
- selection_and_freeze: 96.253602 s / 1.604227 min / 0.026737 h; completed.
- training_only_preflight: 15.791072 s / 0.263185 min / 0.004386 h; completed.
- test_binding: 9.417810 s / 0.156964 min / 0.002616 h; completed.
- green_evaluation: 108.725594 s / 1.812093 min / 0.030202 h; completed.
- raw_verification: 65.751183 s / 1.095853 min / 0.018264 h; completed.
- final_integrity: 14.373696 s / 0.239562 min / 0.003993 h; completed.
- render: 5.793412 s / 0.096557 min / 0.001609 h; completed.
