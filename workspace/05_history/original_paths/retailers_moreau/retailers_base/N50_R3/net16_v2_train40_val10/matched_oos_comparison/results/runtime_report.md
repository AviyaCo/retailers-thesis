# Matched OOS runtime report

Per-configuration current evaluation wall times, seconds. Concurrent times are not summed.

| Method | N train | Radius | Completed n | Mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
| new | 40 | 0 | 10 | 4.975551 | 4.854942 | 4.768969 | 5.414531 |
| new | 40 | 0.001 | 10 | 4.872779 | 4.863754 | 4.752441 | 5.076111 |
| new | 40 | 0.003 | 10 | 4.901813 | 4.936554 | 4.791707 | 5.032402 |
| new | 40 | 0.01 | 10 | 4.887495 | 4.871381 | 4.777526 | 5.039906 |
| new | 40 | 0.03 | 10 | 4.937011 | 4.844758 | 4.800475 | 5.520378 |
| new | 40 | 0.2 | 10 | 4.941365 | 4.863092 | 4.777153 | 5.494508 |
| new | 40 | 0.4 | 10 | 4.916858 | 4.854200 | 4.787035 | 5.118960 |
| new | 40 | 0.7 | 10 | 4.921576 | 4.863573 | 4.818364 | 5.066100 |
| new | 40 | 1.5 | 10 | 4.928537 | 4.912312 | 4.781794 | 5.109366 |
| old | 50 | 0.001 | 10 | 4.997856 | 4.915764 | 4.786548 | 5.540256 |
| old | 50 | 0.0015 | 10 | 4.958132 | 4.918761 | 4.784285 | 5.218243 |
| old | 50 | 0.002 | 10 | 5.010147 | 4.967435 | 4.814243 | 5.431023 |
| old | 50 | 0.003 | 10 | 4.920099 | 4.838335 | 4.754109 | 5.419264 |
| old | 50 | 0.005 | 10 | 4.875004 | 4.817069 | 4.757224 | 5.131562 |
| old | 50 | 0.01 | 10 | 4.976018 | 4.929843 | 4.807632 | 5.231532 |
| old | 50 | 0.02 | 10 | 4.943880 | 4.890247 | 4.809269 | 5.362579 |
| old | 50 | 0.05 | 10 | 4.940488 | 4.898980 | 4.802658 | 5.330754 |
| old | 50 | 0.1 | 10 | 4.911342 | 4.841628 | 4.732226 | 5.526116 |
| old | 50 | 0.15 | 10 | 4.944396 | 4.855104 | 4.798497 | 5.478143 |
| old | 50 | 0.2 | 10 | 4.857047 | 4.832784 | 4.740015 | 5.082540 |
| old | 50 | 0.3 | 10 | 4.887189 | 4.857054 | 4.725189 | 5.083596 |
| old | 50 | 0.4 | 10 | 4.820876 | 4.829445 | 4.700599 | 4.955867 |
| old | 50 | 0.45 | 10 | 4.995886 | 5.015006 | 4.747061 | 5.225529 |
| old | 50 | 0.5 | 10 | 4.929965 | 4.907153 | 4.705923 | 5.182483 |
| old | 50 | 0.55 | 10 | 5.050155 | 4.975943 | 4.721617 | 5.598956 |
| old | 50 | 0.6 | 10 | 4.980853 | 4.901840 | 4.751405 | 5.442766 |
| old | 50 | 0.65 | 10 | 4.918260 | 4.847827 | 4.736096 | 5.520671 |
| old | 50 | 0.7 | 10 | 4.899173 | 4.842500 | 4.732024 | 5.220663 |
| old | 50 | 0.75 | 10 | 4.906354 | 4.905372 | 4.804614 | 5.056849 |
| old | 50 | 0.8 | 10 | 4.874132 | 4.835415 | 4.692697 | 5.322287 |
| old | 50 | 1 | 10 | 4.908805 | 4.842064 | 4.664330 | 5.393297 |
| old | 50 | 1.5 | 10 | 4.794511 | 4.806051 | 4.500380 | 5.051785 |

Overall invocation: 203.397953 s / 3.389966 min / 0.056499 h; status: completed.
Failed/incomplete: 0; excluded above.
Cached-only evaluations: zero (identical states are intentionally re-evaluated).

| Phase/category | Current wall time |
|---|---|
| verify_frozen_inputs (completed) | 0.158378 s / 0.002640 min / 0.000044 h |
| test_generation (completed) | 0.023599 s / 0.000393 min / 0.000007 h |
| parallel_evaluation (completed) | 202.396956 s / 3.373283 min / 0.056221 h |
| plot (completed) | 0.645304 s / 0.010755 min / 0.000179 h |
| Nominal neural training | Not run; historical per-model timing not imported; rerun needed if unavailable |
| Robust neural/CCG AP, MP, total | Not run; historical training artifacts remain separate |
| SRO-FA partition, solver, total | Not run; reference cached, not a current timing |
| Hyperparameter selection / CV | Not run; choices frozen |
| Neural final test evaluation | Per-model table and raw phase timings above |

Environment: {"host": "gold-srv2.iem.technion.ac.il", "platform": "Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28", "processor": "x86_64", "cpu_count": 72}; 8 workers, one solver/Torch/BLAS thread each.
Unknown historical timings are not inferred from cache loading. Shared-host contention is uncontrolled.
