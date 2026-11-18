# Stopping-checkpoint comparison

Both GREEN (540) and BLACK (570) use their actual stopping checkpoints, not minimum-AP checkpoints. All 11 nonconverged BLACK iteration-cap policies are retained at stop 40; successful replay does not imply training convergence or globally certified robust optimality. Both are freshly evaluated on the same 10,000 test scenarios per R (seed 9999). This historically known test is descriptive, not a new holdout. No new validation, radius selection, ranking test, or test-based policy selection is performed. Fixed saved nodes only; connecting lines are visual. Bands are pointwise seed 20/80 percentiles, not confidence intervals. Dashed neural IS curves are means of per-center maxima on each checkpoint's own finite pool, not a full adversarial supremum. SCS, SRO-FA, SAA-kNN and all original reference levels and qualifications are unchanged historical comparators, not fresh results. SCS pools are unrepaired and not newly box-certified; historical test solver status is not newly certified. Gray references retain their historical sampling/test-fit qualifications and are not newly certified population bounds. Only BLACK extends to radius 2 for R10. Current evaluation and persisted training timings are separate; unavailable historical totals require a rerun.

Counts: GREEN 540 converged; BLACK 559 converged + 11 capped. All 1,110 fresh replays completed.
Cap identities: black_R10_N200_s6_rho0.7, black_R10_N50_s0_rho0.7, black_R10_N50_s2_rho0.7, black_R10_N50_s2_rho1.5, black_R10_N50_s4_rho0.7, black_R10_N50_s5_rho0.4, black_R10_N50_s6_rho0.7, black_R10_N50_s6_rho1.5, black_R10_N50_s8_rho0.2, black_R10_N50_s8_rho0.7, black_R3_N50_s9_rho0.7.

## Fixed-radius matched costs

| R | N | radius | GREEN OOS | BLACK OOS | BLACK − GREEN |
|---|---|---|---|---|---|
| 3 | 50 | 0 | 20.591139 | 20.591139 | 0.000000 |
| 3 | 50 | 0.001 | 20.591139 | 20.591139 | 0.000000 |
| 3 | 50 | 0.003 | 20.591139 | 20.591139 | 0.000000 |
| 3 | 50 | 0.01 | 20.591139 | 20.591139 | 0.000000 |
| 3 | 50 | 0.03 | 20.591139 | 20.636376 | 0.045236 |
| 3 | 50 | 0.2 | 20.573817 | 20.476167 | -0.097650 |
| 3 | 50 | 0.4 | 20.529497 | 20.470175 | -0.059322 |
| 3 | 50 | 0.7 | 20.733442 | 20.666352 | -0.067090 |
| 3 | 50 | 1.5 | 21.484426 | 21.375432 | -0.108994 |
| 3 | 200 | 0 | 19.947783 | 19.947783 | 0.000000 |
| 3 | 200 | 0.001 | 19.947783 | 19.947783 | 0.000000 |
| 3 | 200 | 0.003 | 19.947783 | 19.947783 | 0.000000 |
| 3 | 200 | 0.01 | 19.947783 | 19.947783 | 0.000000 |
| 3 | 200 | 0.03 | 19.947783 | 19.917962 | -0.029821 |
| 3 | 200 | 0.2 | 19.888235 | 19.867081 | -0.021154 |
| 3 | 200 | 0.4 | 19.873891 | 19.850942 | -0.022950 |
| 3 | 200 | 0.7 | 20.060598 | 20.003549 | -0.057049 |
| 3 | 200 | 1.5 | 21.345981 | 21.340129 | -0.005852 |
| 3 | 800 | 0 | 19.820354 | 19.820354 | 0.000000 |
| 3 | 800 | 0.001 | 19.820354 | 19.820354 | 0.000000 |
| 3 | 800 | 0.003 | 19.820354 | 19.820354 | 0.000000 |
| 3 | 800 | 0.01 | 19.820354 | 19.811819 | -0.008535 |
| 3 | 800 | 0.03 | 19.820354 | 19.794715 | -0.025638 |
| 3 | 800 | 0.2 | 19.824071 | 19.719183 | -0.104888 |
| 3 | 800 | 0.4 | 19.753921 | 19.640055 | -0.113865 |
| 3 | 800 | 0.7 | 19.946312 | 19.868154 | -0.078158 |
| 3 | 800 | 1.5 | 21.291767 | 21.338853 | 0.047086 |
| 10 | 50 | 0 | 78.234062 | 78.234062 | 0.000000 |
| 10 | 50 | 0.001 | 78.234062 | 78.234062 | 0.000000 |
| 10 | 50 | 0.003 | 78.234062 | 78.234062 | 0.000000 |
| 10 | 50 | 0.01 | 78.234062 | 78.234062 | 0.000000 |
| 10 | 50 | 0.03 | 78.309461 | 78.376540 | 0.067079 |
| 10 | 50 | 0.2 | 76.498006 | 76.572047 | 0.074041 |
| 10 | 50 | 0.4 | 74.434274 | 74.515848 | 0.081574 |
| 10 | 50 | 0.7 | 73.235910 | 73.323959 | 0.088049 |
| 10 | 50 | 1.5 | 72.669066 | 72.570100 | -0.098966 |
| 10 | 50 | 2 | unavailable (BLACK only) | 73.321306 | unavailable (BLACK only) |
| 10 | 200 | 0 | 71.632591 | 71.632591 | 0.000000 |
| 10 | 200 | 0.001 | 71.632591 | 71.632591 | 0.000000 |
| 10 | 200 | 0.003 | 71.632591 | 71.632591 | 0.000000 |
| 10 | 200 | 0.01 | 71.632591 | 71.632591 | 0.000000 |
| 10 | 200 | 0.03 | 71.632591 | 70.732902 | -0.899689 |
| 10 | 200 | 0.2 | 69.990449 | 69.656062 | -0.334388 |
| 10 | 200 | 0.4 | 69.537462 | 69.101868 | -0.435594 |
| 10 | 200 | 0.7 | 69.470363 | 68.787042 | -0.683321 |
| 10 | 200 | 1.5 | 71.378079 | 71.554935 | 0.176856 |
| 10 | 200 | 2 | unavailable (BLACK only) | 72.408512 | unavailable (BLACK only) |
| 10 | 800 | 0 | 70.268129 | 70.268129 | 0.000000 |
| 10 | 800 | 0.001 | 70.268129 | 70.268129 | 0.000000 |
| 10 | 800 | 0.003 | 70.268129 | 70.268129 | 0.000000 |
| 10 | 800 | 0.01 | 70.268129 | 70.268129 | 0.000000 |
| 10 | 800 | 0.03 | 70.268129 | 69.752182 | -0.515947 |
| 10 | 800 | 0.2 | 69.110382 | 68.097071 | -1.013311 |
| 10 | 800 | 0.4 | 68.419964 | 68.066875 | -0.353089 |
| 10 | 800 | 0.7 | 68.506557 | 68.342644 | -0.163914 |
| 10 | 800 | 1.5 | 71.443245 | 71.592201 | 0.148956 |
| 10 | 800 | 2 | unavailable (BLACK only) | 72.499506 | unavailable (BLACK only) |

## Runtime

Per-policy wall time, NOT summed concurrent sweep elapsed. Persisted active phase sums exclude nominal and are NOT end-to-end wall time. Caps remain separate from converged runs.
Unavailable historical totals require rerun; never inferred from phase sums.
Each cell is count; mean (s/min/h); median / min / max (seconds). No concurrent times are summed.
Raw phase/iteration timings, timestamps and supplied thread/worker counts are in runtime_report.json.
Hardware/environment: {"versions": {"python": "3.11.13 (main, Jan 28 2026, 00:22:30) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)]", "numpy": "2.4.2", "scipy": "1.17.1", "torch": "2.10.0+cu128", "gurobi": [13, 0, 1]}, "executable": "/data/home/aviyac/ap_env/bin/python", "platform": "Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28", "hostname": "gold-srv2.iem.technion.ac.il", "machine": "x86_64", "workers": 16, "threads": 1, "torch_threads": 1, "torch_interop_threads": 1, "thread_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}}.

| Method | R/N/radius/status | Evaluation | Persisted training total | Persisted active phases |
|---|---|---|---|---|
| Nominal selected arm | 3/50/—/completed | not separately measured | 10; 37.620 s (0.627 min; 0.0105 h); 37.242 / 29.627 / 45.877; missing=0 | — |
| Nominal selected arm | 3/200/—/completed | not separately measured | 10; 149.802 s (2.497 min; 0.0416 h); 149.807 / 149.042 / 150.498; missing=0 | — |
| Nominal selected arm | 3/800/—/completed | not separately measured | 10; 170.070 s (2.834 min; 0.0472 h); 171.988 / 157.940 / 174.612; missing=0 | — |
| Nominal selected arm | 10/50/—/completed | not separately measured | 10; 113.648 s (1.894 min; 0.0316 h); 125.832 / 83.230 / 129.866; missing=0 | — |
| Nominal selected arm | 10/200/—/completed | not separately measured | 10; 161.969 s (2.699 min; 0.0450 h); 169.707 / 96.384 / 170.892; missing=0 | — |
| Nominal selected arm | 10/800/—/completed | not separately measured | 10; 198.557 s (3.309 min; 0.0552 h); 215.968 / 121.084 / 216.452; missing=0 | — |
| black | 3/50/0/converged | 10; 5.471 s (0.091 min; 0.0015 h); 5.397 / 5.264 / 5.935; missing=0 | 10; 2.087 s (0.035 min; 0.0006 h); 2.000 / 1.757 / 2.684; missing=0 | 10; 1.909 s (0.032 min; 0.0005 h); 1.829 / 1.583 / 2.448; missing=0 |
| black | 3/50/0.001/converged | 10; 5.426 s (0.090 min; 0.0015 h); 5.412 / 5.215 / 5.744; missing=0 | 10; 2.028 s (0.034 min; 0.0006 h); 1.863 / 1.680 / 2.686; missing=0 | 10; 1.849 s (0.031 min; 0.0005 h); 1.695 / 1.491 / 2.470; missing=0 |
| black | 3/50/0.003/converged | 10; 5.455 s (0.091 min; 0.0015 h); 5.406 / 5.242 / 6.062; missing=0 | 10; 2.032 s (0.034 min; 0.0006 h); 2.072 / 1.573 / 2.528; missing=0 | 10; 1.863 s (0.031 min; 0.0005 h); 1.901 / 1.404 / 2.368; missing=0 |
| black | 3/50/0.01/converged | 10; 5.527 s (0.092 min; 0.0015 h); 5.444 / 5.254 / 6.270; missing=0 | 10; 1.947 s (0.032 min; 0.0005 h); 1.900 / 1.643 / 2.491; missing=0 | 10; 1.764 s (0.029 min; 0.0005 h); 1.723 / 1.508 / 2.329; missing=0 |
| black | 3/50/0.03/converged | 10; 5.534 s (0.092 min; 0.0015 h); 5.486 / 5.289 / 5.761; missing=0 | 10; 98.535 s (1.642 min; 0.0274 h); 82.643 / 1.712 / 208.361; missing=0 | 10; 93.363 s (1.556 min; 0.0259 h); 77.946 / 1.576 / 197.640; missing=0 |
| black | 3/50/0.2/converged | 10; 5.598 s (0.093 min; 0.0016 h); 5.535 / 5.327 / 6.084; missing=0 | 10; 715.089 s (11.918 min; 0.1986 h); 624.722 / 338.278 / 1597.339; missing=0 | 10; 666.755 s (11.113 min; 0.1852 h); 583.617 / 319.052 / 1479.017; missing=0 |
| black | 3/50/0.4/converged | 10; 5.589 s (0.093 min; 0.0016 h); 5.543 / 5.375 / 6.054; missing=0 | 10; 1412.568 s (23.543 min; 0.3924 h); 1400.907 / 889.616 / 2300.128; missing=0 | 10; 1306.289 s (21.771 min; 0.3629 h); 1297.474 / 828.213 / 2107.559; missing=0 |
| black | 3/50/0.7/capped | 1; 5.777 s (0.096 min; 0.0016 h); 5.777 / 5.777 / 5.777; missing=0 | 1; 11411.629 s (190.194 min; 3.1699 h); 11411.629 / 11411.629 / 11411.629; missing=0 | 1; 10302.616 s (171.710 min; 2.8618 h); 10302.616 / 10302.616 / 10302.616; missing=0 |
| black | 3/50/0.7/converged | 9; 5.607 s (0.093 min; 0.0016 h); 5.588 / 5.355 / 6.070; missing=0 | 9; 4017.667 s (66.961 min; 1.1160 h); 4090.433 / 1230.665 / 8251.298; missing=0 | 9; 3677.568 s (61.293 min; 1.0215 h); 3739.076 / 1146.340 / 7526.121; missing=0 |
| black | 3/50/1.5/converged | 10; 5.749 s (0.096 min; 0.0016 h); 5.791 / 5.532 / 6.025; missing=0 | 10; 3859.725 s (64.329 min; 1.0721 h); 3056.512 / 734.514 / 9307.253; missing=0 | 10; 3522.961 s (58.716 min; 0.9786 h); 2798.608 / 672.437 / 8428.786; missing=0 |
| black | 3/200/0/converged | 10; 5.556 s (0.093 min; 0.0015 h); 5.607 / 5.305 / 5.761; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3.436 s (0.057 min; 0.0010 h); 3.392 / 3.285 / 3.743; missing=0 |
| black | 3/200/0.001/converged | 10; 5.568 s (0.093 min; 0.0015 h); 5.534 / 5.357 / 5.903; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3.588 s (0.060 min; 0.0010 h); 3.568 / 3.477 / 3.794; missing=0 |
| black | 3/200/0.003/converged | 10; 5.471 s (0.091 min; 0.0015 h); 5.498 / 5.242 / 5.714; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3.646 s (0.061 min; 0.0010 h); 3.645 / 3.453 / 3.850; missing=0 |
| black | 3/200/0.01/converged | 10; 5.589 s (0.093 min; 0.0016 h); 5.607 / 5.329 / 5.796; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 6.558 s (0.109 min; 0.0018 h); 3.746 / 3.600 / 14.632; missing=0 |
| black | 3/200/0.03/converged | 10; 5.717 s (0.095 min; 0.0016 h); 5.658 / 5.452 / 6.150; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 118.558 s (1.976 min; 0.0329 h); 124.912 / 12.496 / 185.045; missing=0 |
| black | 3/200/0.2/converged | 10; 5.773 s (0.096 min; 0.0016 h); 5.782 / 5.515 / 6.014; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 464.982 s (7.750 min; 0.1292 h); 471.889 / 154.425 / 697.503; missing=0 |
| black | 3/200/0.4/converged | 10; 5.799 s (0.097 min; 0.0016 h); 5.822 / 5.595 / 5.898; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 842.182 s (14.036 min; 0.2339 h); 829.096 / 384.135 / 1235.343; missing=0 |
| black | 3/200/0.7/converged | 10; 6.064 s (0.101 min; 0.0017 h); 6.012 / 5.633 / 6.461; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1192.484 s (19.875 min; 0.3312 h); 1163.479 / 737.138 / 1724.910; missing=0 |
| black | 3/200/1.5/converged | 10; 6.102 s (0.102 min; 0.0017 h); 6.051 / 5.825 / 6.485; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 870.743 s (14.512 min; 0.2419 h); 882.460 / 407.141 / 1457.373; missing=0 |
| black | 3/800/0/converged | 10; 5.934 s (0.099 min; 0.0016 h); 5.896 / 5.632 / 6.438; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 13.321 s (0.222 min; 0.0037 h); 13.331 / 12.980 / 13.754; missing=0 |
| black | 3/800/0.001/converged | 10; 5.803 s (0.097 min; 0.0016 h); 5.807 / 5.644 / 5.958; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 14.185 s (0.236 min; 0.0039 h); 14.321 / 13.706 / 14.387; missing=0 |
| black | 3/800/0.003/converged | 10; 5.918 s (0.099 min; 0.0016 h); 5.870 / 5.806 / 6.119; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 14.238 s (0.237 min; 0.0040 h); 14.222 / 13.984 / 14.557; missing=0 |
| black | 3/800/0.01/converged | 10; 5.835 s (0.097 min; 0.0016 h); 5.782 / 5.631 / 6.119; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 31.526 s (0.525 min; 0.0088 h); 14.703 / 13.932 / 183.387; missing=0 |
| black | 3/800/0.03/converged | 10; 6.238 s (0.104 min; 0.0017 h); 6.257 / 6.023 / 6.391; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 228.980 s (3.816 min; 0.0636 h); 201.581 / 45.637 / 539.285; missing=0 |
| black | 3/800/0.2/converged | 10; 6.594 s (0.110 min; 0.0018 h); 6.528 / 6.286 / 6.926; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 887.238 s (14.787 min; 0.2465 h); 650.437 / 517.019 / 2121.836; missing=0 |
| black | 3/800/0.4/converged | 10; 6.960 s (0.116 min; 0.0019 h); 6.958 / 6.550 / 7.358; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1249.436 s (20.824 min; 0.3471 h); 1231.847 / 783.778 / 2035.286; missing=0 |
| black | 3/800/0.7/converged | 10; 7.154 s (0.119 min; 0.0020 h); 7.153 / 6.823 / 7.468; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1673.823 s (27.897 min; 0.4650 h); 1656.495 / 942.851 / 2396.399; missing=0 |
| black | 3/800/1.5/converged | 10; 7.511 s (0.125 min; 0.0021 h); 7.325 / 6.879 / 8.295; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1188.702 s (19.812 min; 0.3302 h); 1070.947 / 860.080 / 1935.378; missing=0 |
| black | 10/50/0/converged | 10; 6.801 s (0.113 min; 0.0019 h); 6.599 / 6.285 / 8.661; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3.972 s (0.066 min; 0.0011 h); 4.741 / 0.679 / 4.804; missing=0 |
| black | 10/50/0.001/converged | 10; 6.578 s (0.110 min; 0.0018 h); 6.530 / 6.324 / 6.959; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 2.055 s (0.034 min; 0.0006 h); 1.329 / 1.164 / 4.853; missing=0 |
| black | 10/50/0.003/converged | 10; 6.530 s (0.109 min; 0.0018 h); 6.391 / 6.174 / 7.834; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.957 s (0.033 min; 0.0005 h); 1.327 / 1.264 / 7.675; missing=0 |
| black | 10/50/0.01/converged | 10; 6.591 s (0.110 min; 0.0018 h); 6.498 / 6.245 / 8.087; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3.125 s (0.052 min; 0.0009 h); 1.422 / 1.287 / 7.866; missing=0 |
| black | 10/50/0.03/converged | 10; 6.692 s (0.112 min; 0.0019 h); 6.557 / 6.292 / 8.073; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 159.046 s (2.651 min; 0.0442 h); 168.973 / 1.537 / 337.576; missing=0 |
| black | 10/50/0.2/capped | 1; 7.264 s (0.121 min; 0.0020 h); 7.264 / 7.264 / 7.264; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=1 | 1; 4991.206 s (83.187 min; 1.3864 h); 4991.206 / 4991.206 / 4991.206; missing=0 |
| black | 10/50/0.2/converged | 9; 7.023 s (0.117 min; 0.0020 h); 6.883 / 6.428 / 8.595; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=9 | 9; 1511.471 s (25.191 min; 0.4199 h); 1390.100 / 397.041 / 3127.309; missing=0 |
| black | 10/50/0.4/capped | 1; 9.584 s (0.160 min; 0.0027 h); 9.584 / 9.584 / 9.584; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=1 | 1; 5543.831 s (92.397 min; 1.5400 h); 5543.831 / 5543.831 / 5543.831; missing=0 |
| black | 10/50/0.4/converged | 9; 7.083 s (0.118 min; 0.0020 h); 7.028 / 6.603 / 8.528; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=9 | 9; 2950.830 s (49.181 min; 0.8197 h); 3159.563 / 1289.395 / 4714.814; missing=0 |
| black | 10/50/0.7/capped | 5; 7.706 s (0.128 min; 0.0021 h); 7.423 / 7.177 / 8.875; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=5 | 5; 6585.127 s (109.752 min; 1.8292 h); 6579.872 / 6140.359 / 7173.425; missing=0 |
| black | 10/50/0.7/converged | 5; 7.199 s (0.120 min; 0.0020 h); 6.760 / 6.404 / 9.343; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=5 | 5; 2613.093 s (43.552 min; 0.7259 h); 1692.449 / 1616.191 / 4837.555; missing=0 |
| black | 10/50/1.5/capped | 2; 8.012 s (0.134 min; 0.0022 h); 8.012 / 7.101 / 8.924; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=2 | 2; 7988.228 s (133.137 min; 2.2190 h); 7988.228 / 6917.596 / 9058.860; missing=0 |
| black | 10/50/1.5/converged | 8; 7.207 s (0.120 min; 0.0020 h); 7.092 / 6.422 / 9.136; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=8 | 8; 4778.211 s (79.637 min; 1.3273 h); 5144.566 / 2337.023 / 7757.232; missing=0 |
| black | 10/50/2/converged | 10; 7.165 s (0.119 min; 0.0020 h); 6.862 / 6.572 / 8.905; missing=0 | 10; 6969.611 s (116.160 min; 1.9360 h); 6484.285 / 2402.881 / 12226.998; missing=0 | 10; 6543.738 s (109.062 min; 1.8177 h); 6187.017 / 2183.656 / 11801.192; missing=0 |
| black | 10/200/0/converged | 10; 6.573 s (0.110 min; 0.0018 h); 6.615 / 6.330 / 6.756; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 8.152 s (0.136 min; 0.0023 h); 8.279 / 7.475 / 8.502; missing=0 |
| black | 10/200/0.001/converged | 10; 6.651 s (0.111 min; 0.0018 h); 6.617 / 6.382 / 7.075; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 5.388 s (0.090 min; 0.0015 h); 4.926 / 4.786 / 8.584; missing=0 |
| black | 10/200/0.003/converged | 10; 6.621 s (0.110 min; 0.0018 h); 6.586 / 6.240 / 7.428; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 5.049 s (0.084 min; 0.0014 h); 5.031 / 4.889 / 5.265; missing=0 |
| black | 10/200/0.01/converged | 10; 6.632 s (0.111 min; 0.0018 h); 6.635 / 6.271 / 7.057; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 5.063 s (0.084 min; 0.0014 h); 5.059 / 4.780 / 5.447; missing=0 |
| black | 10/200/0.03/converged | 10; 6.700 s (0.112 min; 0.0019 h); 6.735 / 6.280 / 7.006; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 109.196 s (1.820 min; 0.0303 h); 115.155 / 5.213 / 219.601; missing=0 |
| black | 10/200/0.2/converged | 10; 7.131 s (0.119 min; 0.0020 h); 7.184 / 6.598 / 7.509; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1209.696 s (20.162 min; 0.3360 h); 1269.220 / 39.813 / 2106.367; missing=0 |
| black | 10/200/0.4/converged | 10; 7.579 s (0.126 min; 0.0021 h); 7.477 / 6.944 / 8.474; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 2521.171 s (42.020 min; 0.7003 h); 2350.438 / 686.908 / 4490.709; missing=0 |
| black | 10/200/0.7/capped | 1; 9.425 s (0.157 min; 0.0026 h); 9.425 / 9.425 / 9.425; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=1 | 1; 14494.552 s (241.576 min; 4.0263 h); 14494.552 / 14494.552 / 14494.552; missing=0 |
| black | 10/200/0.7/converged | 9; 7.957 s (0.133 min; 0.0022 h); 7.803 / 6.997 / 9.611; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=9 | 9; 4864.340 s (81.072 min; 1.3512 h); 3048.871 / 953.109 / 16120.988; missing=0 |
| black | 10/200/1.5/converged | 10; 8.246 s (0.137 min; 0.0023 h); 8.245 / 7.321 / 9.040; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 7622.766 s (127.046 min; 2.1174 h); 7782.083 / 3515.280 / 10732.828; missing=0 |
| black | 10/200/2/converged | 10; 8.548 s (0.142 min; 0.0024 h); 8.692 / 7.702 / 9.253; missing=0 | 10; 12653.868 s (210.898 min; 3.5150 h); 13935.355 / 4447.698 / 19311.588; missing=0 | 10; 12065.815 s (201.097 min; 3.3516 h); 13156.047 / 4249.361 / 18387.327; missing=0 |
| black | 10/800/0/converged | 10; 6.895 s (0.115 min; 0.0019 h); 6.918 / 6.563 / 7.256; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 22.123 s (0.369 min; 0.0061 h); 22.211 / 21.290 / 22.499; missing=0 |
| black | 10/800/0.001/converged | 10; 6.889 s (0.115 min; 0.0019 h); 6.871 / 6.633 / 7.379; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 19.327 s (0.322 min; 0.0054 h); 19.265 / 18.978 / 19.684; missing=0 |
| black | 10/800/0.003/converged | 10; 6.901 s (0.115 min; 0.0019 h); 6.888 / 6.587 / 7.429; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 19.633 s (0.327 min; 0.0055 h); 19.767 / 18.941 / 19.887; missing=0 |
| black | 10/800/0.01/converged | 10; 7.035 s (0.117 min; 0.0020 h); 6.965 / 6.701 / 7.856; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 20.116 s (0.335 min; 0.0056 h); 20.124 / 19.299 / 20.734; missing=0 |
| black | 10/800/0.03/converged | 10; 7.240 s (0.121 min; 0.0020 h); 7.280 / 6.765 / 7.618; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 186.984 s (3.116 min; 0.0519 h); 136.566 / 20.424 / 534.470; missing=0 |
| black | 10/800/0.2/converged | 10; 9.384 s (0.156 min; 0.0026 h); 9.210 / 8.803 / 10.146; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 2831.778 s (47.196 min; 0.7866 h); 2621.138 / 1275.251 / 5235.814; missing=0 |
| black | 10/800/0.4/converged | 10; 9.403 s (0.157 min; 0.0026 h); 9.663 / 8.033 / 10.185; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 3100.949 s (51.682 min; 0.8614 h); 3146.899 / 993.865 / 5299.639; missing=0 |
| black | 10/800/0.7/converged | 10; 10.249 s (0.171 min; 0.0028 h); 9.896 / 9.066 / 12.784; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 5281.435 s (88.024 min; 1.4671 h); 4616.287 / 2401.682 / 10961.382; missing=0 |
| black | 10/800/1.5/converged | 10; 12.995 s (0.217 min; 0.0036 h); 12.261 / 10.072 / 15.996; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 14424.292 s (240.405 min; 4.0067 h); 12568.138 / 5634.449 / 36162.905; missing=0 |
| black | 10/800/2/converged | 10; 14.154 s (0.236 min; 0.0039 h); 14.493 / 11.279 / 16.061; missing=0 | 10; 20486.585 s (341.443 min; 5.6907 h); 20683.874 / 6850.297 / 33628.073; missing=0 | 10; 19795.937 s (329.932 min; 5.4989 h); 19843.422 / 6586.371 / 32566.360; missing=0 |
| green | 3/50/0/converged | 10; 5.531 s (0.092 min; 0.0015 h); 5.480 / 5.305 / 5.911; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.909 s (0.032 min; 0.0005 h); 1.829 / 1.583 / 2.448; missing=0 |
| green | 3/50/0.001/converged | 10; 5.555 s (0.093 min; 0.0015 h); 5.434 / 5.274 / 6.504; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.849 s (0.031 min; 0.0005 h); 1.695 / 1.491 / 2.470; missing=0 |
| green | 3/50/0.003/converged | 10; 5.595 s (0.093 min; 0.0016 h); 5.581 / 5.364 / 5.968; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.863 s (0.031 min; 0.0005 h); 1.901 / 1.404 / 2.368; missing=0 |
| green | 3/50/0.01/converged | 10; 5.507 s (0.092 min; 0.0015 h); 5.479 / 5.376 / 5.715; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.764 s (0.029 min; 0.0005 h); 1.723 / 1.508 / 2.329; missing=0 |
| green | 3/50/0.03/converged | 10; 5.502 s (0.092 min; 0.0015 h); 5.481 / 5.305 / 5.671; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 1.834 s (0.031 min; 0.0005 h); 1.789 / 1.571 / 2.265; missing=0 |
| green | 3/50/0.2/converged | 10; 5.596 s (0.093 min; 0.0016 h); 5.551 / 5.271 / 6.282; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 160.808 s (2.680 min; 0.0447 h); 146.594 / 134.004 / 243.524; missing=0 |
| green | 3/50/0.4/converged | 10; 5.527 s (0.092 min; 0.0015 h); 5.449 / 5.291 / 6.214; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 392.600 s (6.543 min; 0.1091 h); 302.388 / 206.841 / 858.822; missing=0 |
| green | 3/50/0.7/converged | 10; 5.609 s (0.093 min; 0.0016 h); 5.558 / 5.369 / 6.030; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 604.863 s (10.081 min; 0.1680 h); 487.570 / 329.838 / 1339.552; missing=0 |
| green | 3/50/1.5/converged | 10; 5.543 s (0.092 min; 0.0015 h); 5.499 / 5.119 / 6.082; missing=0 | 0; unavailable; unavailable / unavailable / unavailable; missing=10 | 10; 837.705 s (13.962 min; 0.2327 h); 489.934 / 321.743 / 2924.820; missing=0 |
| green | 3/200/0/converged | 10; 5.548 s (0.092 min; 0.0015 h); 5.542 / 5.237 / 5.972; missing=0 | 10; 4.101 s (0.068 min; 0.0011 h); 4.059 / 3.938 / 4.386; missing=0 | 10; 3.436 s (0.057 min; 0.0010 h); 3.392 / 3.285 / 3.743; missing=0 |
| green | 3/200/0.001/converged | 10; 5.610 s (0.094 min; 0.0016 h); 5.618 / 5.375 / 5.785; missing=0 | 10; 4.296 s (0.072 min; 0.0012 h); 4.271 / 4.113 / 4.522; missing=0 | 10; 3.588 s (0.060 min; 0.0010 h); 3.568 / 3.477 / 3.794; missing=0 |
| green | 3/200/0.003/converged | 10; 5.584 s (0.093 min; 0.0016 h); 5.578 / 5.432 / 5.761; missing=0 | 10; 4.341 s (0.072 min; 0.0012 h); 4.363 / 4.151 / 4.507; missing=0 | 10; 3.646 s (0.061 min; 0.0010 h); 3.645 / 3.453 / 3.850; missing=0 |
| green | 3/200/0.01/converged | 10; 5.629 s (0.094 min; 0.0016 h); 5.627 / 5.401 / 5.905; missing=0 | 10; 4.387 s (0.073 min; 0.0012 h); 4.394 / 4.229 / 4.542; missing=0 | 10; 3.719 s (0.062 min; 0.0010 h); 3.706 / 3.600 / 3.836; missing=0 |
| green | 3/200/0.03/converged | 10; 5.524 s (0.092 min; 0.0015 h); 5.575 / 5.223 / 5.676; missing=0 | 10; 4.451 s (0.074 min; 0.0012 h); 4.451 / 4.150 / 4.686; missing=0 | 10; 3.807 s (0.063 min; 0.0011 h); 3.821 / 3.516 / 4.002; missing=0 |
| green | 3/200/0.2/converged | 10; 5.590 s (0.093 min; 0.0016 h); 5.601 / 5.428 / 5.738; missing=0 | 10; 292.790 s (4.880 min; 0.0813 h); 285.969 / 143.384 / 450.227; missing=0 | 10; 279.131 s (4.652 min; 0.0775 h); 273.213 / 138.319 / 427.669; missing=0 |
| green | 3/200/0.4/converged | 10; 5.737 s (0.096 min; 0.0016 h); 5.733 / 5.513 / 5.990; missing=0 | 10; 434.954 s (7.249 min; 0.1208 h); 450.474 / 242.685 / 581.982; missing=0 | 10; 411.235 s (6.854 min; 0.1142 h); 426.978 / 231.244 / 544.887; missing=0 |
| green | 3/200/0.7/converged | 10; 5.883 s (0.098 min; 0.0016 h); 5.825 / 5.568 / 6.393; missing=0 | 10; 600.519 s (10.009 min; 0.1668 h); 577.834 / 423.976 / 787.155; missing=0 | 10; 559.013 s (9.317 min; 0.1553 h); 540.737 / 402.431 / 722.550; missing=0 |
| green | 3/200/1.5/converged | 10; 5.892 s (0.098 min; 0.0016 h); 5.861 / 5.581 / 6.189; missing=0 | 10; 456.573 s (7.610 min; 0.1268 h); 450.935 / 261.476 / 647.909; missing=0 | 10; 417.379 s (6.956 min; 0.1159 h); 411.730 / 242.683 / 584.420; missing=0 |
| green | 3/800/0/converged | 10; 5.907 s (0.098 min; 0.0016 h); 5.818 / 5.728 / 6.265; missing=0 | 10; 15.940 s (0.266 min; 0.0044 h); 15.844 / 15.596 / 16.609; missing=0 | 10; 13.321 s (0.222 min; 0.0037 h); 13.331 / 12.980 / 13.754; missing=0 |
| green | 3/800/0.001/converged | 10; 6.026 s (0.100 min; 0.0017 h); 5.982 / 5.829 / 6.274; missing=0 | 10; 16.767 s (0.279 min; 0.0047 h); 16.820 / 16.281 / 17.045; missing=0 | 10; 14.185 s (0.236 min; 0.0039 h); 14.321 / 13.706 / 14.387; missing=0 |
| green | 3/800/0.003/converged | 10; 5.841 s (0.097 min; 0.0016 h); 5.824 / 5.668 / 6.221; missing=0 | 10; 16.827 s (0.280 min; 0.0047 h); 16.834 / 16.481 / 17.137; missing=0 | 10; 14.238 s (0.237 min; 0.0040 h); 14.222 / 13.984 / 14.557; missing=0 |
| green | 3/800/0.01/converged | 10; 5.924 s (0.099 min; 0.0016 h); 5.914 / 5.692 / 6.128; missing=0 | 10; 17.271 s (0.288 min; 0.0048 h); 17.186 / 16.349 / 18.343; missing=0 | 10; 14.744 s (0.246 min; 0.0041 h); 14.703 / 13.932 / 15.571; missing=0 |
| green | 3/800/0.03/converged | 10; 5.861 s (0.098 min; 0.0016 h); 5.821 / 5.734 / 6.224; missing=0 | 10; 18.254 s (0.304 min; 0.0051 h); 18.206 / 17.386 / 19.091; missing=0 | 10; 15.692 s (0.262 min; 0.0044 h); 15.637 / 14.847 / 16.343; missing=0 |
| green | 3/800/0.2/converged | 10; 6.203 s (0.103 min; 0.0017 h); 6.155 / 5.966 / 6.484; missing=0 | 10; 281.614 s (4.694 min; 0.0782 h); 262.924 / 234.157 / 484.015; missing=0 | 10; 270.847 s (4.514 min; 0.0752 h); 253.221 / 224.946 / 463.035; missing=0 |
| green | 3/800/0.4/converged | 10; 6.425 s (0.107 min; 0.0018 h); 6.415 / 6.183 / 6.746; missing=0 | 10; 528.981 s (8.816 min; 0.1469 h); 518.489 / 344.702 / 775.867; missing=0 | 10; 503.917 s (8.399 min; 0.1400 h); 496.082 / 325.732 / 740.106; missing=0 |
| green | 3/800/0.7/converged | 10; 6.649 s (0.111 min; 0.0018 h); 6.597 / 6.314 / 7.005; missing=0 | 10; 670.410 s (11.174 min; 0.1862 h); 719.415 / 454.591 / 801.347; missing=0 | 10; 637.913 s (10.632 min; 0.1772 h); 680.613 / 433.332 / 765.098; missing=0 |
| green | 3/800/1.5/converged | 10; 6.850 s (0.114 min; 0.0019 h); 6.848 / 6.472 / 7.428; missing=0 | 10; 662.394 s (11.040 min; 0.1840 h); 680.662 / 517.209 / 769.838; missing=0 | 10; 622.588 s (10.376 min; 0.1729 h); 635.171 / 487.356 / 726.122; missing=0 |
| green | 10/50/0/converged | 10; 6.616 s (0.110 min; 0.0018 h); 6.429 / 6.180 / 8.102; missing=0 | 10; 4.242 s (0.071 min; 0.0012 h); 5.020 / 0.873 / 5.111; missing=0 | 10; 3.972 s (0.066 min; 0.0011 h); 4.741 / 0.679 / 4.804; missing=0 |
| green | 10/50/0.001/converged | 10; 6.530 s (0.109 min; 0.0018 h); 6.461 / 6.184 / 7.717; missing=0 | 10; 3.469 s (0.058 min; 0.0010 h); 2.429 / 2.271 / 5.215; missing=0 | 10; 2.055 s (0.034 min; 0.0006 h); 1.329 / 1.164 / 4.853; missing=0 |
| green | 10/50/0.003/converged | 10; 6.433 s (0.107 min; 0.0018 h); 6.456 / 6.229 / 6.586; missing=0 | 10; 1.851 s (0.031 min; 0.0005 h); 1.644 / 1.536 / 2.458; missing=0 | 10; 1.316 s (0.022 min; 0.0004 h); 1.315 / 1.264 / 1.403; missing=0 |
| green | 10/50/0.01/converged | 10; 6.718 s (0.112 min; 0.0019 h); 6.505 / 6.383 / 8.758; missing=0 | 10; 1.698 s (0.028 min; 0.0005 h); 1.690 / 1.609 / 1.802; missing=0 | 10; 1.391 s (0.023 min; 0.0004 h); 1.374 / 1.287 / 1.516; missing=0 |
| green | 10/50/0.03/converged | 10; 6.691 s (0.112 min; 0.0019 h); 6.555 / 6.294 / 8.230; missing=0 | 10; 35.495 s (0.592 min; 0.0099 h); 1.890 / 1.750 / 229.094; missing=0 | 10; 33.427 s (0.557 min; 0.0093 h); 1.603 / 1.464 / 219.110; missing=0 |
| green | 10/50/0.2/converged | 10; 6.730 s (0.112 min; 0.0019 h); 6.545 / 6.275 / 8.220; missing=0 | 10; 924.420 s (15.407 min; 0.2568 h); 945.767 / 426.801 / 1325.076; missing=0 | 10; 839.820 s (13.997 min; 0.2333 h); 858.810 / 397.041 / 1186.953; missing=0 |
| green | 10/50/0.4/converged | 10; 7.065 s (0.118 min; 0.0020 h); 6.891 / 6.540 / 8.810; missing=0 | 10; 1901.607 s (31.693 min; 0.5282 h); 2010.812 / 837.817 / 3370.829; missing=0 | 10; 1648.835 s (27.481 min; 0.4580 h); 1778.091 / 773.116 / 2736.098; missing=0 |
| green | 10/50/0.7/converged | 10; 7.196 s (0.120 min; 0.0020 h); 6.833 / 6.500 / 9.353; missing=0 | 10; 2652.544 s (44.209 min; 0.7368 h); 2201.673 / 1342.637 / 5096.248; missing=0 | 10; 2253.314 s (37.555 min; 0.6259 h); 1871.665 / 1228.456 / 4243.046; missing=0 |
| green | 10/50/1.5/converged | 10; 7.159 s (0.119 min; 0.0020 h); 6.770 / 6.434 / 9.144; missing=0 | 10; 3571.289 s (59.521 min; 0.9920 h); 3839.900 / 1084.061 / 5106.144; missing=0 | 10; 3335.674 s (55.595 min; 0.9266 h); 3512.693 / 979.163 / 4865.911; missing=0 |
| green | 10/200/0/converged | 10; 6.628 s (0.110 min; 0.0018 h); 6.583 / 6.451 / 6.997; missing=0 | 10; 9.192 s (0.153 min; 0.0026 h); 9.276 / 8.438 / 9.653; missing=0 | 10; 8.152 s (0.136 min; 0.0023 h); 8.279 / 7.475 / 8.502; missing=0 |
| green | 10/200/0.001/converged | 10; 6.547 s (0.109 min; 0.0018 h); 6.556 / 6.331 / 6.772; missing=0 | 10; 7.828 s (0.130 min; 0.0022 h); 6.838 / 6.649 / 9.568; missing=0 | 10; 5.388 s (0.090 min; 0.0015 h); 4.926 / 4.786 / 8.584; missing=0 |
| green | 10/200/0.003/converged | 10; 6.542 s (0.109 min; 0.0018 h); 6.510 / 6.279 / 7.108; missing=0 | 10; 6.265 s (0.104 min; 0.0017 h); 6.135 / 5.857 / 6.768; missing=0 | 10; 5.049 s (0.084 min; 0.0014 h); 5.031 / 4.889 / 5.265; missing=0 |
| green | 10/200/0.01/converged | 10; 6.597 s (0.110 min; 0.0018 h); 6.542 / 6.277 / 6.972; missing=0 | 10; 5.959 s (0.099 min; 0.0017 h); 5.949 / 5.633 / 6.374; missing=0 | 10; 5.063 s (0.084 min; 0.0014 h); 5.059 / 4.780 / 5.447; missing=0 |
| green | 10/200/0.03/converged | 10; 6.483 s (0.108 min; 0.0018 h); 6.447 / 6.270 / 6.753; missing=0 | 10; 6.385 s (0.106 min; 0.0018 h); 6.354 / 6.056 / 6.800; missing=0 | 10; 5.469 s (0.091 min; 0.0015 h); 5.430 / 5.213 / 5.920; missing=0 |
| green | 10/200/0.2/converged | 10; 6.820 s (0.114 min; 0.0019 h); 6.791 / 6.470 / 7.246; missing=0 | 10; 403.812 s (6.730 min; 0.1122 h); 416.938 / 42.839 / 639.684; missing=0 | 10; 388.959 s (6.483 min; 0.1080 h); 402.747 / 39.813 / 613.969; missing=0 |
| green | 10/200/0.4/converged | 10; 7.005 s (0.117 min; 0.0019 h); 7.019 / 6.566 / 7.496; missing=0 | 10; 895.840 s (14.931 min; 0.2488 h); 950.674 / 687.085 / 1314.161; missing=0 | 10; 857.490 s (14.291 min; 0.2382 h); 908.721 / 660.500 / 1250.074; missing=0 |
| green | 10/200/0.7/converged | 10; 7.249 s (0.121 min; 0.0020 h); 7.304 / 6.729 / 7.738; missing=0 | 10; 1694.136 s (28.236 min; 0.4706 h); 1570.348 / 966.665 / 2704.543; missing=0 | 10; 1610.410 s (26.840 min; 0.4473 h); 1500.691 / 926.168 / 2557.496; missing=0 |
| green | 10/200/1.5/converged | 10; 7.435 s (0.124 min; 0.0021 h); 7.477 / 6.773 / 7.848; missing=0 | 10; 4841.734 s (80.696 min; 1.3449 h); 4483.177 / 2449.988 / 6557.406; missing=0 | 10; 4706.680 s (78.445 min; 1.3074 h); 4314.468 / 2332.092 / 6453.173; missing=0 |
| green | 10/800/0/converged | 10; 7.036 s (0.117 min; 0.0020 h); 6.960 / 6.786 / 7.401; missing=0 | 10; 25.792 s (0.430 min; 0.0072 h); 25.889 / 24.691 / 26.297; missing=0 | 10; 22.123 s (0.369 min; 0.0061 h); 22.211 / 21.290 / 22.499; missing=0 |
| green | 10/800/0.001/converged | 10; 6.928 s (0.115 min; 0.0019 h); 6.920 / 6.641 / 7.325; missing=0 | 10; 24.568 s (0.409 min; 0.0068 h); 23.760 / 23.515 / 26.732; missing=0 | 10; 19.327 s (0.322 min; 0.0054 h); 19.265 / 18.978 / 19.684; missing=0 |
| green | 10/800/0.003/converged | 10; 6.911 s (0.115 min; 0.0019 h); 6.886 / 6.771 / 7.125; missing=0 | 10; 23.574 s (0.393 min; 0.0065 h); 23.497 / 22.978 / 24.269; missing=0 | 10; 19.633 s (0.327 min; 0.0055 h); 19.767 / 18.941 / 19.887; missing=0 |
| green | 10/800/0.01/converged | 10; 6.993 s (0.117 min; 0.0019 h); 6.925 / 6.661 / 7.846; missing=0 | 10; 23.737 s (0.396 min; 0.0066 h); 23.707 / 22.840 / 24.422; missing=0 | 10; 20.116 s (0.335 min; 0.0056 h); 20.124 / 19.299 / 20.734; missing=0 |
| green | 10/800/0.03/converged | 10; 6.885 s (0.115 min; 0.0019 h); 6.850 / 6.652 / 7.292; missing=0 | 10; 25.411 s (0.424 min; 0.0071 h); 25.088 / 23.961 / 26.757; missing=0 | 10; 21.868 s (0.364 min; 0.0061 h); 21.530 / 20.424 / 23.217; missing=0 |
| green | 10/800/0.2/converged | 10; 7.529 s (0.125 min; 0.0021 h); 7.436 / 7.021 / 8.067; missing=0 | 10; 584.322 s (9.739 min; 0.1623 h); 454.996 / 417.594 / 943.511; missing=0 | 10; 566.614 s (9.444 min; 0.1574 h); 441.963 / 405.190 / 914.163; missing=0 |
| green | 10/800/0.4/converged | 10; 7.901 s (0.132 min; 0.0022 h); 8.000 / 7.450 / 8.489; missing=0 | 10; 1361.164 s (22.686 min; 0.3781 h); 1505.529 / 946.989 / 1711.746; missing=0 | 10; 1319.355 s (21.989 min; 0.3665 h); 1456.095 / 917.982 / 1660.489; missing=0 |
| green | 10/800/0.7/converged | 10; 8.707 s (0.145 min; 0.0024 h); 8.606 / 7.765 / 9.961; missing=0 | 10; 2774.969 s (46.249 min; 0.7708 h); 2504.321 / 1626.347 / 4868.850; missing=0 | 10; 2696.676 s (44.945 min; 0.7491 h); 2431.870 / 1577.250 / 4739.869; missing=0 |
| green | 10/800/1.5/converged | 10; 10.097 s (0.168 min; 0.0028 h); 10.017 / 8.986 / 11.389; missing=0 | 10; 7990.363 s (133.173 min; 2.2195 h); 6882.736 / 3895.838 / 18468.406; missing=0 | 10; 7833.878 s (130.565 min; 2.1761 h); 6773.941 / 3767.814 / 18300.106; missing=0 |
| SRO-FA | historical | unavailable | unavailable; rerun required | unavailable |
| Validation/CV | no new selection | not run | historical unavailable | — |

No new training was performed. Nominal selected-arm timings are counted once per R/N/seed, not per radius. Unmeasured nominal evaluation and historical CV are not zero.
Whole-pipeline elapsed time: pending parent completion (not a sum of policy wall times).

Parent elapsed through rendering/report creation: 857.957477 s / 14.299291 min / 0.238322 h.
Authoritative parent phases and sweep elapsed: ../pipeline_summary.json. The sealed runtime_report.json whole_pipeline_elapsed_s remains null intentionally. Full process time INCLUDING imports and final serialization/publication is in the sibling launch directory child_runtime.json; unavailable until that wrapper exits. No sum of concurrent worker times is an elapsed sweep time.
