# BLACK fixed-policy comparison

Fresh replay only; warm-started historical continuation is not training from scratch. All 570 policies and all 11 capped/nonconverged training policies are retained. The historically known 10,000-row seed9999 tests are reused, not a newly untouched holdout. No OOS-based checkpoint/radius selection or interpolation is performed.

Own-pool IS is a mean of per-center maxima, not an adversarial supremum. Original green/SCS/SRO-FA/SAA-kNN numbers, grids, gray references and caveats remain unchanged. R10 black alone extends to radius 2; other curves are not extrapolated.

## Fixed common radii (cost differences, not selected winners)

| R | N | Radius | Black OOS | Green OOS | Black − green | Black own-pool IS | Green own-pool IS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 50 | 0 | 20.591139 | 20.591139 | 0 | 19.365727 | 19.365727 |
| 3 | 50 | 0.001 | 20.591139 | 20.591139 | 0 | 19.365727 | 19.365727 |
| 3 | 50 | 0.003 | 20.591139 | 20.591139 | 0 | 19.365727 | 19.365727 |
| 3 | 50 | 0.01 | 20.591139 | 20.591139 | 0 | 19.365727 | 19.365727 |
| 3 | 50 | 0.03 | 20.636376 | 20.591139 | 0.045236437 | 19.313699 | 19.365727 |
| 3 | 50 | 0.2 | 20.476167 | 20.63774 | -0.16157321 | 19.617618 | 19.647229 |
| 3 | 50 | 0.4 | 20.470175 | 20.529497 | -0.059322412 | 20.021869 | 20.022256 |
| 3 | 50 | 0.7 | 20.653866 | 20.733442 | -0.079575637 | 20.782968 | 20.924093 |
| 3 | 50 | 1.5 | 21.375432 | 21.484426 | -0.10899368 | 23.28584 | 23.401688 |
| 3 | 200 | 0 | 19.947783 | 19.947783 | 0 | 19.413979 | 19.413979 |
| 3 | 200 | 0.001 | 19.947783 | 19.947783 | 0 | 19.413979 | 19.413979 |
| 3 | 200 | 0.003 | 19.947783 | 19.947783 | 0 | 19.413979 | 19.413979 |
| 3 | 200 | 0.01 | 19.947783 | 19.947783 | 0 | 19.413979 | 19.413979 |
| 3 | 200 | 0.03 | 19.917962 | 19.947783 | -0.029821021 | 19.432164 | 19.413979 |
| 3 | 200 | 0.2 | 19.867081 | 19.888235 | -0.021154055 | 19.832422 | 19.820836 |
| 3 | 200 | 0.4 | 19.850229 | 19.873891 | -0.023662886 | 20.395958 | 20.366686 |
| 3 | 200 | 0.7 | 20.003549 | 20.060598 | -0.057049241 | 21.347035 | 21.357545 |
| 3 | 200 | 1.5 | 21.338038 | 21.345981 | -0.0079433427 | 23.396817 | 23.383316 |
| 3 | 800 | 0 | 19.820354 | 19.820354 | 0 | 19.5525 | 19.5525 |
| 3 | 800 | 0.001 | 19.820354 | 19.820354 | 0 | 19.5525 | 19.5525 |
| 3 | 800 | 0.003 | 19.820354 | 19.820354 | 0 | 19.5525 | 19.5525 |
| 3 | 800 | 0.01 | 19.811819 | 19.820354 | -0.0085345251 | 19.548596 | 19.5525 |
| 3 | 800 | 0.03 | 19.794715 | 19.820354 | -0.025638304 | 19.582247 | 19.5525 |
| 3 | 800 | 0.2 | 19.719183 | 19.81902 | -0.099836795 | 20.0018 | 19.982731 |
| 3 | 800 | 0.4 | 19.640055 | 19.753921 | -0.11386548 | 20.484615 | 20.545928 |
| 3 | 800 | 0.7 | 19.868154 | 19.946312 | -0.078157961 | 21.542033 | 21.496836 |
| 3 | 800 | 1.5 | 21.342645 | 21.291767 | 0.050877781 | 23.459898 | 23.40623 |
| 10 | 50 | 0 | 78.234062 | 78.234062 | 0 | 62.705004 | 62.705004 |
| 10 | 50 | 0.001 | 78.234062 | 78.234062 | 0 | 62.705004 | 62.705004 |
| 10 | 50 | 0.003 | 78.234062 | 78.234062 | 0 | 62.705004 | 62.705004 |
| 10 | 50 | 0.01 | 78.234062 | 78.234062 | 0 | 62.705004 | 62.705004 |
| 10 | 50 | 0.03 | 78.37654 | 78.309461 | 0.067079048 | 61.27238 | 62.559681 |
| 10 | 50 | 0.2 | 76.572047 | 76.498006 | 0.074041064 | 62.680151 | 63.137297 |
| 10 | 50 | 0.4 | 74.528147 | 74.440684 | 0.087462999 | 65.877278 | 66.129707 |
| 10 | 50 | 0.7 | 73.302378 | 73.233594 | 0.068784508 | 69.142497 | 69.337228 |
| 10 | 50 | 1.5 | 72.631189 | 72.669066 | -0.037877099 | 76.545974 | 76.50669 |
| 10 | 200 | 0 | 71.632591 | 71.632591 | 0 | 68.579003 | 68.579003 |
| 10 | 200 | 0.001 | 71.632591 | 71.632591 | 0 | 68.579003 | 68.579003 |
| 10 | 200 | 0.003 | 71.632591 | 71.632591 | 0 | 68.579003 | 68.579003 |
| 10 | 200 | 0.01 | 71.632591 | 71.632591 | 0 | 68.579003 | 68.579003 |
| 10 | 200 | 0.03 | 70.732902 | 71.632591 | -0.8996889 | 67.815073 | 68.579003 |
| 10 | 200 | 0.2 | 69.656062 | 69.990449 | -0.33438771 | 68.201851 | 68.416895 |
| 10 | 200 | 0.4 | 69.101868 | 69.537462 | -0.4355941 | 70.05988 | 70.193437 |
| 10 | 200 | 0.7 | 68.73768 | 69.470363 | -0.73268283 | 72.34846 | 72.648131 |
| 10 | 200 | 1.5 | 71.526949 | 71.378079 | 0.1488703 | 78.829869 | 78.587107 |
| 10 | 800 | 0 | 70.268129 | 70.268129 | 0 | 69.290098 | 69.290098 |
| 10 | 800 | 0.001 | 70.268129 | 70.268129 | 0 | 69.290098 | 69.290098 |
| 10 | 800 | 0.003 | 70.268129 | 70.268129 | 0 | 69.290098 | 69.290098 |
| 10 | 800 | 0.01 | 70.268129 | 70.268129 | 0 | 69.290098 | 69.290098 |
| 10 | 800 | 0.03 | 69.752182 | 70.268129 | -0.51594657 | 68.881541 | 69.290098 |
| 10 | 800 | 0.2 | 68.097071 | 69.110382 | -1.0133112 | 68.915041 | 69.477733 |
| 10 | 800 | 0.4 | 68.066875 | 68.419964 | -0.3530888 | 70.644644 | 70.668885 |
| 10 | 800 | 0.7 | 68.342644 | 68.506557 | -0.16391379 | 73.248393 | 73.119255 |
| 10 | 800 | 1.5 | 71.594107 | 71.416299 | 0.17780851 | 79.109963 | 78.810218 |

## Current replay wall times (completed, journal-verified models only)

| R | N | Radius | Runs | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 50 | 0 | 10 | 20.938356 | 20.850049 | 20.510339 | 21.661383 |
| 3 | 50 | 0.001 | 10 | 21.008605 | 21.002285 | 20.418432 | 21.756055 |
| 3 | 50 | 0.003 | 10 | 21.013422 | 20.951467 | 20.479712 | 21.55181 |
| 3 | 50 | 0.01 | 10 | 20.948291 | 20.921947 | 20.45371 | 21.673914 |
| 3 | 50 | 0.03 | 10 | 21.00888 | 20.889764 | 20.525562 | 22.396043 |
| 3 | 50 | 0.2 | 10 | 21.134186 | 21.030708 | 20.732479 | 21.960164 |
| 3 | 50 | 0.4 | 10 | 21.382994 | 21.285295 | 20.99438 | 22.128225 |
| 3 | 50 | 0.7 | 10 | 22.16195 | 22.077078 | 20.954329 | 22.827454 |
| 3 | 50 | 1.5 | 10 | 21.960123 | 22.103512 | 20.797971 | 23.079939 |
| 3 | 200 | 0 | 10 | 21.153454 | 21.190346 | 20.788668 | 21.509448 |
| 3 | 200 | 0.001 | 10 | 21.11661 | 21.017492 | 20.764333 | 21.65142 |
| 3 | 200 | 0.003 | 10 | 21.07331 | 21.001007 | 20.563695 | 22.090548 |
| 3 | 200 | 0.01 | 10 | 21.100461 | 20.847479 | 20.570588 | 22.68752 |
| 3 | 200 | 0.03 | 10 | 21.117929 | 21.103214 | 20.869207 | 21.583967 |
| 3 | 200 | 0.2 | 10 | 21.469009 | 21.464992 | 20.939785 | 22.020106 |
| 3 | 200 | 0.4 | 10 | 21.429801 | 21.449813 | 20.888874 | 21.978782 |
| 3 | 200 | 0.7 | 10 | 22.191366 | 22.112897 | 21.585698 | 23.183526 |
| 3 | 200 | 1.5 | 10 | 21.881567 | 21.897435 | 21.239553 | 22.337141 |
| 3 | 800 | 0 | 10 | 21.45686 | 21.494788 | 20.727861 | 22.115151 |
| 3 | 800 | 0.001 | 10 | 21.245328 | 21.260272 | 20.828221 | 21.861527 |
| 3 | 800 | 0.003 | 10 | 21.160664 | 21.170717 | 20.773451 | 22.139443 |
| 3 | 800 | 0.01 | 10 | 21.600183 | 21.545282 | 20.915387 | 22.277966 |
| 3 | 800 | 0.03 | 10 | 21.567251 | 21.518925 | 20.890151 | 22.682576 |
| 3 | 800 | 0.2 | 10 | 22.261931 | 22.07338 | 21.779474 | 23.581919 |
| 3 | 800 | 0.4 | 10 | 22.585296 | 22.482906 | 22.030774 | 23.439828 |
| 3 | 800 | 0.7 | 10 | 23.136454 | 22.954464 | 22.620253 | 24.454639 |
| 3 | 800 | 1.5 | 10 | 23.008383 | 22.901099 | 22.501284 | 23.95099 |
| 10 | 50 | 0 | 10 | 30.86721 | 30.743681 | 30.169918 | 31.812 |
| 10 | 50 | 0.001 | 10 | 30.574969 | 30.418879 | 29.883552 | 31.973817 |
| 10 | 50 | 0.003 | 10 | 30.230079 | 30.233944 | 29.821076 | 30.861207 |
| 10 | 50 | 0.01 | 10 | 30.538036 | 30.409973 | 29.906991 | 31.994626 |
| 10 | 50 | 0.03 | 10 | 30.409811 | 30.264916 | 30.123939 | 31.256609 |
| 10 | 50 | 0.2 | 10 | 31.467669 | 30.816676 | 30.134449 | 33.546834 |
| 10 | 50 | 0.4 | 10 | 32.296921 | 32.393699 | 30.708984 | 34.136602 |
| 10 | 50 | 0.7 | 10 | 32.700094 | 33.006075 | 30.790544 | 34.769855 |
| 10 | 50 | 1.5 | 10 | 32.437685 | 32.662032 | 31.246535 | 33.871055 |
| 10 | 50 | 2 | 10 | 31.82174 | 31.438214 | 30.889991 | 33.518256 |
| 10 | 200 | 0 | 10 | 30.434171 | 30.367751 | 29.666973 | 31.985614 |
| 10 | 200 | 0.001 | 10 | 30.268057 | 30.173546 | 29.543879 | 31.247099 |
| 10 | 200 | 0.003 | 10 | 30.49247 | 30.286753 | 29.794619 | 31.42569 |
| 10 | 200 | 0.01 | 10 | 30.420825 | 30.360009 | 29.888459 | 31.021083 |
| 10 | 200 | 0.03 | 10 | 30.29219 | 30.160824 | 29.958844 | 30.882785 |
| 10 | 200 | 0.2 | 10 | 31.810715 | 31.771611 | 30.031397 | 34.581291 |
| 10 | 200 | 0.4 | 10 | 31.992483 | 32.078944 | 30.579183 | 33.523177 |
| 10 | 200 | 0.7 | 10 | 33.123443 | 32.372974 | 30.916295 | 36.532671 |
| 10 | 200 | 1.5 | 10 | 33.655075 | 33.556862 | 31.81907 | 36.106846 |
| 10 | 200 | 2 | 10 | 33.646975 | 34.025785 | 31.691817 | 35.026817 |
| 10 | 800 | 0 | 10 | 30.748379 | 30.594661 | 30.307383 | 31.744417 |
| 10 | 800 | 0.001 | 10 | 31.07046 | 31.018397 | 30.378147 | 31.858779 |
| 10 | 800 | 0.003 | 10 | 30.697206 | 30.551888 | 30.116525 | 31.898193 |
| 10 | 800 | 0.01 | 10 | 30.713615 | 30.549585 | 30.288128 | 31.590794 |
| 10 | 800 | 0.03 | 10 | 30.836698 | 30.890783 | 30.265444 | 31.45448 |
| 10 | 800 | 0.2 | 10 | 33.395301 | 33.203313 | 32.503083 | 34.600751 |
| 10 | 800 | 0.4 | 10 | 33.735874 | 33.53983 | 31.57446 | 36.217788 |
| 10 | 800 | 0.7 | 10 | 34.195227 | 33.736975 | 32.688426 | 36.963686 |
| 10 | 800 | 1.5 | 10 | 37.936526 | 36.862531 | 34.971668 | 41.596184 |
| 10 | 800 | 2 | 10 | 39.101251 | 40.356643 | 34.44916 | 42.554126 |

## Persisted nominal training (once per R/N/seed, not repeated per radius)

| R | N | Seeds | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 50 | 10 | 37.620406 | 37.241888 | 29.626559 | 45.87712 |
| 3 | 200 | 10 | 149.80154 | 149.80706 | 149.04216 | 150.49819 |
| 3 | 800 | 10 | 170.06978 | 171.98813 | 157.94017 | 174.61177 |
| 10 | 50 | 10 | 113.64844 | 125.83173 | 83.229825 | 129.86619 |
| 10 | 200 | 10 | 161.9686 | 169.70667 | 96.384184 | 170.89157 |
| 10 | 800 | 10 | 198.55716 | 215.96808 | 121.08424 | 216.45162 |

| Runtime category | Evidence / time |
|---|---|
| Nominal neural training | No new training; persisted per-seed evidence in runtime_report.json; missing fields unavailable, rerun required |
| Robust neural/CCG | No new AP/MP; persisted per-iteration AP/MP and nominal initialization kept separately |
| Historical BLACK continuation whole invocation (nonconverged) | 62823.446786 s / 1047.057446 min / 17.450957 h; 16 workers, one solver/Torch thread each; not a completed-model aggregate or current timing |
| SRO-FA partition / optimization / OOS / total | No new run; original base provenance unchanged; missing historical fields unavailable, rerun required |
| Hyperparameter selection / CV / benchmark construction | No new run; cached original evidence only |
| Final neural test and own-pool evaluation | Per-model table; separate oos/is targets, projection and cost phases in raw results |
| Current preflight_validation (completed) | 151.645607 s / 2.527427 min / 0.042124 h |
| Current resources (completed) | 0.033046 s / 0.000551 min / 0.000009 h |
| Current test_and_base_binding (completed) | 6.149330 s / 0.102489 min / 0.001708 h |
| Current final_manifest_seal (completed) | 5.787089 s / 0.096451 min / 0.001608 h |
| Current neural_cohorts (completed) | 1083.598741 s / 18.059979 min / 0.301000 h |
| Current journal_verification (completed) | 114.748420 s / 1.912474 min / 0.031875 h |
| Current render (completed) | 42.068550 s / 0.701142 min / 0.011686 h |

Preflight unit/probe time is not model training or production evaluation time. Concurrent model times are not added to obtain sweep elapsed. The launcher records whole-child wall time including final serialization; raw model totals exclude final result serialization. Failed, interrupted and unverified jobs are excluded from completed aggregates.

Concurrency: 16 joint subprocess slots, one numerical thread per process. Environment: {"affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], "captured_at_utc": "2026-09-28T08:56:32.578798+00:00", "cgroup": "12:cpuset:/\n11:memory:/user.slice/user-1754.slice/session-1404.scope\n10:devices:/user.slice\n9:freezer:/\n8:hugetlb:/\n7:pids:/user.slice/user-1754.slice/session-1404.scope\n6:rdma:/\n5:cpu,cpuacct:/\n4:blkio:/\n3:net_cls,net_prio:/\n2:perf_event:/\n1:name=systemd:/user.slice/user-1754.slice/session-1404.scope\n", "cgroup_files": {"/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us": "100000", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us": "-1", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat": "nr_periods 0\nnr_throttled 0\nthrottled_time 0", "/sys/fs/cgroup/cpuset/cpuset.cpus": "0-71", "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/memory.usage_in_bytes": "495648530432", "/sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes": "504610410496", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/memory.usage_in_bytes": "376917532672", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/session-1404.scope/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/session-1404.scope/memory.usage_in_bytes": "336239099904", "/sys/fs/cgroup/pids/user.slice/pids.current": "2516", "/sys/fs/cgroup/pids/user.slice/pids.max": "max", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/pids.current": "656", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/pids.max": "3295194", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/session-1404.scope/pids.current": "61", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/session-1404.scope/pids.max": "max"}, "cgroup_memory_headrooms_bytes": [9223371541206241280, 9223371532244361216, 9223371659937239040, 9223371700615671808], "cpu_quota_cores": null, "executable": "/data/home/aviyac/ap_env/bin/python", "host": "gold-srv2.iem.technion.ac.il", "host_memory": {"MemAvailable_bytes": 504674795520, "MemTotal_bytes": 539924553728, "SwapFree_bytes": 32317272064, "SwapTotal_bytes": 34359734272}, "load_average": [1.96, 1.95, 1.2], "logical_cpus": 72, "memory_available_bytes": 504674795520, "platform": "Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28", "python": "3.11.13 (main, Jan 28 2026, 00:22:30) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)]"}

Raw journals certify recorded strict OPTIMAL/physical acceptance and center mapping, not independent cost recomputation: original evaluators do not persist scenario costs or QP solutions.

Publication status and final report/final-verification wall times are recorded in the sealed pipeline summary one directory above this report. This report is constructed before atomic publication.
