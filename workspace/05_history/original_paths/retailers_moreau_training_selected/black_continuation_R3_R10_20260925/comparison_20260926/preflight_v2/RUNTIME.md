# Invocation runtime (not historical training)

Status: completed; mode: preflight.

| Category | Measured time / status |
|---|---|
| Nominal neural training | Not run; persisted history separate; missing fields unavailable, rerun required |
| Robust neural/CCG AP / MP / total | Not run; persisted history separate |
| SRO-FA partition / optimization / evaluation / total | Not run; original references unchanged |
| Hyperparameter selection / CV / benchmark construction | Not run |
| Final neural evaluation | Production raw per-model results only; preflight probes are not model training |
| syntax (completed) | 0.036550 s / 0.000609 min / 0.000010 h |
| unit_tests (completed) | 14.061996 s / 0.234367 min / 0.003906 h |
| collection (completed) | 983.496839 s / 16.391614 min / 0.273194 h |
| resources (completed) | 0.278739 s / 0.004646 min / 0.000077 h |
| source_copy (completed) | 332.247685 s / 5.537461 min / 0.092291 h |
| policy_seal (completed) | 286.797648 s / 4.779961 min / 0.079666 h |
| training_probes (completed) | 37.036382 s / 0.617273 min / 0.010288 h |
| journal_verification (completed) | 1.129011 s / 0.018817 min / 0.000314 h |
| final_verification (completed) | 196.880911 s / 3.281349 min / 0.054689 h |

| Policy | Process status | Process wall time | Backend model wall time |
|---|---|---|---|
| R10_N200_s6_rho0.7 | completed | 36.922737 s / 0.615379 min / 0.010256 h | 32.221525 s / 0.537025 min / 0.008950 h |
| R10_N50_s8_rho0.2 | completed | 35.164315 s / 0.586072 min / 0.009768 h | 30.465141 s / 0.507752 min / 0.008463 h |
| R10_N800_s0_rho2 | completed | 33.781321 s / 0.563022 min / 0.009384 h | 29.447382 s / 0.490790 min / 0.008180 h |
| R3_N200_s0_rho0.4 | completed | 25.959533 s / 0.432659 min / 0.007211 h | 21.110396 s / 0.351840 min / 0.005864 h |
| R3_N50_s9_rho0.7 | completed | 26.811610 s / 0.446860 min / 0.007448 h | 22.535687 s / 0.375595 min / 0.006260 h |
| R3_N800_s0_rho0.4 | completed | 26.119663 s / 0.435328 min / 0.007255 h | 21.354693 s / 0.355912 min / 0.005932 h |

These per-process statuses do not replace the global journal/publication barrier. No failed or unverified result is a certified completed-model aggregate. Raw worker records retain separate OOS/IS phases and iterations. Final elapsed time and this reporting phase are in the sealed summary; whole-child time including serialization is in the launcher runtime artifact. Concurrent wall times are not summed. Missing unreturned timings are unavailable, not zero.

Workers: 16; numerical threads per worker: 1. Environment: {"affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], "captured_at_utc": "2026-09-26T18:45:08.901235+00:00", "cgroup": "12:cpuset:/\n11:memory:/user.slice/user-1754.slice/session-1404.scope\n10:devices:/user.slice\n9:freezer:/\n8:hugetlb:/\n7:pids:/user.slice/user-1754.slice/session-1404.scope\n6:rdma:/\n5:cpu,cpuacct:/\n4:blkio:/\n3:net_cls,net_prio:/\n2:perf_event:/\n1:name=systemd:/user.slice/user-1754.slice/session-1404.scope\n", "cgroup_files": {"/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us": "100000", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us": "-1", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat": "nr_periods 0\nnr_throttled 0\nthrottled_time 0", "/sys/fs/cgroup/cpuset/cpuset.cpus": "0-71", "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/memory.usage_in_bytes": "509974867968", "/sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes": "519496310784", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/memory.usage_in_bytes": "330846797824", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/session-1404.scope/memory.limit_in_bytes": "9223372036854771712", "/sys/fs/cgroup/memory/user.slice/user-1754.slice/session-1404.scope/memory.usage_in_bytes": "294832033792", "/sys/fs/cgroup/pids/user.slice/pids.current": "2468", "/sys/fs/cgroup/pids/user.slice/pids.max": "max", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/pids.current": "586", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/pids.max": "3295194", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/session-1404.scope/pids.current": "61", "/sys/fs/cgroup/pids/user.slice/user-1754.slice/session-1404.scope/pids.max": "max"}, "cgroup_memory_headrooms_bytes": [9223371526879903744, 9223371517358460928, 9223371706007973888, 9223371742022737920], "cpu_quota_cores": null, "executable": "/data/home/aviyac/ap_env/bin/python", "host": "gold-srv2.iem.technion.ac.il", "host_memory": {"MemAvailable_bytes": 514104782848, "MemTotal_bytes": 539924553728, "SwapFree_bytes": 32414433280, "SwapTotal_bytes": 34359734272}, "load_average": [1.82, 1.72, 1.5], "logical_cpus": 72, "memory_available_bytes": 514104782848, "platform": "Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28", "python": "3.11.13 (main, Jan 28 2026, 00:22:30) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)]"}
