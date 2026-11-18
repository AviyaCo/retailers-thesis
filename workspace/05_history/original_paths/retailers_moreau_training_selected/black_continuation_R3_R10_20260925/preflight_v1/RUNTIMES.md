# Runtime accounting

Current sweep: 412.666 s / 6.878 min / 0.115 h.
Concurrent model durations are never summed as sweep elapsed.

| Phase | R | N | Radius | Status | n/measured | Mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cached_black | 3 | 50 | 0.0 | cached_only | 10/10 | 2.087 | 2.000 | 1.757 | 2.684 |
| cached_black | 3 | 50 | 0.001 | cached_only | 10/10 | 2.028 | 1.863 | 1.680 | 2.686 |
| cached_black | 3 | 50 | 0.003 | cached_only | 10/10 | 2.032 | 2.072 | 1.573 | 2.528 |
| cached_black | 3 | 50 | 0.01 | cached_only | 10/10 | 1.947 | 1.900 | 1.643 | 2.491 |
| cached_black | 3 | 50 | 0.03 | cached_only | 10/10 | 98.535 | 82.643 | 1.712 | 208.361 |
| cached_black | 3 | 50 | 0.2 | cached_only | 10/10 | 715.089 | 624.722 | 338.278 | 1597.339 |
| cached_black | 3 | 50 | 0.4 | cached_only | 10/10 | 1412.568 | 1400.907 | 889.616 | 2300.128 |
| cached_black | 3 | 50 | 0.7 | cached_only | 10/10 | 4757.063 | 4359.540 | 1230.665 | 11411.629 |
| cached_black | 3 | 50 | 1.5 | cached_only | 10/10 | 3859.725 | 3056.512 | 734.514 | 9307.253 |
| cached_nominal_training | 10 | 200 | 0.0 | cached_only | 10/10 | 161.969 | 169.707 | 96.384 | 170.892 |
| cached_nominal_training | 10 | 50 | 0.0 | cached_only | 10/10 | 113.648 | 125.832 | 83.230 | 129.866 |
| cached_nominal_training | 10 | 800 | 0.0 | cached_only | 10/10 | 198.557 | 215.968 | 121.084 | 216.452 |
| cached_nominal_training | 3 | 200 | 0.0 | cached_only | 10/10 | 149.802 | 149.807 | 149.042 | 150.498 |
| cached_nominal_training | 3 | 50 | 0.0 | cached_only | 10/10 | 37.620 | 37.242 | 29.627 | 45.877 |
| cached_nominal_training | 3 | 800 | 0.0 | cached_only | 10/10 | 170.070 | 171.988 | 157.940 | 174.612 |

| Additional phase | Runtime |
|---|---|
| SRO-FA partition / solve / evaluation / total | Not run; unavailable |
| New nominal training / validation / CV / final test | Not run |
| Historical LR search / validation | Separate original campaign phase timings in manifest |

Unmeasured historical fields require a rerun; cached loading is not training time.
