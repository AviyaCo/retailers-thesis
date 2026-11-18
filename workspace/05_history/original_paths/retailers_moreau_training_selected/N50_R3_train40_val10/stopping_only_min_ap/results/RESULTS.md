# Stopping-only minimum-AP reporting

R=3; N=50 = 40 train + 10 validation; seeds 0–9; network seed 0; split seed 100000 + seed. Frozen historical test: 10,000 scenarios, seed 9999. Native validation and strict OPTIMAL-only test evaluation are unchanged.

## Interpretation and provenance

- Both rules return the earliest minimum mean AP incumbent in their own stopping prefix. Green is initial-AP eligible; black retains every-center stopping. Stopping and returned checkpoints are distinct. No new global optimum or every-center certificate is claimed for the returned minimum-AP policy; AP incumbent rankings do not resolve differences within AP bounds.
- Black seed 9 at radius 0.7 remains capped at 40 MPs even when an earlier checkpoint is returned. It is included in the complete ten-seed cost curve but separated from stopping-rule-satisfied runtime cohorts.
- All fixed-radius costs use all ten seeds. Supplied radius selections were frozen using validation only (smaller-radius ties, nominal eligible) before test lookup. Reporting never selects, tunes, clips costs, or interpolates red.
- Cache reuse requires exact seed, checkpoint SHA256 and learned-state SHA256 under the same evaluator and data contract. Reuse across radii is legitimate for the same policy because evaluation is independent of training radius. Repeated cache references are neither new observations nor current runtime.
- Red SRO-FA, yellow kNN, blue, purple and horizontal reference numbers are unchanged. Historical training/selection budgets and reference samples differ; this is not a matched-protocol or statistical-significance comparison. The previously inspected seed-9999 test and test-informed historical radius grid permit descriptive, not confirmatory, conclusions. Wins are not guaranteed.
- Bands are empirical 20th/80th percentiles across seed means, not confidence intervals. Red comparisons exist only at exactly shared radius values.
- Saved nonoverlapping phases through STOP, not the returned checkpoint; excludes nominal training, selection/publication and uninstrumented overhead. Not fresh end-to-end runtime or parallel sweep elapsed time.
- Fresh green and black protocol end-to-end times are unavailable. No new training, AP search, MP optimization, CV, or benchmark construction was run. Cached durations must not be counted as current work. Missing historical timings require a rerun to measure; SRO-FA runtime is unavailable.
- Current invocation total and reporting-phase completion are not yet available here: consult summary.json finalized by the caller. Phase snapshots may show reporting running. Do not sum nested phases or concurrent model wall times and present the sum as elapsed campaign runtime.

## Fixed-radius means and descriptive references

Negative differences indicate lower cost. All rows contain ten seeds, including capped policies. No red entry means that exact radius is absent, not an interpolated estimate.

| Color | Radius | Test mean | 20th / 80th percentile | Minus yellow | Minus red |
|---|---:|---:|---:|---:|---:|
| green | 0 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | not shared |
| green | 0.001 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.367270 |
| green | 0.003 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.374216 |
| green | 0.01 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.361897 |
| green | 0.03 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | not shared |
| green | 0.2 | 20.637740 | 19.985036 / 20.974815 | -0.159976 | -0.915025 |
| green | 0.4 | 20.529497 | 19.965308 / 21.090209 | -0.268219 | -0.756489 |
| green | 0.7 | 20.733442 | 20.393726 / 21.243231 | -0.064274 | -0.436787 |
| green | 1.5 | 21.484426 | 21.250424 / 21.735782 | 0.686710 | 0.052331 |
| black | 0 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | not shared |
| black | 0.001 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.367270 |
| black | 0.003 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.374216 |
| black | 0.01 | 20.591139 | 20.215946 / 20.988469 | -0.206577 | -1.361897 |
| black | 0.03 | 20.636376 | 20.289914 / 20.968996 | -0.161340 | not shared |
| black | 0.2 | 20.476167 | 20.194685 / 20.716874 | -0.321550 | -1.076599 |
| black | 0.4 | 20.470175 | 20.010670 / 21.098484 | -0.327541 | -0.815811 |
| black | 0.7 | 20.653866 | 20.215409 / 21.079144 | -0.143850 | -0.516363 |
| black | 1.5 | 21.375432 | 21.152493 / 21.543447 | 0.577716 | -0.056663 |

green vs yellow: lower at 8/9 compared radii, equal at 0, higher at 1. Descriptive counts only; not significance tests.

green vs red: lower at 6/7 compared radii, equal at 0, higher at 1. Descriptive counts only; not significance tests.

black vs yellow: lower at 8/9 compared radii, equal at 0, higher at 1. Descriptive counts only; not significance tests.

black vs red: lower at 7/7 compared radii, equal at 0, higher at 0. Descriptive counts only; not significance tests.

## Supplied validation-selected policies

| Color | Seed | Radius | Validation mean | Test mean | Returned / stop candidate | Terminal status |
|---|---:|---:|---:|---:|---|---|
| green | 0 | 0.4 | 19.083233 | 20.339569 | 3 / 3 | mean_gap_satisfied |
| green | 1 | 1.5 | 18.129061 | 21.734690 | 5 / 5 | mean_gap_satisfied |
| green | 2 | 0.4 | 21.954051 | 19.926301 | 3 / 3 | mean_gap_satisfied |
| green | 3 | 0.2 | 19.194097 | 20.878259 | 2 / 2 | mean_gap_satisfied |
| green | 4 | 0 | 18.079470 | 20.365244 | 0 / 0 | nominal_reference |
| green | 5 | 0 | 17.165652 | 20.600806 | 0 / 0 | nominal_reference |
| green | 6 | 0 | 21.196551 | 20.095758 | 0 / 0 | nominal_reference |
| green | 7 | 0 | 14.158781 | 21.534351 | 0 / 0 | nominal_reference |
| green | 8 | 0.4 | 19.717625 | 19.975059 | 5 / 5 | mean_gap_satisfied |
| green | 9 | 0.2 | 18.740452 | 19.999166 | 3 / 3 | mean_gap_satisfied |

green selected test costs (n=10): mean / median / min / max = 20.544920 / 20.352407 / 19.926301 / 21.734690.
| black | 0 | 0.4 | 19.092806 | 20.182440 | 11 / 11 | separation_satisfied |
| black | 1 | 1.5 | 17.987906 | 21.399181 | 10 / 11 | separation_satisfied |
| black | 2 | 0.03 | 21.980868 | 19.903388 | 2 / 2 | separation_satisfied |
| black | 3 | 0 | 19.214488 | 20.851999 | 0 / 0 | separation_satisfied |
| black | 4 | 0.4 | 17.998818 | 20.219934 | 8 / 8 | separation_satisfied |
| black | 5 | 0.03 | 17.154974 | 20.648812 | 2 / 2 | separation_satisfied |
| black | 6 | 0 | 21.196551 | 20.095758 | 0 / 0 | separation_satisfied |
| black | 7 | 0.2 | 13.955365 | 20.576590 | 12 / 12 | separation_satisfied |
| black | 8 | 0.4 | 19.661620 | 19.931924 | 8 / 8 | separation_satisfied |
| black | 9 | 0 | 18.757749 | 20.245992 | 0 / 0 | separation_satisfied |

black selected test costs (n=10): mean / median / min / max = 20.405602 / 20.232963 / 19.903388 / 21.399181.

## Capped configuration (not converged)

Black seed 9, radius 0.7: returned candidate 39; stop candidate 40; 40 MPs; test mean 20.421154. Status remains iteration_cap independently of the earlier return or whether this radius was validation-selected.

## Runtime overview

Durations below are seconds / minutes. Historical source nominal training used 10 workers; source CCG used 16 workers, one thread each. New evaluations use one worker and one thread. Saved historical environment metadata are retained; unrecorded historical machine load is unknown.

Current environment: {"cpu_count": 72, "host": "gold-srv2.iem.technion.ac.il", "platform": "Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28", "processor": "x86_64", "python": "3.11.13 (main, Jan 28 2026, 00:22:30) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)]"}

| Scope | Wall time |
|---|---|
| Fresh green protocol end-to-end | unavailable (not run) |
| Fresh black protocol end-to-end | unavailable (not run) |
| SRO-FA partition / solver / evaluation / total | unavailable; rerun required |
| New nominal training / AP / MP / CV / benchmark construction | not run |
| Current total, including reporting | pending; caller finalizes summary.json |
| Historical campaign nominal_lr_search_s (separate, not additive) | 121.796206 s / 2.029937 min |
| Historical campaign ccg_sweep_s (separate, not additive) | 14942.580206 s / 249.043003 min |
| Historical campaign validation_s (separate, not additive) | 1.658737 s / 0.027646 min |
| Historical campaign overall_s (separate, not additive) | 15092.965248 s / 251.549421 min |
| Current validation: sum of 3 actual calls (not whole phase) | 0.017471 s / 0.000291 min |
| Current test: sum of 3 actual calls (not whole phase) | 15.095724 s / 0.251595 min |
| Current phase reconstruct_and_freeze: completed (snapshot, not additive) | 31.164853 s / 0.519414 min |
| Current phase validation_and_radius_selection: completed (snapshot, not additive) | 0.672499 s / 0.011208 min |
| Current phase radius_selection: completed (snapshot, not additive) | 0.001176 s / 0.000020 min |
| Current phase test_lookup_and_evaluation: completed (snapshot, not additive) | 15.750403 s / 0.262507 min |
| Current phase fresh_test_evaluation: completed (snapshot, not additive) | 15.724281 s / 0.262071 min |
| Current phase end_verification: completed (snapshot, not additive) | 2.098892 s / 0.034982 min |
| Current phase reporting: running (snapshot, not additive) | unavailable |

## Historical selected nominal training (not full LR search)

| Seed | Epochs evaluated | Training seconds / minutes |
|---:|---:|---|
| 0 | 2496 | 29.626559 s / 0.493776 min |
| 1 | 3000 | 33.889487 s / 0.564825 min |
| 2 | 2536 | 37.515047 s / 0.625251 min |
| 3 | 3000 | 45.877120 s / 0.764619 min |
| 4 | 3000 | 40.610397 s / 0.676840 min |
| 5 | 3000 | 40.767370 s / 0.679456 min |
| 6 | 3000 | 40.005789 s / 0.666763 min |
| 7 | 3000 | 35.182774 s / 0.586380 min |
| 8 | 3000 | 36.968728 s / 0.616145 min |
| 9 | 3000 | 35.760788 s / 0.596013 min |

## Runtime statistics

Each row reports n and mean / median / min / max. Seconds are stored in JSON; counts are dimensionless. Capped and nominal-reference cohorts stay separate. Black source wall time is historical parent-trajectory time, not a new min-AP total. Per-candidate AP/MP phases, identities, source timestamps and persisted evaluation durations are retained in runtime_comparison.json; repeated cached evaluations are not summed.

| Cohort | Metric | n | Mean / median / min / max (s unless count) | Mean minutes |
|---|---|---:|---|---:|
| Historical nominal selected winners | training_wall_s | 10 | 37.620406 / 37.241888 / 29.626559 / 45.877120 | 0.627007 |
| Historical nominal selected winners | epochs_evaluated | 10 | 2903.200000 / 3000.000000 / 2496.000000 / 3000.000000 | — |
| Current validation calls | wall_time_s | 3 | 0.005824 / 0.002134 / 0.002109 / 0.013228 | 0.000097 |
| Current test calls | wall_time_s | 3 | 5.031908 / 4.867627 / 4.863434 / 5.364664 | 0.083865 |
| black, radius 0, completed/separation_satisfied | reconstructed_phase_s | 10 | 1.929114 / 1.845262 / 1.627561 / 2.468371 | 0.032152 |
| black, radius 0, completed/separation_satisfied | ap_s | 10 | 1.902812 / 1.824002 / 1.575930 / 2.442822 | 0.031714 |
| black, radius 0, completed/separation_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| black, radius 0, completed/separation_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| black, radius 0, completed/separation_satisfied | source_ccg_wall_s | 10 | 2.087166 / 1.999984 / 1.756980 / 2.684249 | 0.034786 |
| black, radius 0.001, completed/separation_satisfied | reconstructed_phase_s | 10 | 1.870207 / 1.726003 / 1.517364 / 2.492965 | 0.031170 |
| black, radius 0.001, completed/separation_satisfied | ap_s | 10 | 1.843538 / 1.689710 / 1.484481 / 2.464243 | 0.030726 |
| black, radius 0.001, completed/separation_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| black, radius 0.001, completed/separation_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| black, radius 0.001, completed/separation_satisfied | source_ccg_wall_s | 10 | 2.027726 / 1.862810 / 1.680292 / 2.686424 | 0.033795 |
| black, radius 0.003, completed/separation_satisfied | reconstructed_phase_s | 10 | 1.883026 / 1.922692 / 1.430179 / 2.389696 | 0.031384 |
| black, radius 0.003, completed/separation_satisfied | ap_s | 10 | 1.856880 / 1.895179 / 1.398168 / 2.361139 | 0.030948 |
| black, radius 0.003, completed/separation_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| black, radius 0.003, completed/separation_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| black, radius 0.003, completed/separation_satisfied | source_ccg_wall_s | 10 | 2.031718 / 2.071909 / 1.572742 / 2.527521 | 0.033862 |
| black, radius 0.01, completed/separation_satisfied | reconstructed_phase_s | 10 | 1.783913 / 1.738477 / 1.526059 / 2.345154 | 0.029732 |
| black, radius 0.01, completed/separation_satisfied | ap_s | 10 | 1.759159 / 1.718464 / 1.502751 / 2.325034 | 0.029319 |
| black, radius 0.01, completed/separation_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| black, radius 0.01, completed/separation_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| black, radius 0.01, completed/separation_satisfied | source_ccg_wall_s | 10 | 1.947422 / 1.899755 / 1.643105 / 2.490503 | 0.032457 |
| black, radius 0.03, completed/separation_satisfied | reconstructed_phase_s | 10 | 93.384408 / 77.961808 / 1.598010 / 197.662532 | 1.556407 |
| black, radius 0.03, completed/separation_satisfied | ap_s | 10 | 42.129196 / 35.192645 / 1.570181 / 101.445001 | 0.702153 |
| black, radius 0.03, completed/separation_satisfied | mp_s | 10 | 51.195405 / 42.708629 / 0.000000 / 102.646798 | 0.853257 |
| black, radius 0.03, completed/separation_satisfied | mp_count | 10 | 1.800000 / 2.000000 / 0.000000 / 3.000000 | — |
| black, radius 0.03, completed/separation_satisfied | source_ccg_wall_s | 10 | 98.535294 / 82.642768 / 1.711565 / 208.361424 | 1.642255 |
| black, radius 0.2, completed/separation_satisfied | reconstructed_phase_s | 10 | 666.775254 / 583.631029 / 319.074839 / 1479.051663 | 11.112921 |
| black, radius 0.2, completed/separation_satisfied | ap_s | 10 | 387.275338 / 321.495295 / 148.869646 / 1000.449150 | 6.454589 |
| black, radius 0.2, completed/separation_satisfied | mp_s | 10 | 279.325860 / 266.591165 / 162.002724 / 478.277297 | 4.655431 |
| black, radius 0.2, completed/separation_satisfied | mp_count | 10 | 6.600000 / 6.000000 / 4.000000 / 12.000000 | — |
| black, radius 0.2, completed/separation_satisfied | source_ccg_wall_s | 10 | 715.089333 / 624.722135 / 338.277539 / 1597.338762 | 11.918156 |
| black, radius 0.4, completed/separation_satisfied | reconstructed_phase_s | 10 | 1306.310088 / 1297.501326 / 828.228555 / 2107.593784 | 21.771835 |
| black, radius 0.4, completed/separation_satisfied | ap_s | 10 | 852.338250 / 816.232474 / 503.606363 / 1519.218823 | 14.205637 |
| black, radius 0.4, completed/separation_satisfied | mp_s | 10 | 453.705350 / 460.968397 / 324.418941 / 661.234301 | 7.561756 |
| black, radius 0.4, completed/separation_satisfied | mp_count | 10 | 10.200000 / 10.000000 / 8.000000 / 15.000000 | — |
| black, radius 0.4, completed/separation_satisfied | source_ccg_wall_s | 10 | 1412.567801 / 1400.907108 / 889.616289 / 2300.128266 | 23.542797 |
| black, radius 0.7, completed/iteration_cap | reconstructed_phase_s | 1 | 10302.639037 / 10302.639037 / 10302.639037 / 10302.639037 | 171.710651 |
| black, radius 0.7, completed/iteration_cap | ap_s | 1 | 8923.770179 / 8923.770179 / 8923.770179 / 8923.770179 | 148.729503 |
| black, radius 0.7, completed/iteration_cap | mp_s | 1 | 1377.828640 / 1377.828640 / 1377.828640 / 1377.828640 | 22.963811 |
| black, radius 0.7, completed/iteration_cap | mp_count | 1 | 40.000000 / 40.000000 / 40.000000 / 40.000000 | — |
| black, radius 0.7, completed/iteration_cap | source_ccg_wall_s | 1 | 11411.629000 / 11411.629000 / 11411.629000 / 11411.629000 | 190.193817 |
| black, radius 0.7, completed/separation_satisfied | reconstructed_phase_s | 9 | 3677.588329 / 3739.113897 / 1146.353697 / 7526.144861 | 61.293139 |
| black, radius 0.7, completed/separation_satisfied | ap_s | 9 | 2734.639766 / 2831.639839 / 667.082529 / 5859.292010 | 45.577329 |
| black, radius 0.7, completed/separation_satisfied | mp_s | 9 | 942.433723 / 937.731903 / 479.024126 / 1665.993364 | 15.707229 |
| black, radius 0.7, completed/separation_satisfied | mp_count | 9 | 18.666667 / 20.000000 / 9.000000 / 28.000000 | — |
| black, radius 0.7, completed/separation_satisfied | source_ccg_wall_s | 9 | 4017.666794 / 4090.432882 / 1230.665084 / 8251.298251 | 66.961113 |
| black, radius 1.5, completed/separation_satisfied | reconstructed_phase_s | 10 | 3522.978507 / 2798.627512 / 672.452374 / 8428.801063 | 58.716308 |
| black, radius 1.5, completed/separation_satisfied | ap_s | 10 | 2693.192329 / 2058.625665 / 489.844723 / 7006.610356 | 44.886539 |
| black, radius 1.5, completed/separation_satisfied | mp_s | 10 | 829.227398 / 739.511699 / 142.150736 / 1421.192508 | 13.820457 |
| black, radius 1.5, completed/separation_satisfied | mp_count | 10 | 19.400000 / 18.500000 / 9.000000 / 35.000000 | — |
| black, radius 1.5, completed/separation_satisfied | source_ccg_wall_s | 10 | 3859.724695 / 3056.511687 / 734.514130 / 9307.252776 | 64.328745 |
| green, radius 0, completed/nominal_reference | reconstructed_phase_s | 10 | 1.929114 / 1.845262 / 1.627561 / 2.468371 | 0.032152 |
| green, radius 0, completed/nominal_reference | ap_s | 10 | 1.902812 / 1.824002 / 1.575930 / 2.442822 | 0.031714 |
| green, radius 0, completed/nominal_reference | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| green, radius 0, completed/nominal_reference | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| green, radius 0, completed/nominal_reference | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.001, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 1.870207 / 1.726003 / 1.517364 / 2.492965 | 0.031170 |
| green, radius 0.001, completed/mean_gap_satisfied | ap_s | 10 | 1.843538 / 1.689710 / 1.484481 / 2.464243 | 0.030726 |
| green, radius 0.001, completed/mean_gap_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| green, radius 0.001, completed/mean_gap_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| green, radius 0.001, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.003, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 1.883026 / 1.922692 / 1.430179 / 2.389696 | 0.031384 |
| green, radius 0.003, completed/mean_gap_satisfied | ap_s | 10 | 1.856880 / 1.895179 / 1.398168 / 2.361139 | 0.030948 |
| green, radius 0.003, completed/mean_gap_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| green, radius 0.003, completed/mean_gap_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| green, radius 0.003, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.01, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 1.783913 / 1.738477 / 1.526059 / 2.345154 | 0.029732 |
| green, radius 0.01, completed/mean_gap_satisfied | ap_s | 10 | 1.759159 / 1.718464 / 1.502751 / 2.325034 | 0.029319 |
| green, radius 0.01, completed/mean_gap_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| green, radius 0.01, completed/mean_gap_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| green, radius 0.01, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.03, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 1.855071 / 1.807203 / 1.598010 / 2.280775 | 0.030918 |
| green, radius 0.03, completed/mean_gap_satisfied | ap_s | 10 | 1.828307 / 1.783004 / 1.565290 / 2.260512 | 0.030472 |
| green, radius 0.03, completed/mean_gap_satisfied | mp_s | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| green, radius 0.03, completed/mean_gap_satisfied | mp_count | 10 | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| green, radius 0.03, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.2, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 160.828642 / 146.614557 / 134.019386 / 243.538241 | 2.680477 |
| green, radius 0.2, completed/mean_gap_satisfied | ap_s | 10 | 63.459936 / 53.447291 / 47.696428 / 104.487291 | 1.057666 |
| green, radius 0.2, completed/mean_gap_satisfied | mp_s | 10 | 97.299326 / 92.847014 / 86.260401 / 138.976942 | 1.621655 |
| green, radius 0.2, completed/mean_gap_satisfied | mp_count | 10 | 2.200000 / 2.000000 / 2.000000 / 3.000000 | — |
| green, radius 0.2, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.4, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 392.621387 / 302.411178 / 206.875215 / 858.837719 | 6.543690 |
| green, radius 0.4, completed/mean_gap_satisfied | ap_s | 10 | 186.651914 / 133.791227 / 97.079721 / 449.668825 | 3.110865 |
| green, radius 0.4, completed/mean_gap_satisfied | mp_s | 10 | 205.853280 / 168.516531 / 107.553024 / 408.977900 | 3.430888 |
| green, radius 0.4, completed/mean_gap_satisfied | mp_count | 10 | 4.100000 / 3.500000 / 3.000000 / 7.000000 | — |
| green, radius 0.4, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 0.7, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 604.883862 / 487.596322 / 329.856168 / 1339.569916 | 10.081398 |
| green, radius 0.7, completed/mean_gap_satisfied | ap_s | 10 | 318.085327 / 239.852327 / 151.141598 / 750.161466 | 5.301422 |
| green, radius 0.7, completed/mean_gap_satisfied | mp_s | 10 | 286.646255 / 247.608636 / 178.612495 / 589.148045 | 4.777438 |
| green, radius 0.7, completed/mean_gap_satisfied | mp_count | 10 | 5.600000 / 5.000000 / 4.000000 / 9.000000 | — |
| green, radius 0.7, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |
| green, radius 1.5, completed/mean_gap_satisfied | reconstructed_phase_s | 10 | 837.722257 / 489.950119 / 321.756710 / 2924.834886 | 13.962038 |
| green, radius 1.5, completed/mean_gap_satisfied | ap_s | 10 | 525.870073 / 265.648320 / 168.187252 / 2066.565468 | 8.764501 |
| green, radius 1.5, completed/mean_gap_satisfied | mp_s | 10 | 311.653528 / 250.881713 / 126.195485 / 857.830094 | 5.194225 |
| green, radius 1.5, completed/mean_gap_satisfied | mp_count | 10 | 7.400000 / 6.000000 / 4.000000 / 17.000000 | — |
| green, radius 1.5, completed/mean_gap_satisfied | source_ccg_wall_s | 0 | unavailable | — |

## Artifacts

- comparison.png: complete fixed-radius curves and unchanged references.
- plot_data.json: all plotted numerical values and original test-array metadata.
- runtime_comparison.json: raw timing evidence, source hashes, status-separated statistics.
- summary.json: caller-owned final invocation timing and provenance.
