# Completed mean-rule retailer comparison

31 isolated continuation jobs and 31 fresh strict evaluations completed; combined with 59 recovered/reference evaluations, the green grid contains exactly 90 unique seed/radius results (10 seeds per radius). Eight positive radii connect; zero remains an isolated nominal reference.

Black: **Moreau: every-center stop (cap 40 MPs)**. Green: **Moreau: mean-gap stop / min-AP***.
The green policy is chosen using training/AP information, not test cost. No new test-based selection, tuning, significance tests, projection solves or retraining occurs in this report. The same 10,000 scenarios (seed 9999) were previously inspected: results are descriptive, not confirmatory. Bands are 20th/80th percentiles across seed costs around the mean, **not confidence intervals**.
Only the green positive curve changes numerically. Other curves, isolated nominal and horizontal references retain their prior numeric values exactly; old outputs are not overwritten.
Black seed 9 at radius .7 exhausted 40 MPs and is **capped, not converged**. Its timing is aggregated separately. Mean-gap satisfaction does not certify every-center separation or global optimality.

## Cost comparison at common radii

Signed differences are black minus comparator; negative means lower black cost. Means below come from the plotted arrays, not a different policy/radius selection.

| Radius | Blue SCS | Black full | Green short | Black − blue | Black − green |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 20.725357672 | 20.591139479 | 20.607406428 | -0.134218193 | -0.016266948 |
| 0.003 | 20.709394936 | 20.591139479 | 20.618098314 | -0.118255457 | -0.026958835 |
| 0.01 | 20.685510902 | 20.591139479 | 20.596219776 | -0.094371422 | -0.005080297 |
| 0.2 | 20.502524077 | 20.476166528 | 20.637739739 | -0.026357549 | -0.161573211 |
| 0.4 | 20.604119210 | 20.470174738 | 20.529497151 | -0.133944471 | -0.059322412 |
| 0.7 | 20.800235546 | 20.666352085 | 20.733442042 | -0.133883461 | -0.067089957 |
| 1.5 | 21.413835180 | 21.375432168 | 21.484425850 | -0.038403011 | -0.108993682 |

## Runtime interpretation

Original input verification/loading + each candidate evaluation/AP + incoming-pool evaluation, MP call and returned-pool evaluation before each next candidate. Nonoverlapping outer phases only; excludes nominal training, selection/checkpoint publication, uninstrumented controller/persistence overhead and new resume loading. NOT fresh short-run end-to-end time.
Full Moreau wall time is the historical per-configuration CCG invocation (nominal training separate). Short full wall time is unavailable: recovered paths were not fresh runs, and continuation wall time covers only the current resume invocation. The comparable short phase sum combines persisted candidate-zero phases with newly measured continuation phases. Saved/current splits and origin cohort counts are retained below and in JSON. Resume input/loading and selection overhead remain in current wall time and raw worker records, but not in the matched phase sum.
Model durations are concurrent wall times, not additive sweep elapsed times. Moreau CCG and continuations use 16 workers, one numerical/solver thread each. Prior strict evaluation used 8 workers; fresh evaluation concurrency is recorded per result. Nominal Moreau used 10 workers. Unrecorded historical nominal-SCS concurrency/threads are unknown.
Environment: {'prior_training_host': 'gold-srv2.iem.technion.ac.il', 'prior_evaluation': {'host': 'gold-srv2.iem.technion.ac.il', 'platform': 'Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28', 'cpu_count': 72}, 'current': {'host': 'gold-srv2.iem.technion.ac.il', 'platform': 'Linux-4.18.0-553.97.1.el8_10.x86_64-x86_64-with-glibc2.28', 'cpu': 'Intel Xeon Gold 6354; shared server'}, 'hardware_note': 'Unrecorded CPU model/memory and historical SCS host are unknown.'}.

## Campaign and evaluation/selection elapsed times

| Phase | Measured seconds / minutes / hours |
|---|---|
| prior_nominal_lr_search_s | 121.796206 s / 2.029937 min / 0.033832 h |
| prior_full_ccg_sweep_s | 14942.580206 s / 249.043003 min / 4.150717 h |
| prior_terminal_validation_s | 1.658737 s / 0.027646 min / 0.000461 h |
| prior_evaluation_invocation_s | 106.856984 s / 1.780950 min / 0.029682 h |
| new_continuation_sweep_s | 147.204327 s / 2.453405 min / 0.040890 h |
| new_evaluation_sweep_s | 231.605939 s / 3.860099 min / 0.064335 h |
| Current parent phase: verify_inputs (completed) | 4.309381 s / 0.071823 min / 0.001197 h |
| Current parent phase: freeze_selected_policies (completed) | 0.051706 s / 0.000862 min / 0.000014 h |
| Current parent phase: copy_test_arrays (completed) | 0.001534 s / 0.000026 min / 0.000000 h |
| Parent final elapsed | Pending caller finalization after this function; not inferred |
| SRO-FA partition / solver / evaluation / total | Cached; timing unavailable, not rerun |
| CV / new hyperparameter selection / new benchmark construction | None |
| Exact plotted fixed-radius SCS training/CCG | Unavailable: all 230 total_time_s fields are zero placeholders; rerun required |
| Rendering / reporting | Separate measured report_timing.json; rendering is nested, not additive |

## Nominal training (separate from CCG)

| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |
|---|---:|---|---|---:|---|---|---:|
| Moreau nominal selected winner | 0 | completed / training_completed | all | 10 | wall_s | 37.620406 / 37.241888 / 29.626559 / 45.877120 | 0.627007 |
| Moreau nominal selected winner | 0 | completed / training_completed | all | 10 | epochs | 2903.200000 / 3000.000000 / 2496.000000 / 3000.000000 | — |
| SCS nominal separate rerun | 0 | completed / training_completed | all | 10 | wall_s | 347.422133 / 369.971440 / 243.749316 / 405.271895 | 5.790369 |
| SCS nominal separate rerun | 0 | completed / training_completed | all | 10 | epochs | 2525.500000 / 2709.500000 / 1716.000000 / 3000.000000 | — |

## Moreau by radius and completion/terminal status

| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |
|---|---:|---|---|---:|---|---|---:|
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | wall_s | 2.087166 / 1.999984 / 1.756980 / 2.684249 | 0.034786 |
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | phase_sum_s | 1.929114 / 1.845262 / 1.627561 / 2.468371 | 0.032152 |
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | ap_s | 1.902812 / 1.824002 / 1.575930 / 2.442822 | 0.031714 |
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | mp_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | ap_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau every-center | 0 | completed / separation_satisfied | all | 10 | mp_count | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | wall_s | 2.027726 / 1.862810 / 1.680292 / 2.686424 | 0.033795 |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | phase_sum_s | 1.870207 / 1.726003 / 1.517364 / 2.492965 | 0.031170 |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | ap_s | 1.843538 / 1.689710 / 1.484481 / 2.464243 | 0.030726 |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | mp_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | ap_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau every-center | 0.001 | completed / separation_satisfied | all | 10 | mp_count | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | wall_s | 2.031718 / 2.071909 / 1.572742 / 2.527521 | 0.033862 |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | phase_sum_s | 1.883026 / 1.922692 / 1.430179 / 2.389696 | 0.031384 |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | ap_s | 1.856880 / 1.895179 / 1.398168 / 2.361139 | 0.030948 |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | mp_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | ap_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau every-center | 0.003 | completed / separation_satisfied | all | 10 | mp_count | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | wall_s | 1.947422 / 1.899755 / 1.643105 / 2.490503 | 0.032457 |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | phase_sum_s | 1.783913 / 1.738477 / 1.526059 / 2.345154 | 0.029732 |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | ap_s | 1.759159 / 1.718464 / 1.502751 / 2.325034 | 0.029319 |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | mp_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | ap_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau every-center | 0.01 | completed / separation_satisfied | all | 10 | mp_count | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | wall_s | 98.535294 / 82.642768 / 1.711565 / 208.361424 | 1.642255 |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | phase_sum_s | 93.384408 / 77.961808 / 1.598010 / 197.662532 | 1.556407 |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | ap_s | 42.129196 / 35.192645 / 1.570181 / 101.445001 | 0.702153 |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | mp_s | 51.195405 / 42.708629 / 0.000000 / 102.646798 | 0.853257 |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | ap_count | 2.800000 / 3.000000 / 1.000000 / 4.000000 | — |
| Moreau every-center | 0.03 | completed / separation_satisfied | all | 10 | mp_count | 1.800000 / 2.000000 / 0.000000 / 3.000000 | — |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | wall_s | 715.089333 / 624.722135 / 338.277539 / 1597.338762 | 11.918156 |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | phase_sum_s | 666.775254 / 583.631029 / 319.074839 / 1479.051663 | 11.112921 |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | ap_s | 387.275338 / 321.495295 / 148.869646 / 1000.449150 | 6.454589 |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | mp_s | 279.325860 / 266.591165 / 162.002724 / 478.277297 | 4.655431 |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | ap_count | 7.600000 / 7.000000 / 5.000000 / 13.000000 | — |
| Moreau every-center | 0.2 | completed / separation_satisfied | all | 10 | mp_count | 6.600000 / 6.000000 / 4.000000 / 12.000000 | — |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | wall_s | 1412.567801 / 1400.907108 / 889.616289 / 2300.128266 | 23.542797 |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | phase_sum_s | 1306.310088 / 1297.501326 / 828.228555 / 2107.593784 | 21.771835 |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | ap_s | 852.338250 / 816.232474 / 503.606363 / 1519.218823 | 14.205637 |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | mp_s | 453.705350 / 460.968397 / 324.418941 / 661.234301 | 7.561756 |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | ap_count | 11.200000 / 11.000000 / 9.000000 / 16.000000 | — |
| Moreau every-center | 0.4 | completed / separation_satisfied | all | 10 | mp_count | 10.200000 / 10.000000 / 8.000000 / 15.000000 | — |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | wall_s | 11411.629000 / 11411.629000 / 11411.629000 / 11411.629000 | 190.193817 |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | phase_sum_s | 10302.639037 / 10302.639037 / 10302.639037 / 10302.639037 | 171.710651 |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | ap_s | 8923.770179 / 8923.770179 / 8923.770179 / 8923.770179 | 148.729503 |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | mp_s | 1377.828640 / 1377.828640 / 1377.828640 / 1377.828640 | 22.963811 |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | ap_count | 41.000000 / 41.000000 / 41.000000 / 41.000000 | — |
| Moreau every-center | 0.7 | completed / iteration_cap | all | 1 | mp_count | 40.000000 / 40.000000 / 40.000000 / 40.000000 | — |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | wall_s | 4017.666794 / 4090.432882 / 1230.665084 / 8251.298251 | 66.961113 |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | phase_sum_s | 3677.588329 / 3739.113897 / 1146.353697 / 7526.144861 | 61.293139 |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | ap_s | 2734.639766 / 2831.639839 / 667.082529 / 5859.292010 | 45.577329 |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | mp_s | 942.433723 / 937.731903 / 479.024126 / 1665.993364 | 15.707229 |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | ap_count | 19.666667 / 21.000000 / 10.000000 / 29.000000 | — |
| Moreau every-center | 0.7 | completed / separation_satisfied | all | 9 | mp_count | 18.666667 / 20.000000 / 9.000000 / 28.000000 | — |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | wall_s | 3859.724695 / 3056.511687 / 734.514130 / 9307.252776 | 64.328745 |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | phase_sum_s | 3522.978507 / 2798.627512 / 672.452374 / 8428.801063 | 58.716308 |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | ap_s | 2693.192329 / 2058.625665 / 489.844723 / 7006.610356 | 44.886539 |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | mp_s | 829.227398 / 739.511699 / 142.150736 / 1421.192508 | 13.820457 |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | ap_count | 20.400000 / 19.500000 / 10.000000 / 36.000000 | — |
| Moreau every-center | 1.5 | completed / separation_satisfied | all | 10 | mp_count | 19.400000 / 18.500000 / 9.000000 / 35.000000 | — |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | phase_sum_s | 1.929114 / 1.845262 / 1.627561 / 2.468371 | 0.032152 |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | ap_s | 1.902812 / 1.824002 / 1.575930 / 2.442822 | 0.031714 |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | mp_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | ap_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau mean-gap | 0 | completed / nominal_reference | all | 10 | mp_count | 0.000000 / 0.000000 / 0.000000 / 0.000000 | — |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 45.949694 / 43.625314 / 13.150911 / 82.627311 | 0.765828 |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | ap_s | 13.996845 / 14.054323 / 7.375848 / 20.449732 | 0.233281 |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | mp_s | 31.901959 / 29.808573 / 5.737277 / 62.364179 | 0.531699 |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | all | 10 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 42.647503 / 40.864868 / 11.847340 / 85.176900 | 0.710792 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | ap_s | 14.202107 / 14.018438 / 6.834282 / 21.102806 | 0.236702 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | mp_s | 28.393737 / 26.787811 / 4.320297 / 64.021107 | 0.473229 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | all | 10 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 45.617094 / 51.251376 / 10.038873 / 73.223472 | 0.760285 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | ap_s | 14.140929 / 15.231107 / 5.602377 / 19.918444 | 0.235682 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | mp_s | 31.429896 / 36.449147 / 4.397561 / 53.262857 | 0.523832 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | all | 10 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 42.630862 / 48.230485 / 8.578662 / 68.287933 | 0.710514 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | ap_s | 14.717978 / 16.022394 / 5.305217 / 21.964071 | 0.245300 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | mp_s | 27.869042 / 31.805477 / 3.232815 / 47.733075 | 0.464484 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | all | 10 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 160.828642 / 146.614557 / 134.019386 / 243.538241 | 2.680477 |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | ap_s | 63.459936 / 53.447291 / 47.696428 / 104.487291 | 1.057666 |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | mp_s | 97.299326 / 92.847014 / 86.260401 / 138.976942 | 1.621655 |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | ap_count | 3.200000 / 3.000000 / 3.000000 / 4.000000 | — |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | all | 10 | mp_count | 2.200000 / 2.000000 / 2.000000 / 3.000000 | — |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 392.621387 / 302.411178 / 206.875215 / 858.837719 | 6.543690 |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | ap_s | 186.651914 / 133.791227 / 97.079721 / 449.668825 | 3.110865 |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | mp_s | 205.853280 / 168.516531 / 107.553024 / 408.977900 | 3.430888 |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | ap_count | 5.100000 / 4.500000 / 4.000000 / 8.000000 | — |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | all | 10 | mp_count | 4.100000 / 3.500000 / 3.000000 / 7.000000 | — |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 604.883862 / 487.596322 / 329.856168 / 1339.569916 | 10.081398 |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | ap_s | 318.085327 / 239.852327 / 151.141598 / 750.161466 | 5.301422 |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | mp_s | 286.646255 / 247.608636 / 178.612495 / 589.148045 | 4.777438 |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | ap_count | 6.600000 / 6.000000 / 5.000000 / 10.000000 | — |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | all | 10 | mp_count | 5.600000 / 5.000000 / 4.000000 / 9.000000 | — |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | phase_sum_s | 837.722257 / 489.950119 / 321.756710 / 2924.834886 | 13.962038 |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | ap_s | 525.870073 / 265.648320 / 168.187252 / 2066.565468 | 8.764501 |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | mp_s | 311.653528 / 250.881713 / 126.195485 / 857.830094 | 5.194225 |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | ap_count | 8.400000 / 7.000000 / 5.000000 / 18.000000 | — |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | all | 10 | mp_count | 7.400000 / 6.000000 / 4.000000 / 17.000000 | — |

## Short-rule saved/current split by origin

| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |
|---|---:|---|---|---:|---|---|---:|
| Moreau mean-gap | 0 | completed / nominal_reference | nominal_reference | 10 | saved_phase_s | 1.929114 / 1.845262 / 1.627561 / 2.468371 | 0.032152 |
| Moreau mean-gap | 0 | completed / nominal_reference | nominal_reference | 10 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0 | completed / nominal_reference | nominal_reference | 10 | current_wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | continued | 10 | saved_phase_s | 1.870207 / 1.726003 / 1.517364 / 2.492965 | 0.031170 |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | continued | 10 | current_phase_s | 44.079488 / 41.928438 / 11.461699 / 81.063300 | 0.734658 |
| Moreau mean-gap | 0.001 | completed / mean_gap_satisfied | continued | 10 | current_wall_s | 44.707399 / 42.486203 / 11.780839 / 82.101939 | 0.745123 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | continued | 10 | saved_phase_s | 1.883026 / 1.922692 / 1.430179 / 2.389696 | 0.031384 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | continued | 10 | current_phase_s | 40.764477 / 38.946063 / 9.770900 / 82.787204 | 0.679408 |
| Moreau mean-gap | 0.003 | completed / mean_gap_satisfied | continued | 10 | current_wall_s | 41.439077 / 39.627529 / 10.062181 / 83.919830 | 0.690651 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | continued | 10 | saved_phase_s | 1.783913 / 1.738477 / 1.526059 / 2.345154 | 0.029732 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | continued | 10 | current_phase_s | 43.833181 / 49.521513 / 8.472423 / 71.697413 | 0.730553 |
| Moreau mean-gap | 0.01 | completed / mean_gap_satisfied | continued | 10 | current_wall_s | 44.502097 / 50.262687 / 8.733031 / 72.524454 | 0.741702 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | continued | 1 | saved_phase_s | 1.598010 / 1.598010 / 1.598010 / 1.598010 | 0.026633 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | continued | 1 | current_phase_s | 17.204022 / 17.204022 / 17.204022 / 17.204022 | 0.286734 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | continued | 1 | current_wall_s | 17.626305 / 17.626305 / 17.626305 / 17.626305 | 0.293772 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | recovered | 9 | saved_phase_s | 45.278510 / 51.578741 / 8.578662 / 68.287933 | 0.754642 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | recovered | 9 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0.03 | completed / mean_gap_satisfied | recovered | 9 | current_wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | recovered | 10 | saved_phase_s | 160.828642 / 146.614557 / 134.019386 / 243.538241 | 2.680477 |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | recovered | 10 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0.2 | completed / mean_gap_satisfied | recovered | 10 | current_wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | recovered | 10 | saved_phase_s | 392.621387 / 302.411178 / 206.875215 / 858.837719 | 6.543690 |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | recovered | 10 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0.4 | completed / mean_gap_satisfied | recovered | 10 | current_wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | recovered | 10 | saved_phase_s | 604.883862 / 487.596322 / 329.856168 / 1339.569916 | 10.081398 |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | recovered | 10 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 0.7 | completed / mean_gap_satisfied | recovered | 10 | current_wall_s | unavailable (rerun required) | — |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | recovered | 10 | saved_phase_s | 837.722257 / 489.950119 / 321.756710 / 2924.834886 | 13.962038 |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | recovered | 10 | current_phase_s | 0.000000 / 0.000000 / 0.000000 / 0.000000 | 0.000000 |
| Moreau mean-gap | 1.5 | completed / mean_gap_satisfied | recovered | 10 | current_wall_s | unavailable (rerun required) | — |

## UNMATCHED selected-radius SCS timing reruns — not the plotted SCS policies

These 10 completed/converged timing reruns used validation-selected radii/policies and 50 training samples versus 40 training / 10 validation for Moreau. The original run launched 30 concurrent jobs across three N sizes. They are **not timings for the fixed-radius blue curve**. Do not infer a solver-only speedup: policies, radii, training sizes and algorithms differ. AP/MP counts use trajectory status fields, not ccg_iters (which can identify an incumbent). Initialization was cached; nominal SCS training is reported separately. All statistics are computed from the actual timing files, never substituted with expected values.

| UNMATCHED overall metric | n | Mean / median / min / max | Mean minutes |
|---|---:|---|---:|
| wall_s | 10 | 2646.285778 / 2128.967273 / 449.094197 / 6244.696002 | 44.104763 |
| ap_s | 10 | 5.291798 / 5.455942 / 1.505831 / 9.522519 | 0.088197 |
| mp_s | 10 | 2640.983543 / 2123.451341 / 446.798551 / 6235.164121 | 44.016392 |
| ap_count | 10 | 3.800000 / 3.500000 / 2.000000 / 6.000000 | — |
| mp_count | 10 | 2.800000 / 2.500000 / 1.000000 / 5.000000 | — |

## UNMATCHED SCS by selected radius/status

| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |
|---|---:|---|---|---:|---|---|---:|
| SCS UNMATCHED selected-radius rerun | 0.001 | completed / converged | all | 1 | wall_s | 1092.076749 / 1092.076749 / 1092.076749 / 1092.076749 | 18.201279 |
| SCS UNMATCHED selected-radius rerun | 0.001 | completed / converged | all | 1 | ap_s | 2.444345 / 2.444345 / 2.444345 / 2.444345 | 0.040739 |
| SCS UNMATCHED selected-radius rerun | 0.001 | completed / converged | all | 1 | mp_s | 1089.624973 / 1089.624973 / 1089.624973 / 1089.624973 | 18.160416 |
| SCS UNMATCHED selected-radius rerun | 0.001 | completed / converged | all | 1 | ap_count | 3.000000 / 3.000000 / 3.000000 / 3.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.001 | completed / converged | all | 1 | mp_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.003 | completed / converged | all | 1 | wall_s | 655.170152 / 655.170152 / 655.170152 / 655.170152 | 10.919503 |
| SCS UNMATCHED selected-radius rerun | 0.003 | completed / converged | all | 1 | ap_s | 1.505831 / 1.505831 / 1.505831 / 1.505831 | 0.025097 |
| SCS UNMATCHED selected-radius rerun | 0.003 | completed / converged | all | 1 | mp_s | 653.655888 / 653.655888 / 653.655888 / 653.655888 | 10.894265 |
| SCS UNMATCHED selected-radius rerun | 0.003 | completed / converged | all | 1 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.003 | completed / converged | all | 1 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.15 | completed / converged | all | 2 | wall_s | 492.869401 / 492.869401 / 449.094197 / 536.644605 | 8.214490 |
| SCS UNMATCHED selected-radius rerun | 0.15 | completed / converged | all | 2 | ap_s | 2.307547 / 2.307547 / 2.284873 / 2.330221 | 0.038459 |
| SCS UNMATCHED selected-radius rerun | 0.15 | completed / converged | all | 2 | mp_s | 490.551105 / 490.551105 / 446.798551 / 534.303659 | 8.175852 |
| SCS UNMATCHED selected-radius rerun | 0.15 | completed / converged | all | 2 | ap_count | 2.000000 / 2.000000 / 2.000000 / 2.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.15 | completed / converged | all | 2 | mp_count | 1.000000 / 1.000000 / 1.000000 / 1.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.2 | completed / converged | all | 3 | wall_s | 3145.359672 / 2825.795753 / 1432.138794 / 5178.144471 | 52.422661 |
| SCS UNMATCHED selected-radius rerun | 0.2 | completed / converged | all | 3 | ap_s | 6.653391 / 6.896986 / 4.114752 / 8.948435 | 0.110890 |
| SCS UNMATCHED selected-radius rerun | 0.2 | completed / converged | all | 3 | mp_s | 3138.694586 / 2818.890684 / 1428.011997 / 5169.181078 | 52.311576 |
| SCS UNMATCHED selected-radius rerun | 0.2 | completed / converged | all | 3 | ap_count | 4.333333 / 4.000000 / 3.000000 / 6.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.2 | completed / converged | all | 3 | mp_count | 3.333333 / 3.000000 / 2.000000 / 5.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.4 | completed / converged | all | 1 | wall_s | 3844.525610 / 3844.525610 / 3844.525610 / 3844.525610 | 64.075427 |
| SCS UNMATCHED selected-radius rerun | 0.4 | completed / converged | all | 1 | ap_s | 6.797131 / 6.797131 / 6.797131 / 6.797131 | 0.113286 |
| SCS UNMATCHED selected-radius rerun | 0.4 | completed / converged | all | 1 | mp_s | 3837.715022 / 3837.715022 / 3837.715022 / 3837.715022 | 63.961917 |
| SCS UNMATCHED selected-radius rerun | 0.4 | completed / converged | all | 1 | ap_count | 5.000000 / 5.000000 / 5.000000 / 5.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.4 | completed / converged | all | 1 | mp_count | 4.000000 / 4.000000 / 4.000000 / 4.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.45 | completed / converged | all | 2 | wall_s | 5224.633724 / 5224.633724 / 4204.571445 / 6244.696002 | 87.077229 |
| SCS UNMATCHED selected-radius rerun | 0.45 | completed / converged | all | 2 | ap_s | 8.797705 / 8.797705 / 8.072891 / 9.522519 | 0.146628 |
| SCS UNMATCHED selected-radius rerun | 0.45 | completed / converged | all | 2 | mp_s | 5215.826788 / 5215.826788 / 4196.489455 / 6235.164121 | 86.930446 |
| SCS UNMATCHED selected-radius rerun | 0.45 | completed / converged | all | 2 | ap_count | 5.500000 / 5.500000 / 5.000000 / 6.000000 | — |
| SCS UNMATCHED selected-radius rerun | 0.45 | completed / converged | all | 2 | mp_count | 4.500000 / 4.500000 / 4.000000 / 5.000000 | — |

## Strict OOS evaluation by radius/status and invocation

| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |
|---|---:|---|---|---:|---|---|---:|
| OOS mean_gap | 0 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.950250 / 4.916411 / 4.803183 / 5.182075 | 0.082504 |
| OOS mean_gap | 0.001 | completed / strict_OPTIMAL | current_evaluation | 10 | wall_s | 5.053032 / 5.100957 / 4.678696 / 5.361926 | 0.084217 |
| OOS mean_gap | 0.003 | completed / strict_OPTIMAL | current_evaluation | 10 | wall_s | 4.931439 / 4.866395 / 4.772180 / 5.154005 | 0.082191 |
| OOS mean_gap | 0.01 | completed / strict_OPTIMAL | current_evaluation | 10 | wall_s | 4.904928 / 4.900100 / 4.771860 / 5.028610 | 0.081749 |
| OOS mean_gap | 0.03 | completed / strict_OPTIMAL | current_evaluation | 1 | wall_s | 4.795108 / 4.795108 / 4.795108 / 4.795108 | 0.079918 |
| OOS mean_gap | 0.03 | completed / strict_OPTIMAL | prior_evaluation | 9 | wall_s | 4.909740 / 4.900127 / 4.820112 / 5.006826 | 0.081829 |
| OOS mean_gap | 0.2 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.907051 / 4.904892 / 4.781459 / 5.094693 | 0.081784 |
| OOS mean_gap | 0.4 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.879101 / 4.876981 / 4.771529 / 4.963597 | 0.081318 |
| OOS mean_gap | 0.7 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.986345 / 4.993649 / 4.823834 / 5.193938 | 0.083106 |
| OOS mean_gap | 1.5 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.809166 / 4.792197 / 4.639488 / 5.083641 | 0.080153 |
| OOS terminal | 0 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 5.044505 / 5.034327 / 4.837062 / 5.223559 | 0.084075 |
| OOS terminal | 0.001 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.896795 / 4.900835 / 4.674373 / 5.164667 | 0.081613 |
| OOS terminal | 0.003 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.922167 / 4.879542 / 4.818928 / 5.169730 | 0.082036 |
| OOS terminal | 0.01 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.972427 / 4.967748 / 4.840788 / 5.182527 | 0.082874 |
| OOS terminal | 0.03 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.947257 / 4.928052 / 4.817691 / 5.152905 | 0.082454 |
| OOS terminal | 0.2 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.976670 / 4.981344 / 4.753182 / 5.193031 | 0.082944 |
| OOS terminal | 0.4 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.943463 / 4.914159 / 4.803091 / 5.161293 | 0.082391 |
| OOS terminal | 0.7 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.967452 / 4.933434 / 4.824115 / 5.265261 | 0.082791 |
| OOS terminal | 1.5 | completed / strict_OPTIMAL | prior_evaluation | 10 | wall_s | 4.834598 / 4.808038 / 4.666740 / 5.077993 | 0.080577 |

## Reproducibility

Raw phase durations, AP/MP counts and trajectories, source paths/SHA-256 hashes, nominal epochs, worker records, timestamps and evaluation identities are in runtime_comparison.json. Unknown metadata/timings remain null, not zero. Continuation records preserve cached_prefix, resume input/loading, selected policy evidence and current invocation totals. No cached timing is labeled as current full-model time. No failed/incomplete or capped configuration is mixed into a converged aggregate.

[Completed plot](matched_oos_completed.png) · [Plot data](plot_data.json) · [Raw timing and provenance](runtime_comparison.json) · [Reporting time](report_timing.json)
