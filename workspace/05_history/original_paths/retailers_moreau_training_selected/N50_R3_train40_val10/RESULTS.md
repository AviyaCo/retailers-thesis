# Retailer training-selected rerun results

Completed 2026-09-16T12:18:09.732301+00:00; overall elapsed **15092.965248s (251.549421min)**. Fresh30 nominal trainings,90 independent CCG configurations; no test evaluation.

Nominal and MP checkpoints, scheduler and early stopping use training loss. Validation chooses the nominal LR after training and the radius after all terminal policies are fixed. The40/10 split, ten seeds, nine radii, zero capacity buffer and native Moreau/AP numerical settings are unchanged. Historical results are not overwritten.

## Validation (selection data, not an independent test)

Nominal LR-selected mean: **18.948014549**. Final radius-selected mean: **18.700114478**. These scores reuse ten validation observations per seed for LR/radius selection; they do not establish generalization improvement or a U-shaped population curve.

| Radius | Validation mean | Initial AP mean at this radius | Terminal AP mean | Terminal AP upper mean | MPs | Separated / stagnated / capped | Nominal weights retained |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 18.948014549 | 19.365727199 | 19.365727199 | 19.365727199 | 0 | 10 / 0 / 0 | 10 |
| 0.001 | 18.948014549 | 19.367903467 | 19.367903467 | 19.367936704 | 0 | 10 / 0 / 0 | 10 |
| 0.003 | 18.948014549 | 19.372596542 | 19.372596542 | 19.372624375 | 0 | 10 / 0 / 0 | 10 |
| 0.01 | 18.948014549 | 19.390358248 | 19.390358248 | 19.390373997 | 0 | 10 / 0 / 0 | 10 |
| 0.03 | 19.069058415 | 19.447943159 | 19.334074276 | 19.334108335 | 18 | 10 / 0 / 0 | 2 |
| 0.2 | 18.970495743 | 20.112295833 | 19.621640120 | 19.621661560 | 66 | 10 / 0 / 0 | 0 |
| 0.4 | 19.064397193 | 20.982105226 | 20.030264232 | 20.030297082 | 102 | 10 / 0 / 0 | 0 |
| 0.7 | 19.279519314 | 22.380348556 | 20.804326582 | 20.804338349 | 208 | 9 / 0 / 1 | 0 |
| 1.5 | 19.701545068 | 26.069258787 | 23.306128994 | 23.306166342 | 194 | 10 / 0 / 0 | 0 |

AP comparisons are paired at the same radius and seed. Increasing AP costs across nested boxes is not a failed U-shape. AP bounds concern the encoded numerical policy approximation, not uniform native-policy or global optimization certification. Stagnated/capped terminal policies remain included with their statuses—not relabelled converged.

## MP checkpoint audit

588 MPs: 577 selected objectives improved, 0 worsened, and 11 returned unchanged weights. Each comparison uses incoming and returned states on the same augmented pool. Selected states are checked against the saved checkpoint and reevaluated before the next AP. This prevents validation restoration from choosing a worse training objective, but does not guarantee full-box or held-out improvement.

## Runtime accounting

Linux gold-srv2, Intel Xeon Gold6354,72 logical CPU slots. Nominal10 workers then CCG16 workers; Torch/Gurobi single-thread; final validation1 worker. Stored unit: wall seconds, monotonic perf_counter. Concurrent model walls are not sweep elapsed; nested phases are not additive.

| Component | Elapsed seconds | Minutes |
|---|---:|---:|
| nominal phase | 121.796206 | 2.029937 |
| ccg phase | 14942.580206 | 249.043003 |
| validation phase | 1.658737 | 0.027646 |

| Nominal training cohort | n | Mean seconds | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Selected LR winners | 10 | 37.620406 | 37.241888 | 29.626559 | 45.877120 |
| LR0.005 | 10 | 38.008518 | 38.378013 | 21.875551 | 45.877120 |
| LR0.01 | 10 | 35.664763 | 35.498384 | 29.626559 | 40.610397 |
| LR0.05 | 10 | 31.841771 | 33.161544 | 22.793271 | 36.974832 |

Winner-only training excludes rejected LRs; full nominal-phase elapsed above includes initialization, all30 trainings, subsequent validation and LR selection.

| Radius | n | Mean CCG seconds | Median | Min | Max | Mean AP | Mean MP | Mean pool-evaluation | Mean final-validation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 2.087166 | 1.999984 | 1.756980 | 2.684249 | 1.902812 | 0.000000 | 0.005794 | 0.002809 |
| 0.001 | 10 | 2.027726 | 1.862810 | 1.680292 | 2.686424 | 1.843538 | 0.000000 | 0.005310 | 0.002243 |
| 0.003 | 10 | 2.031718 | 2.071909 | 1.572742 | 2.527521 | 1.856880 | 0.000000 | 0.006077 | 0.002347 |
| 0.01 | 10 | 1.947422 | 1.899755 | 1.643105 | 2.490503 | 1.759159 | 0.000000 | 0.005165 | 0.002346 |
| 0.03 | 10 | 98.535294 | 82.642768 | 1.711565 | 208.361424 | 42.129196 | 51.195405 | 0.038604 | 0.002362 |
| 0.2 | 10 | 715.089333 | 624.722135 | 338.277539 | 1597.338762 | 387.275338 | 279.325860 | 0.153644 | 0.002446 |
| 0.4 | 10 | 1412.567801 | 1400.907108 | 889.616289 | 2300.128266 | 852.338250 | 453.705350 | 0.245393 | 0.002732 |
| 0.7 | 10 | 4757.063014 | 4359.539535 | 1230.665084 | 11411.629000 | 3353.552807 | 985.973214 | 0.546372 | 0.002358 |
| 1.5 | 10 | 3859.724695 | 3056.511687 | 734.514130 | 9307.252776 | 2693.192329 | 829.227398 | 0.541263 | 0.002287 |

CCG totals exclude nominal training. AP includes native replay and persistence; MP includes training snapshots/journals. Radius0 is nominal-policy assessment, not additional nominal training. Completed execution is not synonymous with separation: capped/stagnated outcomes are included in the descriptive all-executed table and stratified below.

| Radius | Terminal status | n | Mean CCG seconds | Median | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 0 | separation_satisfied | 10 | 2.087166 | 1.999984 | 1.756980 | 2.684249 |
| 0.001 | separation_satisfied | 10 | 2.027726 | 1.862810 | 1.680292 | 2.686424 |
| 0.003 | separation_satisfied | 10 | 2.031718 | 2.071909 | 1.572742 | 2.527521 |
| 0.01 | separation_satisfied | 10 | 1.947422 | 1.899755 | 1.643105 | 2.490503 |
| 0.03 | separation_satisfied | 10 | 98.535294 | 82.642768 | 1.711565 | 208.361424 |
| 0.2 | separation_satisfied | 10 | 715.089333 | 624.722135 | 338.277539 | 1597.338762 |
| 0.4 | separation_satisfied | 10 | 1412.567801 | 1400.907108 | 889.616289 | 2300.128266 |
| 0.7 | separation_satisfied | 9 | 4017.666794 | 4090.432882 | 1230.665084 | 8251.298251 |
| 0.7 | iteration_cap | 1 | 11411.629000 | 11411.629000 | 11411.629000 | 11411.629000 |
| 1.5 | separation_satisfied | 10 | 3859.724695 | 3056.511687 | 734.514130 | 9307.252776 |

| Other component | Status |
|---|---|
| SRO-FA | Not run |
| CV | Not run; fixed40/10 split |
| Independent test | Not evaluated |
| Failed/interrupted campaign attempts | Separate timestamped artifacts; never counted as completed reruns |

## Reproducibility

Started 2026-09-16T08:06:36.767038+00:00; ended 2026-09-16T12:18:09.732301+00:00. Exact command: `/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py`. Cwd this folder. Named tmux session `retailers_training_selected_20260916`; launch `tmux new-session -d -s retailers_training_selected_20260916 -c /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 bash launch.sh`. While active attach `tmux attach -t retailers_training_selected_20260916`; check session `tmux has-session -t retailers_training_selected_20260916`. Completed sessions disappear; persistent logs and exit artifacts remain.

- [results/20260916T080636.766867Z/summary.json](results/20260916T080636.766867Z/summary.json)
- [results/20260916T080636.766867Z/aggregates.json](results/20260916T080636.766867Z/aggregates.json)
- [results/20260916T080636.766867Z/provenance.json](results/20260916T080636.766867Z/provenance.json)
- [results/20260916T080636.766867Z/pre_validation_terminal_policies.json](results/20260916T080636.766867Z/pre_validation_terminal_policies.json)
- [Preflight and protocol review](PRELAUNCH_REVIEW.md)
- [Thesis notes](../../unit_commitment/two_stage/uc_surrogate_mp_experiment/thesis_modeling_notes.tex)

- [campaign_20260916T080634.exit](campaign_20260916T080634.exit)
- [campaign_20260916T080634.log](campaign_20260916T080634.log)
