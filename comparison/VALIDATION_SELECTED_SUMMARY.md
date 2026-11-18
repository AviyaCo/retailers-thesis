# Validation-selected retailer comparison — 6 October 2026

Read-only summary of existing results. No experiment, training, validation,
test evaluation, or hyperparameter selection was run to produce this report.
All numerical score entries average ten training seeds, each with its own
validation-selected radius; SAA-kNN instead uses per-seed leave-one-out
selection of k. Lower test cost is better. Black is excluded.

**Runtime definitions:** nominal fit is separate from the selected-radius CCG
trajectory. Neither includes the full hyperparameter search. SRO time is the
sum of the saved selected-candidate total (partition, model construction,
optimization, validation and associated overhead) and its test evaluation;
it excludes rejected radii, shared data generation, and outer serialization.
This is a sum of measured stages, not a continuously measured whole workflow.
Do not interpret these differing scopes as matched end-to-end speedups.

`Pending` means the required current selection/result is not complete or not
available in the inspected current publication. `Unmeasured` means a suitable
historical timing is missing; a new timed run is needed, not an estimate.
An em dash means the nominal-neural stage does not apply.

## 1. Shipping: R=3, fixed charge 0.1

Neural scores use the completed October 6 validation selection and the common
strict-Gurobi test evaluator with raw q>0 charges. SRO-FA and SAA-kNN retain
their historical thresholds (q>1e-10 and q>1e-6 respectively); they have NOT
been recomputed under the neural q>0 convention.

| Method | N | Mean test cost | Mean nominal fit | Mean selected-radius CCG / SRO time |
|---|---:|---:|---:|---:|
| SRO-FA | 50 | 21.6861 | — | 0.252 s |
| SRO-FA | 200 | 20.6898 | — | 6.260 s |
| SRO-FA | 800 | Pending | — | Pending |
| SAA-kNN | 50 | 21.4575 | — | Unmeasured |
| SAA-kNN | 200 | 20.7016 | — | Unmeasured |
| SAA-kNN | 800 | 20.4992 | — | Unmeasured |
| Neural CCG / Clarabel | 50 | 20.6042 | Unmeasured | 139.07 min |
| Neural CCG / Clarabel | 200 | 20.0912 | Unmeasured separately | 303.34 min |
| Neural CCG / Clarabel | 800 | 19.8580 | Unmeasured separately | 257.63 min |
| Moreau green / STOP | 50 | 20.8173 | 91.40 s | 2.17 min |
| Moreau green / STOP | 200 | 20.3424 | 109.25 s | 5.71 min |
| Moreau green / STOP | 800 | 19.9670 | 131.55 s | 9.71 min |

Shipping SRO-FA N800 has an incomplete saved validation sweep, with two seed
entries and no completed ten-seed selected-test aggregate. The saved word
`running` is historical state, not evidence of a currently live process.
Clarabel nominal workflow means do exist for N200/N800: 24,217.077609 s
(6.73 h) and 83,568.354220 s (23.21 h). These bundle SAA initialization,
CV, fitting and hard evaluation, so are not isolated nominal-fit times.
All thirty selected Clarabel trajectories report convergence. All thirty
selected green trajectories have completed uninterrupted CCG timing records;
none is one of the five resumed shipping trajectories. The full green grid
still retains its documented mixed original/continuation provenance.

## 2. R3: no shipping charge

| Method | N | Mean test cost | Mean nominal fit | Mean selected-radius CCG / SRO time |
|---|---:|---:|---:|---:|
| SRO-FA | 50 | 21.2344 | — | 0.087 s |
| SRO-FA | 200 | 20.4744 | — | 0.383 s |
| SRO-FA | 800 | 19.9429 | — | 2.527 s |
| SAA-kNN | 50 | 20.7977 | — | Unmeasured |
| SAA-kNN | 200 | 20.2686 | — | Unmeasured |
| SAA-kNN | 800 | 19.9988 | — | Unmeasured |
| Neural CCG / SCS | 50 | 20.4124 | 347.42 s | 44.10 min |
| Neural CCG / SCS | 200 | 19.6970 | 2,507.35 s | 165.62 min |
| Neural CCG / SCS | 800 | 19.5756 | 10,851.85 s | 291.23 min |
| Moreau green / STOP | 50 | Pending | 37.62 s | Pending selection |
| Moreau green / STOP | 200 | Pending | 149.80 s | Pending selection |
| Moreau green / STOP | 800 | Pending | 170.07 s | Pending selection |

SCS nominal and selected-radius CCG times are separate historical timing
reruns, not recovered creation times of the test-scored checkpoints. All
thirty timed selected-radius trajectories report convergence. Do not add
these independently run cohorts and label the result observed end-to-end time.

## 3. R10: no shipping charge

| Method | N | Mean test cost | Mean nominal fit | Mean selected-radius CCG / SRO time |
|---|---:|---:|---:|---:|
| SRO-FA | 50 | 73.5547 | — | 0.555 s |
| SRO-FA | 200 | 72.5167 | — | 21.525 s |
| SRO-FA | 800 | 72.0639 | — | 526.045 s (8.77 min) |
| SAA-kNN | 50 | 75.4717 | — | Unmeasured |
| SAA-kNN | 200 | 72.8765 | — | Unmeasured |
| SAA-kNN | 800 | 70.4544 | — | Unmeasured |
| Neural CCG / SCS | 50 | 71.6681 | Unmeasured separately | 577.12 min |
| Neural CCG / SCS | 200 | 68.6142 | Unmeasured separately | 613.97 min |
| Neural CCG / SCS | 800 | 68.5410 | Unmeasured separately | 560.63 min |
| Moreau green / STOP | 50 | Pending | 113.65 s | Pending selection |
| Moreau green / STOP | 200 | Pending | 161.97 s | Pending selection |
| Moreau green / STOP | 800 | Pending | 198.56 s | Pending selection |

All thirty selected SCS trajectories report convergence. The nominal-directory
`total_time_s` covers a seed workflow that also runs CCG; it must not be
relabeled isolated nominal fitting. Eight N800 nominal summaries are absent.

## Why the green R3/R10 scores are pending

The current September 28 publication evaluates STOP checkpoints but explicitly
does not perform validation/radius selection. Its reporter removes the old
validation-selected claims from current results and preserves them only as
historical minimum-AP metadata. Current fixed-radius STOP test curves exist;
they are not validation-selected test scores. The old radius winners must not
be silently transferred to changed policies. No new selection is launched here.

## Evaluation and selection runtime coverage

All following values are seconds, with size order N50/N200/N800.

- Shipping green: mean full saved-policy validation grid per seed
  0.248 / 0.342 / 0.888 s; selected-policy original test evaluation
  5.301 / 5.017 / 4.986 s. These evaluation times are not training times.
- Shipping Clarabel: corresponding validation grid means
  0.272 / 0.575 / 1.799 s; test means 4.964 / 4.926 / 5.104 s.
- October 6 shipping selection: eight single-thread workers, parallel validation
  9.087025 s; complete child 48.780893 s (0.813015 min). Test was retrieved from
  verified existing evaluations; lookup time is not a fresh evaluation duration.
- R3 SCS: validation stage for all ten seeds per size
  1.769356 / 4.033970 / 12.582113 s. R10 SCS:
  1.533543 / 4.165266 / 15.171312 s. Their selected test costs were cached;
  selection-time cache reads do not measure original test evaluation.
- R3 SCS shared LR-search invocations: 1,101.984 / 4,266.084 /
  24,822.863 s, separate from both nominal fit and CCG timings.
- SRO full candidate-grid measured stage sums, mean per seed (not single
  selected-model time): shipping 3.334260 / 73.519472 / incomplete;
  R3 1.402788 / 6.357511 / 87.889738;
  R10 2.819214 / 47.374821 / 5,998.930246 s.
- SRO selected test means: shipping 0.018018 / 0.066441 / incomplete;
  R3 0.020476 / 0.066055 / 0.255624;
  R10 0.024112 / 0.087766 / 0.265770 s. Already included in SRO table times.
- SAA-kNN fitting, LOO k selection and test timing are unmeasured separately.
  Missing historical timings require reruns; none were performed here.

## Runtime distribution details

Each entry below has **n=10** and gives **mean; median; minimum–maximum**, all
in seconds. These are per-model durations, not sums of concurrent elapsed time.
All selected SRO solves have recorded OPTIMAL status (shipping within its MIP
tolerance, not necessarily zero gap). Incomplete shipping N800 is excluded.

- Shipping SRO N50: 0.252339; 0.222185; 0.085673–0.560088.
- Shipping SRO N200: 6.259819; 5.254989; 1.283430–15.606822.
- R3 SRO N50: 0.087207; 0.080251; 0.055018–0.147751.
- R3 SRO N200: 0.382707; 0.369850; 0.195498–0.682725.
- R3 SRO N800: 2.526570; 2.657837; 2.054836–2.889342.
- R10 SRO N50: 0.554696; 0.651078; 0.262940–0.747286.
- R10 SRO N200: 21.524999; 26.299162; 5.714693–31.295771.
- R10 SRO N800: 526.045185; 502.945202; 427.587526–687.634860.
- Shipping Clarabel CCG N50: 8344.110000; 8840.715000; 1047.120000–13975.620000.
- Shipping Clarabel CCG N200: 18200.191000; 15615.215000; 6859.120000–36596.420000.
- Shipping Clarabel CCG N800: 15457.731000; 8560.020000; 6870.630000–36242.280000.
- Shipping green CCG N50: 130.230202; 3.526181; 3.333975–505.752991.
- Shipping green CCG N200: 342.556705; 388.999777; 8.125651–758.685147.
- Shipping green CCG N800: 582.778323; 623.012168; 28.386114–936.612680.
- R3 SCS timed CCG N50: 2646.285778; 2128.967273; 449.094197–6244.696002.
- R3 SCS timed CCG N200: 9937.463116; 6999.168646; 2389.345664–27267.185110.
- R3 SCS timed CCG N800: 17473.783280; 13346.123564; 10007.221740–41812.691190.
- R10 SCS CCG N50: 34627.197323; 38243.707518; 4863.189397–72379.820295.
- R10 SCS CCG N200: 36838.019606; 29632.035683; 16782.042257–88941.558927.
- R10 SCS CCG N800: 33637.576308; 30647.957244; 11698.235006–71093.089844.

Nominal fit distributions, same units and ordering:

- Shipping green N50: 91.401937; 97.809597; 54.809915–105.290021.
- Shipping green N200: 109.253598; 109.106485; 75.079683–129.524771.
- Shipping green N800: 131.554789; 119.575288; 82.101536–175.521144.
- R3 green N50: 37.620406; 37.241888; 29.626559–45.877120.
- R3 green N200: 149.801540; 149.807060; 149.042159–150.498188.
- R3 green N800: 170.069779; 171.988134; 157.940169–174.611765.
- R10 green N50: 113.648442; 125.831726; 83.229825–129.866195.
- R10 green N200: 161.968598; 169.706665; 96.384184–170.891570.
- R10 green N800: 198.557159; 215.968078; 121.084237–216.451620.
- R3 SCS nominal rerun N50: 347.422133; 369.971440; 243.749316–405.271895.
- R3 SCS nominal rerun N200: 2507.346407; 2669.526587; 1421.100739–3178.310878.
- R3 SCS nominal rerun N800: 10851.847154; 11037.180511; 8835.194680–11777.323840.

## Interpretation and reproducibility

The test is the historical 10,000-scenario seed-9999 draw, already inspected;
historical grids were test-informed. These are descriptive results, not fresh
confirmatory evidence. Green uses 40/160/640 training observations plus
10/40/160 validation observations. Legacy neural and SRO use 50/200/800
training observations and independent validation of size 13/50/200, seed
88880+s. kNN's LOO excludes a neighbor but does not refit the full SAA label
pipeline per fold. No new leakage repair or experimental change was made here.

Known environment: Linux gold-srv2, dual Xeon Gold 6354, 72 logical CPUs,
approximately 502 GiB RAM. Historical per-model resource allocation and load
are not consistently known. Shipping green CCG used 62 single-thread workers;
R3 selected SCS timing reruns used up to 30 workers; the SRO selection solves
record one solver thread. Different protocols, concurrency, stopping and
historical timing boundaries preclude a controlled solver-speedup claim.
Overall elapsed time for every original sweep is not available; it is not
reconstructed by summing overlapping model durations.

## Source artifacts and update locations

- [Shipping completed current selection](../workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_validation_qpositive_20261006/COMPLETION.md), [scores](../workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_validation_qpositive_20261006/production_v1/score_summary.json), [selected test and training-time references](../workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_validation_qpositive_20261006/production_v1/selected_test_results.json), [validation/evaluation timing](../workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_validation_qpositive_20261006/production_v1/runtime_report.json).
- [R3 neural selections](../workspace/05_history/original_paths/retailers_problem/R3/rho_validation_results_net16_v2.json), [R10 neural selections](../workspace/05_history/original_paths/retailers_problem/R10/rho_validation_results_net16_v2.json).
- [R3 SRO selections and raw stage times](../workspace/05_history/original_paths/retailers_problem/R3/sro_fa_rho_validation_results.json), [R10 SRO selections and raw stage times](../workspace/05_history/original_paths/retailers_problem/R10/sro_fa_rho_validation_results.json), [shipping SRO selections and incomplete N800](../workspace/05_history/original_paths/retailers_with_shipping/sro_fa_rho_validation_results.json).
- [Saved kNN seed costs and selected k](../workspace/05_history/original_paths/retailers_problem/neural_vs_knn_paired_tests.json).
- [R3 SCS selected timing campaign and per-job summary paths](../workspace/05_history/original_paths/retailers_problem/R3/selected_timing_net16_v2_run2/campaign_runtime.json), [R3 nominal timing scope](../workspace/05_history/original_paths/retailers_moreau_training_selected/TIMING_COMPARISON_20260924.md).
- R10 SCS per-selected-seed runtime: the corresponding N directory, then rho_sweep_net16_v2/rho_<three-decimal-radius>/seed_<seed>_summary.json, field total_time_s. Radius and seed are joined to the validation selection above, never chosen using test cost.
- [Current STOP protocol without radius selection](stopping_checkpoint_20260928/README.md), [explicit exclusion of historical validation claims](stopping_checkpoint_20260928/report.py#L249-L265), [green nominal timing distributions](stopping_checkpoint_20260928/production_v1/published/runtime_report.json).

Update pending cells only after completed validation-selected results for the
corresponding policies exist. Do not fill them with fixed-radius test minima,
old minimum-AP selections, partial-seed averages, or cache-loading durations.