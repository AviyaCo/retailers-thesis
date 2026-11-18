# Radius re-selection under the rule of 2026-11-18

Rule: **largest radius within 1e-6 of the minimum validation score** (replaces "minimum validation score; exact tie -> smaller radius"). No policy was retrained or re-evaluated:
every test cost is a stored evaluation of the same policy with the same evaluator on the same
10,000-scenario seed-9999 test set. Per-seed records (old/new radius, tied set and scores, gaps, test
files): `selections_new_rule.json`. Created by `reselect.py`.

## Tolerance check

- 422 selections over all sources and seeds. The stored choice is reproduced by the old rule in 421 of 421 cases (1 legacy selection(s) have no stored choice).
- Smallest nonzero gap between the best and second-best validation score: 5.15e-04; genuine gaps at or below 1e-6: 0.
- All ties are exact (bit-identical scores): True. No tie arises from rounding; the old rule did fire, always choosing the smaller radius.

## Reported Protocol A selections, per setting

| Method | Setting | N | Changed seeds | Test cost, old rule (mean ± sd) | Test cost, new rule (mean ± sd) | Selected radii, old | Selected radii, new |
|---|---|---:|---:|---:|---:|---|---|
| Neural CCG | R10 | 50 | 0/10 | 73.0281 ± 1.7643 | 73.0281 ± 1.7643 | {'0.2': 1, '0.4': 3, '0.7': 1, '1.5': 5} | {'0.2': 1, '0.4': 3, '0.7': 1, '1.5': 5} |
| Neural CCG | R10 | 200 | 0/10 | 69.2914 ± 0.6228 | 69.2914 ± 0.6228 | {'0.2': 3, '0.4': 2, '0.7': 5} | {'0.2': 3, '0.4': 2, '0.7': 5} |
| Neural CCG | R10 | 800 | 0/10 | 68.2466 ± 0.4328 | 68.2466 ± 0.4328 | {'0.2': 2, '0.4': 3, '0.7': 5} | {'0.2': 2, '0.4': 3, '0.7': 5} |
| Neural CCG | R3 | 50 | 4/10 | 20.5449 ± 0.6490 | 20.5449 ± 0.6490 | {'0': 4, '0.2': 2, '0.4': 3, '1.5': 1} | {'0.03': 4, '0.2': 2, '0.4': 3, '1.5': 1} |
| Neural CCG | R3 | 200 | 3/10 | 19.8664 ± 0.2912 | 19.8664 ± 0.2912 | {'0': 3, '0.2': 2, '0.4': 5} | {'0.03': 3, '0.2': 2, '0.4': 5} |
| Neural CCG | R3 | 800 | 4/10 | 19.6818 ± 0.1391 | 19.6818 ± 0.1391 | {'0': 4, '0.2': 1, '0.4': 4, '0.7': 1} | {'0.03': 4, '0.2': 1, '0.4': 4, '0.7': 1} |
| Neural CCG | R3_shipping | 50 | 6/10 | 20.8173 ± 0.4446 | 20.8173 ± 0.4446 | {'0': 6, '0.2': 1, '0.4': 1, '0.7': 2} | {'0.03': 6, '0.2': 1, '0.4': 1, '0.7': 2} |
| Neural CCG | R3_shipping | 200 | 3/10 | 20.3424 ± 0.2478 | 20.3424 ± 0.2478 | {'0': 3, '0.2': 3, '0.4': 3, '0.7': 1} | {'0.03': 3, '0.2': 3, '0.4': 3, '0.7': 1} |
| Neural CCG | R3_shipping | 800 | 1/10 | 19.9670 ± 0.1172 | 19.9670 ± 0.1172 | {'0': 1, '0.2': 4, '0.4': 3, '0.7': 2} | {'0.03': 1, '0.2': 4, '0.4': 3, '0.7': 2} |
| SRO-FA | R10 | 50 | 0/10 | 73.8966 ± 0.9076 | 73.8966 ± 0.9076 | {'0.75': 1, '1': 4, '1.5': 5} | {'0.75': 1, '1': 4, '1.5': 5} |
| SRO-FA | R10 | 200 | 0/10 | 72.5903 ± 0.2273 | 72.5903 ± 0.2273 | {'1.5': 10} | {'1.5': 10} |
| SRO-FA | R10 | 800 | 0/10 | 72.2893 ± 0.1127 | 72.2893 ± 0.1127 | {'0.8': 1, '1': 8, '1.5': 1} | {'0.8': 1, '1': 8, '1.5': 1} |
| SRO-FA | R3 | 50 | 0/10 | 21.3598 ± 0.3181 | 21.3598 ± 0.3181 | {'0.2': 1, '0.45': 1, '0.55': 1, '0.6': 1, '0.65': 1, '0.7': 2, '0.75': 2, '1': 1} | {'0.2': 1, '0.45': 1, '0.55': 1, '0.6': 1, '0.65': 1, '0.7': 2, '0.75': 2, '1': 1} |
| SRO-FA | R3 | 200 | 0/10 | 20.5517 ± 0.1349 | 20.5517 ± 0.1349 | {'0.2': 2, '0.4': 1, '0.45': 1, '0.5': 1, '0.6': 2, '0.75': 2, '0.8': 1} | {'0.2': 2, '0.4': 1, '0.45': 1, '0.5': 1, '0.6': 2, '0.75': 2, '0.8': 1} |
| SRO-FA | R3 | 800 | 0/10 | 19.9866 ± 0.0486 | 19.9866 ± 0.0486 | {'0.3': 2, '0.4': 1, '0.45': 1, '0.5': 5, '0.6': 1} | {'0.3': 2, '0.4': 1, '0.45': 1, '0.5': 5, '0.6': 1} |
| SRO-FA | R3_shipping | 50 | 0/10 | 21.5533 ± 0.2647 | 21.5533 ± 0.2647 | {'0.5': 2, '0.7': 1, '0.8': 1, '0.9': 4, '1.2': 1, '1.3': 1} | {'0.5': 2, '0.7': 1, '0.8': 1, '0.9': 4, '1.2': 1, '1.3': 1} |
| SRO-FA | R3_shipping | 200 | 0/10 | 20.8645 ± 0.1155 | 20.8645 ± 0.1155 | {'0.2': 2, '0.3': 2, '0.4': 1, '0.6': 3, '0.7': 2} | {'0.2': 2, '0.3': 2, '0.4': 1, '0.6': 3, '0.7': 2} |
| SRO-FA | R3_shipping | 800 | 0/10 | 20.2395 ± 0.0360 | 20.2395 ± 0.0360 | {'0.3': 1, '0.4': 3, '0.5': 5, '0.7': 1} | {'0.3': 1, '0.4': 3, '0.5': 5, '0.7': 1} |
| SRO-FA+LDR (not reported) | R3 | 50 | 0/10 | 21.3014 ± 0.4283 | 21.3014 ± 0.4283 | {'0.4': 1, '0.5': 1, '0.6': 1, '0.7': 1, '0.8': 2, '1': 3, '1.5': 1} | {'0.4': 1, '0.5': 1, '0.6': 1, '0.7': 1, '0.8': 2, '1': 3, '1.5': 1} |
| SRO-FA+LDR (not reported) | R3 | 200 | 0/10 | 20.5451 ± 0.1309 | 20.5451 ± 0.1309 | {'0.3': 1, '0.4': 1, '0.5': 2, '0.6': 3, '0.75': 2, '0.8': 1} | {'0.3': 1, '0.4': 1, '0.5': 2, '0.6': 3, '0.75': 2, '0.8': 1} |
| SRO-FA+LDR (not reported) | R3 | 800 | 0/10 | 19.9650 ± 0.0379 | 19.9650 ± 0.0379 | {'0.4': 2, '0.5': 4, '0.6': 3, '0.75': 1} | {'0.4': 2, '0.5': 4, '0.6': 3, '0.75': 1} |

## Every changed selection

| Source | Setting | N | Seed | Old | New | Tied radii (all at the score) | Tied score | Same policy | Test cost old → new | Stored test evaluation used |
|---|---|---:|---:|---:|---:|---|---:|---|---|---|
| neural_ccg | R3 | 50 | 4 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.079470 | True | 20.365244 → 20.365244 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N50_s4_rho0.03/result.json` |
| neural_ccg | R3 | 50 | 5 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 17.165653 | True | 20.600806 → 20.600806 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N50_s5_rho0.03/result.json` |
| neural_ccg | R3 | 50 | 6 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 21.196551 | True | 20.095758 → 20.095758 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N50_s6_rho0.03/result.json` |
| neural_ccg | R3 | 50 | 7 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 14.158781 | True | 21.534351 → 21.534351 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N50_s7_rho0.03/result.json` |
| neural_ccg | R3 | 200 | 6 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 19.360797 | True | 19.691109 → 19.691109 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N200_s6_rho0.03/result.json` |
| neural_ccg | R3 | 200 | 7 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.698747 | True | 20.681116 → 20.681116 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N200_s7_rho0.03/result.json` |
| neural_ccg | R3 | 200 | 8 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.079249 | True | 19.814484 → 19.814484 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N200_s8_rho0.03/result.json` |
| neural_ccg | R3 | 800 | 1 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.288593 | True | 19.929321 → 19.929321 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N800_s1_rho0.03/result.json` |
| neural_ccg | R3 | 800 | 2 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.540628 | True | 19.563415 → 19.563415 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N800_s2_rho0.03/result.json` |
| neural_ccg | R3 | 800 | 3 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.215782 | True | 19.581627 → 19.581627 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N800_s3_rho0.03/result.json` |
| neural_ccg | R3 | 800 | 9 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.272160 | True | 19.924502 → 19.924502 | `comparison/stopping_checkpoint_20260928/production_v1/jobs/green_R3_N800_s9_rho0.03/result.json` |
| neural_ccg | R3_shipping | 200 | 1 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 22.530628 | True | 20.261377 → 20.261377 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N200_r0p03_s1/result.json` |
| neural_ccg | R3_shipping | 200 | 5 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 19.885950 | True | 20.246324 → 20.246324 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N200_r0p03_s5/result.json` |
| neural_ccg | R3_shipping | 200 | 6 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.369166 | True | 20.639064 → 20.639064 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N200_r0p03_s6/result.json` |
| neural_ccg | R3_shipping | 50 | 1 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.587061 | True | 21.598501 → 21.598501 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s1/result.json` |
| neural_ccg | R3_shipping | 50 | 2 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 22.041174 | True | 20.272913 → 20.272913 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s2/result.json` |
| neural_ccg | R3_shipping | 50 | 4 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.166581 | True | 20.986390 → 20.986390 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s4/result.json` |
| neural_ccg | R3_shipping | 50 | 6 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 21.702588 | True | 20.437394 → 20.437394 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s6/result.json` |
| neural_ccg | R3_shipping | 50 | 8 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 20.207844 | True | 20.593589 → 20.593589 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s8/result.json` |
| neural_ccg | R3_shipping | 50 | 9 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 18.994847 | True | 20.490985 → 20.490985 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N50_r0p03_s9/result.json` |
| neural_ccg | R3_shipping | 800 | 5 | 0 | 0.03 | 0, 0.001, 0.003, 0.01, 0.03 | 19.723967 | True | 19.881709 → 19.881709 | `workspace/05_history/original_paths/retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/evaluations/green_N800_r0p03_s5/result.json` |
| legacy_blue_nn | R3 | 50 | 6 | 0.001 | 0.002 | 0.001, 0.0015, 0.002 | 16.265156 | same test cost at all tied radii | 21.009016 → 21.009016 | `workspace/05_history/original_paths/retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json` |

## Affected timing entry (Experiment 1, Protocol A, "Neural selected-radius CCG")

Mean over ten seeds of the recorded CCG active time through the stopping checkpoint of the selected run
(historical seconds, excluding nominal training). The newly selected ρ=0.03 runs return the same policy
as ρ=0 but are separate CCG runs with their own recorded adversarial-verification time.

| N | Old rule | New rule |
|---:|---:|---:|
| 50 | 189.9 | 189.8 |
| 200 | 294.3 | 294.4 |
| 800 | 369.9 | 370.8 |

Re-selection wall time: 0.90 s, one process; stored-record lookups only.
