# Paired statistical tests: final NN policy vs SRO-FA

Paired statistical comparison of the final NN policy and SRO-FA, R=3 and R=10 (2026-11-02).

Test A (between seeds): for each data seed s = 0..9 both methods are trained on the SAME N training
samples (`np.random.seed(s)`, the same truncated-normal generator; verified by hash below) and evaluated on
the SAME 10,000 test scenarios (seed 9999). The unit of analysis is the per-seed out-of-sample mean
cost; the paired difference is d_s = NN_s - SRO-FA_s (negative = NN cheaper). Reported per comparison:
mean and SE of d, wins, a two-sided paired t-test and an exact two-sided Wilcoxon signed-rank test (n=10),
and a Holm correction within each family of comparisons.

Comparisons: (1) matched radius, for every radius shared by both grids; (2) each method at its own
best fixed radius on this test (hindsight; the same rule for both methods, so descriptive only). No
radius is selected on validation data in either saved campaign.

Sources (read-only): final full-N NN plot data (per-seed OOS means, SHA-256 pinned in render/full_n.py)
and the historical SRO-FA result JSONs (per-seed OOS means). No training or evaluation is performed.
The test set has been used before, so these are descriptive statistics on a known test set, not a fresh
holdout.

## R=3, N=50

Best fixed radius on this test (hindsight, same rule for both): NN rho=0.4 20.4958 vs SRO-FA rho=0.6 21.0950; NN - SRO-FA = -0.5992 (SE 0.1233); NN better in 9/10 seeds; paired t p=0.0009; Wilcoxon exact p=0.0059.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 20.6133 | 21.9584 | -1.3451 (0.1417) | 10/10 | 5.5e-06 (2.7e-05) | 0.002 (0.014) |
| 0.003 | 20.6133 | 21.9654 | -1.3520 (0.1389) | 10/10 | 4.5e-06 (2.7e-05) | 0.002 (0.014) |
| 0.01 | 20.6133 | 21.9530 | -1.3397 (0.1376) | 10/10 | 4.5e-06 (2.7e-05) | 0.002 (0.014) |
| 0.2 | 20.5254 | 21.5528 | -1.0274 (0.0662) | 10/10 | 8.4e-08 (5.9e-07) | 0.002 (0.014) |
| 0.4 | 20.4958 | 21.2860 | -0.7902 (0.1185) | 9/10 | 9.2e-05 (0.00028) | 0.0039 (0.014) |
| 0.7 | 20.6392 | 21.1702 | -0.5310 (0.1045) | 10/10 | 0.00066 (0.0013) | 0.002 (0.014) |
| 1.5 | 21.5370 | 21.4321 | 0.1049 (0.0581) | 5/10 | 0.1 (0.1) | 0.23 (0.23) |

## R=3, N=200

Best fixed radius on this test (hindsight, same rule for both): NN rho=0.4 19.9136 vs SRO-FA rho=0.6 20.3882; NN - SRO-FA = -0.4746 (SE 0.0577); NN better in 10/10 seeds; paired t p=1.8e-05; Wilcoxon exact p=0.002.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 20.0000 | 21.3170 | -1.3170 (0.0869) | 10/10 | 1e-07 (5.2e-07) | 0.002 (0.0098) |
| 0.01 | 20.0000 | 21.2603 | -1.2603 (0.0849) | 10/10 | 1.2e-07 (5.2e-07) | 0.002 (0.0098) |
| 0.2 | 19.9595 | 20.6865 | -0.7270 (0.1085) | 9/10 | 8.9e-05 (0.00018) | 0.0039 (0.0098) |
| 0.4 | 19.9136 | 20.4541 | -0.5405 (0.0445) | 10/10 | 6.9e-07 (2.1e-06) | 0.002 (0.0098) |
| 1.5 | 21.2871 | 21.2519 | 0.0352 (0.0340) | 5/10 | 0.33 (0.33) | 0.62 (0.62) |

## R=3, N=800

Best fixed radius on this test (hindsight, same rule for both): NN rho=0.4 19.7060 vs SRO-FA rho=0.5 19.9417; NN - SRO-FA = -0.2357 (SE 0.0544); NN better in 8/10 seeds; paired t p=0.0019; Wilcoxon exact p=0.0098.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 19.7960 | 20.7112 | -0.9152 (0.0543) | 10/10 | 4.1e-08 (2e-07) | 0.002 (0.0098) |
| 0.01 | 19.7960 | 20.6417 | -0.8457 (0.0525) | 10/10 | 6.1e-08 (2.4e-07) | 0.002 (0.0098) |
| 0.2 | 19.7177 | 20.1045 | -0.3867 (0.0564) | 10/10 | 7.5e-05 (0.00022) | 0.002 (0.0098) |
| 0.4 | 19.7060 | 19.9609 | -0.2549 (0.0612) | 8/10 | 0.0024 (0.0049) | 0.0098 (0.02) |
| 1.5 | 21.2730 | 21.1655 | 0.1074 (0.0302) | 1/10 | 0.0062 (0.0062) | 0.014 (0.02) |

## R=10, N=50

Best fixed radius on this test (hindsight, same rule for both): NN rho=1.5 72.3586 vs SRO-FA rho=1.5 73.1719; NN - SRO-FA = -0.8133 (SE 0.0990); NN better in 10/10 seeds; paired t p=1.8e-05; Wilcoxon exact p=0.002.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 76.9512 | 83.5470 | -6.5958 (0.5712) | 10/10 | 1.1e-06 (5.3e-06) | 0.002 (0.0098) |
| 0.01 | 76.9512 | 83.3453 | -6.3941 (0.6647) | 10/10 | 4.9e-06 (1.5e-05) | 0.002 (0.0098) |
| 0.2 | 75.9743 | 80.9699 | -4.9957 (0.5867) | 10/10 | 1.3e-05 (2.7e-05) | 0.002 (0.0098) |
| 0.4 | 74.2680 | 79.0614 | -4.7934 (0.4834) | 10/10 | 3.8e-06 (1.5e-05) | 0.002 (0.0098) |
| 1.5 | 72.3586 | 73.1719 | -0.8133 (0.0990) | 10/10 | 1.8e-05 (2.7e-05) | 0.002 (0.0098) |

## R=10, N=200

Best fixed radius on this test (hindsight, same rule for both): NN rho=0.7 69.0892 vs SRO-FA rho=1.5 72.4679; NN - SRO-FA = -3.3788 (SE 0.2561); NN better in 10/10 seeds; paired t p=3.4e-07; Wilcoxon exact p=0.002.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 71.2736 | 83.9745 | -12.7009 (0.4247) | 10/10 | 2.6e-10 (1.3e-09) | 0.002 (0.0098) |
| 0.01 | 71.2736 | 83.9284 | -12.6548 (0.4281) | 10/10 | 2.8e-10 (1.3e-09) | 0.002 (0.0098) |
| 0.2 | 69.7233 | 80.8360 | -11.1127 (0.4161) | 10/10 | 7e-10 (2.1e-09) | 0.002 (0.0098) |
| 0.4 | 69.1856 | 77.5120 | -8.3263 (0.3896) | 10/10 | 5.1e-09 (1e-08) | 0.002 (0.0098) |
| 1.5 | 71.4050 | 72.4679 | -1.0629 (0.1084) | 10/10 | 4.2e-06 (4.2e-06) | 0.002 (0.0098) |

## R=10, N=800

Best fixed radius on this test (hindsight, same rule for both): NN rho=0.7 68.3931 vs SRO-FA rho=1 72.0639; NN - SRO-FA = -3.6708 (SE 0.2016); NN better in 10/10 seeds; paired t p=2.1e-08; Wilcoxon exact p=0.002.

| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 71.0188 | 83.8371 | -12.8183 (0.2745) | 10/10 | 4.7e-12 (2.1e-11) | 0.002 (0.0098) |
| 0.01 | 71.0188 | 83.7462 | -12.7274 (0.3026) | 10/10 | 1.2e-11 (3.6e-11) | 0.002 (0.0098) |
| 0.2 | 69.1255 | 79.7754 | -10.6499 (0.2655) | 10/10 | 1.9e-11 (3.7e-11) | 0.002 (0.0098) |
| 0.4 | 68.4571 | 75.5939 | -7.1368 (0.1506) | 10/10 | 4.1e-12 (2.1e-11) | 0.002 (0.0098) |
| 1.5 | 71.2573 | 72.3559 | -1.0987 (0.0625) | 10/10 | 2.8e-08 (2.8e-08) | 0.002 (0.0098) |

Note: with n=10 the smallest possible two-sided exact Wilcoxon p-value is 2/1024 = 0.00195.
Holm correction is applied within each R/N table (matched radii).

## Adversary (AP) timing, R=10, final full-N campaign

Source: `green_fullN_cv_20261011/runs/production_v1/jobs/ccg_R10_*.result.json` (`ap_total_time_s`,
`ap_sweeps_completed`). One AP pass = solving the adversarial problem for every training point once,
sequentially (1 worker, 1 Gurobi thread). "Per point" = one adversarial solve for one training point inside
one pass (= total AP time / (passes x N)). Medians over 10 seeds; historical timings on a shared machine.
Raw values: `R10_adversary_timing_fullN.json`.

| N | radius | AP passes | total AP time per policy, min (median / max) | per-point solve, s | whole robust training, min |
|---|---|---:|---:|---:|---:|
| 50 | <=0.03 | 1 | 0.03 / 0.04 | 0.03 | 0.04 |
| 50 | 0.2 | 6 | 0.53 / 0.8 | 0.10 | 7.1 |
| 50 | 0.4 | 11 | 2.3 / 3.8 | 0.25 | 18.0 |
| 50 | 0.7 | 16 | 8.1 / 16.9 | 0.59 | 36.0 |
| 50 | 1.5 | 12 | 35.3 / 79.7 | 3.0 | 59.2 |
| 200 | <=0.03 | 1 | 0.10 / 0.2 | 0.03 | 0.12 |
| 200 | 0.2 | 3 | 0.70 / 1.0 | 0.07 | 4.4 |
| 200 | 0.4 | 4 | 1.8 / 2.9 | 0.13 | 7.7 |
| 200 | 0.7 | 6 | 5.8 / 13.4 | 0.31 | 18.6 |
| 200 | 1.5 | 10 | 54.9 / 125.9 | 1.7 | 77.2 |
| 800 | <=0.03 | 1 | 0.39 / 0.5 | 0.03 | 0.45 |
| 800 | 0.2 | 2 | 1.8 / 2.6 | 0.06 | 6.9 |
| 800 | 0.4 | 3 | 4.2 / 6.2 | 0.10 | 13.5 |
| 800 | 0.7 | 5 | 16.6 / 25.2 | 0.25 | 38.0 |
| 800 | 1.5 | 8 | 105.9 / 269.6 | 1.0 | 139.2 |

## Best radius in hindsight (graph data, `best_radius_hindsight.json`)

Source: exactly the data plotted in the final figures (`full_n.load_full_n_data`). Each method gets its own best radius, chosen with hindsight on the test set (oracle radius selection for both methods). Paired across the 10 seeds (same training data per seed, same 10k test scenarios).

**(A) One best radius per method per (R,N), chosen by seed-mean**

| R | N | NN best ρ: cost | SRO-FA best ρ: cost | NN−SRO-FA (SE) | % | NN cheaper | t p | Wilcoxon p | SAA-kNN |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 50 | 0.4: 20.496 | 0.6: 21.095 | −0.599 (0.123) | −2.8% | 9/10 | 0.0009 | 0.006 | 20.798 |
| 3 | 200 | 0.4: 19.914 | 0.6: 20.388 | −0.475 (0.058) | −2.3% | 10/10 | 2e-05 | 0.002 | 20.269 |
| 3 | 800 | 0.4: 19.706 | 0.5: 19.942 | −0.236 (0.054) | −1.2% | 8/10 | 0.002 | 0.010 | 19.999 |
| 10 | 50 | 1.5: 72.359 | 1.5: 73.172 | −0.813 (0.099) | −1.1% | 10/10 | 2e-05 | 0.002 | 75.472 |
| 10 | 200 | 0.7: 69.089 | 1.5: 72.468 | −3.379 (0.256) | −4.7% | 10/10 | 3e-07 | 0.002 | 72.877 |
| 10 | 800 | 0.7: 68.393 | 1.0: 72.064 | −3.671 (0.202) | −5.1% | 10/10 | 2e-08 | 0.002 | 70.454 |

**(B) Best radius per seed (oracle per seed, per method)**

| R | N | NN | SRO-FA | NN−SRO-FA (SE) | NN cheaper | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 3 | 50 | 20.318 | 21.067 | −0.749 (0.062) | 10/10 | 8e-07 | 0.002 |
| 3 | 200 | 19.775 | 20.357 | −0.582 (0.030) | 10/10 | 1e-08 | 0.002 |
| 3 | 800 | 19.622 | 19.933 | −0.311 (0.029) | 10/10 | 2e-06 | 0.002 |
| 10 | 50 | 72.163 | 73.172 | −1.009 (0.125) | 10/10 | 2e-05 | 0.002 |
| 10 | 200 | 68.892 | 72.468 | −3.576 (0.240) | 10/10 | 1e-07 | 0.002 |
| 10 | 800 | 68.251 | 72.064 | −3.813 (0.179) | 10/10 | 5e-09 | 0.002 |

Wilcoxon 0.002 = the exact two-sided minimum with n=10 (all 10 seeds in favour). Caveat: radius chosen on the test set → descriptive comparison of the best attainable cost of each method.

### Shipping (R=3), best radius in hindsight (`best_radius_hindsight_shipping.json`)

Same analysis on the final shipping graph data (`load_full_n_data('shipping_R3')`), script `../best_radius_shipping.py`. Training data verified identical by hash for all 30 (N, seed) pairs; same 10k test set.

**(A) One best radius per method, chosen by seed-mean**

| N | NN best ρ: cost | SRO-FA best ρ: cost | NN−SRO-FA (SE) | % | NN cheaper | t p | Wilcoxon p | SAA-kNN |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.4: 20.763 | 0.6: 21.400 | −0.637 (0.096) | −3.0% | 10/10 | 1e-04 | 0.002 | 21.458 |
| 200 | 0.4: 20.227 | 0.5: 20.634 | −0.407 (0.074) | −2.0% | 9/10 | 4e-04 | 0.004 | 20.702 |
| 800 | 0.4: 19.996 | 0.4: 20.214 | −0.218 (0.048) | −1.1% | 8/10 | 0.002 | 0.010 | 20.499 |

**(B) Best radius per seed**

| N | NN | SRO-FA | NN−SRO-FA (SE) | % | NN cheaper | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 50 | 20.651 | 21.315 | −0.664 (0.080) | −3.1% | 10/10 | 2e-05 | 0.002 |
| 200 | 20.160 | 20.598 | −0.439 (0.069) | −2.1% | 10/10 | 1e-04 | 0.002 |
| 800 | 19.968 | 20.191 | −0.223 (0.041) | −1.1% | 8/10 | 4e-04 | 0.010 |

## Summary: best radius in hindsight, all settings (one table)

Each method at its own best radius (by seed-mean, hindsight on the test set); paired over 10 seeds with identical training data and the same 10k test set. Last column: best radius chosen separately per seed.

| Setting | N | NN (ρ*) | SRO-FA (ρ*) | NN−SRO-FA (SE) | % | NN cheaper | t-test p | Wilcoxon p | SAA-kNN | Per-seed best: diff (NN cheaper) |
|---|---|---|---|---|---|---|---|---|---|---|
| R=3 | 50 | 20.496 (0.4) | 21.095 (0.6) | −0.599 (0.123) | −2.8% | 9/10 | 0.0009 | 0.006 | 20.798 | −0.749 (10/10) |
| R=3 | 200 | 19.914 (0.4) | 20.388 (0.6) | −0.475 (0.058) | −2.3% | 10/10 | 2e-05 | 0.002 | 20.269 | −0.582 (10/10) |
| R=3 | 800 | 19.706 (0.4) | 19.942 (0.5) | −0.236 (0.054) | −1.2% | 8/10 | 0.002 | 0.010 | 19.999 | −0.311 (10/10) |
| R=10 | 50 | 72.359 (1.5) | 73.172 (1.5) | −0.813 (0.099) | −1.1% | 10/10 | 2e-05 | 0.002 | 75.472 | −1.009 (10/10) |
| R=10 | 200 | 69.089 (0.7) | 72.468 (1.5) | −3.379 (0.256) | −4.7% | 10/10 | 3e-07 | 0.002 | 72.877 | −3.576 (10/10) |
| R=10 | 800 | 68.393 (0.7) | 72.064 (1.0) | −3.671 (0.202) | −5.1% | 10/10 | 2e-08 | 0.002 | 70.454 | −3.813 (10/10) |
| R=3 shipping | 50 | 20.763 (0.4) | 21.400 (0.6) | −0.637 (0.096) | −3.0% | 10/10 | 1e-04 | 0.002 | 21.458 | −0.664 (10/10) |
| R=3 shipping | 200 | 20.227 (0.4) | 20.634 (0.5) | −0.407 (0.074) | −2.0% | 9/10 | 4e-04 | 0.004 | 20.702 | −0.439 (10/10) |
| R=3 shipping | 800 | 19.996 (0.4) | 20.214 (0.4) | −0.218 (0.048) | −1.1% | 8/10 | 0.002 | 0.010 | 20.499 | −0.223 (8/10) |

Significance threshold 0.05; with 10 seeds the smallest attainable exact Wilcoxon p is 0.002 (all 10 seeds favour one method).
