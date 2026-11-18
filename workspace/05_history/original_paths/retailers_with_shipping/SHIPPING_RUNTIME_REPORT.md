# Fixed-Shipping R=3 Runtime Report

## Measurement scope

This report summarizes persisted wall-clock measurements for the fixed-shipping
retailers experiments with 10 training seeds. Times are measured values from the
saved experiment artifacts; no missing times are inferred. Neural CCG
`total_time_s` is the end-to-end robust run for one radius and seed and excludes
the separately trained nominal model. Nominal `seed_time_s` includes the full
seed workflow, including nominal cross-validation. SRO-FA `time_total` includes
partition construction, optimization, OOS evaluation, and other per-run
overhead.

The runs were executed on a server with two Intel Xeon Gold 6354 CPUs (72
logical CPUs total), 502 GiB RAM, and two NVIDIA A30 GPUs with 24 GiB each.
Concurrent neural workers were used during parts of the sweep, so per-model
wall times must not be summed and interpreted as elapsed sweep time. Overall
sweep elapsed time was not persisted and is therefore unavailable.

## Nominal neural model

| Training size | Measured seeds | Mean | Median | Minimum | Maximum |
|---:|---:|---:|---:|---:|---:|
| 50 | 0 | unavailable | unavailable | unavailable | unavailable |
| 200 | 10 | 24,217 s (6.73 h) | 23,842 s (6.62 h) | 22,845 s (6.35 h) | 26,030 s (7.23 h) |
| 800 | 10 | 83,568 s (23.21 h) | 83,537 s (23.20 h) | 81,730 s (22.70 h) | 85,902 s (23.86 h) |

The historical N=50 nominal summaries do not contain a runtime field. A rerun
is required to measure that case. Nominal evaluation time is not stored
separately from the complete seed workflow.

## Robust neural CCG

Each row reports complete end-to-end radius/seed CCG runs. `Iterations` is the
mean number of saved AP/MP trajectory entries. AP and MP are mean sums over all
iterations for one model. MP training dominates the observed runtime.

| N | Radius | Runs | Total mean | Total median | Total range | Iterations | AP mean | MP mean | Converged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.001 | 10 | 1,011 s (16.8 min) | 1,073 s | 424--1,322 s | 2.2 | 0.0 s* | 1,011 s | 10/10 |
| 50 | 0.003 | 10 | 1,172 s (19.5 min) | 1,125 s | 341--2,026 s | 2.3 | 0.3 s* | 1,172 s | 10/10 |
| 50 | 0.005 | 10 | 1,173 s (19.6 min) | 1,181 s | 364--1,991 s | 2.3 | 3.0 s | 1,170 s | 10/10 |
| 50 | 0.01 | 10 | 1,033 s (17.2 min) | 1,075 s | 378--1,670 s | 2.2 | 3.0 s | 1,030 s | 10/10 |
| 50 | 0.02 | 10 | 1,033 s (17.2 min) | 1,019 s | 637--1,287 s | 2.0 | 2.7 s | 1,030 s | 10/10 |
| 50 | 0.05 | 10 | 1,102 s (18.4 min) | 1,174 s | 544--1,622 s | 2.1 | 3.4 s | 1,098 s | 10/10 |
| 50 | 0.1 | 10 | 2,813 s (46.9 min) | 2,904 s | 742--5,544 s | 3.0 | 5.1 s | 2,808 s | 10/10 |
| 50 | 0.15 | 10 | 3,665 s (61.1 min) | 3,362 s | 1,711--5,331 s | 3.6 | 6.8 s | 3,658 s | 9/10 |
| 50 | 0.2 | 10 | 4,900 s (81.7 min) | 4,524 s | 2,709--10,173 s | 4.1 | 8.1 s | 4,892 s | 10/10 |
| 50 | 0.3 | 10 | 9,900 s (2.75 h) | 7,619 s | 2,753--29,127 s | 5.6 | 11.5 s | 9,888 s | 10/10 |
| 50 | 0.4 | 10 | 10,290 s (2.86 h) | 10,377 s | 4,328--17,734 s | 6.0 | 12.7 s | 10,277 s | 10/10 |
| 50 | 0.5 | 10 | 27,098 s (7.53 h) | 12,272 s | 5,208--165,455 s | 10.4 | 14.8 s | 27,084 s | 7/10 |
| 50 | 0.7 | 10 | 26,056 s (7.24 h) | 12,723 s | 5,277--86,319 s | 8.9 | 14.2 s | 26,041 s | 5/10 |
| 50 | 1.0 | 10 | 14,521 s (4.03 h) | 9,814 s | 5,871--47,017 s | 9.8 | 12.6 s | 14,509 s | 4/10 |
| 50 | 1.5 | 10 | 14,306 s (3.97 h) | 3,913 s | 3,044--84,303 s | 8.0 | 11.2 s | 14,295 s | 7/10 |
| 50 | 2.0 | 10 | 23,894 s (6.64 h) | 14,873 s | 991--88,785 s | 10.1 | 19.5 s | 23,874 s | 9/10 |
| 200 | 0.001 | 10 | 3,668 s (61.1 min) | 3,322 s | 1,625--5,391 s | 2.6 | 18.2 s | 3,649 s | 10/10 |
| 200 | 0.003 | 10 | 3,963 s (66.0 min) | 4,036 s | 2,402--5,090 s | 2.5 | 8.6 s | 3,954 s | 10/10 |
| 200 | 0.005 | 10 | 3,041 s (50.7 min) | 2,567 s | 2,045--4,574 s | 2.4 | 8.0 s | 3,033 s | 10/10 |
| 200 | 0.01 | 10 | 2,845 s (47.4 min) | 2,715 s | 1,998--4,505 s | 2.3 | 7.3 s | 2,838 s | 10/10 |
| 200 | 0.02 | 10 | 2,847 s (47.4 min) | 2,647 s | 1,554--4,605 s | 2.2 | 8.7 s | 2,838 s | 10/10 |
| 200 | 0.05 | 10 | 2,818 s (47.0 min) | 2,772 s | 1,586--4,649 s | 2.1 | 7.7 s | 2,811 s | 10/10 |
| 200 | 0.1 | 10 | 4,446 s (74.1 min) | 2,913 s | 1,643--10,037 s | 2.5 | 10.4 s | 4,436 s | 10/10 |
| 200 | 0.15 | 10 | 7,165 s (1.99 h) | 6,835 s | 2,051--10,101 s | 3.2 | 15.0 s | 7,150 s | 10/10 |
| 200 | 0.2 | 10 | 8,009 s (2.22 h) | 6,885 s | 2,340--12,530 s | 3.4 | 18.1 s | 7,991 s | 10/10 |
| 200 | 0.3 | 10 | 14,835 s (4.12 h) | 13,618 s | 2,504--33,409 s | 4.1 | 24.9 s | 14,810 s | 10/10 |
| 200 | 0.4 | 10 | 25,738 s (7.15 h) | 24,979 s | 8,639--56,058 s | 5.1 | 38.2 s | 25,699 s | 10/10 |
| 200 | 0.5 | 10 | 30,473 s (8.46 h) | 26,850 s | 13,810--67,211 s | 5.6 | 44.4 s | 30,429 s | 9/10 |
| 200 | 0.7 | 10 | 34,304 s (9.53 h) | 38,023 s | 18,626--52,178 s | 6.5 | 43.7 s | 34,260 s | 10/10 |
| 200 | 1.0 | 10 | 31,528 s (8.76 h) | 32,157 s | 13,943--54,459 s | 6.7 | 46.0 s | 31,481 s | 8/10 |
| 200 | 1.5 | 10 | 26,056 s (7.24 h) | 25,379 s | 15,453--37,706 s | 6.4 | 44.1 s | 26,012 s | 10/10 |
| 200 | 2.0 | 10 | 29,602 s (8.22 h) | 27,791 s | 16,052--55,222 s | 6.9 | 48.5 s | 29,553 s | 9/10 |
| 800 | 0.001 | 10 | 11,790 s (3.27 h) | 11,188 s | 7,494--19,224 s | 2.2 | 43.1 s | 11,746 s | 10/10 |
| 800 | 0.003 | 10 | 11,401 s (3.17 h) | 10,726 s | 8,413--16,754 s | 2.2 | 32.8 s | 11,368 s | 10/10 |
| 800 | 0.005 | 10 | 11,097 s (3.08 h) | 10,696 s | 7,163--17,524 s | 2.2 | 31.1 s | 11,066 s | 10/10 |
| 800 | 0.01 | 10 | 11,704 s (3.25 h) | 10,571 s | 7,679--17,676 s | 2.3 | 31.8 s | 11,672 s | 10/10 |
| 800 | 0.02 | 10 | 10,751 s (2.99 h) | 10,967 s | 7,982--16,081 s | 2.2 | 31.2 s | 10,720 s | 10/10 |
| 800 | 0.05 | 10 | 11,480 s (3.19 h) | 10,479 s | 9,120--16,835 s | 2.3 | 34.8 s | 11,446 s | 10/10 |
| 800 | 0.1 | 10 | 16,228 s (4.51 h) | 15,079 s | 9,278--36,956 s | 2.7 | 41.3 s | 16,186 s | 10/10 |
| 800 | 0.15 | 10 | 25,131 s (6.98 h) | 28,232 s | 8,295--31,539 s | 2.9 | 53.2 s | 25,078 s | 10/10 |
| 800 | 0.2 | 10 | 33,708 s (9.36 h) | 35,969 s | 23,808--49,201 s | 3.6 | 71.4 s | 33,637 s | 10/10 |
| 800 | 0.3 | 10 | 39,160 s (10.88 h) | 43,723 s | 21,742--61,511 s | 3.7 | 90.0 s | 39,070 s | 10/10 |
| 800 | 0.4 | 10 | 42,374 s (11.77 h) | 42,527 s | 24,948--68,907 s | 3.8 | 94.6 s | 42,279 s | 10/10 |
| 800 | 0.5 | 10 | 88,652 s (24.63 h) | 81,073 s | 61,934--122,726 s | 4.4 | 114.7 s | 88,538 s | 10/10 |
| 800 | 0.7 | 6 | 77,383 s (21.50 h) | 75,601 s | 43,114--128,714 s | 4.5 | 122.9 s | 77,260 s | 6/6 |
| 800 | 1.0 | 3 | 79,858 s (22.18 h) | 87,672 s | 39,218--112,682 s | 5.0 | 142.1 s | 79,715 s | 3/3 |
| 800 | 1.5 | 5 | 93,664 s (26.02 h) | 88,936 s | 79,685--112,959 s | 5.4 | 137.2 s | 93,527 s | 5/5 |
| 800 | 2.0 | 10 | 113,749 s (31.60 h) | 110,514 s | 76,382--168,848 s | 6.9 | 171.1 s | 113,578 s | 9/10 |

*The near-zero AP aggregates at the smallest N=50 radii reflect historical
artifact values and should not be interpreted as newly measured zero-cost AP
runs. N=800 radii 0.7, 1.0, and 1.5 are incomplete and are excluded from
complete-radius comparisons. A saved run can be complete without satisfying
the CCG convergence criterion; the final column reports this distinction.

Across all 160 complete saved radius/seed records for each of N=50 and N=200,
mean per-model CCG time was 8,998 s (2.50 h) and 14,459 s (4.02 h),
respectively. For N=800, 144 currently saved records have mean 37,809 s
(10.50 h), but this aggregate is not directly comparable because three radii
are incomplete.

## SRO-FA

All entries below contain 10 seeds. Stage columns are means per radius/seed.

| N | Radius | Total mean | Total median | Total range | Partition mean | Solver mean | OOS evaluation mean | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.0 | 0.08 s | 0.08 s | 0.06--0.13 s | <0.01 s | 0.01 s | 0.03 s | historical status unavailable |
| 50 | 0.1 | 0.15 s | 0.15 s | 0.09--0.22 s | <0.01 s | 0.09 s | 0.02 s | historical status unavailable |
| 50 | 0.5 | 0.48 s | 0.54 s | 0.20--0.77 s | <0.01 s | 0.37 s | 0.02 s | historical status unavailable |
| 50 | 1.0 | 0.55 s | 0.38 s | 0.28--1.30 s | <0.01 s | 0.39 s | 0.02 s | historical status unavailable |
| 50 | 1.5 | 0.75 s | 0.49 s | 0.42--1.71 s | <0.01 s | 0.56 s | 0.02 s | historical status unavailable |
| 200 | 0.0 | 2.56 s | 0.24 s | 0.22--23.42 s | 0.01 s | 0.04 s | 0.10 s | historical status unavailable |
| 200 | 0.1 | 1.52 s | 1.40 s | 1.04--2.04 s | 0.01 s | 1.25 s | 0.09 s | historical status unavailable |
| 200 | 0.5 | 9.35 s | 8.96 s | 2.30--18.71 s | 0.01 s | 8.88 s | 0.08 s | historical status unavailable |
| 200 | 1.0 | 27.02 s | 22.42 s | 3.39--86.26 s | 0.01 s | 26.11 s | 0.09 s | historical status unavailable |
| 200 | 1.5 | 64.65 s | 22.99 s | 8.06--234.42 s | 0.01 s | 62.98 s | 0.09 s | historical status unavailable |
| 800 | 0.0 | 0.95 s | 0.96 s | 0.86--1.03 s | 0.05 s | 0.19 s | 0.35 s | historical status unavailable |
| 800 | 0.1 | 23.57 s | 25.07 s | 10.82--37.13 s | 0.05 s | 22.52 s | 0.35 s | historical status unavailable |
| 800 | 0.5 | 473.80 s (7.90 min) | 488.32 s | 261.29--738.26 s | 0.07 s | 469.85 s | 0.46 s | historical status unavailable |
| 800 | 1.0 | 3,008.17 s (50.14 min) | 3,008.22 s | 3,007.41--3,009.02 s | 0.06 s | 3,000.17 s | 0.33 s | time limit, incumbent available (10/10) |
| 800 | 1.5 | 3,011.27 s (50.19 min) | 3,011.28 s | 3,010.47--3,011.67 s | 0.05 s | 3,000.25 s | 0.28 s | time limit, incumbent available (10/10) |

For N=800 at radii 1.0 and 1.5, Gurobi status is 9 (time limit) for all
seeds. The saved incumbents have mean relative gaps of 1.70% and 1.29%, with
maximum gaps of 1.93% and 1.59%, respectively. These results must not be called
optimal. Across all 340 saved SRO-FA records per training size, mean per-model
times are 0.28 s (N=50), 10.92 s (N=200), and 806.95 s (N=800); these pooled
means mix radii of very different difficulty.

## Corrective 10k evaluation

After historical N=50 summaries were repointed to the generated policy with
minimum empirical AP mean, all 12 affected radii were reevaluated on the fixed
10k test set. Persisted monotonic per-radius evaluation times sum to 474.76 s
(7.91 min), with mean 39.56 s, median 39.29 s, and range 38.26--44.01 s.
The first three radii were saved before an external interruption. The resumed
invocation reevaluated the remaining nine radii in 359.55 s (5.99 min), from
2026-07-27 13:25:48 UTC through 13:31:47 UTC. These are corrective evaluation
times, not neural training or original sweep runtimes.

## Unavailable timing

- N=50 nominal neural training: historical summaries contain no timing.
- Separate nominal CV, training, and evaluation stages: only complete
  `seed_time_s` is persisted for N=200 and N=800.
- Historical kNN cross-validation, benchmark construction, and 10k evaluation:
  the evaluator did not persist stage-level timing. The N=800 evaluation
  completed, but its runtime cannot be reconstructed from the saved benchmark
  artifact. The later N=50 corrective evaluation is measured separately above.
- Overall concurrent sweep elapsed time: not persisted. Summing per-model wall
  times would overstate elapsed time because models ran concurrently.
- Hardware allocation per historical model (exact CPU/GPU assignment and
  concurrent-worker count): not persisted consistently.

## Source artifacts

- Nominal: `N{N}/seed_{seed}_summary.json`
- Neural CCG: `N{N}/rho_sweep/rho_*/seed_{seed}_summary.json`
- SRO-FA: `N{N}/sro_fa/sro_fa_shipping_results_N{N}_R3.json`

All stored timing units are seconds. The historical scripts use wall-clock
`time.time()` measurements; future runs should use monotonic
`time.perf_counter()` while retaining seconds as the persisted unit.