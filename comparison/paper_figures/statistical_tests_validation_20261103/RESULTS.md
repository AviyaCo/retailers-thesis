# Paired tests at validation-selected radii (Protocol A): NN vs SRO-FA

Both methods: same 40/160/640 training rows, radius chosen per seed on the same 10/40/160
validation rows, same 10,000-row seed-9999 test. Difference = NN - SRO-FA (negative = NN cheaper).
Shipping: 0.1 per raw q2 > 0 for both. Read-only analysis of saved per-seed test means.

| Setting | N | NN | SRO-FA | NN-SRO-FA (SE) | % | NN cheaper | t-test p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R=3 | 50 | 20.545 | 21.360 | -0.815 (0.182) | -3.8% | 9/10 | 0.0015 | 0.0059 |
| R=3 | 200 | 19.866 | 20.552 | -0.685 (0.101) | -3.3% | 9/10 | 7.8e-05 | 0.0039 |
| R=3 | 800 | 19.682 | 19.987 | -0.305 (0.042) | -1.5% | 10/10 | 5.2e-05 | 0.002 |
| R=10 | 50 | 73.028 | 73.897 | -0.868 (0.604) | -1.2% | 9/10 | 0.18 | 0.084 |
| R=10 | 200 | 69.291 | 72.590 | -3.299 (0.201) | -4.5% | 10/10 | 5.2e-08 | 0.002 |
| R=10 | 800 | 68.247 | 72.289 | -4.043 (0.160) | -5.6% | 10/10 | 1.2e-09 | 0.002 |
| R=3 shipping | 50 | 20.817 | 21.553 | -0.736 (0.112) | -3.4% | 10/10 | 0.0001 | 0.002 |
| R=3 shipping | 200 | 20.342 | 20.865 | -0.522 (0.069) | -2.5% | 10/10 | 3.6e-05 | 0.002 |
| R=3 shipping | 800 | 19.967 | 20.239 | -0.272 (0.032) | -1.3% | 10/10 | 1.3e-05 | 0.002 |

With n=10 the smallest exact two-sided Wilcoxon p is 2/1024 = 0.002.
Caveats: validation rows also chose the NN learning rate; the test set was previously
inspected, so results are descriptive. Sources and SHA-256: paired_tests_validation.json.
