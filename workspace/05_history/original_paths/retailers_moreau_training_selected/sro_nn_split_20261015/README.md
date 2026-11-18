# SRO-FA and SRO-FA+LDR on the NN split — 15 October 2026

> **18 November 2026 — radius selection rule changed.** Validation-based radius selection now uses the shared rule "largest radius within 1e-6 of the minimum validation score" (`src/config.py`), replacing "exact tie -> smaller radius". The sealed outputs of this run record the old rule and are unchanged; the current selections, which change no test cost, are in `comparison/paper_figures/radius_selection_20261118/`.

Protocol, methods and caveats are in the [run.py](run.py) docstring. In short:
SRO-FA (R3, R10, shipping R3) and SRO-FA with linear decision rules (R3 only)
are trained on exactly the NN's 40/160/640 training rows. The radius is selected
per seed on the NN's 10/40/160 validation rows, the selections are sealed, and
only then are the selected policies scored on the shared 10,000-row seed-9999
test set.

- Launched 15 Oct 2026 06:55 UTC, host gold-srv2, 28 single-thread workers,
  2,670 training jobs (FA 2,100 + FA+LDR 570).
- Session: `tmux attach -t sro_nn_split_20261015`
- Command: `/data/home/aviyac/ap_env/bin/python -B -u run.py --output runs/production_v1 --workers 28`
- Log: [production_v1.log](production_v1.log); per-job records in
  `runs/production_v1/jobs/`, results in `runs/production_v1/RESULTS.md`,
  `selection_seal.json` and `test_results.json`.
- Re-running the same command resumes; finished job files are never overwritten.

Pre-launch checks: the split was identical to `src/config.split_data` for R3 and
R10. The LDR objective was always at or below SRO-FA's, with no material
infeasibility inside the training boxes. One N=800 LDR solve took about 28 s.

## Validation-split run: completed 15 Oct 2026 19:11 UTC

Exit 0: 2,670/2,670 jobs, no failures, 120 selections sealed before the test
set was generated. Elapsed time was 12.26 h with 28 workers. Almost all of it was
the tail of large-radius N=800 LDR solves (up to 8.8 h each); the median N=800
LDR job took 42 s. See `runs/production_v1/RESULTS.md`. Mean share of test points
needing a material LDR correction: clip 6.3/3.6/1.3%, scale 4.9/4.2/1.4% for
N = 50/200/800. Shipping costs are identical under the q2 > 0 and q2 > 1e-10 fees.

## Full-N LDR sweep (`run_full_n.py`)

Launched 15 Oct 2026 11:29 UTC in tmux `sro_ldr_fullN_20261015` with 24 workers:
570 R3 jobs over all N rows, scored on the test set, for the comparison plot.
Log: [full_n_v1.log](full_n_v1.log).
Completed 16 Oct 2026 05:15 UTC, exit 0: 570/570, no failures, 17.8 h elapsed.
The elapsed time is almost entirely 4 N=800 radius-1.5 solves, up to 17.8 h each;
the median N=800 job took 80 s. Best mean test cost on the grid: 21.147 @0.75, 20.381 @0.6,
19.923 @0.5 for N = 50/200/800. At small radii the boxes are tiny and LDR
extrapolates badly outside them (test cost well above 22). Averaged over all radii,
30-53% of test points needed clipping/scaling. Curve: green_fullN_cv_20261011/runs/
production_v1/published_with_comparators/R3/.
