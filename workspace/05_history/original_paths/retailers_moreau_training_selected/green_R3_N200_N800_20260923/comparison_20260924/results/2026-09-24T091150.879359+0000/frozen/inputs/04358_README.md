# Matched historical R3 comparison, September24

Authorized scope: three panels N50/N200/N800; new green OOS and finite-pool
IS, SCS-trained OOS and finite-pool IS, cached SAA-kNN and SRO-FA OOS, and
the existing gray reference levels. No model fitting, AP/MP optimization,
validation reselection, radius refinement or test-based choice is performed.

## Protocol

- Freeze all180 green policies at N200/N800, their selected candidate pools,
  saved20 validation-selected radius winners, and all600 SCS policies on the
  actual historical grids (24/18/18 radii including zero, through1.5).
- Preserve completed N50 green90-policy OOS/IS results and their proof chain.
- Test draw: original SciPy/global-NumPy truncated-normal generator,
 10000 pairs, seed9999. Float64 arrays must exactly match the persisted N50
  array hashes. This historically inspected test supports a **descriptive
  comparison**, not a fresh untouched-test claim.
- Freeze identities before test generation. Workers verify learned state,
  checkpoint, pool, manifest and array identities. No test score selects a
  checkpoint or radius. Each plotted radius uses all ten seeds0–9.
- Evaluate all780 policies freshly: strict Gurobi `OPTIMAL`, finite output,
  physical residual at most1e-4; zero capacity buffer. Preserve original
  float32 network/target/inventory and float64 allocation/cost arithmetic.
  No retry, clipping, zero substitution, degraded tolerance or solver fallback.
- IS is the mean of per-training-center maximum cost on that returned
  policy's own saved MP pool, not AP mean or a later stopping pool. Green uses
 40/160/640 centers; legacy SCS uses50/200/800. SCS pools contain nominal
  centers plus every AP point actually supplied through its selected MP index,
  excluding later APs. Missing files/centers or nonfinite data are fatal.
- Legacy SCS pool audit found small box residuals (up to about7.15e-6).
  The user explicitly approved evaluating the **unchanged historical coordinates**
  and disclosing residuals; this is observational IS, not certification that
  those points are inside the box. Every positive residual is preserved with
  source/ball identity; counts above1e-9 are diagnostic, not a relaxed tolerance.
  Green's1e-9 membership gate remains unchanged. Strict physical QP acceptance
  remains identical for both methods.
- SAA-kNN/SRO-FA scores are frozen historical references, not newly certified
  strict solves. kNN uses the persisted training LOO choice, not test selection;
  its LOO interpolation does not refit the entire SAA initialization per fold.
- Green reserves20% of its budget for validation; historical baselines train
  on the full50/200/800. This is not an isolated projection-backend comparison.
-20th/80th percentile bands describe seed variation, not confidence intervals.
  Plot actual grids only; no fabricated interpolated evaluation points.

## Validation and execution

`preflight.py --output NEW_DIRECTORY` runs all solver-free suites, a full
disposable780-policy freeze, and20 strict training-row QPs from ten fixed
policies covering all methods/sizes and radii0/0.2. It forbids test generation.
Its sealed summary binds the exact code/docs/entrypoint, historical sources,
policies and saved pools. Any input change requires fresh preflight.

`run_comparison.py --preflight ABSOLUTE_PREFLIGHT_SUMMARY` validates preflight,
creates an exclusive timestamped output and fresh freeze, then evaluates with
up to32 spawn workers. All numerical libraries and Gurobi use one thread.
The worker budget respects CPU affinity/quota, existing load, four reserved
CPU slots,16GiB host memory reserve and at least2GiB per worker, increased
using measured preflight peak RSS. Disk checks reserve snapshot and journal
space without truncating outputs or changing the experiment.

Long runs are launched inside a named tmux session using `bash launch.sh
ABSOLUTE_PREFLIGHT_SUMMARY`; stdout/stderr and exit code persist under
campaign timestamp files. `launch.json` records the exact output, command,
session and log paths. Attach with `tmux attach -t retailers_comparison_R3_20260924`;
check presence with `tmux has-session -t retailers_comparison_R3_20260924`.
The intended working directory is this directory, using ap_env Python3.11.

## Outputs and runtime accounting

The run publishes `comparison_R3.png` (300dpi), vector `comparison_R3.pdf`,
`plot_data.json`, `summary.json`, `runtime_aggregates.json` and `RESULTS.md`
only after the full strict grid and final source checks pass. Each policy
retains OOS/IS solve journals, phase times, counts, iterations and failures.
No partial cohort is silently plotted. `plot_combined_R3_net16v2.py
--green-comparison` rerenders the completed comparison without numerical
evaluation; its default historical-only behavior is preserved.

Timings use perf_counter seconds with UTC start/end, size/radius/seed/status
and worker/thread metadata. Current OOS, IS, dispatch/freeze/verification/render
and total wall times are distinct. Historical nominal/CCG/AP/MP and SRO-FA
timings remain source-labeled; cache-loading time is never model time.
Missing historical durations are unavailable and require rerun; no times are
inferred from another run. Failed/incomplete cohorts remain separate from
completed mean/median/min/max aggregates. Concurrent durations are not summed
and called elapsed time. No benchmark retraining or CV occurs in this launch.

Earlier syntax-edit failures occurred before numerical execution; after user
approval to continue, parser repairs and regression tests passed. No broken
source or disposable preflight policy is used as a production model.