# Retailers base — N50 R3 net16_v2 — SCS versus Moreau

Fresh-output restart authorized2026-09-09. Solver comparison is named in the
top-level directory, followed by problem, sample/retailer counts and network.
This runner reuses tested source from retailers_moreau_N50_R3_net16_v2,
not its failed output, trained weights or timing records. Its metadata-only
unlimited-time serialization fix leaves actual solver settings unchanged.

## This timing trial

- Original total50 samples all used for training, no validation or test data.
  This retains the agreed timing baseline; a future train/validation-within50
  protocol is a different experiment and is not silently applied here.
- Seed0 data, seed0 fresh network, one shared fresh SAA inventory initialization.
  R3, width16--16, LR .05/.01, full batch50,3000-epoch cap, patience300.
- Both SCS and Moreau use the corrected evaluated-pre-update checkpoint rule.
  Training epochs may differ because numerical gradients affect early stopping.
  Report per-epoch and total time, not just total-time ratios.
- Synthetic gradient fixtures, real-initial-target Gurobi projection check,
  alternating fixed-weight forward/backward measurements precede training.
  SCS trains first, then Moreau in the same process; no concurrent model workers.
- Post-selection training-data cost is measured with the own solver and common
  strict Gurobi projection. No validation/test/generalization claim.
- Raw timings, seeds, settings, solver iterations/status and epoch counts are
  in results/trial.json. No trial-level fallback or retry; legacy diffcp SCS
  inaccurate-solution retry/acceptance is retained and disclosed.
- One completed run per arm implies mean=median=min=max of its training time.
  Fixed-weight repetitions have separate first-call and subsequent statistics.
  SAA, setup, training, own evaluation, reference evaluation and total trial
  timers are separate. Imports/initial metadata/final persistence are outside
  the top-level timer. CCG/AP/robust MP, SRO-FA, CV and test are not run.
- Torch/BLAS caps1 do not cap all solver pools. No full-core sweep or equal
  effective-thread claim. Failed prior diagnostics are not used as successful
  timing evidence; historical seed0 training243.749316s/1716epochs is only a
  reference under the old checkpoint convention.

## Execution

Session: retailers_solver_compare_N50_R3_20260909.
From this directory the exact command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/home/aviyac/ap_env/bin/python -B -u run_comparison.py > comparison.log 2>&1
```

Attach: `tmux attach -t retailers_solver_compare_N50_R3_20260909`.
Status: `tmux has-session -t retailers_solver_compare_N50_R3_20260909`.
Progress: `tail -n 10 comparison.log`; structured status: results/trial.json.
Output creation is exclusive. Do not overwrite existing results or run log.

## Proposed later protocol — not launched

Define N50 as the total development budget, split into training and validation
before SAA initialization, scaling or fitting. Train-loss gradients use training
only; validation may select checkpoints and radius without entering gradient
updates. Use the same split/budget for neural and comparator methods. Freeze
the split, radius grid, stopping/selection rules and any refit rule before
running. Generate a fresh final test only after those choices are frozen;
previously inspected test outcomes cannot become an unseen test again.

The current canonical R3 selector uses separate validation seed88880+seed
(13 samples for50 training samples), whereas9999 is the saved10k test seed.
Independent validation is not inherently leakage, but its extra data budget
differs from a strict50-total-sample design. Neither new split counts nor a
reduced radius grid have yet been agreed. Parallelism requires explicit model
worker and native/Gurobi thread caps plus available-memory/load checks.