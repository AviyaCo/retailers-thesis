# Full-N green sweeps with five-fold nominal LR CV — 11 October 2026

New orchestration only; numerical code is reused from the original R3, R10 and
shipping experiments. No original code, practical validation result, or saved
plot is overwritten. `backend.py` binds the original generator, SAA, models,
train-only trainer, AP and CCG controller. `reporting.py` reuses the strict
evaluators and existing renderer. Source hashes are verified in fresh processes.

## Fixed protocol

- Cases: no-shipping R3, no-shipping R10, shipping R3.
- Actual final training N: 50, 200, 800; seeds 0–9; network seed 0.
- Five-fold shuffled scikit-learn KFold, random state 100000 + data seed.
- LRs: .005, .01, .05. Same folds and fold-local SAA initialization across LRs;
  no held-out rows in SAA, initialization, loss, epoch selection or scheduler.
- Minimum mean held-out native **hard** cost selects LR; exact ties use smaller
  LR. Shipping charges .1 for every raw q>0, without clipping/cutoff.
- Fresh all-N SAA and nominal refit; no fold checkpoint is used as the refit.
- All nine original green radii: 0, .001, .003, .01, .03, .2, .4, .7, 1.5.
  Each starts independently from the same full-N nominal policy for its seed.
- Training-best preupdate checkpoint; original 3000 epochs / patience 300,
  fresh Adam, inventory LR .01, original scheduler and projection.
- CCG: first strict aggregate gap <1%, including candidate0; at most 40 MPs;
  return STOP and its own pre-insertion pool, never earlier minimum AP.
- User explicitly approved the previously used shipping **advisory** activation
  and exact-duplicate checks for every fresh configuration on 11 October.
  Finite values, physical feasibility, OPTIMAL status, bound ordering and
  material nonidentical float32-collision checks remain enforced. This is not
  exact hard-fee robust certification. No automatic relaxation or retry.

There are 1,350 CV fits, 90 full-N refits, and 810 radius configurations. All
must complete/converge before any test generation. Failure or nonconvergence
blocks publication rather than silently dropping cases. Other jobs in the
already-running stage finish and retain their evidence; no next stage starts.

## Evaluation and interpretation

After a sealed complete training grid, replay the already-inspected 10,000-row
seed9999 test, using float32 targets, original float64 Gurobi QP/cost arithmetic,
OPTIMAL status, physical residual <=1e-6 and shipping raw q>0 fees. This is not
a fresh confirmatory test. The native own-pool membership tolerance is retained
and raw pool coordinates are never clipped.

Three green-only PNG/PDF plots show all radii and all ten seeds (mean and
pointwise 20–80% band, not confidence intervals). The minimum of the displayed
mean curve is descriptive fixed-grid hindsight, not practical radius selection
or a continuous-radius optimum. Old validation-selected results remain separate.
No new blue or SRO-FA training is performed.

## Timing and execution

Every model/configuration retains monotonic wall time, timestamps, sample size,
seed, parameters and thread/worker counts. CV SAA, training, held-out scoring,
LR selection, all-N SAA/refit, every AP/MP and final evaluation are separate.
Original training and AP journals retain epochs, iterations and solver status.
Cached nominal time is not relabelled CCG runtime. Concurrent model walls are
not summed as sweep elapsed. Final runtime reports give count/mean/median/min/max
and preserve failed/nonconverged histories separately.

Launch from this physical directory with the configured Python interpreter:

    /data/home/aviyac/ap_env/bin/python -B -u launch.py --name production_v1 --workers 16

The launcher must run **inside a named tmux session started on the host**. It
invokes the existing historical-path compatibility wrapper with only this
experiment's runs directory writable. Both original-path and canonical aliases
of all old experiments stay read-only. Logs and invocation receipts live beside
this README; detailed outputs live in runs/production_v1. No source edits after
launch. A new invocation requires a fresh output name; no implicit resume.

Planned session: `green_fullN_cv_20261011`.
Attach: `tmux attach -t green_fullN_cv_20261011`.
Presence: `tmux has-session -t green_fullN_cv_20261011`.
Progress: inspect production_v1.log and runs/production_v1/summary.json.
Completion: production_v1.invocation.json must say completed/exit0; session
absence alone does not establish success. Expected final plots are under
runs/production_v1/published/{R3,R10,shipping_R3}/.

Short source-parity tests and train-only projection probes are **not** production
performance results. Their successful and failed scratch artifacts are preserved.