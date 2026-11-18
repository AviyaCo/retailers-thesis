# Prospective green R3, no shipping: N200 and N800

Authorized 23 September 2026. Fresh isolated runs, not a resume of SCS or
relabeling of black trajectories. Existing experiments are immutable.

## Frozen statistical and numerical protocol

- R=3, f=0, total N=200/800 with one paired 80/20 split: 160/40 and 640/160.
  Exact canonical truncnorm generator, data seeds0–9, split seed100000+s,
  network seed0, width16–16. No test data accessed; no refit or five-fold CV.
- Training-only SAA LP initialization, strictly OPTIMAL. All three nominal
  LRs .005/.01/.05 start with identical per-seed weights. Fresh full-batch
  Adam per LR/MP, inventory LR.01, clipping1, 3000 evaluated epochs maximum,
  patience300, training-loss scheduler factor.7/patience150/threshold1e-3.
  Save evaluated preupdate training-best states with improvement>1e-4;
  incoming policy eligible. No validation in trainer or CCG.
- Native Moreau0.4.0 projection and existing R3 AP/numerical helpers reused
  unchanged; zero capacity buffer. No fallback/clipping/retry/relaxed tolerance.
- Radii0/.001/.003/.01/.03/.2/.4/.7/1.5. Fresh independent CCG starts from the
  validation-selected nominal LR winner. All centers must pass the original
  strict AP/status/replay/membership/duplicate gates. Pool insertion unchanged.
- **Live green:** assess candidate0; stop immediately at first
  abs(APmean-poolmean)/max(abs(poolmean),1e-9)<.01. Return earliest minimum mean
  AP incumbent through and including that stop. Separate stop/return IDs and
  gaps. No MP after stopping. At most40 MPs plus initial/terminal AP.
- Repeated-state/no-addition stagnation or cap without a green pass is explicitly
  nonconverged; no selected policy published. Solver failure stays failed.
  Parent does not publish authoritative radius selections from a partial grid.
- All60 nominal arms finish before LR validation; all180 CCG jobs finish with
  green stops before final validation and per-seed minimum-cost radius selection
  (smaller radius tie). Validation reuse for LR/radius is disclosed tuning, not
  an unbiased performance estimate. Saved test9999 is already inspected;
  future OOS comparisons remain descriptive. This launch does not evaluate test
  data or train kNN/SRO-FA references.

## Concurrency and timing

All numerical/solver workers are single-threaded. Initialization20 tasks,
nominal60 tasks, CCG180 tasks and validation20 tasks use spawned workers.
Maximum concurrency is selected once at launch from affinity/cgroup limits,
sampled shared-host load (ceil max1/5/15-minute load plus4 CPU reserve), and
available memory (16GiB host reserve, at least4GiB/worker or twice measured
preflight peak). This is not a guarantee on future MIP-tree memory. No solver
iteration/time budget is changed for resources. No other user's processes are
modified. Production resource snapshot and chosen limits are frozen in provenance.

All nominal/SAA/training/validation/selection/CCG phases have monotonic raw
seconds, UTC boundaries, sizes/seeds/radii, solver status/iterations and worker
counts. Every AP and MP has a separate timing; CCG excludes shared nominal
training and includes work through stopping, not just returned policy. Failed,
capped/stagnated and completed cohorts are separate. Per-model wall times are
not summed and presented as concurrent sweep elapsed. Model timings exclude
their final JSON serialization; outer worker/phase timers record broader scopes.
Postrun runtime aggregates preserve n/mean/median/min/max. SRO-FA/CV/test are
not run, not invented zero-second model runtimes.

## Preflight and reproducibility

preflight.py checks solver-free fixtures, actual training-only LPs, short
disposable training/replay, radius0 live-controller coverage, AP checks, and
a constructed maximum-pool resource fixture (not experimental AP data).
No preflight weights are reused in production. These short checks do not
change the production3000/40 budgets and do not measure full-training speed.

Run working directory is this folder. Production exact command:
`/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py --preflight ABSOLUTE_PREFLIGHT_SUMMARY`.
`bash launch.sh ABSOLUTE_PREFLIGHT_SUMMARY` records stdout/stderr in
campaign_TIMESTAMP.log and exit status in campaign_TIMESTAMP.exit. Named
production tmux session: `retailers_green_R3_200_800_20260923`.
Attach: `tmux attach -t retailers_green_R3_200_800_20260923`;
status: `tmux has-session -t retailers_green_R3_200_800_20260923`.
launch.json records actual output/worker limits; per-job logs persist beside
results. Fresh timestamped output only; no automatic resume or retries.

Shipping is not part of this launch. Its approved cost is f times1{q2>0}, with
smooth MP training and hard AP/assessment. A finite closed MILP cannot directly
represent strict positivity in general; the adversarial supremum need not be
attained. No epsilon cutoff, positive minimum shipment, or paid-zero relaxation
has been authorized as a replacement, so shipping remains paused.