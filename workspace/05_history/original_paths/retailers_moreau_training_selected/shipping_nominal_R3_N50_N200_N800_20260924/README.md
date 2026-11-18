# Shipping R3: isolated nominal stage only

## Approved boundary

Implement 30 fresh shipping-SAA initializations (N=50/200/800, seeds 0..9),
then a strict complete-grid barrier, then 90 independent nominal LR arms; STOP.
No LR selection, validation/test evaluation, AP, CCG, radius sweep, refit,
SRO-FA, benchmark or publication. Full shipping AP remains unresolved; see the
sibling shipping_exact_activation_diagnostic_20260924 diagnostic. Nothing here
purports to fix it or certify a real-arithmetic hard-fee policy.

Canonical paired truncated-normal demands and split seed 100000+s yield train
40/160/640 and heldout 10/40/160. Only training demands reach SAA or gradients.
Generation may construct the complete paired sample; saved split metadata contains
identities/hashes, not heldout values. Nominal workers load only training data and
an immutable initial checkpoint; all three LRs start with identical learned state,
network seed0, fresh Adam, 16-16 hidden layers. No legacy CV launcher is imported.

## Numerical method

Same native frozen green R3 full-batch float32 network, float64 Moreau active-set
projection/exact differentiation, zero capacity buffer, no shipment clipping.
The SCS-named historical argument dictionary is only a checked compatibility
argument: no SCS/Clarabel solve or CVXPY layer is constructed. The original
trainer is definition-adapted only to admit Ntrain40 in addition to160/640:
3000 epochs, patience300, threshold1e-4, preupdate training-best including incoming
state, final evaluated epoch updated before restoration, LR .005/.01/.05,
inventory LR .01, scheduler factor .7/patience150/threshold1e-3, gradient clip1.
Absolute economic costs remain c=.25, h=.05, b=.5, f=.1, gamma5.
Training uses .1 sum(1-exp(-5q2)). Diagnostic hard cost uses .1 per returned
component q2>0, with NO epsilon, on the very same unmodified shipment tensor.
Smooth replay is not a hard-cost improvement guarantee. Numerical signs are
not exact-real arithmetic or solver-support certification.

Training-only SAA is the historical minimum MILP with .1 per binary and
Q2<=M*z, M=max(train_x1+train_x2)+1. No positive lower link is needed: at a
minimum with positive fee, an unneeded binary can be set to zero. A bounded
optimum exists without overdelivery: once q2 exceeds a retailer's total demand,
the first recourse branch is below the shipment-independent holding branch;
reducing q2 to total demand cannot increase their maximum (nor the third branch),
relaxes capacity, and cannot increase shipping cost. Positive c,h similarly
never reward purchasing unusable inventory. Thus this M excludes no required
optimal solution. This minimization argument does NOT justify a hard-fee AP.
Require Gurobi OPTIMAL; record every optimization status/runtime/iterations/nodes
and, only when available, incumbent allocations, binaries, epigraphs and residuals.
Solver tolerances are numerical, not exact arithmetic. Zero SAA inventories can
produce -inf through the unchanged inverse softplus: KEEP FAILURE, no floor,
replacement initialization, fallback, tolerance relaxation or budget cap.

## Disposable validation and production gate

`test_nominal.py` contains bookkeeping fixtures and tiny native projection/spawn
tests, not research policies. Preflight runs these separately, then three serial
seed0 SAA+two-epoch native model children, with smooth restore/replay and finite
gradients on training rows only. No 30-MIP sweep is needed for preflight. Two
epochs are NOT a fit or a production runtime estimate. These artifacts must never
initialize production. A failed preflight stops and cannot authorize a launch.

Use ap_env Python to invoke `launch.py preflight --output ABSENT_DIRECTORY
--session UNIQUE_NAME`. This launches only disposable checks. Production is a
separate explicit invocation of `launch.py production --output ABSENT_DIRECTORY
--session UNIQUE_NAME --preflight SUMMARY_PATH --preflight-sha256 EXACT_SHA`.
Production is NOT launched by implementation/preflight. The launch gate verifies
the exact summary, unique three-size children, statuses, complete phase set,
two-epoch training/replay, child/phase/checkpoint hashes, tests, source checks and
all local Python/shell/README and historical dependency hashes, including native
Moreau binary. Mismatch aborts; it never reruns or overwrites evidence.

Launch JSON records exact command, cwd, named tmux session, persistent merged
stdout/stderr log, attach/status commands and completion channel. Adjacent exit
and invocation JSON preserve whole-child monotonic time and UTC endpoints. Use
`tmux attach -t NAME`, `tmux has-session -t NAME`, or `tmux wait-for NAME_done`;
after completion inspect the exit/log/invocation artifacts. No nohup is used.
Resource snapshot at launch intersects affinity, mounted cgroup ancestors and
cpusets, subtracts ceil(max load averages)+4 CPU slots, reserves16GiB host memory,
and budgets max(4GiB,2x preflight peak) per worker. Caps are30 initializations and
90 nominal workers, not a fixed concurrency. External/R10 workloads are unmodified.

## Runtime interpretation

Raw structured seconds use perf_counter and UTC endpoints; each job identifies
size, seed, split/network seeds, LR, method, status, iterations/epochs, thread and
worker counts. Initialization separates data generation, MILP, model creation;
nominal separates loading and training. Preflight additionally records training
replay. Sweeps record actual parallel elapsed separately, never sum concurrent
model times as elapsed. Completed-only mean/median/min/max/count aggregates are
grouped by method/size/LR; failed/incomplete/cancelled runs remain separate.

| Method | Production runtime |
|---|---|
| Shipping SAA initialization | Not run; future invocation measures it |
| Nominal neural training | Not run; two-epoch preflight is not training timing |
| Robust neural/CCG | Not run; AP unresolved |
| SRO-FA | Not run |
| Validation/test/CV | Not run; prohibited in this stage |

The mandatory repository methodology note is updated outside all frozen sources.