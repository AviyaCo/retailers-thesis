# AP replay: both failures reproduced, native batch discrepancy captured

Completed 28 September 2026, 09:25:41.733786--09:25:49.727803 UTC.
**Diagnostic execution completed; the historical AP/controller failures remain
failures. No campaign repair or numerical-policy change was performed.**

## Fixed experiment and reproduction

Three prespecified saved training cases, each executed once unchanged and once
with logging/capture only. Same policies, centers, radius, Gurobi numerical
parameters, epsilon, native projection, and fee rule. Fresh model instances;
the preceding MP/AP sequence was not rerun. No training, validation/test, CCG
continuation, refit, SRO-FA or CV. Baseline/logged are diagnostic repetitions,
not independent performance samples.

All six calls exactly reproduced the historical AP status, bound, full incumbent,
binary flags, AP shipments, link residual, native shipments and native hard cost.
The two activation-failure cases failed their AP postcheck in both modes; the
N800 AP returned in both modes and then failed the unchanged duplicate guard.
Thus there were **four AP-postcheck failures and two controller-recheck failures**,
not six accepted APs. Logging changed no compared numerical result.

## 1. Activation-link failures: solver itself confirms violation

For N200/seed5/radius1.5/candidate1/sample6 and N50/seed7/radius1.5/candidate9/
sample33, the exported original constraint is correctly present:

$$q_3-10^{-6}z_3\geq0.$$

Gurobi returns $q_3=0,z_3=1$ in both cases. Direct row recomputation and Gurobi's
own `MaxVio`, `ConstrVio`, and shipping-row `Slack` give **1e-6**, versus effective
`FeasibilityTol=1e-9`. Both solver logs explicitly say:

> Optimal solution found (tolerance 1.00e-04)
> Warning: max constraint violation (1.0000e-06) exceeds tolerance

The first tolerance is the optimality MIP-gap tolerance, not permission for the
row violation. Gurobi status2/zero gap therefore does not establish acceptable
original-model feasibility. Raw `.X` values, exported solutions and postcheck
values agree: no extraction rounding/clamp created the violation. N200 additionally
has a first-layer equality residual about1.506263e-7 above the configured
feasibility tolerance. The postcheck was correctly rejecting numerical infeasibility.

**Unresolved:** which internal numerical/presolve/postsolve mechanism led to
these returned values. Presolve and scaling remained automatic; no intervention
was made. This evidence is not enough to call it a known solver bug or to claim
that disabling presolve would fix it. Do not relax the guard or publish the
failed incumbent/bound as a valid certificate based on optimal status alone.

Evidence: [N200 log](replay_v1/N200_seed5_radius1.5_candidate1_sample6/logged/solver.log),
[N200 raw capture](replay_v1/N200_seed5_radius1.5_candidate1_sample6/logged/result.json),
[N50 log](replay_v1/N50_seed7_radius1.5_candidate9_sample33/logged/solver.log),
[N50 raw capture](replay_v1/N50_seed7_radius1.5_candidate9_sample33/logged/result.json).
Original LP/MPS and raw solver solution are saved beside each logged result.

## 2. Sample271: exact same point, different numerical execution context

N800/seed7/radius.2/candidate1/sample271: both modes reproduced all640 saved pool
maxima. The rerun AP witness exactly equals its historical point and own-pool
index1, flattened index538 within the1265-point pool evaluation. Same learned
policy and capacity. Passive hooks captured:

| Quantity | Full-pool batch | Native AP singleton |
|---|---:|---:|
| First network target | 41.9108772277832 | 41.9108772277832 |
| Second network target | 21.800281524658203 | 21.80028533935547 |
| Third network target | 21.800283432006836 | 21.800283432006836 |
| Capacity | 20.110593795776367 | 20.110593795776367 |
| First shipment | 20.110593795776367 | 20.11059315999349 |
| Second shipment | 0 | 1.2715657540240954e-6 |
| Third shipment | 0 | -6.357828787884046e-7 |
| Strict-positive shipping fee | 0.1 | 0.2 |
| Native hard cost at this point | 12.393165702372789 | 12.493165702372789 |

The second target differs by3.814697265625e-6. The singleton's positive second
shipment adds a0.1 fee; the recorded hard-cost difference is0.1 to floating
precision. The singleton's small negative third shipment is retained unchanged
and passes the approved physical1e-6 guard; no clipping is applied. The AP itself
returns three positive shipments, charges0.3, and has cost/bound
12.593166339263046. Its link residual is0; Gurobi's maximum violation is only
3.552714e-15. This case is **not** an activation-link failure.

The repeated-point controller rejection therefore reproduces a numerical
evaluation disagreement, not an absent/new point. The native target difference
is directly observed upstream of projection, and fee-count arithmetic explains
the native cost jump. The exact matrix kernel/operation ordering is not traced;
a fixed-target control separating all projection effects was not run in this
new diagnostic. Do not claim bitwise batch invariance or a universal direction
of bias. This is not an exact native-hard robust certificate.

[Full same-point evidence](replay_v1/N800_seed7_radius0.2_candidate1_sample271/logged/result.json).

## Runtime and execution environment

Linux gold-srv2.iem.technion.ac.il; Intel Xeon Gold6354,72 logical CPUs; Python
3.11.13, Torch2.10.0, NumPy2.4.2, Gurobi13.0.1, Moreau CPU0.4.0. One sequential
worker, one Torch/Gurobi thread, Gurobi Seed0. Data/network seeds and candidate
identities are recorded with every result; no new random data are generated.

| N / seed / radius / sample | Mode | Count | Total case s | AP call s | Optimize-only s | Pool evaluation s | Capture s |
|---|---|---:|---:|---:|---:|---:|---:|
| 200 / 5 / 1.5 / 6 | Baseline | 1 | .445897 | .418480 | .009450 | not run | not run |
| 200 / 5 / 1.5 / 6 | Logged | 1 | .065618 | .061094 | .008540 | not run | .005124 |
| 50 / 7 / 1.5 / 33 | Baseline | 1 | .056069 | .052127 | .009597 | not run | not run |
| 50 / 7 / 1.5 / 33 | Logged | 1 | .067124 | .062977 | .010069 | not run | .004791 |
| 800 / 7 / .2 / 271 | Baseline | 1 | .057198 | .031872 | .004443 | .020161 | not run |
| 800 / 7 / .2 / 271 | Logged | 1 | .063688 | .040991 | .004639 | .018345 | .004904 |

Each diagnostic configuration/mode has one execution, so its runtime mean,
median, minimum and maximum equal the listed total. These are **diagnostic
durations of reproduced failures**, not a completed-model aggregate. Baseline
first-call overhead and logged export/capture overhead are present, with separate
scopes retained; do not interpret logged/baseline time differences as speedups.
Logged AP `solver_wall_time_s` wraps exports/capture too: optimize-only values
above use the inner timer. Raw standalone capture snapshots have timing status
`running` because written inside the capture block; the final embedded capture
in each result supplies the authoritative completed status and elapsed time.

| Overall/model scope | Measured duration or status |
|---|---|
| Diagnostic body | 2.925375 s (0.048756 min) |
| Whole invocation, imports/teardown included | **7.994011 s (0.133234 min)** |
| Nominal neural training | Not rerun; historical training timings remain in campaign artifacts |
| Robust neural/CCG total | Not rerun; only fixed AP steps above |
| Native pool/singleton evaluation | Pool separately above; singleton included in AP replay timer |
| SRO-FA / CV / validation / test | Not run; unavailable for this diagnostic |

Nested timings are not added as whole elapsed. Every solve records solver status,
iterations, nodes, UTC/performance-counter times and unchanged numerical settings.

## Launch, provenance and review

Working directory: /data/home/aviyac/retailers_moreau_training_selected/shipping_ap_replay_20260928.
Exact launch: `/data/home/aviyac/ap_env/bin/python -B launch.py --output replay_v1 --session shipping_ap_replay_20260928`.
Session: `shipping_ap_replay_20260928` (finished; no active replay).
Persistent log: [replay_v1.log](replay_v1.log).
Exact child command/cwd: [replay_v1.launch.json](replay_v1.launch.json).
Whole-child timing: [replay_v1.invocation.json](replay_v1.invocation.json).
Raw summary: [replay_v1/summary.json](replay_v1/summary.json).

Monitoring commands during execution: `tmux attach -t shipping_ap_replay_20260928`
and `tmux has-session -t shipping_ap_replay_20260928`. Sources were isolated and
frozen by the run; no historical computational artifact was changed.
Independent read-only review verified3 local-source,13 direct-input,5176
frozen-source and364 production-input bindings, all six substantive historical
comparisons and exact point identity. The numerical causes still unresolved
above were not replaced by assumptions. No further solver-setting experiment
or numerical-policy change has been performed.