# Completed initial-eligible green / matched minimum-AP update

**Completed, exit 0, 23 September 2026.** All180 training-derived choices were
frozen before validation. Both methods return the earliest minimum mean AP
incumbent **through and including their own stopping assessment**; green may
stop at candidate0. Black's every-center stopping and capped status are unchanged.
No historical sources, artifacts or reference curves were overwritten.

Only three black policies needed new evaluation: radius0.7, seeds1/7/9,
candidates11/12/39. All30 native validation projections returned Solved and
all30,000 test Gurobi projections returned OPTIMAL. The 802 protected input
hashes were verified unchanged. Other evaluations were reused by exact seed,
checkpoint-byte hash and learned-state hash with matching evaluator/data
contracts; reuse across radii does not create independent observations.

## Timing of the new rule (mean per configuration)

These are reconstructed **nonoverlapping saved phase sums through stopping**,
not fresh full-protocol elapsed times. They include original loading and pool/AP/MP
phases, exclude nominal training and uninstrumented controller/selection overhead.
They count all work before stopping even when the selected checkpoint is earlier.
Ten seeds per row unless marked; same training budget40, fixed validation10.

| Radius | Green phase seconds (minutes) | Black phase seconds (minutes) | Green mean MPs | Black mean MPs |
|---:|---:|---:|---:|---:|
| .001 | 1.870207 (.031170) | 1.870207 (.031170) | 0 | 0 |
| .003 | 1.883026 (.031384) | 1.883026 (.031384) | 0 | 0 |
| .01 | 1.783913 (.029732) | 1.783913 (.029732) | 0 | 0 |
| .03 | 1.855071 (.030918) | 93.384408 (1.556407) | 0 | 1.8 |
| .2 | 160.828642 (2.680477) | 666.775254 (11.112921) | 2.2 | 6.6 |
| .4 | 392.621387 (6.543690) | 1306.310088 (21.771835) | 4.1 | 10.2 |
| .7 | 604.883862 (10.081398) | 3677.588329 (61.293139), n9 separated | 5.6 | 18.666667, n9 |
| 1.5 | 837.722257 (13.962038) | 3522.978507 (58.716308) | 7.4 | 19.4 |

Black radius0.7 seed9 remains **capped, not converged**:40MPs, recorded phase
sum10302.639037s (171.710651min), historical CCG wall11411.629000s
(190.193817min). Returning candidate39 does not erase that last iteration.
Green radius0 and all four positive radii through0.03 stop with zeroMPs:
50/90 configurations. The corresponding nominal policy is not newly trained.

Full n/mean/median/min/max for every radius, AP/MP/total phase scope and
separate status cohort are in [RESULTS.md](RESULTS.md) and
[runtime_comparison.json](runtime_comparison.json). Raw per-iteration phases,
timestamps, counts and cached nominal histories are retained in those artifacts.

| Other timing scope | Measured seconds / interpretation |
|---|---|
| Cached nominal training, n10 | Mean37.620406s (.627007min), median37.241888, min29.626559, max45.877120; shared green/black initialization, excludes rejected LR arms |
| Prior nominal LR-search campaign | 121.796206s (2.029937min), 30 trainings,10 workers; not rerun |
| Prior black CCG sweep | 14942.580206s (4.150717h),16 workers; not the elapsed time of a new min-AP campaign |
| Current no-solver preflight | 30.974498s (.516242min);9 tests and complete archive checks |
| Current update invocation | **50.974514s (.849575min)**, including reconstruction, validation, test and reporting |
| Reconstruction/freeze | 31.164853s |
| Validation plus radius-selection parent phase | .672499s; three new evals plus cached lookup/serialization |
| Three new native val10 calls, n3 | Mean.005824s; median.002134; min.002109; max.013228; model loading measured separately in per-record outer timers |
| Radius selection | .001176s, nested in validation parent phase |
| Three new test calls, n3 | Mean5.031908s; median4.867627; min4.863434; max5.364664 |
| New test-evaluation parent phase | 15.724281s (.262071min), three sequential calls plus persistence |
| Test lookup/evaluation parent phase | 15.750403s, includes previous row; do not add twice |
| Final input verification | 2.098892s |
| Report/render | .873191s |
| New nominal / MP / AP / CV / SRO-FA | Not run; SRO-FA partition/solver/evaluation/total timings unavailable |

The definitive final invocation timestamps are07:56:05.540085--07:56:56.514467UTC.
The invocation clock starts at entry to main, after module imports, and excludes
the final summary serialization; it is not shell-start-to-process-exit timing.
The reporting-phase snapshot predates caller finalization; [summary.json](summary.json)
is authoritative for completed phase and overall times. New work ran one CPU
worker, one numerical/solver thread, on gold-srv2 (Intel Xeon Gold6354 shared Linux
host). Historical CCG used16 single-thread workers; concurrent walls must not be
summed into elapsed sweep time. No fresh green/black full-training speedup is claimed.

## Costs and reference comparison

Per-seed **validation-selected** mean test cost:
- Green:20.544920252 (previous green20.547568019; different allowed stopping prefix).
- Black:20.405601829 (selected-radius results unchanged; some fixed-radius returns change).
- Yellow cached kNN reference:20.797716108, not an equal-protocol paired benchmark.

Green is below yellow at8/9 displayed radii including nominal (7/8 positive
radii), and below red SRO-FA at6/7 exactly shared radii. It is **above both at1.5**.
Black is below red at all7 shared radii, but also above yellow at1.5.
Neither red nor yellow has been rerun, interpolated or optimized to favor a color.
There is no single fair ``selected red'' number computed by this update; its
per-radius curve and historical protocol are preserved.

The new green candidate at radii0.001/.003/.01/.03 is the original nominal
policy, mean test20.591139479. At.2/.4/.7/1.5 the green curve remains
20.637739739/20.529497151/20.733442042/21.484425850. Green's lower computation
does not prove that continuing optimization could not improve the robust
objective. Neither stopping rule certifies global nonconvex optimality.

The already-inspected seed9999 test and test-informed historical radius grid
support **descriptive comparisons only**, not a confirmatory superiority claim
or justification to retune radii on test outcomes. Validation itself was reused
for historical LR selection and current radius selection. The red/yellow data
and selection budgets differ. No final method choice is made automatically.

## Reproduction and artifacts

Session `retailers_min_ap_20260923` completed normally; it is no longer running.
Exact command: `/data/home/aviyac/ap_env/bin/python -B -u run_update.py` from
the parent directory of this report, with numerical thread environment variables1.
Launcher [../launch.sh](../launch.sh), log [../campaign.log](../campaign.log),
exit [../campaign.exit](../campaign.exit), preflight [../preflight.json](../preflight.json).
While active, attach using `tmux attach -t retailers_min_ap_20260923` and check
with `tmux has-session -t retailers_min_ap_20260923`. Existing outputs refuse
overwrite; this is not an instruction to relaunch into the completed directory.

- [Updated figure](comparison.png)
- [Full report and distributions](RESULTS.md)
- [Raw completed summary and phase times](summary.json)
- [Training-only checkpoint decisions](training_only_choices.json)
- [Radius choices frozen before test lookup](radius_choices_before_test.json)