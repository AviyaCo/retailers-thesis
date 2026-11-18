# Standalone green paper figure

Completed 23 September 2026, exit 0. [Vector PDF](green_paper.pdf) and
[300 dpi PNG](green_paper.png) show the **new green** Neural CCG OOS and
finite-pool IS curves, SRO-FA, SAA-kNN, and both reference levels.
Historical SCS, purple, and black policy curves are removed. The original
comparison and supervisor-review manuscript remain unchanged.

## Definition and checks

IS uses the same aggregation as the historical SCS figure: evaluate the fixed
returned policy on its own saved augmented training pools, maximize within
each of the 40 pools, then average the 40 maxima. This is **not the AP mean**,
and it does not use later scenarios collected after the returned checkpoint.
In particular, radius 0.2 seeds 1 and 7 use candidate 1 pools, not their later
stopping pools. Green's selective scenario insertion is unchanged; identical
aggregation does not mean identical pool construction to historical SCS.

All 90 green configurations (10 seeds, 9 radii including zero) completed.
All **9,197** fresh projection QPs returned OPTIMAL and passed the original
physical-residual gate. The unchanged strict OOS evaluator supplies network
inference, Gurobi projection, and scenario cost arithmetic. No training,
AP search, MP update, validation, test, or reference reevaluation occurred.
Four unit tests passed. All **868** protected hashes and every raw solve
journal were independently rechecked after completion. Maximum absolute
strict IS versus saved native-pool difference: **0.00000244817**.

The complete green OOS seed arrays, SRO-FA object, horizontal reference values,
and test-array identities match the previous comparison exactly. The figure
retains the largest radius, 1.5, without clipping. Visual inspection confirmed
the legend, labels, and full curves are visible; the vector PDF has one page.

| Radius | Finite-pool IS mean | Cached OOS mean |
|---:|---:|---:|
| 0 | 19.365727276 | 20.591139479 |
| 0.001 | 19.365727276 | 20.591139479 |
| 0.003 | 19.365727276 | 20.591139479 |
| 0.01 | 19.365727276 | 20.591139479 |
| 0.03 | 19.365727276 | 20.591139479 |
| 0.2 | 19.647229050 | 20.637739739 |
| 0.4 | 20.022256390 | 20.529497151 |
| 0.7 | 20.924093408 | 20.733442042 |
| 1.5 | 23.401688494 | 21.484425850 |

Small radii return the nominal checkpoint and its singleton pools, hence
the identical IS values. An IS mean below the separate clairvoyant reference
estimate is not a violation of a common-sample lower bound: training and
reference samples differ. Likewise, an uncertainty-pool IS cost exceeding
ordinary OOS cost at large radii is not an inconsistency.

## Runtime accounting

Raw monotonic seconds, UTC boundaries, seed/radius/selection identities,
solver iterations/settings, phase timers, and completed per-radius statistics
are in [summary.json](summary.json). The current run used **one CPU worker and
one numerical/solver thread** on gold-srv2.iem.technion.ac.il, shared Linux host
(Intel Xeon Gold 6354; 72 logical CPUs). Python 3.11.13, Torch 2.10.0+cu128,
NumPy 2.4.2, SciPy 1.17.1, Gurobi 13.0.1; model evaluation used CPU.

| Scope | Wall-clock seconds | Human-readable / status |
|---|---:|---|
| Current complete invocation | 41.649932 | 0.694166 min, completed |
| Tests including interpreter/import startup | 2.402710 | 2.403 s, completed |
| Input verification and freeze | 28.327535 | 0.472126 min, completed |
| New IS evaluation parent phase | 8.078856 | 8.079 s, all 90 completed; includes bookkeeping/persistence |
| New per-configuration IS evaluation, n=90 | Mean 0.074329; median 0.055048; min 0.024433; max 0.483161 | Sum 6.689640 s; not the full invocation |
| Final verification and aggregation | 2.210538 | 2.211 s, completed |
| PDF/PNG rendering | 0.625781 | 0.626 s, completed |
| Cached nominal neural training, n=10 | Mean 37.620406; median 37.241888; min 29.626559; max 45.877120 | Mean 0.627007 min; not rerun; excludes rejected LR arms |
| Cached nominal LR search | 121.796206 | 2.029937 min; 30 models, 10 workers; not rerun |
| Green robust neural/CCG | Saved phase sums by radius in table below | Work through stopping; fresh full-protocol total **unmeasured** |
| Current OOS / validation / CV | Not run | OOS scores and historical timings reused; no current evaluation timing to report |
| SRO-FA partition / solver / OOS / total | Not run | Historical curve reused; historical timing fields not audited in this figure task |

Current invocation ran 08:29:12.304615–08:29:53.954447 UTC. Its clock begins
after module imports and excludes final-summary serialization. Report writing
and the later independent checks are outside this figure-generation timing.
The first configuration includes cold-start overhead; it is not discarded.
There were no failed, incomplete, or cached-only IS evaluation runs.

### Completed IS evaluation times by radius

Every entry below is seconds, n=10 seeds, training size 40. This is **fixed-policy
evaluation time**, not training time. All nine radii were freshly evaluated.

| Radius | Mean | Median | Minimum | Maximum |
|---:|---:|---:|---:|---:|
| 0 | 0.088042 | 0.041706 | 0.025084 | 0.483161 |
| 0.001 | 0.040212 | 0.035333 | 0.035224 | 0.060791 |
| 0.003 | 0.038807 | 0.035203 | 0.035070 | 0.059902 |
| 0.01 | 0.035443 | 0.035243 | 0.024805 | 0.046483 |
| 0.03 | 0.037201 | 0.030747 | 0.024433 | 0.059281 |
| 0.2 | 0.064169 | 0.060384 | 0.044927 | 0.103013 |
| 0.4 | 0.106306 | 0.115283 | 0.065406 | 0.144204 |
| 0.7 | 0.112534 | 0.101973 | 0.085911 | 0.172067 |
| 1.5 | 0.146250 | 0.133990 | 0.085745 | 0.259009 |

### Historical green CCG work, not a fresh training runtime

| Radius | Mean reconstructed phase seconds | Minutes |
|---:|---:|---:|
| 0 | 1.929114 | 0.032152 |
| 0.001 | 1.870207 | 0.031170 |
| 0.003 | 1.883026 | 0.031384 |
| 0.01 | 1.783913 | 0.029732 |
| 0.03 | 1.855071 | 0.030918 |
| 0.2 | 160.828642 | 2.680477 |
| 0.4 | 392.621387 | 6.543690 |
| 0.7 | 604.883862 | 10.081398 |
| 1.5 | 837.722257 | 13.962038 |

These n=10-per-radius figures sum nonoverlapping historical loading/pool/AP/MP
phases through **stopping**, even if an earlier policy is returned. They exclude
shared nominal training, selection, and uninstrumented controller overhead.
Full mean/median/min/max for these saved phases are in the new summary;
per-iteration AP/MP records and original concurrency are retained in the
[previous runtime artifact](../../results/runtime_comparison.json).
The historical black source sweep took 14,942.580206 s (4.150717 h) with
16 workers; that is not the wall time of a new green sweep. Fresh end-to-end
green timing requires a rerun and is not inferred from these records.

## Suggested caption

Mean performance over ten training seeds for the no-shipping retailer problem
(R=3, f=0), with total sample budget N=50 split into 40 training and 10
validation observations. Dark green denotes out-of-sample cost for the
initial-eligible mean-gap Neural CCG protocol with minimum-mean-AP return;
light green denotes the returned policy's in-sample augmented finite-pool
objective, evaluated by strict Gurobi projection. The latter averages the
worst cost within each of that checkpoint's training pools, not over the full
uncertainty sets. SRO-FA, SAA-kNN, and the two clairvoyant references retain
their historical values and data budgets. Shading denotes the 20th–80th
percentiles of seed-level OOS costs, not confidence intervals. The 10,000-point
seed-9999 test draw has previously been inspected; comparisons are descriptive.
The 2k SAA-LP and separate 1M conditional reference use different samples.
The horizontal scale is linear through radius 0.001 and logarithmic above it.

No radius or policy was selected from this plot. In particular, the existing
test-informed grid and unequal reference protocols preclude a confirmatory
superiority claim.

## Reproduction

The completed named tmux session was `retailers_green_paper_20260923`.
Working directory: the parent of this report's directory. Exact command:
`/data/home/aviyac/ap_env/bin/python -B -u make_figure.py` with numerical thread
environment variables set to 1 by [the launcher](../launch.sh).
Persistent [log](../campaign.log) and [exit code](../campaign.exit) are retained.
While active: `tmux attach -t retailers_green_paper_20260923`; status:
`tmux has-session -t retailers_green_paper_20260923` and `tail -n 10 campaign.log`.
Outputs refuse overwrite; do not rerun into the completed results directory.

Sources, launcher, original caption draft, and README are hash-protected as
run inputs. This completion report was added afterwards and uses the final
timings. The figure data are available in [plot_data.json](plot_data.json),
with policy/pool identities in [frozen_jobs.json](frozen_jobs.json).