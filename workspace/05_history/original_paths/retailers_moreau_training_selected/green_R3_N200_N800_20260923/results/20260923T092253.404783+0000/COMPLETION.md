# Completed: R3/no-shipping, N200 and N800

Verified2026-09-24 at08:13UTC. Production ran2026-09-23 from
09:22:53.404921 to09:44:01.791289UTC; exit0. Total measured monotonic wall
time: **1268.386533 seconds =21min8.39s**. The tmux session has exited normally.

## Status

-20/20 training-only SAA initializations completed.
-60/60 nominal LR/seed/size runs completed.
-20/20 nominal LR selections completed after60 fixed-policy validations.
-180/180 CCG configurations completed:90 N200 and90 N800, all with
  `green_gap_satisfied` at the stopping candidate. No failures,
  nonconvergence or40-MP caps.
-20/20 final radius selections completed after180 fixed-policy validations.
-No test evaluation or benchmark rerun was part of this launch.
-Shipping and R10 were not launched.
-Current manifest hash, all35 protected inputs, all180 CCG summary hashes
  and all180 selected-checkpoint hashes matched in the read-only verification.

This is convergence of the agreed **mean AP/current-pool gap stopping rule**,
not proof of global robust optimality. The earliest minimum-AP return rule
selected an earlier-than-stopping checkpoint for one configuration per size.
Consequently178/180 returned checkpoints pass their own recorded green gap;
the other two are valid returns under the agreed selection rule, not claimed
to inherit the stopping checkpoint's gap certificate.

## Environment and scope

Host gold-srv2.iem.technion.ac.il, Xeon Gold6354,72 logical CPUs; CPU execution.
Up to59 single-threaded nominal/CCG workers,20 initialization/validation workers.
Stages run sequentially. Ntotal200/800 corresponds to160/640 training and40/160
validation observations. Data seeds0–9, network seed0, split seed100000+data seed.
Full-batch Adam; LR grid0.005/0.01/0.05;3000-epoch budget, patience300.

Durations below are seconds from persisted monotonic timers. CCG time includes
the entire executed trajectory through stopping and excludes shared nominal
training. AP includes replay/checks/persistence; MP is training-call wall time.
Other verification/loading/I/O explains the difference between AP+MP and total.
Concurrent per-model times must not be summed as sweep elapsed. Runtime
differences also reflect iteration counts and changing shared-host contention;
these are not isolated scaling benchmarks.

## Parallel stage elapsed time

| Stage | Wall seconds | Human-readable |
|---|---:|---|
|SAA initialization,20 jobs|3.993812|3.99s|
|Nominal training,60 jobs|206.209766|3min26.21s|
|Nominal validation/LR selection,20 jobs|5.899489|5.90s|
|Robust neural CCG,180 jobs|1045.852028|17min25.85s|
|Radius validation/selection,20 jobs|6.362978|6.36s|
|Entire production campaign|1268.386533|21min8.39s|
|Separate preflight, excluded from production|45.502146|45.50s|
|Final OOS evaluation|Not run|Not part of launch|
|Cross-validation|Not applicable|Single paired split, not CV|
|SRO-FA partition/solver/evaluation/total|Not run|No new benchmark timing|

## Nominal training per model

Includes all three LR arms, not just winners. Evaluation is separate.

| Ntotal | n | Mean s | Median s | Min s | Max s | Epoch range |
|---:|---:|---:|---:|---:|---:|---:|
|200|30|147.571|149.628|120.658|150.864|2428–3000|
|800|30|168.290|172.782|78.268|175.696|2473–3000|

Mean training durations:2.46min and2.80min. N20026/30 and N80021/30 reached
the3000-epoch budget; the remaining13 stopped by training patience. Finishing
the nominal budget is not an optimization convergence certificate. Per-LR
training and evaluation statistics are in the [earlier detailed timing report](RUNTIME_SNAPSHOT_093049.md);
its nominal measurements are final, but its CCG measurements are superseded here.

## Final CCG per configuration, ten completed seeds per row

| Ntotal | Radius | n | Mean total s | Median s | Min s | Max s | Mean AP s | Mean MP s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|200|0|10|4.101|4.059|3.938|4.386|3.423|0|
|200|0.001|10|4.296|4.271|4.113|4.522|3.575|0|
|200|0.003|10|4.341|4.363|4.151|4.507|3.635|0|
|200|0.01|10|4.387|4.394|4.229|4.542|3.709|0|
|200|0.03|10|4.451|4.451|4.150|4.686|3.798|0|
|200|0.2|10|292.790|285.969|143.384|450.227|16.502|262.508|
|200|0.4|10|434.954|450.474|242.685|581.982|22.522|388.536|
|200|0.7|10|600.519|577.834|423.976|787.155|31.449|527.307|
|200|1.5|10|456.573|450.935|261.476|647.909|35.045|382.108|
|800|0|10|15.940|15.844|15.596|16.609|13.298|0|
|800|0.001|10|16.767|16.820|16.281|17.045|14.167|0|
|800|0.003|10|16.827|16.834|16.481|17.137|14.221|0|
|800|0.01|10|17.271|17.186|16.349|18.343|14.725|0|
|800|0.03|10|18.254|18.206|17.386|19.091|15.674|0|
|800|0.2|10|281.614|262.924|234.157|484.015|45.048|225.664|
|800|0.4|10|528.981|518.489|344.702|775.867|69.881|433.792|
|800|0.7|10|670.410|719.415|454.591|801.347|87.143|550.482|
|800|1.5|10|662.394|680.662|517.209|769.838|105.198|517.066|

Mean minutes at radii0.2/0.4/0.7/1.5: N2004.88/7.25/10.01/7.61;
N8004.69/8.82/11.17/11.04. All100 configurations at radii up to0.03
used zero MPs. N200 used138 MPs and228 AP sweeps total; N800 used96 MPs
and186 AP sweeps. Maximum MPs per configuration:6 and4 respectively.
Failed/interrupted/capped cohorts are empty, not pooled into these statistics.

## Selection and remaining evaluation

Validation-selected radius counts:

- N200: radius0 for3 seeds,0.2 for2,0.4 for5.
- N800: radius0 for4 seeds,0.2 for1,0.4 for4,0.7 for1.

These selections are frozen before any test evaluation. No OOS performance or
superiority claim is available from this campaign alone. Completing OOS and
paper-style comparisons is a separate step; no new evaluation was launched
while checking status.

Raw per-model/per-iteration timings, starts/ends, solver statuses and epoch
counts remain in [summary.json](summary.json) and the individual job artifacts.
See [runtime_aggregates.json](runtime_aggregates.json) for machine-readable
per-phase/size/radius statistics (nominal aggregates there use outer job wall,
whereas the training table above uses the nested training-call wall).
The [provenance manifest](provenance.json) records the frozen inputs/environment.

This completion report is observational; no experiment inputs or methods changed.