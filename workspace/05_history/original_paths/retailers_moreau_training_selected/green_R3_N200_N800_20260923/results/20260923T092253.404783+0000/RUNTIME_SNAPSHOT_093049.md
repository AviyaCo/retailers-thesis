# Interim runtime detail: R3, N200/N800

Snapshot: **2026-09-23 09:30:49.170619 UTC**. Campaign started
09:22:53.404921 UTC; still running. UTC timestamp difference:475.765698s
(7min55.77s), **not a final monotonic campaign measurement**.
All completed durations below are persisted monotonic wall-clock seconds.

At this snapshot:60/60 nominal runs and104/180 CCG configurations collected
as completed; no collected failed/nonconverged jobs. The remaining76 CCG
configurations were unfinished or queued and are excluded from completed-run
statistics. This is not a forecast of their runtimes or convergence.

## Execution and timing scopes

- R3, no shipping; Ntotal200/800 means Ntrain160/640 and Nval40/160.
- Data seeds0–9; network seed0; split seed100000+data seed.
- Full-batch Adam, nominal LR0.005/0.01/0.05, inventory LR0.01;
  maximum3000 evaluated epochs, patience300. CCG permits40 MPs plus terminal AP.
- Shared host gold-srv2.iem.technion.ac.il, Xeon Gold6354,72 logical CPUs;
  CPU execution, up to59 simultaneous nominal/CCG workers, one numerical and
  solver thread each; initialization/validation use up to20 workers.
- Nominal training includes setup, epochs, journals, checkpointing and restoration;
  excludes final JSON serialization and separate train/validation evaluation.
- CCG totals include loading, all executed AP/MP work through the stop,
  verification and persistence, not merely work through the returned checkpoint.
  Shared nominal initialization/training is excluded, not charged again per radius.
- AP time includes native replay, integrity checks and sample persistence, not
  only Gurobi solver runtime. MP time is the training-call wall time.
- AP+MP can be smaller than CCG total because of other checks and I/O.
- Concurrent per-model durations must not be summed and called sweep elapsed.

## Parallel stage elapsed time

| Stage | Completed work | Wall seconds | Human-readable |
|---|---|---:|---|
| Preflight, before production |36 tests plus both numerical size checks|45.502146|45.50s|
| Training-only SAA initialization |20 jobs|3.993812|3.99s|
| Nominal training stage |60 LR/seed/size jobs|206.209766|3min26.21s|
| Fixed nominal validation and LR selection |20 jobs,60 policy evaluations|5.899489|5.90s|
| Robust neural/CCG stage |104/180 collected complete|Pending|Running; final elapsed unavailable|
| Radius validation/selection |Not started at snapshot|Pending|Not measured yet|
| Final OOS evaluation |Not part of this launch|Not run|No test access|
| Cross-validation |Single paired split, not CV|Not applicable|No CV timing|
| SRO-FA partition/solver/evaluation/total |Not part of this launch|Not run|No new benchmark timing|

The CCG stage began09:26:29.545074UTC; timestamp elapsed at snapshot was
259.625545s (4min19.63s), not its final monotonic duration. Stage times include
process-pool startup/teardown; therefore short validation calls do not imply
the whole parallel validation stage takes only milliseconds.

## Nominal training: ten seeds per size/LR

All entries are completed training calls, not queue wait or validation time.

| Ntotal | Network LR | n | Mean s | Median s | Min s | Max s | Evaluated epochs, min–max |
|---:|---:|---:|---:|---:|---:|---:|---:|
|200|0.005|10|145.956073|148.545875|130.206712|150.864416|2594–3000|
|200|0.01|10|146.932876|149.686960|120.658370|150.650212|2428–3000|
|200|0.05|10|149.825400|149.810415|149.042159|150.498188|3000–3000|
|800|0.005|10|172.068380|173.037508|159.095002|175.696126|2497–3000|
|800|0.01|10|172.407222|172.979931|167.758590|174.611765|2769–3000|
|800|0.05|10|160.395143|170.130002|78.268261|174.192956|2473–3000|

All30 arms combined: N200 mean147.571450s (2.46min), median149.627769s,
range120.658370–150.864416s; N800 mean168.290248s (2.80min),
median172.781994s, range78.268261–175.696126s.

Stopping counts: N20026 at the3000-epoch budget and4 by training patience;
N80021 at budget and9 by patience. Budget completion does not certify
optimization convergence. The unusually fast N800 arm was seed9/LR0.05:
2789 epochs in78.268261s,09:25:03.394757–09:26:21.663006UTC. It started late
after another worker became free and overlapped less of the fully occupied
phase; differing contention is a plausible contributor, not a controlled
speedup result.

For the validation-selected nominal arms only (10 per size), mean training
time was149.801540s for N200 and170.069779s for N800. These exclude rejected
LR arms and must not be described as the full hyperparameter-search cost.

## Completed CCG configurations by radius

Every row below is a separate completed-run cohort. Counts at radius0.2
are incomplete and favor early finishers; do not extrapolate them to all seeds.

| Ntotal | Radius | n/10 | Mean total s | Median s | Min s | Max s | Mean AP s | Mean MP s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|200|0|10/10|4.101434|4.058549|3.937547|4.386118|3.423213|0|
|200|0.001|10/10|4.295686|4.271014|4.113239|4.522029|3.575482|0|
|200|0.003|10/10|4.340859|4.362591|4.150713|4.506569|3.634818|0|
|200|0.01|10/10|4.386743|4.393581|4.228857|4.542099|3.708810|0|
|200|0.03|10/10|4.451386|4.451112|4.149529|4.685711|3.798311|0|
|200|0.2|2/10|153.661993|153.661993|143.384295|163.939691|12.388508|133.903798|
|800|0|10/10|15.940348|15.843909|15.596135|16.609379|13.297688|0|
|800|0.001|10/10|16.767357|16.819831|16.280943|17.044772|14.166954|0|
|800|0.003|10/10|16.826721|16.834473|16.481388|17.137034|14.220893|0|
|800|0.01|10/10|17.271492|17.186482|16.348614|18.343165|14.724542|0|
|800|0.03|10/10|18.253867|18.206001|17.385657|19.091472|15.674500|0|
|800|0.2|2/10|235.191198|235.191198|234.157411|236.224984|42.871793|183.075760|

Radii0,0.001,0.003,0.01,0.03 all stopped at their initial AP assessment,
with zero MPs. Radius0.2 completed means correspond to2.56min (N200) and
3.92min (N800). Radii0.4,0.7,1.5 had no collected completed configurations
at this snapshot; no final mean/median/range is available.

### Positive-radius per-seed and per-iteration evidence

These are the four completed radius0.2 jobs at the snapshot. AP lists are
in candidate order; MP lists are in update order and include call overhead.

| Ntotal | Seed | CCG total s | AP sweep seconds | MP call seconds | MP evaluated epochs |
|---:|---:|---:|---|---|---|
|200|7|143.384295|4.107000,5.339911|128.806138|2993|
|200|4|163.939691|4.750883,5.695108,4.884114|128.354626,10.646832|3000,301|
|800|0|236.224984|18.922978,23.236829|184.903511|2148|
|800|9|234.157411|19.943551,23.640227|181.248008|2102|

MP training is the largest recorded component in these four finished
positive-radius jobs. This observation does not establish the eventual
bottleneck or total duration for larger radii.

## Separate initialization/evaluation timings

| Operation | Ntotal | n | Mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---:|---:|---:|
|SAA initialization call|200|10|0.072854|0.064899|0.043008|0.103976|
|SAA initialization call|800|10|0.194561|0.197395|0.154621|0.251990|
|Selected-checkpoint train replay|200|30|0.014042|0.012957|0.007344|0.024077|
|Selected-checkpoint train replay|800|30|0.011035|0.010368|0.007835|0.016090|
|One nominal validation evaluation|200|30|0.003410|0.003166|0.002883|0.004391|
|One nominal validation evaluation|800|30|0.004891|0.004793|0.003934|0.006865|
|Three-arm LR validation/selection job|200|10|0.052167|0.056063|0.037054|0.065448|
|Three-arm LR validation/selection job|800|10|0.064128|0.067808|0.044236|0.077764|

## Evidence

- [Live parent summary](summary.json): all completed nominal/selection records
  and compact collected CCG records, with seeds, starts/ends and status.
- [Frozen environment/provenance](provenance.json): source identities,
  worker policy, environment and versions.
- Detailed CCG summaries under N200/N800 contain every iteration's AP/MP
  timings; training artifacts preserve epoch journals. These structured raw
  timings are the evidence, not estimates inferred from logs or older runs.
- [Launch and attach/status instructions](../../STARTUP.md).

This report is a post-launch observation, not an experiment input. No solver,
optimizer, data, selection protocol or running source was changed to produce it.