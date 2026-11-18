# R3 runtime comparison — 24 September 2026

## Conclusion

The latest green N200/N800 training-and-selection campaign completed in
**1,268.386533 s (21 min 8.39 s)**. The subsequent comparison pipeline took
**341.443545 s (5 min 41.44 s)**, including **226.575250 s (3 min 46.58 s)**
for fresh OOS/IS evaluation. These are separate invocations, not per-model
training times. N50 green training was reused, not rerun in either campaign.

Historical records support substantially shorter green neural training than
historical SCS training. They do **not** establish a controlled end-to-end
speedup factor. SRO-FA's recorded per-case totals are substantially shorter
than green CCG times. SAA-kNN fitting/selection time is unavailable. A claim
of being faster than every baseline is therefore unsupported.

This is a read-only comparison of persisted measurements. No new training,
solver runs, test-based selection, experimental changes or timing estimates
from file modification dates were performed.

## Latest green campaign: actual parallel elapsed time

| Phase | Completed work | Wall seconds | Human-readable |
|---|---|---:|---|
|SAA initialization|20 jobs|3.993812|3.99 s|
|Nominal neural training|60 LR/seed/size jobs|206.209766|3 min 26.21 s|
|Nominal validation / LR selection|60 evaluations|5.899489|5.90 s|
|Robust neural CCG|180 configurations|1045.852028|17 min 25.85 s|
|Radius validation / selection|180 evaluations|6.362978|6.36 s|
|Entire training-and-selection invocation|All completed|1268.386533|21 min 8.39 s|
|Subsequent fresh OOS/IS evaluation sweep|780 policies|226.575250|3 min 46.58 s|
|Entire subsequent comparison pipeline|Freeze, evaluation, verification, figure/report|341.443545|5 min 41.44 s|
|Cross-validation|Not performed; single train/validation split|N/A|N/A|

All 180 green CCG configurations satisfied the agreed stopping rule; no
failed or capped configurations are mixed into these aggregates. A stopping
gap is not a global robust-optimality certificate. All 780 fresh evaluation
policies completed. Failed preliminary comparison attempts remain separately
recorded and are not counted as successful per-model runtime.

Green ran on gold-srv2.iem.technion.ac.il, Xeon Gold 6354, 72 logical CPUs:
up to 59 single-threaded training/CCG workers, 20 initialization/validation
workers, and 32 workers for the subsequent comparison. Per-model wall times
overlap; their sum is not the campaign elapsed time.

## Nominal neural training per model

Seconds; completed records only. Green aggregates include all three LR arms
and ten seeds (30 models per size). SCS records are separate nominal timing
reruns, ten seeds per size, not timings recovered from original checkpoint
creation. Both tables exclude the cost of shared hyperparameter selection.

| Method / timing cohort | N total | n | Mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---:|---:|---:|
|Green, all nominal LR arms|200|30|147.571|149.628|120.658|150.864|
|Green, all nominal LR arms|800|30|168.290|172.782|78.268|175.696|
|SCS, separate nominal rerun|50|10|347.422|369.971|243.749|405.272|
|SCS, separate nominal rerun|200|10|2507.346|2669.527|1421.101|3178.311|
|SCS, separate nominal rerun|800|10|10851.847|11037.181|8835.195|11777.324|

Green nominal mean times are 2.46 and 2.80 minutes. For the ten
validation-selected green arms only, means are 149.801540 and 170.069779 s;
these omit rejected LR arms and are not the complete selection cost.

SCS nominal field: `nominal_training_time_s`. Its broader seed-level
`total_time_s` includes initialization/overhead, with means
347.679 / 2507.730 / 10853.045 s at N50/200/800. Separate shared LR-search
times are 1101.984 / 4266.084 / 24822.863 s. Entire nominal rerun invocation
times, including LR search and ten seeds, are 4578.971 / 29343.700 /
133353.869 s. These differently scheduled invocation times are not a
controlled comparison with green's parallel campaign.

Matching final nominal losses across reruns does not prove checkpoint
identity or recover original historical execution duration.

## Robust neural CCG per configuration

Seconds; each row has all ten seeds. Green includes every executed AP/MP
through stopping, not just work through the returned checkpoint. Shared
nominal training and later validation/evaluation are excluded.

| Method / cohort | N total | Radius | n | Mean s | Median s | Min s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
|Green|200|0.2|10|292.790|285.969|143.384|450.227|
|Green|200|0.4|10|434.954|450.474|242.685|581.982|
|Green|200|1.5|10|456.573|450.935|261.476|647.909|
|Green|800|0.2|10|281.614|262.924|234.157|484.015|
|Green|800|0.4|10|528.981|518.489|344.702|775.867|
|Green|800|1.5|10|662.394|680.662|517.209|769.838|
|Historical SCS|800|0.2|10|19541.498|13817.404|11216.162|38353.722|
|Historical SCS|800|0.4|10|36425.199|29669.026|25209.037|64391.292|
|Historical SCS|800|1.5|10|58466.450|57114.820|40892.270|93688.578|

At radius 0.4, green means are 7.25 min (N200) and 8.82 min (N800).
Original SCS N50/N200 CCG summaries at these radii store `total_time_s=0`
and lack AP/MP timings: treat these as **unavailable**, not zero-cost runs.
Original N800 SCS summaries have positive totals but no structured AP/MP
breakdown. Timed selected-radius SCS reruns exist, but do not cover a complete
ten-seed fixed-radius cohort and are not substituted into this table.

Do not add separate nominal rerun time to original CCG time and label the
sum an observed end-to-end run. Full original checkpoint-bound training and
selection timing is not established for all SCS policies.

## SRO-FA: recorded total per radius and seed

Historical `time_total`, seconds; each row contains all ten seeds. This
broader total includes training-data generation, partition construction,
the LP wrapper and historical 10,000-sample OOS evaluation. It excludes
shared test-data generation and subsequent result serialization.

| N total | Radius | n | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
|50|0.2|10|0.086349|0.085993|0.075275|0.099905|
|50|0.4|10|0.102009|0.097643|0.088993|0.119672|
|50|1.5|10|0.275758|0.277951|0.242279|0.308817|
|200|0.2|10|0.379918|0.382678|0.332767|0.426580|
|200|0.4|10|0.580922|0.578069|0.518715|0.642601|
|200|1.5|10|2.971193|2.948177|2.735070|3.260847|
|800|0.2|10|2.161293|2.155015|1.993910|2.300984|
|800|0.4|10|4.229342|4.254969|3.444928|5.038521|
|800|1.5|10|84.121188|84.718669|74.097831|91.405340|

Separate recorded fields are `time_partition`, `time_lp` (only
`m.optimize()`, not model construction), and `time_eval`. Their per-radius
statistics are in the existing comparison report. Do not add them to the
already inclusive `time_total`. Historical SRO timers use `time.time()`;
green current timers use monotonic wall time. No new SRO solve was run.

## SAA-kNN and evaluation

Ten cached kNN outcomes exist per size, but dedicated fitting, leave-one-out
k selection, and OOS timing are unavailable. A mixed evaluation invocation's
elapsed time cannot be relabeled as kNN fitting time. A rerun with explicit
phase boundaries is required to measure these missing durations.

Current green/SCS OOS replay uses the same strict Gurobi projection evaluator.
Its approximately five seconds per 10,000-row policy evaluation does not
measure the relative training backends. Per-policy OOS/IS count, mean, median,
minimum and maximum timings by method, size and radius are preserved in the
existing comparison report and structured aggregates.

## Interpretation of quality versus runtime

At the fixed illustrative radius 0.4, N800 mean OOS costs are green
19.753921, SCS 19.619379, SRO-FA 19.960933 and SAA-kNN 19.998804. This
suggests a quality/runtime trade-off: green is cheaper to train than the
historical SCS implementation, whereas SRO-FA is much faster but has higher
OOS cost at this radius. These are not test-selected deployment results or
statistically established superiority claims.

Green trains on 40/160/640 observations plus validation; historical baselines
use 50/200/800. Green uses full-batch Adam; N800 historical SCS MP uses
minibatches of 200. Stopping/selection and execution settings also differ.
Historical SCS launchers used different concurrency (20 workers for original
N800, 30 for selected timing reruns); nominal rerun records include 36 Torch
interop threads despite one intra-op thread. Equivalent historical hardware
load and complete thread settings are not established. The inspected test
draw is historical, not an untouched confirmatory test.

A controlled speedup claim would require matched data, model/optimization
protocol, resources, timing boundaries and complete selection accounting.
No such new benchmark has been launched here.

## Evidence locations

- [Green completed campaign and full CCG statistics](retailers_moreau_training_selected/green_R3_N200_N800_20260923/results/20260923T092253.404783+0000/COMPLETION.md)
- [Green structured runtime aggregates](retailers_moreau_training_selected/green_R3_N200_N800_20260923/results/20260923T092253.404783+0000/runtime_aggregates.json)
- [Current evaluation and historical SRO component timing tables](retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/results/2026-09-24T091945.582854+0000/RESULTS.md)
- [SCS N50 nominal timing example](retailers_problem/R3/N50/nominal_timing_net16_v2/seed_0_summary.json)
- [SCS N200 nominal timing example](retailers_problem/R3/N200/nominal_timing_net16_v2/seed_0_summary.json)
- [SCS N800 nominal timing example](retailers_problem/R3/N800/nominal_timing_net16_v2/seed_0_summary.json)
- [SCS N800 original CCG timing example](retailers_problem/R3/N800/rho_sweep_net16_v2/rho_0p200/seed_0_summary.json)
- [SRO N50 raw results](retailers_problem/R3/N50/sro_fa/sro_fa_results_N50_R3.json)
- [SRO N200 raw results](retailers_problem/R3/N200/sro_fa/sro_fa_results_N200_R3.json)
- [SRO N800 raw results](retailers_problem/R3/N800/sro_fa/sro_fa_results_N800_R3.json)

The independent read-only audit verified 636 baseline source/frozen-copy
bindings with no mismatches. Missing timings remain unavailable; no source
artifacts or numerical methods were modified.