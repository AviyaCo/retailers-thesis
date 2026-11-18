# Black/green comparison

One shared location for figures and evaluation provenance. All links below
resolve to verified copies inside this backup. Frozen artifact bytes remain
unchanged. Back up the whole main directory, not only these shortcuts.

## Paper-sized green NN / SRO-FA / reference comparison

The [paper figures](paper_figures/README.md) show only the green Moreau NN
policy (in/out-of-sample), SRO-FA and the saved lower-bound/reference levels,
for both R3 and R10 with N=50/200/800 panels. They have no overall title, use
epsilon_N on the horizontal axis, and measure 17 by 5.5 cm including the legend.
This is a display-only revision of the current STOP results, not a new experiment.

## Current stopping-checkpoint protocol

On 28 September the return rule changed to the **stopping checkpoint for both
green and black**. The isolated [stopping-checkpoint workflow](stopping_checkpoint_20260928/README.md)
completed all 540 green and 570 black replays, including all 11 caps,
without retraining. Full preflight passed at 15:47 UTC: 70 tests, 31,865 verified
files, all 1,110 selections and 12 training-only numerical parity probes.
Whole-child preflight elapsed was 483.141406 seconds (8.052357 minutes).
The separate shared regression suite passed 65 tests in 31.007 seconds.
See [exact preflight command and attach/status commands](stopping_checkpoint_20260928/preflight_v1.launch/invocation.json),
[persistent log](stopping_checkpoint_20260928/preflight_v1.launch/child.log), and
[whole-child status/runtime](stopping_checkpoint_20260928/preflight_v1.launch/child_runtime.json).
Production completed at 16:01:33 UTC with exit 0 in tmux session
`retailers_stop_test_20260928`, using 16 single-thread workers on Linux host
`gold-srv2`. All 1,110 replays and parent raw verification passed, with
12,250,483 strict optimal physically accepted projections. All 15 published
artifact hashes were independently checked; both PNGs were visually inspected.
Comparator panels are unchanged and every curve retains all ten seeds.

- **Current R3:** [PNG](stopping_checkpoint_20260928/production_v1/published/comparison_R3_stopping.png) / [PDF](stopping_checkpoint_20260928/production_v1/published/comparison_R3_stopping.pdf).
- **Current R10:** [PNG](stopping_checkpoint_20260928/production_v1/published/comparison_R10_stopping.png) / [PDF](stopping_checkpoint_20260928/production_v1/published/comparison_R10_stopping.pdf).
- [Fixed-radius costs and per-configuration runtime tables](stopping_checkpoint_20260928/production_v1/published/RESULTS.md).

| Runtime scope | Seconds | Minutes |
|---|---:|---:|
| Preflight, whole child | 483.141406 | 8.052357 |
| Production, whole child | 858.768170 | 14.312803 |
| Parallel replay stage (included above) | 472.603779 | 7.876730 |
| Parent raw verification (included above) | 212.087518 | 3.534792 |
| Nominal training / robust CCG / SRO-FA fitting / CV | Not rerun | Not rerun |

Persisted nominal and robust training timings remain separate in the linked
runtime tables; missing historical totals, SRO-FA and CV timings are unavailable,
not zero. The sealed report was written before process exit; its pending
whole-process timing is resolved by the completed
[whole-child receipt](stopping_checkpoint_20260928/production_v1.launch/child_runtime.json)
and [pipeline summary](stopping_checkpoint_20260928/production_v1/pipeline_summary.json).
The [production launch record](stopping_checkpoint_20260928/production_v1.launch/invocation.json)
records the exact command, working directory and attach/status commands;
[production log](stopping_checkpoint_20260928/production_v1.launch/child.log).
The completed tmux session has exited; use the saved log and receipts.
The archived figures below retain their **historical minimum-AP return rule**;
they are not relabelled as stopping-checkpoint results.

## Local directory links

- [black_green_figures/](black_green_figures/): final R3/R10 PNG/PDF figures,
	plot data, results and runtime report.
- [black_green_evaluation/](black_green_evaluation/): full frozen evaluation
	code, preflight, preserved failed attempt and completed September 28 retry.
- [green_R3/](green_R3/): original R3 green/reference comparison code and results.
- [green_R10/](green_R10/): original R10 green/reference comparison code and results.

These are relative links to local archive copies. Keep archived bytes unchanged.
Raw policy evaluations and journals are in the full evaluation archives, not
only the published-figure directory. Recorded old paths can be resolved with
[archive_paths.py](../archive_paths.py); this does not certify historical scripts
as portable executables.

## Available green/reference figures

- [R3 PNG](green_R3/results/2026-09-24T091945.582854+0000/comparison_R3.png)
- [R10 PNG](green_R10/production_v1/published/comparison_R10.png)

## Black overlays

**Published on 28 September 2026.** All 570 fixed-policy evaluations and final
integrity checks passed, with 6,451,480 accepted strict optimal projections.
Whole invocation: 1,561.301943 s (26.021699 min), using 16 single-thread workers;
the parallel evaluation stage took 1,083.598741 s (18.059979 min). No training,
CCG, SRO-FA fitting or CV was performed. The earlier resource-gate failure is
preserved separately and was not a solver failure.

- [R3 black/green PNG](black_green_figures/comparison_R3_with_black.png) / [PDF](black_green_figures/comparison_R3_with_black.pdf)
- [R10 black/green PNG](black_green_figures/comparison_R10_with_black.png) / [PDF](black_green_figures/comparison_R10_with_black.pdf)
- [Results and runtime tables](black_green_figures/RESULTS.md)

- [Retry log](black_green_evaluation/production_v2_20260928.launch/child.log)
- [Retry runtime/status](black_green_evaluation/production_v2_20260928.launch/child_runtime.json)

- [Status](black_green_evaluation/STATUS.md)
- [Failed production summary](black_green_evaluation/production_v1/summary.json)
- [Whole-child runtime](black_green_evaluation/production_v1.launch/child_runtime.json)

The requested figures retain every original comparator, all ten seeds and all
eleven capped black policies. Black R10 alone extends to radius two. Costs on
the already-inspected seed9999 test are descriptive, not fresh confirmatory
evidence. The new shared plotting module has not replaced the frozen comparison.