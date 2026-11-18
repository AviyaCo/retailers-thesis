# Correction: explicit N50-only SCS timing benchmark exists

The earlier comparison omitted the dedicated N50 R3 width16 benchmark under
[timing_test](../../../retailers_problem/R3/N50/timing_test/run_timing_seed.py).
It is distinct from the ten selected-radius timing reruns already reported.
SCS timing is available; only the original plotted sweep's exact per-policy
timings remain unavailable. No additional training was needed to establish this.

## Verified dedicated benchmark

Nine completed configurations: data seeds10,11,12 at each radius0.001,0.3,1.5;
50 training observations, network seed0, width16--16, network LR0.05 and inventory
LR0.01, full-batch SCS/CVXPYLayers training, maximum3000 epochs/patience300.
Each job starts from a fresh SAA solution and fresh nominal training; no cached
nominal checkpoint. The launcher starts nine jobs concurrently. AP uses one
Gurobi thread; numerical-library/diffcp worker counts and hardware metadata
were not recorded in these JSON files. No historical sweep elapsed time is
inferred by summing the concurrent durations.

All nine JSON files agree with completed logs, all AP sweeps report50 centers,
and AP/MP sums agree with reported CCG times to rounding. All runs record
convergence; their final signed discrepancies are positive and below1%.
The implementation uses a signed stopping diagnostic and post-update weights
against pre-update training losses, rather than the current corrected Moreau
protocol. No source modification or retraining is performed by this correction.

| Radius | n | Mean nominal minutes | Mean CCG minutes | Mean total minutes | Actual MP range |
|---:|---:|---:|---:|---:|---:|
| .001 | 3 | 5.96 | 15.38 | 21.36 | 1–2 |
| .3 | 3 | 6.01 | 79.86 | 85.90 | 2–4 |
| 1.5 | 3 | 5.97 | 152.86 | 158.86 | 5–7 |

Raw statistics below are seconds, mean / median / min / max, three completed
runs per radius:

| Radius | Nominal training | CCG phase sum | Total wall time |
|---:|---|---|---|
| .001 | 357.373333 / 410.31 / 237.57 / 424.24 | 922.87 / 428.97 / 423.64 / 1916.00 | 1281.53 / 849.14 / 667.91 / 2327.54 |
| .3 | 360.716667 / 410.32 / 239.60 / 432.23 | 4791.733333 / 6445.60 / 1240.58 / 6689.02 | 5153.78 / 6879.16 / 1481.50 / 7100.68 |
| 1.5 | 358.163333 / 410.29 / 239.88 / 424.32 | 9171.80 / 10181.13 / 6550.72 / 10783.55 | 9531.393333 / 10422.41 / 6976.41 / 11195.36 |

These are historical measured wall-clock intervals using time.time(), rounded
to0.01seconds, not perf_counter measurements. CCG is the sum of recorded AP
and MP phase times; total includes SAA, nominal and surrounding overhead, but
not test evaluation. Clock-adjustment immunity, exact start/end timestamps and
full historical runtime environment are not available; do not invent them.

## How to use it in the three-way comparison

This is valid historical timing evidence and should be displayed, rather than
replaced by “no SCS timings.” It is not a paired solver-only benchmark:
SCS uses seeds10–12 and50 training samples; current Moreau uses seeds0–9 and
40train/10validation, with different selection, AP and stopping settings.
Radius.3 is absent from the current Moreau grid; no interpolation is supplied.

At the two common radii, the recorded CCG phase means are:

| Radius | SCS dedicated timing (n3) | Black Moreau phases (n10) | Green Moreau phases (n10) |
|---:|---:|---:|---:|
| .001 | 15.38 min | 1.87 s | 45.95 s |
| 1.5 | 152.86 min | 58.72 min | 13.96 min |

Moreau phases include loading/pool evaluations and exclude some controller
overhead; green includes recovered timing. These are descriptive runtime
comparisons across recorded protocols, not an exact solver speedup. Nominal
training is separate. New OOS evaluation, CV and SRO-FA were not part of the
N50-only SCS timing benchmark; their times are unavailable/not applicable.

## Sources

- [Timing worker](../../../retailers_problem/R3/N50/timing_test/run_timing_seed.py)
- [Nine-job launcher](../../../retailers_problem/R3/N50/timing_test/launch_timing.sh)
- [Original timing collector](../../../retailers_problem/R3/N50/timing_test/collect_timing.py)
- [Example raw result](../../../retailers_problem/R3/N50/timing_test/rho_0p001/seed_10_timing.json)
- [Previously reported Moreau and selected-radius SCS timings](REPORT.md)

No frozen experimental outputs, training sources, plotted values or supervisor
manuscript were modified. This is a correction to the timing evidence inventory.