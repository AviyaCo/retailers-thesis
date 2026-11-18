# Complete advisory AP/native replay logging

Final isolated revision of the user-selected replay-only rejection removal.
Native support, native-epsilon and cost disagreements are advisory. Raw q>0
fees, allocations and costs are unchanged. AP internal validity, finite output,
native physical feasibility at 1e-9 and signed AP/pool safeguards remain hard.

The first advisory run had all raw costs but omitted explicit cost discrepancies
for its five physical failures. This revision records a separate timed
`native_cost_observation` before validity rejection. A completed observation is
not a successful physical check or accepted AP result. No additional native call
is made. Counts cover physical failures as well as finalized replays.

Same fixed grid: N50/200/800, all 40/160/640 training centers, seed0 LR0.01,
radii0/0.2: 1680 cases, one worker and one thread. No training, full CCG,
selection, held-out evaluation or SRO-FA. All failures block readiness.
All prior sources, result comparison JSONs and the saved test fixture are
hash-bound; no historical outputs are overwritten. Durations are monotonic
seconds with separate scopes, UTC timestamps and solver telemetry.