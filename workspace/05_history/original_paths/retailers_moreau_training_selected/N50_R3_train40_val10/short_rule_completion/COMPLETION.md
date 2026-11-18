# Completed graph and three-way runtime interpretation

**Timing evidence correction:** a separate nine-run N50-only SCS benchmark was
subsequently located and verified. See [SCS timing correction](SCS_TIMING_CORRECTION.md).
The original plotted-policy timing remains unavailable, but SCS timing evidence
also includes this dedicated benchmark, not only the selected-radius reruns below.

The original generated report precedes parent finalization. Final authoritative
summary: status completed, exit0, 2026-09-16T13:23:38.941813+00:00 through
2026-09-16T13:30:54.524292+00:00. Overall current invocation435.582503seconds
(7.259708minutes), separate prelaunch freeze12.401051seconds.

All31 continuations completed after one MP and one new40-center AP sweep each:
1240 new strict AP solves, 50833 evaluated MP epochs, seven minimum-AP selections
returned the initial policy and24 returned the new policy. These are legitimate
training/AP selections, not missing-point substitutions. The31 fresh policy
evaluations accepted310000 OPTIMAL QPs. Combined with59 explicitly reused
evaluations, all nine green/reference radius cohorts now have ten seeds.
Original nongreen plotted arrays and previous green points are exactly unchanged.

## Per-model runtime comparison

Seconds below use the same measured nonoverlapping phase scope for both Moreau
rules: original loading, AP, MP and pool evaluations through the stopping AP.
Nominal training, selection and uninstrumented overhead are excluded. Green
mixes saved prefixes and saved-initial-plus-current continuations; neither
column is a freshly measured complete shorter-training campaign. See the full
report for n/mean/median/min/max, raw phases, saved/current split and statuses.

| Radius | Black phase mean seconds | Green phase mean seconds | Green mean MPs |
|---:|---:|---:|---:|
| .001 | 1.870207 | 45.949694 | 1 |
| .003 | 1.883026 | 42.647503 | 1 |
| .01 | 1.783913 | 45.617094 | 1 |
| .03 | 93.384408 | 42.630862 | 1 |
| .2 | 666.775254 | 160.828642 | 2.2 |
| .4 | 1306.310088 | 392.621387 | 4.1 |
| .7 | 3677.588329 (9 separated); 10302.639037 (1 capped) | 604.883862 (10 mean-stopped) | 5.6 |
| 1.5 | 3522.978507 | 837.722257 | 7.4 |

Black is cap40, not40 MPs at each radius. At .001/.003/.01 it executes zero
MPs, whereas green forbids stopping at the initial AP and executes one.
The black capped seed9/.7 actual end-to-end CCG duration is11411.629seconds;
its phase sum above excludes overhead. Do not call it converged.

Exact plotted fixed-radius SCS timings: unavailable (230zero placeholders),
rerun required. Separate unequal validation-selected-radius SCS timing reruns
(n10): mean2646.285778s(44.104763min), median2128.967273s,
min449.094197s, max6244.696002s, excluding nominal training. These are UNMATCHED
references, not evidence of an exact solver speedup. Training sizes, policies,
radii, settings and concurrency differ.

| Additional category | Measured time / scope |
|---|---|
| Moreau nominal selected winner | Mean37.620406s per seed; excludes rejected LR candidates |
| SCS nominal separate rerun | Mean347.422133s per seed; unmatched50train vs40 |
| Prior Moreau nominal LR search | 121.796206s overall |
| Prior full Moreau CCG sweep | 14942.580206s overall |
| Current continuation sweep | 147.204327s;16single-thread workers |
| Current evaluation phase | 231.605939s;8single-thread workers, mean4.957713s per policy |
| Current report/render | 39.810010s; separate from evaluation |
| SRO-FA | Cached; timing not imported, not rerun |
| New CV/radius/LR selection | Not run |

The recorded evaluation phase includes executor startup/shutdown and scheduling.
Its first-job-start to last-job-end timestamps span about20.55seconds, much
less than the measured231.61-second parent phase. The remaining overhead was
not individually instrumented; its cause is unestablished. Do not replace the
recorded parent duration with the sum of concurrent per-policy durations or
attribute all231.61seconds to QP solves. Environment: shared Linux XeonGold6354
host gold-srv2.iem.technion.ac.il.

Black has lower means than blue at all seven shared radii. Green is lower than
blue at .001/.003/.01/.4/.7, higher at .2/1.5; SCS has no exact .03 grid point.
No significance claim or test-guided selection follows. The reusedseed9999
evaluation is descriptive; training/API protocols differ.

[Completed graph](matched_oos_completed.png) · [Full runtime report](REPORT.md)
· [Raw timings](runtime_comparison.json) · [Final campaign summary](summary.json)