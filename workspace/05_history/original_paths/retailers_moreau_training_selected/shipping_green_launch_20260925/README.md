# Shipping green launch integration

The user requested full launch after the approved replay-only tolerance gate.
Before production, check the missing hard-pool/signed-gap integration on the
same fixed seed 0, LR 0.01, Ntotal 50/200/800 and radii 0/0.2, now covering
every training center: 1,680 APs. This is an initial-policy readiness gate,
not a CCG run or a claim that later MP policies will pass.

The hard objective uses unchanged native `q > 0` fees and raw pool scenarios.
The smooth objective stays separate for MP training. Only generic replay
acceptance is adapted: the already-approved exception is recomputed from its
logged support and cost evidence. Signed bound/pool and bound/incumbent checks,
membership, duplicate/collision and unresolved-bound checks remain unchanged.
No cost substitutions, clipping, dropped failed samples, solver relaxation,
or signed-gap waiver. Every AP in the fixed grid is retained; any failed case
blocks full launch. A green non-pass alone does not fail readiness.

No validation/test access or retraining in this gate. Previous nominal times
are labeled cached. Per-AP solver/activation/replay/controller timings and
per-configuration pool/total times, statuses, iterations, seeds and thread
counts persist. The tmux supervisor records whole-child elapsed time separately
from nested timings. Sources are frozen before running; old sources unchanged.