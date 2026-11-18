# AP/native replay observation only

The user explicitly selected **remove only AP/native replay checks**, retaining
signed AP/pool safeguards. This isolated revision removes native support,
native-epsilon and cost agreement as rejection conditions. It still evaluates
the same native witness once and records its raw shipments, fees, costs,
support and discrepancies, with consistency_status `recorded_only` rather than
an ordinary pass or tolerance exception. Execution status remains separate.

Unchanged hard rejection conditions: solver status, finite/valid outputs,
AP binary integrality, epsilon links, AP-internal rational support/positive
epsilon witness checks, native physical feasibility at 1e-9, reported-value
integrity, membership, duplicate/collision, unresolved bounds and signed
bound/incumbent and bound/pool ordering. No fee cutoff, q clipping, objective
replacement, weakened physical tolerance or signed-gap waiver.

The same full-center readiness grid is rerun: seed 0, LR 0.01, Ntotal 50/200/800,
radii 0/0.2, all 1,680 training-center APs. All failures remain in the output;
partial success does not count as a passing configuration. Prior sources and
artifacts stay unchanged. No nominal/MP training, full CCG, selection, held-out
evaluation or SRO-FA. Removing replay rejection does not certify that the
epsilon-restricted folded-policy AP bounds the native hard-cost objective.

The named-tmux supervisor persists stdout/stderr and whole-child monotonic
runtime. Raw per-case, AP/solver, replay and pool timings and metadata retain
their existing scopes. Historical nominal training times remain labeled cached.