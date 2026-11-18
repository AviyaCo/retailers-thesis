# Shipping green with epsilon: launch preparation

Authorized prospective scope: R3, Ntotal50/200/800, seeds0–9, the existing
80/20 splits and nine radii0,.001,.003,.01,.03,.2,.4,.7,1.5. Reuse completed
shipping nominal fits; no refit or test access. Smooth MP fee remains
0.1 sum(1-exp(-5q)); native hard assessment remains0.1 per returned q>0.

The new Gurobi AP uses epsilon*z<=q<=Q*z with epsilon1e-6, Threads1,
MIPGap1e-4, feasibility/integrality/optimality tolerances1e-9, NumericFocus3,
and no time cap, solver fallback or retry relaxation. This is an explicitly
restricted-domain approximation: it is not an original-box robustness
certificate, nor a change of the deployed projection into a minimum-shipment
policy. Fees at epsilon are paid. No shipment clipping or fee cutoff is used.

Before full controller integration/launch, an isolated AP gate checks saved
seed0/lr.01 nominal policies (the predeclared middle LR, not validation-selected)
at N50/200/800 and radii0/.2 on training center0 only, plus synthetic contracts.
All six saved-policy cases are attempted even if one fails, with explicit
failure records; no failed case is dropped from the gate. No heldout data or
training/selection is used. The gate is not a full-grid resource/convergence test.

Returned AP binary support must agree componentwise with both the exact real
interpretation of folded MIP coefficients and the unchanged native float32
network/Moreau output. The rational point replay reuses the diagnostic's standard
R3 projection formula; it is not an alternative deployed policy or proof of a
global solver bound. Positive allocations must meet epsilon within the declared
1e-9 numerical feasibility tolerance; this tolerance never licenses a paid-zero
or a zero/positive support disagreement. Existing cost replay tolerance1e-4 is
additional, not a substitute for the support gate. Raw shipments/flags, solver
status, bounds, KKT diagnostics, rational and native replay, and timings persist
even when acceptance fails. Empty restricted domains remain failures.

Full green integration may proceed only after this gate passes and tests cover
smooth-training versus hard-pool bookkeeping, unchanged original pools, signed
negative-gap rejection, initial-eligible stop and prefix-only minAP return.
All270 successful CCG jobs would be required before final radius validation.
Any passing heuristic gap would not certify separation of the original box.

Every numerical stage runs in a named tmux session with a persistent merged log.
Structured JSON records perf_counter seconds, UTC endpoints, sample/seed/radius,
statuses, iterations and worker/thread counts. Whole invocation includes startup
and teardown; nested AP/solver/replay times are not added to it. Historical
nominal/SAA times remain cached and are not reported as current training.

Frozen historical and nominal sources and checkpoints must remain unchanged.