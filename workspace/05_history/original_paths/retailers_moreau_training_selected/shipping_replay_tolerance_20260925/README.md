# Shipping consistency-check-only tolerance

User-authorized revision on25September2026: tolerate only the known zero versus
tiny positive residue in AP/native consistency replay. Preserve the earlier
failed gate and all nominal/model sources in place; this sibling binds them by
their previous recorded hashes. No retraining, new grid, test, validation or
full CCG launch is performed in this change.

## Acceptance rule

AP epsilon remains1e-6; native shipments, hard fees q>0, smooth training,
solver settings and green stopping rule are unchanged. The separate consistency
zero tolerance is1e-9, the existing numerical feasibility tolerance, not a fee
cutoff and not a broad increase of allowable cost error.

1. Require AP flags to agree with rational folded-policy support at every
   component. Genuine exact positive shipments are never classified as zero.
2. Every native support mismatch must be an inactive AP flag, an exactly zero
   AP shipment, an exactly zero rational projection, and0<native q<=1e-9.
   Reverse mismatches, nonzero AP values and larger native differences fail.
3. All solver status, finiteness, activation links, active epsilon and physical
   checks remain enforced. No early return skips these checks.
4. Use one native invocation for both shipments and costs. For an exception,
   subtract the known extra fee0.1 per affected component **only to explain the
   discrepancy**, then require the unexplained residual to meet the existing
   1e-4 floor-scaled cost tolerance, scaled by fee-free costs. Smooth cost is
   not used as fee-free cost. With matching support, the original raw-cost
   replay formula is unchanged.
5. Preserve raw costs, shipments, support differences and projection telemetry.
   Label an accepted exception `accepted_with_replay_tolerance`, with
   `support_matches=false`, reason, component indices, tolerance, raw difference,
   expected fee delta and unexplained residual. It is not an ordinary pass.

This tests a replay acceptance convention, not the root cause of numerical
residues or equality of the AP and native policy. Exceptions are accepted
despite a real difference in the raw hard costs. The restricted AP still does
not certify original-box worst-case cost. No AP/pool signed-gap gate is waived.

## Validation and timing

Regression fixtures include the saved failed-case arithmetic and unrelated
failure cases. Repeat only the same six predeclared saved nominal policies:
Ntotal50/200/800, seed0, LR.01, training center0, radii0/.2; no selection.
Persist all six cases; tests or case failure fail the gate. Nominal fits remain
cached and are not reported as fresh training time.

Use launch_gate.py with a fresh --output and --session. It runs ap_gate.py in
named tmux, preserves merged log and launch/whole-invocation JSON, and waits
for completion. JSON records perf_counter seconds and UTC endpoints per case,
solver, replay and whole invocation, with seed/radius/size/thread metadata.
Nested times are not added to elapsed wall time. A passed six-case gate is not
a full CCG integration test and does not launch a campaign automatically.