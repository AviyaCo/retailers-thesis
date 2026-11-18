# Shipping native projection diagnostic

Authorized investigation only: fees, solver settings, tolerances, checkpoints,
training and stopping rules are unchanged. No AP optimization, CCG, new fitting,
validation or test access. Reuse every saved readiness row (1,680), not just
the failures: seed 0, LR 0.01, Ntotal 50/200/800, radii 0/0.2.

For each configuration, load a fresh model, capture one native original-center
batch and then all saved AP-witness singleton calls in their original order.
Passive projection hooks return None and record actual float32 target/capacity
and float64 returned q. After this sequence, a separate fresh projection layer
solves each captured batch target singly, without re-running the network. Repeat
the complete sequence once from fresh state to check reproducibility, not tune.

Use the frozen standard sorting simplex projection in exact Fraction arithmetic
on the actual represented native target and capacity. This is an independent
diagnostic reference, never a replacement policy or cost. Compare its error
against native q separately from differences to the saved folded-policy AP
reference. At radius zero verify point identity before attributing target
differences to batch execution; at positive radius original centers and AP
points differ and that interpretation is forbidden.

Finite float32-to-float64 promotion is exact. AP uses Gurobi; native projection
uses Moreau active-set, despite the historical SCS argument label. Moreau's
active-set primal_tol defaults to 1e-6, and the native wrapper allows physical
residual up to 1e-6; AP witness checks require 1e-9. A negative q is infeasible
in exact arithmetic even if a numerical acceptance threshold permits it.

Persist inputs/outputs, rational references, effective settings, statuses and
iteration telemetry, per-native-call, reference, fixed-target, configuration
and whole-child monotonic times. Retain failures with unavailable q explicit;
never bypass a failed projection to obtain replacement output. Cached nominal
times are separate. Preserve all sources and consumed readiness outputs by
hash; the named tmux supervisor writes a persistent log and invocation record.