# Authorized approximate shipping readiness guards

On 26 September the user explicitly approved (1) removing the negative
AP-bound/native-pool readiness rejection and (2) setting the native physical
feasibility guard to **1e-6** instead of 1e-9. Both changes are isolated here;
historical sources and results remain untouched and hash-bound.

The signed gap, raw bound and native pool cost are still recorded; negative
cross-implementation gaps are diagnostic-only. Bound/incumbent ordering,
solver status, finite output/evidence integrity, membership, duplicate/collision
and unresolved-bound checks remain enforced. AP binary integrality, epsilon
links and rational folded-policy checks keep their **1e-9** tolerances. The
native check permits q >= -1e-6 and sum(q) <= Q + 1e-6, and logs the actual
residual and tolerance. It does not add ReLU, clip shipments, change solver
parameters, change exact q>0 fees or alter the smooth MP loss.

AP/native support/cost/epsilon observations remain advisory, with complete
cost evidence even when a retained validity check fails. A passing readiness
result means only acceptance under this approximate convention. Neither the
epsilon-restricted AP bound nor the cross-implementation stopping gap certifies
the exact native hard-cost problem.

Validation repeats exactly the same grid: Ntotal 50/200/800, seed0 LR0.01,
radii0/0.2, every 40/160/640 training center, totaling 1680 AP cases. One worker
and one Torch/Gurobi thread. All failures are retained. No new nominal/MP
training, full CCG, hyperparameter selection, held-out evaluation or SRO-FA.
Raw per-configuration/per-AP wall-clock timings, UTC endpoints, solver status
and iterations are persisted; whole-child runtime and stdout/stderr are recorded
by the named-tmux supervisor. Cached nominal time is not new training time.