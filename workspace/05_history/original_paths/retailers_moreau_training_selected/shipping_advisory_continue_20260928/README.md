# Shipping advisory continuation — 28 September 2026

User approved trying continuation with activation-consistency and exact-duplicate
separation discrepancies diagnostic-only. This revision accepts possibly
infeasible activation links for an explicitly approximate experiment; it does
not certify their feasibility or global robust optimality. Preserve raw q>0
fees, shipments, AP settings, epsilon1e-6 and smooth training unchanged.

Only activation link, folded-support and AP positive/epsilon consistency checks
are advisory; those are related checks of the same on/zero discrepancy, not a
new solver formulation. Exact-duplicate material separation is advisory. Do not
insert exact duplicates. Finite values, shipping flag integrality/range, solver
OPTIMAL requirement, AP physical and uncertainty-box membership, native physical
1e-6, bound/incumbent ordering, unresolved bounds and float32 collisions remain
enforced. No silent fallback, corrected costs, clipping or fee cutoff.

Resume the five prior failures at saved AP boundaries:
N50 seed7 radius1.5 candidate9; N200 seed5 radius1.5 candidate1;
N800 seed6 radius.7 candidate3; N800 seed7 radius.2 candidate1;
N800 seed9 radius.4 candidate1. Verify the entire consumed prefix/checkpoint/pool
chain, retain15 prior MPs without retraining, and redo each failed AP sweep in
full. Original source/results remain untouched. Restore repeated-state history;
green pass precedes stagnation, then the original40-total-MP cap. Keep full
inclusive prefix for earliest-minimum-AP selection. If stagnation or another
guard occurs, report it rather than resetting history or adding budget.

Analytic regressions plus one native failed-AP probe per configuration run before
five parallel continuation workers; one Torch/Gurobi thread each. Resource gate
requires capacity for allfive, no degraded retry. Structured monotonic runtimes
separate cached nominal, prior failed-job, cached valid-prefix, failed historical
assessment, and current AP/MP/evaluation/whole invocation. Logical phase sums
are not sweep elapsed. Completed-only count/mean/median/min/max exclude failures
and nonconvergence. Parent models used62workers; currentfive-worker times are
not directly comparable speed measurements. Prior N50 peakRSS and4GiB minimum
reservation are not a guarantee of future large-pool memory.

This is mixed acceptance-protocol continuation, not a homogeneous full rerun.
No nominal or radius validation, test, refit, CV, SRO-FA or benchmark runs here.
Final radius selection stays gated on all270 completed/converged configurations;
this five-case runner reports outcomes without automatically selecting a radius.
Named tmux logs preserve exact command/cwd and current whole-child timing.