# Initial-eligible mean stop, matched minimum-AP return (23 September 2026)

Authorized: reconstruct the saved retailer R3, N=50 (40 train / 10 validation)
comparison and run only missing fixed-checkpoint evaluations. No training,
new AP search, new radius, changed tolerance, or SRO-FA/kNN rerun.

Both colors use the same nominal weights, training-only MP checkpointing,
scenario insertion, strict AP/replay checks, and 40-MP maximum. Black retains
its every-center bound/pool 1% criterion. Green stops at the FIRST complete AP
assessment with abs(mean(AP incumbent)-mean(pool))/max(abs(mean(pool)),1e-9)
<0.01, including candidate zero. Both return the earliest minimum mean AP
incumbent checkpoint among candidates available BEFORE their own stop.
The stopping policy and returned policy are recorded separately. Numerical
minimum ranking is not a claim that differences within AP bounds are resolved.
Black seed9/radius0.7 remains capped, not converged, even if returning earlier.

Preserve all prior artifacts. Derive all180 choices from training records before
validation or test lookup. Verify saved job hashes, checkpoint bytes and learned
state. Reuse validation/test only by exact seed, checkpoint and state identity
under the existing evaluator and data contract. Evaluate the three missing black
checkpoints (radius0.7, seeds1/7/9, candidates11/12/39) on their original val10
and byte-identical archived test10000 seed9999. No test regeneration. Freeze the
per-seed minimum-validation radius choices (ties smaller radius, nominal eligible)
before reading cached test costs or executing new test evaluations. All other
validation/test records are explicitly cached, not newly measured observations.

New evaluations use existing native validation and strict OPTIMAL-only Gurobi
test evaluation unchanged, one thread and one worker. Failure stops publication;
no retries, substituted checkpoints, partial curves, or relaxed tolerances.
Raw data/checkpoint provenance and current source versions are preserved. Prior
artifacts are verified again before final publication.

Recorded per-candidate AP/MP/pool durations give reconstructed nonoverlapping
phase sums through the STOP, not through the returned checkpoint. They exclude
nominal training and uninstrumented controller/selection overhead; neither a
fresh end-to-end green/black runtime nor a fresh training sweep is measured.
The old black wall time is shown only as historical trajectory time. Preserve
per-configuration timings, statuses, seeds/radii, source timestamps and 16-worker
single-thread concurrency. Aggregate completed and capped cohorts separately
with n/mean/median/min/max. Current reconstruction, validation, test evaluation,
selection, reporting and end-to-end invocation use perf_counter/UTC. Cached
nominal training, prior LR-search and evaluation times stay separate; SRO-FA
runtime is unavailable and CV is not run.

The red SRO-FA and yellow kNN cached references, plus blue/purple, remain
numerically unchanged. Compare red only at exact common radii; do not interpolate
or choose radii on test costs. Different historical training/selection budgets
remain caveats. The inspected seed9999 test and test-informed historical grid
are descriptive, not confirmatory evidence; this update does not remove that
limitation. Do not claim or tune to guaranteed wins over reference lines.

Execution: `bash launch.sh`, under tmux `retailers_min_ap_20260923`.
The launcher runs `/data/home/aviyac/ap_env/bin/python -B -u run_update.py` from
this directory, logging stdout/stderr to `campaign.log`, exit to `campaign.exit`.
Attach: `tmux attach -t retailers_min_ap_20260923`.
Status: `tmux has-session -t retailers_min_ap_20260923` and
`tail -n 15 campaign.log`. Structured output: `results/summary.json`.