# Shipping live-green full campaign

User authorization: 26 September, hard shipping cost for BOTH nominal LR and
robust radius validation selection, then launch the full shipping campaign.

Reuse completed 30 training-only initializations and 90 nominal fits without
refit. All Ntotal50/200/800, seeds0..9, LR.005/.01/.05 must pass a complete-grid
and input-hash barrier before any LR validation. Paired 80/20 split regenerated
and verified against saved IDs and raw-array hashes; validation sizes10/40/160.
Choose minimum native q>0 hard validation cost, smaller LR exact tie.

Run270 CCG jobs: every size/seed and radii0,.001,.003,.01,.03,.2,.4,.7,1.5.
CCG reads only train-only JSON, never validation. Native hard per-pool maxima
drive AP comparisons and the strict abs(APmean-poolmean)/max(abs(poolmean),1e-9)
<.01 green stop, including candidate0. Return earliest minimum AP incumbent
over the fully executed prefix including stop. No future candidates/pools.
40 MP maximum, 3000 epochs/patience300, fresh Adam each MP, incoming state
eligible, preupdate training-best, smooth training objective and scheduler.
Smooth MP loss replay is separate from hard stopping; smooth improvement does
not guarantee hard improvement. Cap/stagnation/solver failure publishes no
selected model. Every AP center is attempted and failures retained.

Approved numerical guards: AP epsilon1e-6, internal tolerances1e-9; native
physical tolerance1e-6. AP/native support/cost/epsilon and negative APbound/native
pool differences are diagnostic only. Retain solver status, finite/integrity,
AP internal rational/integrality/links, bound/incumbent, membership, duplicates
and unresolved-bound checks. No ReLU/clipping/fee cutoff/fallback. This is an
approximate numerical experiment, not exact native hard-cost certification.

All270 jobs must converge and complete before radius validation. Select minimum
hard validation mean, smaller radius exact tie; no selection on a partial grid.
Test, SRO-FA, benchmark, refit and CV are not run. Reusing validation for LR and
radius is tuning, not unbiased performance estimation. All ordinary failed jobs
are retained and the other270 jobs still run; no error-saving substitute.

ProcessPool spawn, one thread per worker, affinity/cgroup/load/memory resource
accounting reserves4CPU slots+16GiB memory and >=4GiB/worker. Per-job/iteration
AP, MP, evaluation and total monotonic times with UTC, statuses/epochs/iterations,
seed/N/radius/workers are persisted. Completed-only runtime aggregates keep
cached nominal time, failed/nonconverged cases and sweep elapsed separate.
Named tmux supervisor writes persistent merged log and whole-child invocation.

Preflight runs analytic tests and approved-guard regressions, then two disposable
full native N50 seed0 LR.01 configurations at radii0/.2 without validation. These
exercise initial stopping and actual MP/AP integration and never initialize
production. Production requires a matching successful preflight source freeze.
The native cases pass through the actual spawned production worker and parent
dispatcher. The production entrypoint itself requires the preflight summary
and SHA256, binds its manifest and sources, and propagates the bindings to each
worker. Memory sizing uses the larger of the bound readiness peak and the new
N50 parent/worker peak, with the minimum reservation above; this is not a
measurement or certification of growing N800-pool memory requirements.