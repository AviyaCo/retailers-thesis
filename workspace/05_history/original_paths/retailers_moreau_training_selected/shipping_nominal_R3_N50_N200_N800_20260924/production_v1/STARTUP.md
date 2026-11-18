# Shipping nominal-only production launched

Started **2026-09-24 15:22:45.117546 UTC** on gold-srv2.iem.technion.ac.il
(Xeon Gold 6354, 72 logical CPUs). Status is running, not completed.

## Exact scope

R3, Ntotal50/200/800, train40/160/640 and reserved10/40/160 observations;
data seeds0–9, split100000+seed, network seed0, width16–16. Fresh
training-only shipping SAA-MILP initializations for all30 size/seed pairs,
then90 full-batch smooth nominal fits: LR .005/.01/.05. Same frozen green
preupdate training-best checkpoint,3000-epoch ceiling/patience300, fresh
Adam, native Moreau projection and zero capacity buffer.

The smooth training fee is0.1 sum(1-exp(-5q)). A separate native diagnostic
charges0.1 for each identical returned shipment strictly greater than zero,
with no epsilon or clipping. This is not an exact-real support certificate.

**The pipeline stops after all90 nominal fits.** No LR winner, validation/test
evaluation, AP/CCG, radius selection, refit, SRO-FA/kNN comparison or paper
figure is produced. The unresolved shipping AP is not bypassed or replaced.
Starting this valid component does not mean the full shipping experiment runs.
No validation values enter initialization or training. No preflight checkpoint
is reused. Failed/nonfinite initializations or fits remain failures, with no
floor, replacement inventory, solver fallback or successful-subset selection.

## Execution and monitoring

Working directory:
/data/home/aviyac/retailers_moreau_training_selected/shipping_nominal_R3_N50_N200_N800_20260924

tmux session: `shipping_nominal_R3_20260924`.

Exact child command:

`/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/shipping_nominal_R3_N50_N200_N800_20260924/run_nominal.py --output /data/home/aviyac/retailers_moreau_training_selected/shipping_nominal_R3_N50_N200_N800_20260924/production_v1 --preflight /data/home/aviyac/retailers_moreau_training_selected/shipping_nominal_R3_N50_N200_N800_20260924/preflight_v2/summary.json --preflight-sha256 92b320e91b441aabe0923744497b3f9acc46385868085471c900915d1431b21a`

- Attach: `tmux attach -t shipping_nominal_R3_20260924`
- Session status: `tmux has-session -t shipping_nominal_R3_20260924`
- Log: ../production_v1.log (merged stdout/stderr).
- Progress: summary.json; per-job worker logs/training journals under N50/N200/N800.
- Exit marker after termination: ../production_v1.exit.
- Whole invocation timing after termination: ../production_v1.invocation.json.
- Exact launch/source snapshot: ../production_v1.launch.json.
- Actual resource/protocol/source evidence: provenance.json.

The launcher uses persistent tmux. No promise of proactive chat notifications
is implied; logs and progress artifacts update without an open editor.

## Resource limits and timing

Actual launch budget: **30 initialization workers,52 nominal workers**, one
numerical/solver thread each. Phases are sequential, not82 concurrent workers.
The launch reserved16 background-load CPU slots plus4 free slots,16GiB host
memory and4GiB per worker. Peak disposable RSS was890044416 bytes.
Affinity and cgroup ancestors were checked. These are sampled resource estimates,
not hard guarantees about future MILP memory or contention. No R10 process,
configuration or source was changed.

| Phase | Runtime/status at launch |
|---|---|
|Validated disposable preflight|142.040 s whole invocation;139.435 s internal|
|Shipping SAA initialization|Running;20/30 completed at15:22:58UTC, no recorded failures|
|Production nominal training|Pending completion of all30 initializations|
|Robust neural/CCG|Not launched: AP unresolved|
|SRO-FA / kNN|Not run|
|Validation/test/CV|Not run; no heldout evaluation in this stage|

Production records preserve perf_counter seconds, UTC endpoints, sizes/seeds/LR,
status/iteration/epoch counts and worker/thread settings. Shared initialization,
nominal training and parallel stage/sweep walls remain separate. At completion,
runtime aggregates contain counts/mean/median/min/max per size/LR, separate from
failures/incomplete work. Concurrent model walls are not summed as elapsed time.

## Verification

Independent read-only review approved only this restricted stage. All44 source
bindings (13 local,31 dependencies, including native binary) matched.25 tests
and14 malformed-evidence rejection checks passed. Each size's seed0 shipping
SAA reached tolerance-based Gurobi OPTIMAL; two-epoch gradient/restoration/replay
checks passed. Larger seed coverage and full production completion are unproven.

The earliest standalone zero-capacity fixture assertion failed because the
native solver returned a tiny negative residual. The test assumption, not the
allocation or fee rule, was corrected. Both preflightv1 andv2 completed; onlyv2
matches the final gate/controller source bindings. No invalid preflight is reused.

Frozen README and validation report describe preparation before launch. This
later output note records the launch without modifying those bound sources.