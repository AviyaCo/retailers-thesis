# R10 green campaign launched — 24 September 2026

Status at launch: running, not a completed result or convergence claim.
Production started **2026-09-24 13:10:56.490346 UTC**.

## Authorized scope

Fresh no-shipping green experiment at R=10, Ntotal=50/200/800, with paired
train/validation splits 40/10, 160/40 and 640/160. The frozen R3 green protocol
is preserved apart from the retailer dimension and the requested size grid.
Network inputs/outputs become 21/10; hidden widths remain 16/16. The max-of-three
cost expression is unchanged; projection has 10 variables and 11 inequalities.

- Ten data seeds 0–9, network seed 0, split seed 100000+data seed.
- 30 training-only SAA initializations, 90 full-batch nominal LR arms.
- LR grid .005/.01/.05; no historical extra .1 LR or minibatching.
- 270 CCG configurations: three sizes × ten seeds × nine fixed radii
  0/.001/.003/.01/.03/.2/.4/.7/1.5.
- Same preupdate training-best checkpoint, 3000 evaluated epochs/patience300,
  fresh Adam, native Moreau projection and strict numerical gates.
- Initial AP eligible; stop at the first strict 1% absolute mean gap;
  return earliest minimum mean-AP incumbent in the inclusive stopping prefix.
- No selected checkpoint for failed, stagnated or capped configurations.
  No final radius winners until all 270 configurations complete successfully.
- Nominal validation selects the initialization/LR before CCG; no validation
  evaluation occurs inside optimizer or CCG calls. Final validation chooses
  30 per-seed/size radius winners only after all CCG policies are frozen.
- No test, SRO-FA, kNN or paper-figure evaluation is part of this training
  invocation. Its completion is required before the corresponding OOS/IS
  comparison can be prepared. No shipping run was launched.

## Persistent execution

Working directory: /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924

Named tmux session: `green_r10_20260924T131053740895134`.

Exact Python command from that working directory:

`/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py --preflight /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/preflight_v1/summary.json`

Launch used `bash launch.sh` with the above absolute preflight summary. The
launcher creates tmux, redirects stdout/stderr and writes its exit code.

- Attach: `tmux attach -t green_r10_20260924T131053740895134`
- Session status: `tmux has-session -t green_r10_20260924T131053740895134`
- Persistent log: ../../campaign_20260924T131053740895134.log
- Exit marker on termination: ../../campaign_20260924T131053740895134.exit
- Actual launch metadata: ../../launch.json
- Current progress: summary.json (per-stage status and completed job records).
- Frozen protocol, environment, source bindings: provenance.json.
- Per-model logs, timing records and AP/MP evidence are under each N directory.

## Resources and runtime accounting

Recorded launch budget: **67 nominal workers, 67 CCG workers**, 30 workers
for initialization and validation. Stages are sequential, not 134 simultaneous
workers. Every numerical/solver worker has one thread. Host gold-srv2.iem.technion.ac.il,
Intel Xeon Gold 6354, 72 logical CPUs. Launch reserved 4 CPU slots plus one
background-load slot, 16 GiB host memory and at least 4 GiB per worker.
Affinity and mounted cgroup ancestors were checked.

This resource snapshot is not a proof of future AP-tree memory use. No solver
time limit, fallback, clipping or relaxed tolerance was added. No other user's
processes were modified.

All runtime fields use monotonic seconds and UTC endpoints, retaining seed,
size, radius/LR, status, epoch/iteration counts and worker/thread settings.
Nominal, per-iteration AP/MP, full CCG through stopping, evaluation and selection
are separately timed. Shared nominal time is excluded from CCG invocation time.
Parallel elapsed time is not the sum of overlapping per-model times.

| Scope | Measured runtime at launch |
|---|---|
|Disposable preflight, 47 tests and three numerical size checks|68.175620 s (1 min 8.18 s)|
|Fresh nominal training|Pending, recorded per seed/LR and overall stage|
|Fresh robust CCG|Pending, recorded per seed/radius and overall stage|
|Validation/LR/radius selection|Pending, separately recorded|
|CV|Not applicable: single paired split|
|SRO-FA / kNN / OOS / IS comparison|Not run in this invocation|

Disposable preflight states and synthetic 41-point stress pools are not reused
as research models or scenarios. The preflight's two-epoch timings do not
predict production training time.

## Independent review and exact launch gate

All 16 local source/doc/launcher files and 37 frozen dependencies matched their
bindings. Independent review verified all 840 radius-zero controller APs plus
nine standalone APs, three strict QP checks, checkpoint/replay/membership
evidence, and real spawned-worker behavior. All APs recorded OPTIMAL.
Ten additional nontraining audit tests passed. Full positive-radius CCG
convergence and total runtime remain unproven.

The inherited launcher checks do not fully reject arbitrary malformed nested
preflight summaries. This launch was therefore explicitly pinned to the exact
independently audited preflight, and the parent separately checked unique size
coverage, completed phases, child-summary hashes, zero return codes, and all
local/frozen source hashes immediately before launch. A shell SHA256 check
also preceded the launcher. No bound source was edited after preflight.

Audited preflight SHA256:
`fdb1036cc4a4cf35db2e6752e779df2e03915e0b2b5d0571a3de458f84be6706`.

The initial unit-test decorator-extraction failure remains preserved; it was
fixed before the successful preflight and before production. No numerical
protocol change or solver retry was used to pass validation.

This post-launch note is outside the frozen source set. Frozen README and
validation report describe their prelaunch state; this note and launch.json
record the later production launch without modifying that evidence.