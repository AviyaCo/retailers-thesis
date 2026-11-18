# Isolated fresh green R10 — same except R10 + N50

**Implementation and disposable preflight only. Production has NOT been launched.**
Parent audit and a successful hash-matching preflight are required before launch.

The frozen reference is ../green_R3_N200_N800_20260923. The ONLY numerical or
statistical changes are R=10 and sizes=(50,200,800), with train/validation counts
40/10,160/40,640/160. Ten data seeds 0..9; network seed 0; split seed 100000+s.
Exact legacy NumPy-global-seeded scipy truncnorm generator, saved/restored RNG;
c=.25,h=.05,b=.5,f=0; network 21→16→16→10. The historical R10 script is a
hash-bound reference, NOT a controller or trainer dependency. Its extra LR .1
is NOT adopted. LR grid remains .005,.01,.05; full-batch fresh Adam at ALL sizes.
The 3000 evaluated epochs, patience300, training-driven scheduler, thresholded
preupdate train-best/incoming eligibility and final evaluated update are unchanged.

Radii: 0,.001,.003,.01,.03,.2,.4,.7,1.5. Initial AP is eligible. Stop at first
strict abs(mean incumbent AP-current pool mean)/max(abs(pool mean),1e-9)<.01;
select earliest minimum incumbent mean AP in the inclusive executed prefix.
40 MPs maximum. Cap/stagnation/failure publishes NO selected checkpoint.
No validation inside training/CCG; no test access. Global phase barriers require
30 initializations, ALL90 nominal arms before nominal validation, ALL270
successful converged CCG jobs before final30 radius winners. No partial selection,
refit, resume, historical checkpoint reuse, budget changes or solver fallbacks.

## Auditable reuse and isolation

source_adapter.py binds original files, the original provenance (including the
Moreau Python/native binary hashes), green sources, strict comparator and R10
reference by SHA256. All new computational modules, tests, this README and shell
launcher are included in SOURCE_NAMES and bound in preflight/production manifests.
Execution uses exact checked fragments in fresh dictionaries/module namespaces;
ADAPTATIONS exposes every changed fragment and occurrence count. No frozen source
is rewritten. No plain historical core/ccg/adversary module is imported or mutated.
Original experiment-main and old training/controller definitions are NOT loaded;
forbidden compatibility sentinels raise if called. Dimension-free functions retain
their exact source. Frozen green controllers/trainers are reused, not redesigned.

r10_core: explicit R10 model config, 10 projection variables/11 constraints/cones;
r10_adversary: inventories columns10:20/20:, retailer loops10; THREE max-cost
branches and NumericFocus3 UNCHANGED. r10_pool:20-dimensional centers/slicing/gates.
r10_strict: independent 21-input/10-output target and strict OPTIMAL Gurobi QP.
All Moreau, AP, membership, replay and physical tolerances are unchanged. Native
physical tolerance1e-6; strict QP physical tolerance1e-4; AP feasibility1e-9.

## Validation and runtime

Use /data/home/aviyac/ap_env/bin/python -B -m unittest -v test_green test_campaign
test_r10 from this directory. The full frozen tests are adapted to R10 fixtures,
plus dimensions, source isolation, source coverage, phase barriers and a real spawn
initializer/thread regression. preflight.py OUTPUTDIR is explicitly disposable;
child flags are --size {50,200,800} --outputdir OUTPUTDIR. Existing outputs are
rejected, failed attempts retained. Preflight runs tests, AST/bash checks, real
train-only SAA, two nominal epochs, AP radii0/.4/1.5 with folded-target/native
replay and membership checks, strict QP comparison, zero-capacity projection,
41-row-per-center SYNTHETIC RESOURCE STRESS ONLY (NOT research data), two stress
MP epochs, varying-batch backward/cache regression, and real radius0 controller
AP over every training center. No validation/test values select any method.

Long preflight must run in a named tmux session from this directory with persistent
stdout/stderr and exit marker. The intended first output is preflight_v1; do not
overwrite it after failures. Production launcher: bash launch.sh followed by the
absolute completed preflight summary path. It creates a named tmux session and
prints exact command, log/exit paths and attach/status commands. Do NOT invoke it
until the independent parent audit approves production.

Runtime records use perf_counter seconds with UTC endpoints, seed/configuration,
status/iterations, worker/thread counts. AP/MP each iteration, nominal training,
evaluation and selection are separate; failed attempts retain partial diagnostics.
Cached nominal time is explicitly prior time, not CCG invocation time. Aggregates
separate failures/nonconvergence and report count/mean/median/min/max by size,
radius (LR for nominal); overall parallel elapsed is never summed job wall time.
SRO-FA/CV/test: not run, no invented timings. Hardware and peak RSS are recorded.

Resource policy: maximum available concurrency, reserves4 CPU slots plus ceiling
of maximum load average,16GiB host memory, at least4GiB/worker or2×preflight peak.
Job caps are270 CCG/90 nominal/30 initialization/30 validation. All mounted
cgroup-v1/v2 ancestors (CPU, cpuset, memory) are respected. Preflight is sequential
across sizes with one numerical worker; future MIP memory is not bounded by sampled
RSS. No solver budget is altered to make a run complete.