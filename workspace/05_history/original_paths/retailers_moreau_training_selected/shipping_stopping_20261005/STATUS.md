# Shipping stopping-checkpoint replay — completed

**production_v2 completed5 October2026 at10:24:01 UTC, exit0. All270 green
stopping policies were freshly evaluated and the updated figure is published.**
The renewed existing WLS license worked; no API key or license-file change was
needed. The prior failed attempt below remains preserved.

Tests, selection/freeze, all three spawned training-only probes and the
16-worker resource gate passed. All2,853,219 production projections were
accepted OPTIMAL with physical residual at most1e-6:2,700,000 OOS rows and
153,219 stopping-own-pool rows. Raw-journal verification and final input checks
passed. A post-completion check rehashed4,236 source/copy pairs,11,547 historical
bindings,270 worker results and all five published artifacts, with no mismatch.
All non-green panel objects are exactly equal to the parent comparison.

- **[Updated green stopping figure (PNG)](production_v2/published/comparison_shipping.png)**
- **[Vector figure (PDF)](production_v2/published/comparison_shipping.pdf)**
- [Results and per-configuration runtime table](production_v2/published/RESULTS.md)
- [Structured runtime report](production_v2/published/runtime_report.json)
- [Whole-child completion receipt](production_v2.invocation.json)

- [Current structured status and timings](production_v2/summary.json)
- [Current stdout/stderr](production_v2.log)
- [Exact child command, working directory and session](production_v2.launch.json)

Named tmux session: `shipping_stop_v2_20261005` (finished). Working directory remains this
original experiment directory, not a navigation alias. Exact launch:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 /data/home/aviyac/ap_env/bin/python -B launch.py --output production_v2 --session shipping_stop_v2_20261005

Attach/status during execution:

    tmux attach -t shipping_stop_v2_20261005
    tmux has-session -t shipping_stop_v2_20261005

The new run repeats the unchanged gates and records fresh evaluation timings;
it does not retrain, alter fees/tolerances, select radii or replace any old
artifact. The480 historical neural and all benchmark curves remain cached.
The PNG was visually inspected. Twelve assessment/pool selections changed,
but only seven learned states changed. Exactly seven per-policy OOS values
changed; every unchanged learned state reproduced its prior OOS value exactly.
This is not a claim of OOS improvement or exact robust optimality. The known
test and mixed265+5 training cohort retain their descriptive limitations.

### Completed retry runtimes

Host gold-srv2.iem.technion.ac.il, Linux;16 workers with one Torch/Gurobi thread
each. Full invocation10:18:15.631878–10:24:01.539553 UTC. Nested phase times are
not added to concurrent job times as elapsed time.

| Component | Seconds | Human-readable / scope |
|---|---:|---|
|Whole child|345.907670|5.765128 min|
|Parallel green OOS/IS evaluation|108.725594|1.812093 min|
|Raw-result verification|65.751183|1.095853 min|
|Tests incl. subprocess overhead|26.064499|22 revision +60 unchanged tests passed|
|Selection/freeze|96.253602|All270 stopping choices sealed|
|Training-only preflight|15.791072|Three probes, six accepted QPs|
|Test binding|9.417810|Original10000 seed9999 arrays|
|Final integrity|14.373696|Completed|
|Rendering/plot-data writing|5.793412|Completed|
|Nominal neural training|—|Not rerun; historical timings retained|
|Robust neural/CCG AP/MP|—|Not rerun; historical timings retained|
|SRO-FA partition/optimization/evaluation/total|—|Not rerun; cached provenance retained|
|CV/radius validation/refit|—|Not run|

Per-model seed/radius/size OOS and IS durations and per-configuration completed
mean/median/min/max over ten runs are in the linked reports. Missing historical
timings remain unavailable; a rerun would be required to measure them. The
failed first invocation remains separate from completed-run aggregates.

Production manifest SHA256:
`a0fc73c005be77c512a33ae7b6b842ba8c5ea8745a92dccaab1845754d26c619`.

## Preserved first attempt — expired license

At the first attempt on5 October2026, all270 stopping choices were sealed,
but no stopping-checkpoint plot had yet been published.

The production attempt failed with exit1 at08:29:06.616495 UTC because Gurobi
reported `License 2759921 has expired` when starting the first training-only
probe environment. No projection was attempted or accepted, no held-out
evaluation started, and no production evaluation manifest or published folder
was created. There is no solver substitution, retry with degraded settings,
test-driven change, omission or relabelling of the old minimum-AP figure.

## Completed selection and checks

- [Sealed returned-policy index](production_v1/selection_manifest.json)
- [Selection seal](production_v1/selection_manifest.sha256)
- [Attempt summary and raw phase timings](production_v1/summary.json)
- [Whole-child runtime receipt](production_v1.invocation.json)
- [Persistent stdout/stderr](production_v1.log)

All270 green entries have `return_rule=stopping_checkpoint`,
`selected_candidate_id=stopping_candidate_id`, and a passing historical mean-gap
assessment. Every checkpoint and its own stopping pool was hash-verified against
the original and frozen copy after the failure. All local frozen source hashes
also remain unchanged. The manifest preserves the480 historical comparator jobs;
they are not newly evaluated or silently changed.

| N | Returned policies | Changed assessment IDs | Changed learned states | Changed pools |
|---|---:|---:|---:|---:|
|50|90|0|0|0|
|200|90|3|3|3|
|800|90|9|4|9|
|Total|270|12|7|12|

Five of the12 later assessments have the same weights but a different pool;
therefore equal policy weights do not justify reusing the earlier IS value.
No OOS improvement is claimed. Original training summaries and parent figures
remain unchanged. Candidate0 is still eligible for the unchanged stopping rule.

All22 revision tests and60 unchanged parent tests passed in the failed
invocation. Unit test elapsed10.860s and10.343s respectively; their parent
subprocess/startup phase is reported below, not added to those nested durations.

## Measured runtimes

Linux host gold-srv2.iem.technion.ac.il;72 affinity CPUs. The requested production
pool was16 one-thread workers, but it was **not launched**: failure occurred in
the single spawned preflight worker, before the resource gate. Durations use
monotonic wall-clock seconds. No completed-model runtime aggregates exist.

| Phase/component | Status | Seconds | Minutes |
|---|---|---:|---:|
|Whole child invocation|Failed, license expired|128.636583|2.143943|
|Tests, including subprocess overhead|Completed|26.752651|0.445878|
|Selection and immutable-input verification|Completed|95.444115|1.590735|
|Spawned training-only preflight phase|Failed before first solve|3.333272|0.055555|
|First probe body (nested within preflight)|Failed before first solve|0.431862|0.007198|
|Nominal neural training|Not rerun; historical timings retained|—|—|
|Robust neural/CCG AP/MP|Not rerun; historical timings retained|—|—|
|SRO-FA partition/optimization/evaluation/total|Not rerun; parent provenance retained|—|—|
|OOS and IS production evaluation|Not started|—|—|
|CV/radius validation/refit|Not run|—|—|

Nested timings are not additive. Missing historical timings remain unavailable;
rerunning would be needed to measure them. This failed invocation is not a
completed model, training runtime or completed replay timing.

## Execution and next action

Start/end:08:26:57.979909–08:29:06.616495 UTC on5 October2026.
Working directory: `/data/home/aviyac/retailers_moreau_training_selected/shipping_stopping_20261005`.
Named tmux session: `shipping_stop_20261005` (finished with failed child).
Persistent log: [production_v1.log](production_v1.log).

Exact launch, with each of OMP/MKL/OPENBLAS/NUMEXPR/RAYON thread counts set to1:

    /data/home/aviyac/ap_env/bin/python -B launch.py --output production_v1 --session shipping_stop_20261005

Attach/status commands during that session:

    tmux attach -t shipping_stop_20261005
    tmux has-session -t shipping_stop_20261005

The failed session is not expected to remain active. The launcher records the
absolute child command/cwd in [production_v1.launch.json](production_v1.launch.json).
At this first attempt, a valid Gurobi license was required before replay could
proceed. The later user-authorized production_v2 replay uses a fresh
output/session and the unchanged gates, preserving this attempt and its frozen
sources. See the current status above.

Selection SHA256: `065d9757d920f817e7fcd969770f2a4f6bdf2214660e0d5e04363137ecf6baab`.