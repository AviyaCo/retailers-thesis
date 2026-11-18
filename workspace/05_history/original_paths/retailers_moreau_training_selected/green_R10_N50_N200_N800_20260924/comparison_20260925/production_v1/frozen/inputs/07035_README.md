# R10 fixed-policy comparison (2026-09-25)

## Scope and protocol

No training, new selection, test-based tuning, interpolation, projection fallback,
checkpoint substitution, retries, or cached green N50. All **810 policies** are
fresh strict replays: 270 green and 540 historical SCS. Sizes are 50/200/800;
seeds are 0–9. Green uses 40/160/640 training centers with the saved 20% validation
split; SCS uses 50/200/800. The completed source is
`../results/20260924T131056.490113+0000`: 30 initialization, 90 nominal,
270 CCG, 30 LR-validation, and 30 radius-validation jobs.

Green radii: 0, .001, .003, .01, .03, .2, .4, .7, 1.5.
SCS radii: 0, .001, .005, .01, .02, .05, .1, .15, .2, .3, .4, .45,
.5, .6, .75, .8, 1, 1.5. Each saved radius has ten seeds.

The approved identical R3 protocol uses the exact locked model/loader, legacy
float32 inference, legacy scenario-cost arithmetic and fresh Gurobi QP per row,
adapted to R10 by the existing reference/evaluator. Only first-stage demand is
passed to policy targets. Acceptance requires OPTIMAL and physical residual
at most 1e-4; no clipping, warm-start replacement, analytical projection or
relaxed tolerance. Green pool membership tolerance remains 1e-9. Historical
SCS coordinates are observational, not box-certified, and are not repaired.

All incoming green checkpoint choices and their own saved pools are retained.
**11 returned green checkpoints have own gap >=1%**; their stopping candidate
passed, but the saved earliest minimum-AP rule can return another checkpoint.
Three historically nonconverged SCS N50 checkpoints are retained: (rho, seed)
= (.15, 4), (.6, 5), (.8, 1). Computational replay success is not CCG convergence.

After all policies and sources are frozen and a policy-only manifest is sealed,
production calls the legacy generator once for 10,000 pairs, seed 9999, each
array float64 shape `(10000, 10)`. Array and file SHA256 hashes are sealed in
the final worker manifest. **No R3 test hashes are reused.** Generation matches
the historical code protocol but is not byte-certified against historical
SRO-FA test arrays. Seed 9999 is historically known, not a new holdout.
OOS is mean test cost; IS is mean per-center maximum on the returned checkpoint's
own finite MP pool, not the full adversarial supremum.

SRO-FA and SAA-kNN are cached descriptive comparators, not fresh strict replays.
Adjacent historical scripts are source references, not certified cache producers.
kNN's saved per-seed LOO selector is not full-pipeline LOO. Only frozen green
validation winners are reported; no selection uses the new replay costs.

The gray **Empirical clairvoyant reference** is ONE fresh joint LP fitted to the
common 10,000-row test, seed 9999, after the policy and test manifests are sealed.
The same scalar appears in all three panels: not eight independent results,
not ten training-seed observations, not deployable-policy OOS performance and
not a population lower bound. Historical gray values are not filled, aggregated,
plotted or overwritten. This explicitly supersedes inherited R3 report statements
about historical gray references or no new benchmark construction. The oracle
runs once before neural replay, sequentially with one solver thread. Its output
never enters neural workers, training, selection, or policy modification.

## APIs and exact sources

- `run_comparison.py`: `freeze_all(output, expected=None)`, `require_grid`,
  `validate_preflight(path, expected_sha256)`, `validate_preflight_evidence`,
  `freeze_test`, `run_worker`, `execute`, `verify_journals`, `main`.
- `preflight.py`: `training_checks(manifest, report)`,
    `oracle_training_check(manifest, report, output)`, `main`.
- `oracle.py`: `solve(xi1, xi2, record, solution_path)` for the training-only
    smoke LP; `run(manifest, output)` for the single common 10k LP.
- `plot_comparison.py`: `build_plot_data(manifest, rows, oracle)` requires the
    fresh oracle record. Runner `oracle_plot_record` explicitly derives plot
    count fields from validated raw LP diagnostics and binds all `oracle.SOURCE_FILES`;
    the raw oracle result (including its scalar historical source digest) is not rewritten.
- `launch.py`: modes `preflight` and `production`; flags `--output`, `--session`,
  and production-only `--preflight`, `--preflight-sha256`. Private `--monitor`
  is used only by tmux; do not invoke it manually.
- `launch.sh`: invokes the launcher using `/data/home/aviyac/ap_env/bin/python`.

`reference.load_definitions` executes named definitions only, in the local
module namespace. `run_comparison.LOCKS` binds historical R3 sources:

| R3 source in green_R3_N200_N800_20260923/comparison_20260924 | SHA256 |
|---|---|
| run_comparison.py | 2c0b3e60a43edc34adb001da508949d8fa16bea02f367bcb1e180eb49e08bb55 |
| preflight.py | 913b7d1515219087bc48a7b4dc86848b69c95baa65773029bc5a7aee867b3b0c |

Every exact replacement checks its occurrence count. Historical top-level code
is never executed and historical files are never edited. The manifest binds all
13 local source/test/doc files (including `oracle.py` and `test_oracle.py`),
these two R3 sources, `evaluation.SOURCE_FILES`, `oracle.SOURCE_FILES`
(including the R3 evaluator and matched R3 compare source), the historical
`retailers_problem/R10/sro_fa/sro_fa_sweep.py`, and all sources/inputs captured
by `evaluation.freeze`, including campaign provenance and relevant historical
nominal/run/evaluation scripts. Output reports and launch metadata are excluded
from local source discovery. Any source/doc edit invalidates prior preflight.
Originals and read-only copies are rehashed; original permissions are unchanged.
SHA256 sidecars are integrity seals, **not cryptographic signatures**.

## Preflight (parent/operator runs this; not run during implementation)

Use a new output directory and a unique tmux session, from this directory:

```bash
bash launch.sh preflight --output /absolute/new/preflight_output --session r10-comparison-preflight
```

This checks Python syntax and shell syntax, all four solver-free unit suites
(`test_evaluation`, `test_plot`, `test_runner`, `test_oracle`),
CPU/cgroup/memory/load/disk gates, and a complete disposable 810-policy freeze.
It never draws seed 9999 or accesses held-out evaluation rows. It then performs
**24 real QPs**: two saved training centers each, seed 0, green/SCS, N50/200/800,
rho 0/.2 (12 cohorts). Per-QP diagnostics, UTC boundaries and monotonic timings,
costs, cohort identities, counts and source hashes are persisted. Separately it
solves **ONE real training-only oracle LP**, using exactly the first two saved
green N50, seed-0, radius-zero training centers (20D, split into two 10D arrays).
No validation/test rows are supplied. The record retains Ntotal=50, Ntrain=40,
LP N=2, seed/sample IDs, training-array and manifest hashes, strict OPTIMAL status,
iterations, finite objective/replay/physical-residual audits, phase and total
monotonic timings, and a checksummed raw solution NPZ. Gate validation reopens
and re-audits that artifact; changed/missing evidence fails closed. The 24-QP
counters explicitly exclude this separate LP. A training-only
generator guard permits only sizes 50/200/800 and seeds 0–9. Unit fixtures use
artificial arrays/mocks, never experiment data or real solvers; the real spawn
wrapper fixture intentionally fails on an absent manifest before computation.

Retain the preflight directory and all copies. Only a completed sealed summary
with passing nested phase/test/cohort/24-QP/one-LP/artifact/timing evidence is eligible.
Production requires its **exact printed SHA256** as well as the summary path.
A failed summary is preserved but is never eligible. No retries or overwrite.
Earlier full-freeze attempts are not reusable evidence for these changed sources;
the parent must run a new preflight after all source/doc changes are finalized.

## Production (parent/operator runs this separately)

```bash
bash launch.sh production --output /absolute/new/production_output --session r10-comparison-production --preflight /absolute/new/preflight_output/summary.json --preflight-sha256 EXACT_PRINTED_64_HEX_SHA256
```

Full preflight/source/version/hash identity is verified before production freeze,
then the complete frozen policy identities and hashset are compared again before
test generation. Read-only policy-only and final manifests each have SHA256
sidecars. A fresh common oracle LP then writes exclusive `oracle/result.json`
and `oracle/solution.npz`; their hashes are recorded in sealed `oracle_binding.json`.
The result and solution become read-only. An oracle failure stops before neural
dispatch and plotting, without retry. Spawn workers use at most 32 processes, one numerical/solver thread
each. The exact R3 resource gate uses CPU affinity, cgroup-v1/v2 ancestor CPU
quotas/memory headroom, host load, four reserved CPUs, 16 GiB memory reserve,
and max(2 GiB, twice preflight peak RSS) per worker. Disk reservation includes
frozen copies and 4096 bytes per expected journal solve. Gates fail closed;
these are sampled headroom estimates, not resource guarantees.

Each policy has an exclusive log/result directory and row-level OOS/IS solve
journals. Failure stops new dispatch; already running workers are collected.
No partial plot, retry, numerical fallback or substituted result is published.
Final verification independently streams journal counts and checks all 810
results, input hashes, test arrays, oracle result/solution/source hashes and plot
provenance. Every neural result JSON is also hashed and rechecked after reporting
immediately before publication, alongside the sealed oracle binding. Plot data, PNG (300 dpi)
and vector PDF are rendered under `publication_staging`; after all validations
and reporting they are published together by atomic directory rename to
`published/`. No existing output directory may be reused.

## Process control and runtime accounting

The launcher uses actual named tmux with this directory as cwd; it does not use
nohup or silently fall back if tmux is absent. An adjacent exclusive
`OUTPUT.launch/` holds immutable invocation JSON, exact child/tmux commands,
source hashes, `child.log`, `child.exit`, and `child_runtime.json`. The monitor
records `time.perf_counter` across the whole child in `child_wall_time_s`,
separately from monitor elapsed including binding checks in `wall_time_s`.
Both have UTC boundaries; status/exit code and exact invocation are retained.
It prints session, log, attach, status and blocking completion commands. Invocation
and runtime metadata include `completion_channel = session + '_done'`. In the
monitor's `finally`, the runtime JSON and exit file are written and fsynced before
`tmux wait-for -S CHANNEL` signals completion (success or failure). This is process
completion only, not an experimental-method change. A signal does not mean success;
read the persisted exit code. If monitor metadata cannot be verified or persisted,
no trustworthy completion signal is promised.

For the preflight example:

```bash
tmux attach-session -t r10-comparison-preflight
tmux has-session -t '=r10-comparison-preflight'
tmux wait-for r10-comparison-preflight_done
cat /absolute/new/preflight_output.launch/child.exit
```

The session disappears when the child/monitor exits; the persistent exit and
runtime files distinguish success/failure. Do not infer success from disappearance.
No polling sleeps are used. Source changes between launch and child startup fail.
Use one blocking waiter on the same tmux server; session/channel names must be
unique per invocation. Production uses `tmux wait-for r10-comparison-production_done`
and `/absolute/new/production_output.launch/child.exit` for the examples above.

`summary.json` contains progress, missing/failed/completed policy counts and phase
clocks. `runtime_aggregates.json` separates current per-model evaluation totals,
OOS and IS by method/N/rho/status, with n, mean/median/min/max, from persisted green
nominal training per N/LR, nominal evaluation, CCG totals per N/rho (including
N50), initialization and both validation phases. Raw epochs, seeds, AP/MP
iteration timings and old concurrency evidence remain in frozen original
summaries. Nominal initialization is not counted again in CCG totals. Current
cache loading is never reported as historical model time. Historical SRO-FA
partition, saved `time_lp`, evaluation and total fields are distinct; `time_lp`
is solver time, not full optimization-wrapper elapsed. Missing kNN timings and
uncertified SCS full training timings are unavailable and require reruns.
Failed/incomplete policies are not mixed into completed aggregates. The sweep
clock and launcher whole-child clock are separate; concurrent model wall times
are never summed and called elapsed. Hardware/concurrency is recorded where
known; missing historical hardware is explicitly unknown.

The fresh common oracle has a separate runtime table and structured
`common_oracle` entry: construction, optimization, combined replay/NPZ
serialization, `oracle.run` total, and parent controller total. The current
oracle API does **not** measure replay and serialization separately; those
fields are unavailable and need instrumentation plus a rerun, not an inferred
split. The raw run total excludes its final result JSON write; the parent phase
includes that write plus audit/binding. Neither clock is nominal/CCG training,
fresh neural evaluation, persisted CV, or SRO-FA time. Failed/unattempted oracle
records stay separate from completed neural runtime aggregates.

`RESULTS.md` reports current and historical runtime tables and scientific caveats.
Implementation tests are not performance measurements of the experiment. Real
preflight, license availability, current headroom, full source freeze, 24 strict
QP checks, one training-only oracle LP and the production 10k oracle/replay remain pending until the parent/operator runs
them. The parent maintains the mandatory thesis modeling notes; this isolated
implementation changes only the five requested files: runner, preflight, runner
tests, launcher and this README. No full freeze, preflight, experiment or launcher
is run during this integration; unit fixtures are not experimental results.