# STOP checkpoint replay

This directory replays **only existing stopping checkpoints**: GREEN 540 and
BLACK 570, including all 11 BLACK iteration caps. No training, AP, MP, policy
optimization, validation, radius selection, test generation, or cache reuse is
performed. Every policy gets a fresh replay, even when learned states coincide.
The test is the historically known, fixed seed-9999 set of 10,000 pairs per R;
this is a descriptive comparison, **not a new holdout**. Historical comparator
curves and their qualifications remain unchanged.

## Entry points (not automatically launched)

Use the configured `/data/home/aviyac/ap_env/bin/python` interpreter with `-B`.
From this directory, the separate commands are:

```sh
/data/home/aviyac/ap_env/bin/python -B launch.py preflight /ABS/NEW_PREFLIGHT --session stop_preflight_20260928 --wait
/data/home/aviyac/ap_env/bin/python -B launch.py production /ABS/NEW_PRODUCTION --session stop_production_20260928 --preflight /ABS/NEW_PREFLIGHT/summary.json --sha EXPLICIT_64_HEX_SUMMARY_SHA --wait
```

Replace paths and the explicit SHA with the completed preflight receipt. Output
and sibling `.launch` directories must not exist. `--wait` is optional; it uses
`tmux wait-for`, not polling. Preflight never starts production. No automatic
worker reduction, retry, resume, fallback, or nohup substitution is provided.
The wrapper uses its own `sys.executable` for every child and tests.

Each launch stores its exact command, working directory, named tmux session,
persistent combined stdout/stderr log, attach/status/log commands, and exit code
under the sibling `.launch` directory. For the example above, attach with
`tmux attach -t stop_preflight_20260928`; inspect the path printed as
`status_command` or `log_command`. A sealed `child_runtime.json` measures the
whole run child, including imports and final serialization/publication. A tmux
session may no longer exist once its child has exited; the logs/receipts persist.

## Mandatory preflight

1. Run **all local `test*.py`** suites in a separate `sys.executable -B` child.
   No skipped/expected-failure tests count as a pass. The shared repository tests
   are a separate parent responsibility, not silently substituted here.
2. Successfully call `selection.collect(output)`. This performs the expensive
   full-prefix audit and roughly 73 GB of binding reads. A selection seal alone
   is insufficient: the completed sealed verification receipt must also bind
   it. Test-file hashes are checked by selection **after** sealing the policies;
   preflight never loads or evaluates test arrays.
3. Bind every local Python file and README, the active evaluator/artifact/package
   modules, archive resolver/mapping, and historical bound sources. Require exact
   historical Python, NumPy, SciPy, Torch and native Gurobi versions. Inspect
   cgroup ancestors, affinity and load; require 16 single-thread workers after
   reserving 8 CPU slots plus background load, 50 GiB memory reserve, and 4 GiB
   per worker. No automatic degradation. Observed probe RSS must fit 4 GiB.
4. Perform **12 real terminal-policy probes**: GREEN and BLACK at seed 0,
   radius 0.4, for R3/R10 × N50/N200/N800. Verify checkpoint, learned state,
   training centers/IDs, own-pool membership and digest. Compare targets on the
   **entire training-center batch** (40/160/640 rows), including exact float32
   inventories. Compare strict QP shipments and float64 costs on its first two
   training rows. Require exact equality, OPTIMAL status 2 and physical residual
   at most 1e-4. All prescribed probes are attempted and retained even on failure.
5. Seal the summary, per-probe timings/statuses, collection/verification hashes,
   unit evidence, source hashes and environment. Any failure blocks production.

The numerical reference is **not a historical module import**. Only AST-selected
`Model`, `load_model`, `state_hash`, `scenario_costs`, and `strict_project`
definitions from SHA-locked `compare.py` (`8285365c…b366`) are executed in an
explicit numerical namespace. R10 takes the eight applicable, exact-count
dimension substitutions from locked `reference.py` (`37627495…105f`); its test
generator and all other historical top-level code are never executed. The
active learned-only evaluator is required to load these existing Moreau states
without adapting their payloads or numerical definitions.

## Production and verification barrier

Production requires the explicit preflight path/hash, passed phases/probes and
all 1,110 identities. Changed sources, incomplete collection, changed versions,
or insufficient resources stop it. Selection artifacts remain untouched. After
validating the sealed selection and preflight, historical test arrays are loaded
and their file-byte and array-content hashes verified. A separate execution
manifest then records that verification and references the complete sealed
selection and its verification receipt, before any worker replay starts.

One spawn `ProcessPoolExecutor(16)` loads the selection once per initializer;
jobs are submitted by index, not by copying the large per-job input map. Each
worker checks sources, environment, test arrays, and its own checkpoint/state,
training centers and native pre-insertion pool. Each policy receives its own
sealed OOS input binding. Strict standalone evaluation writes separate `oos`
and `own_pool` directories containing targets, shipments, costs, raw result and
one diagnostic per solve. Pool costs are means of per-center maxima on that
checkpoint's own finite pool, not a full adversarial supremum.

Workers and the parent independently read **every** journal row, checking exact
ordinals/counts, OPTIMAL acceptance, finite timings/iterations, physical
acceptance/residuals, shipment-array agreement and aggregate counts. Both check
array dimensions/dtypes, full-batch targets, inventories, exactly recomputed
costs, and pool maxima. The parent checks raw-result hashes, policy identities,
input seals and model states. Finally it rehashes **all** selection inputs and
sources. Only the parent adds `journal_verified=True` to publication rows;
raw results are not rewritten.

Any failed job retains diagnostics and timing but no claimed means; a crashed
worker without a returned duration is explicitly unmeasured, not zero. Failed
or incomplete sweeps do not publish plots. Both figures and reports are created
in a new empty staging directory only after the global 1,110-job barrier, then
published together by directory rename. Diagnostic staging is retained on error.

## Timing and provenance

All new measured durations use `time.perf_counter()` seconds with UTC endpoints.
Records distinguish input loading, OOS, own-pool, verification, per-policy wall
time, whole-worker serialization-inclusive time, initializer/scheduling-inclusive
parallel sweep elapsed, selection, report generation and parent pipeline time.
Concurrent worker times are **never summed** as elapsed sweep time. Thread and
worker counts and the runtime environment are retained. The report groups
completed policies by method, size, radius and training convergence/cap status,
with counts, mean/median/min/max times. Failed/incomplete jobs stay outside those
aggregates and cannot pass publication.

Historical training remains solely in the sealed selection manifest, referenced
by raw job results; it is not copied into each worker artifact. Reporting retains
persisted nominal and AP/MP phase timings separately from current replay time.
Missing historical end-to-end, SRO-FA, CV and nominal-evaluation timings remain
unavailable, requiring a rerun where applicable. No training timing is invented.

The publisher seals `runtime_report.json`; its `whole_pipeline_elapsed_s` remains
**null**, not silently patched/resealed. The appended RESULTS note points to the
separate parent `pipeline_summary.json` for measured phase/sweep/parent elapsed
time, and to the sibling launch `child_runtime.json` for full subprocess elapsed
including imports and final serialization. The final parent summary binds every
published artifact, including the updated RESULTS. Parent elapsed excludes its
own final serialization; only the wrapper measures the full run child.

## Bounded developer check

```sh
/data/home/aviyac/ap_env/bin/python -B -m unittest -v test_run
```

These tests use temporary synthetic fixtures/mocks only: no collection, real
policy probe, test-array load, production launch, or frozen-artifact write.
Passing unit tests is **not** evidence that the real parity preflight passed.