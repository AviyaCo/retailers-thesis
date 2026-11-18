# Black continuation: R3 and R10

Implementation only. The parent owns launch, tmux, execution, and the corresponding
thesis-methodology note. No experiment or test was run while writing these files.

## Fixed cohort and method

Exactly **570** rows: **450** original green jobs, **30** fresh R10 radius-two
jobs, and **90** cached R3/N50 black rows. Seeds are 0–9. Original radii are
0, .001, .003, .01, .03, .2, .4, .7, 1.5; R10 additionally uses 2.
R3/N200,N800 contribute 180 existing jobs; R10/N50,N200,N800 contribute 270.
All 480 adapter jobs are scheduled once, including already-closed green STOPs.
The original black cohort remains 89 separated and one capped (seed 9, radius .7).

Continue **the last green STOP checkpoint on its own saved scenario pool**, not
the green minimum-AP returned checkpoint. The budget is **40 total MPs including
the original MPs**, not 40 additional MPs. Stop when every training center has
`max(0, bound - pool_cost) / max(1, abs(bound), abs(pool_cost)) <= .01`.
Selection is the earliest minimum AP-incumbent mean over the complete original
plus new prefix, including STOP. The adapter publishes a selection only on
convergence. The already-capped historical black selection is preserved separately.

The expected green-last pass/continue inventories are R3/N200 38/52, R3/N800
39/51, R10/N50 42/48, R10/N200 45/45, and R10/N800 45/45. Inventory mismatch,
unknown status, altered source, missing input, or failed numerical checks fail
closed. There is no restart, retry, solver fallback, or tolerance change.

The 80/20 split, network seed 0, split seed `100000 + seed`, saved LR winner,
and original native numerical settings are unchanged. No new nominal training,
validation selection, test evaluation, or OOS-data access is performed. The
saved campaign summaries supply winners from `phases.lr_validation` and source
jobs from `phases.ccg`; no fresh validation is used. Historical black is
reconstructed with its hash-locked pure reconstruction module from the original
20260916T080636.766867Z trajectories and checked against the **training-only**
choice artifact written before validation/test. The OOS comparison summary and
test arrays are not opened, and evaluator modules are not imported.

## Source and input integrity

The original R3 and R10 provenance maps contain 35 and 54 bindings respectively;
their union uses the **original recorded hashes**, never newly blessed hashes of
changed historical sources. Provenance root hashes are pinned in the controller.
Each full original trajectory is bound to its source campaign's recorded SHA.
Every candidate checkpoint, pool, assessment, sample JSON, and recursive MP file
is frozen, including controller-training records. Original per-job provenance,
train-only data and nominal copies, saved winners, and the reused black selections
are also covered. Full samples remain in original full summaries, not duplicated
inside manifest jobs. Original inputs are not copied, chmodded, or rewritten.
The adapter owns private new disposable/production copies.

Local source capture is the explicit **nonrecursive `*.py` plus README.md**
pattern, including the parent's separately authored launcher when present. Finish
all local files before preflight. Adding or changing one afterward invalidates
the gate. The manifest and SHA sidecar are read-only. Inputs are initially hashed
once by the inventory builder, verified before/after each applicable job, and
verified globally before production and after the sweep. No continuation backend
is imported in the scheduling parent.

## CLI (parent executes; all OUTPUT directories must be new)

From this directory, with the intended environment's Python:

- `python -B -u campaign.py freeze OUTPUT`: inventory and sealed manifest only;
  no tests, APs, MPs, or production work.
- `python -B -u campaign.py preflight OUTPUT`: freeze the full inventory, run
  all local `test_*.py` modules in a separate subprocess, then six isolated real
  checks: seed 0, radius 1.5 at R3/N200,N800 and R10/N50,N200,N800, plus R10/N50
  radius 2. No production is launched. The adapter checks the entire source
  prefix, replays the native saved pool, solves one center AP, and performs an
  explicitly disposable two-epoch MP.
- `python -B -u campaign.py production OUTPUT --preflight PREFLIGHT/summary.json --sha EXPECTED_SHA`:
  require the externally supplied summary hash, complete exact smoke grid,
  successful unit-test process, unchanged logs/raw adapter summaries, nested
  success evidence and all frozen inputs. Preserve the complete original
  preflight manifest and make a new runtime manifest with selected worker count.

The parent launcher must run long work in a named tmux session from this working
directory with persistent stdout/stderr, record the exact command and log, and
report attach/status commands. No implicit production launch follows preflight.
Internal `_job` and `_smoke` entrypoints receive a manifest SHA and a single job ID.

One shared `ThreadPoolExecutor` launches **one fresh Python subprocess per job**
across both R values, avoiding unqualified backend-module collisions. All numerical
thread environment variables are one. The adapter enforces Torch and Gurobi
thread counts. Scheduling interleaves R3/R10 size groups and seeds at each radius.
One job failure does not cancel any of the other 479 jobs. Exceptions and nonzero
exits are persisted, never treated as successful convergence.

Worker selection reuses only hash-bound pure resource definitions via AST, not
backend/preflight imports. It accounts for affinity, mounted cgroup ancestors,
CPU quota, cpusets, memory headroom and current load. The single joint pool caps
at 16 workers, reserving at least four CPU slots and 10% effective CPUs, at least
8 GiB and 10% available memory; each worker budgets at least 4 GiB and 1.5 times
the largest measured smoke RSS. Insufficient resources fail before scheduling.
This is resource adjustment, not degraded numerical settings; sampled RSS cannot
guarantee a bound on future AP search-tree memory.

## Adapter integration (actual API)

`continuation.run_job(job, manifest_path, output_folder)` and
`continuation.preflight_job(job, manifest_path, output_folder)` return compact
JSON-safe result dictionaries and own the initially nonexistent output folder.
The controller calls these APIs, not the adapter's independent command-line API.

Job fields: `id`, `R`, `Ntotal`, `seed`, `radius`, `source_summary_path`,
`source_summary_sha256` (both null only for fresh R10/radius2), `nominal_record`
(actual saved winner), `input_sha256`, and `expected` (STOP identity, original
MP count, remaining budget, closure and historical wall time).

Manifest fields required by the adapter: `source_sha256`,
`sourceconfig["3"]` and `sourceconfig["10"]` (complete original provenance,
including original protocols and source hashes), positive `workers.ccg`, and
`jobs`. New manifest protocol does not replace the original backend protocol.

Results bind `summary_path`/`summary_sha256`, `current_timings.wall_time_s`,
`mp_count`, `terminal_status`, `status`, `complete`, `converged`, `cached_only`,
and per-process `peak_rss_bytes`. Completed closure is `terminal_status="converged"`;
`iteration_cap`/`stagnation` with `status="nonconverged"` remain nonconverged even
though `complete` is false. Already-closed jobs are reported `cached_only`.
For smoke evidence the parent opens the adapter's hash-bound raw summary and
checks input verification, native pool replay, one optimal center AP, exactly two
MP epochs, incoming/returned pool replay, disposal flags, and nested statuses.

## Persistence and runtime accounting

The live summary exposes counts, manifest, worker budget, environment, finished
job count and status groups. Each child has command/cwd, persistent stdout/stderr,
exit code, timestamps, monotonic invocation duration, result hash and raw adapter
evidence. Failed runs retain partial adapter AP/MP timings. Production returns
exit 1 for any failed jobs, 2 for nonconvergence without failure, and 0 otherwise.

Raw seconds are persisted; reports show seconds/minutes/hours. Separate timing
tables cover current subprocess invocation, current CCG, historical CCG prefix,
historical nominal winner training per N/seed, and explicitly labeled
`previous_ccg_wall_time_s + current_ccg_wall_time_s` active-phase time. This sum
is **not** a fresh end-to-end invocation. Current cache verification is not
training time. Completed, failed, nonconverged and cached-only rows never mix in
aggregates; each R/N/radius group reports n, measured n, mean/median/min/max.
Concurrent job times are never summed as sweep elapsed. Whole-invocation and
joint-pool elapsed times are separately recorded. Original campaign LR-search,
validation, CCG-sweep and overall measured times remain separate in the manifest.
New evaluation/CV and SRO-FA are not run; unavailable historical fields are
explicitly unavailable, requiring a rerun rather than invented timings.

The new unit tests cover cohort/closure counts, cap accounting, identity and hash
rejection, no old writes, fair scheduling, independent failure handling, resource
reserves, nested preflight failures, and disjoint runtime groups. Execution and
real numerical integration remain the parent's responsibility.