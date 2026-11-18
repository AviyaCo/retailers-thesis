# BLACK continuation: fixed-policy comparison, 2026-09-26

User-authorized fixed-policy test comparison, including explicit approval to
retain and label capped trajectories using the training-only rule below.
Production requires successful preflight and independent review. Adding these
files alone is not evidence that numerical evaluation has completed.

## Scope and unchanged experiment

Add black OOS and dashed light-gray **selected-policy own-pool IS** to the
existing R3/R10 three-panel green comparisons. All original comparators,
numbers, method-specific grids, captions/qualifications, references, and saved
validation-selection evidence remain intact. New prose explicitly distinguishes
the old figures' timings and “fresh” claims from this invocation.

Exactly **570 fresh evaluations**: R3 has 270, R10 has 300, covering total
N=50/200/800, training seeds 0–9 and radii
0, .001, .003, .01, .03, .2, .4, .7, 1.5; black R10 alone also has 2.
Training counts are 40/160/640, validation reservations 10/40/160,
split seed 100000+training seed, network seed 0. Historical nominal/learning-rate
choices are reused. This is replay of warm-started continuation, **not retraining
from scratch** and not new validation/CV. There is no interpolation or newly
selected best test radius.

Selection is the earliest exact minimum mean AP incumbent over the complete
retained-plus-new assessed prefix, including terminal candidate 40. All eleven
capped/nonconverged policies remain: ten continuation caps plus the historical
R3/N50/seed9/radius.7 cap. A completed replay is not a claim of training
convergence. Backend collection validates full raw training prefix chains,
checkpoints and pools, without opening test arrays or selecting on OOS costs.

Information-access scope: historical comparison manifests can contain cached
OOS values, and isolated plotting tests read historical OOS-containing plot
datasets before policy sealing. These are provenance/display checks, not
selection inputs. No held-out scenario arrays are loaded and no new OOS
evaluation occurs before sealing. `test_bytes_accessed` refers to scenario-array
bytes, not to every artifact containing a historical test metric.

The exact original 10,000 scenario pairs, seed 9999, are copied and hash-bound,
never regenerated. This historically known test is not advertised as a newly
untouched holdout. Historical SRO-FA/kNN test certification caveats remain.
OOS bands are pointwise seed 20/80 percentiles, not confidence intervals.
IS is the mean of per-training-center maxima on the selected checkpoint's
own finite pool, not the terminal pool or the full adversarial supremum.

## Interfaces

- [run.py](run.py): import-safe orchestration; backend imported lazily.
- [backend.py](backend.py): owns training-prefix collection, immutable copying,
  version-locked evaluators and individual strict replays.
- [plotting.py](plotting.py): owns display-only overlay validation/rendering.
- [test_run.py](test_run.py), [test_backend.py](test_backend.py), and
  [test_plotting.py](test_plotting.py): all required preflight unit suites.
- [launch.py](launch.py): launches **this directory's runner**, not the parent
  continuation campaign, in an explicitly named tmux session.

CLI schemas (absolute OUTPUT must not exist):

```text
python -B -u run.py preflight OUTPUT
python -B -u run.py production OUTPUT --preflight PREFLIGHT/summary.json --sha SHA256
python -B -u launch.py preflight OUTPUT --session NAME [--wait]
python -B -u launch.py production OUTPUT --session NAME --preflight PREFLIGHT/summary.json --sha SHA256 [--wait]
```

Use the launcher for actual runs; its interpreter is
/data/home/aviyac/ap_env/bin/python. A preflight invocation **never** starts
production. Production requires the exact successful summary SHA printed at
preflight completion, not an unverified completed flag. `--wait` blocks using
tmux's completion channel, without polling. The launcher prints/persists its
exact command, working directory, session name, log path, attach command and
status command. Sibling OUTPUT.launch stores invocation metadata, persistent
combined stdout/stderr, whole-child monotonic timings and an exit marker.
Missing tmux stops explicitly; no silent nohup fallback, retry or resume.

Internal runner modes `_unit OUTPUT` and `_worker OUTPUT --manifest MANIFEST
--job ID [--smoke]` are implementation details, not an approved way to bypass
preflight. Production exit 0 means both figures passed and were published;
failure returns 1. CLI syntax errors use argparse's exit 2. Training caps are
retained metadata and do not themselves make successful fixed-policy replay fail.

## Preflight and immutability

1. Compile every top-level local Python source without bytecode writes. Run all
   three unit suites in a separate process, with a persistent log and structured
   discovered/executed test IDs/counts. Skips, expected failures, incomplete
   suites and subprocess errors are not passing evidence. Plot tests exercise
   actual large original base schemas with explicitly artificial test fixtures;
   these fixtures are never production observations or policy selectors.
2. Collect/validate all 570 policies once. Capture the exact seven local Python
   sources and README, all backend/evaluator dependencies, both original renderer
   sources and existing resource-auditor sources. Adding a top-level Python
   file or editing any frozen source/doc invalidates the preflight.
3. Resource/disk gates precede copying/scheduling. Copy the entire collected
   training dependency prefix (potentially tens of GB or more), not just selected
   checkpoints. Preserve original base manifests byte-for-byte as private
   inputs. Their redundant large `files` maps are replaced in the collection's
   `base_comparisons` metadata by an explicit reference to those original
   manifests; the backend still reads the original copied version-lock manifest.
4. Seal a full 570-policy manifest and same-stem SHA256 sidecar, read-only,
   before any new test binding. Neither `tests` nor base plot-data inputs are
   added to this policy-only manifest.
5. Execute exactly six **real** training-only probes, two centers each:

   | R | N total | Seed | Radius | Reason |
   |---:|---:|---:|---:|---|
   | 3 | 50 | 9 | .7 | Historical cap |
   | 10 | 50 | 8 | .2 | Continuation cap |
   | 10 | 200 | 6 | .7 | Continuation cap |
   | 3 | 200 | 0 | .4 | Size/dimension coverage |
   | 3 | 800 | 0 | .4 | Size/dimension coverage |
   | 10 | 800 | 0 | 2 | Radius-2 and size coverage |

   The first two saved training centers only: twelve strict QPs, no test arrays.
6. Independently verify raw probe journals, then rehash all origins/copies.
   Preserve the policy manifest and copied inputs permanently: production
   reuses them directly, without recollecting or recopying huge training inputs.
   A failed preflight still writes timestamps, monotonic phase durations,
   failure traceback and a sealed summary; it cannot authorize production.

## Production and numerical acceptance

Production revalidates preflight tests, all six raw probe journals, source
inventory, manifest seals and all original/private-copy hashes. It reuses the
unchanged preflight `files` entries, then binds original test arrays and base
plot data **after** verifying the policy seal. The original R10 summary is
copied/hash-bound and its completed status, plot-data pointer and locked
manifest digest are checked. Both raw-byte and canonical-JSON base hashes are
recorded. Final manifest retains all collection fields and adds frozen status,
16 workers, exact files/tests metadata and `policies_manifest_sha256`.

A single 16-slot ThreadPool supervises fresh one-policy Python subprocesses,
interleaving R3/R10. Numerical work is never done in the supervisor threads.
Each subprocess initializes only its own R-specific backend context, so bare
module aliases and cross-R state cannot collide. This deliberately accepts
570 numerical-import startups for isolation. All 570 manifest policies are
kept in memory; JSON writing is streamed and worker IPC contains small envelopes,
not enormous printed records. No cached evaluation, skipped seed, retry or
fallback is allowed. Ordinary worker failures do not cancel the other scheduled
policies; all outcomes are collected before the global verification failure.

Each model must have exactly 10,000 + sum(own-pool sizes) accepted projections,
Gurobi status 2, physical residual at most 1e-4, and unchanged strict membership
tolerance 1e-9. Parent journal verification checks:

- exact journal path/hash, phase, sequential row IDs and complete row count;
- strict status 2, `accepted=true`, finite nonnegative residual at most 1e-4;
- monotonic durations, UTC timestamps, solver iterations and thread settings;
- IS center IDs and original sample-ID mapping, including every own-pool row;
- agreement of actual journal counts/iterations with phase and model counters.

**Proof boundary:** the inherited evaluators do not persist scenario costs or
QP solutions in these journals. The parent verifies saved per-center maxima
and their mean, binds the unmodified evaluator/source/result bytes, and verifies
actual strict/physical acceptance. It does not falsely claim independent
cost recomputation from journals. Any stronger solution/cost audit would need
an explicitly approved backend extension.

Raw backend result bytes remain untouched: post-serialization returned hash
fields are excluded from the returned-versus-disk equality check. Display-only
rows gain `source_prefix_timings=training_runtime_reference` and
`solves_physical_accepted` **only after** journal verification. Every global
plot binding contains exact job identities, result SHA256 and raw-proof hashes.
The global barrier requires all 570 results, not just the current R cohort.

Both PNG/PDF pairs and plot datasets are rendered in an unpublished staging
directory. Only after both figures pass, reports are written, all sources,
copies, raw results, journals and seals are rechecked, and staged plot hashes
match, a single directory rename publishes the complete set. Failures leave
diagnostic staging only, not a partial published figure.

## Resource and runtime accounting

Existing cgroup-v1/v2 ancestor-aware resource definitions are reused under
source hashes. CPU reserve is max(4, 10% effective cores), plus observed load;
memory reserve is max(8 GiB, 10% available headroom). Per-worker budget is at
least 4 GiB, 1.5 times measured probe peak RSS and eight times serialized
manifest size. Production requires room for all 16 workers, otherwise fails
without silently reducing concurrency. Each worker uses one Torch/interop,
BLAS and Gurobi thread. Estimates are not guarantees against changing shared
host load or future memory usage. Disk gates include immutable inputs,
journals and large metadata/report artifacts.

Every invocation records `time.perf_counter()` durations, UTC start/end,
status, command, environment and concurrency. Preflight collection, copying,
policy sealing, unit suites and probes remain separate from production test
binding, cohort elapsed, journal verification, rendering and reporting.
Raw model results retain per-model total, target/projection/cost phase times,
training seed/size/radius, solver counts and iterations. Wrapper envelopes
separate fresh process startup/loading/serialization from backend model time.
The launcher records the whole child, including final summary serialization.
Every attempt also writes an invocation runtime table and structured runtime
artifact, including syntax/unit failures. The sealed final summary is authoritative
for report/final-verification durations and publication status; staged reports
are necessarily constructed before the final atomic rename.

Reports include fixed common-radius black-versus-green costs without a new
test-based winner; completed current replay count/mean/median/min/max by R/N/radius;
and a runtime category table for nominal training, robust AP/MP, SRO-FA,
evaluation/CV and benchmark construction. No nominal/CCG/SRO-FA/CV is newly run.
Historical per-seed nominal and per-iteration prefix timings remain distinct.
The persisted parent continuation whole-child time is read from its original
artifact (approximately 62,823.45 seconds, nonconverged), not hardcoded as a
new measurement or included in completed-model aggregates. Its actual training
concurrency is documented in parent artifacts, not inferred from replay workers.
Missing historical fields are unavailable and require a rerun to measure.
Concurrent wall times are never summed and called sweep elapsed. Preflight
unit/probe timings are never relabeled as model training or production replay.

## Review status

Initial syntax/schema repairs passed 56 unit tests. The first preserved
preflight failed during training-evidence collection after 26.940099 seconds,
before any numerical probe. The adapter had reassociated the producer's
`point - (center + radius)` as `point - center - radius`, introducing a spurious
4.8e-15 membership residual. The original arithmetic is now retained exactly,
with an explicit regression test and no tolerance relaxation. Independent
read-only review permits production only after a successful corrected-source
preflight, all six real probes, journal verification and source/input freezing.
There is no automatic production launch.