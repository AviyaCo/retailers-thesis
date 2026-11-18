# Physical consolidation and verified duplicate removal — 5 October 2026

The user approved replacing the shortcut-only arrangement with physical
consolidation of both Hydro and Retailers. No scientific implementation,
dataset, protocol, checkpoint, result, source hash, or frozen manifest is
rewritten. These scripts are migration/execution support, not new experiments.

**Completed:** duplicate removal finished on 5 October at 12:27:50 UTC with
status `deduplicated_verified`. All 1,417,160 planned duplicate files were
removed: 261,101,931,226 logical bytes (261.10 GB). Removal took 2,530.928 seconds
(42.182 minutes), one worker. The maintenance restriction is lifted; keep frozen
evidence unchanged and use the documented execution tasks for historical code.

The [final post-cleanup verification](final_validation.json) passed: all 23
canonical inventories unchanged, 365 archive aliases and 112 workspace links
valid, all 16 unique entries retained, no temporary holding directory. Source
compatibility passed again (210 Hydro hashes, 14 shipping bindings), as did all
nine scratch tests. No numerical reproduction is claimed.

## Ownership

- Hydro's 17 original trees now physically occupy their grouped locations in
  the Hydro workspace's History section.
- Retailers' six current trees now physically occupy the Retailers workspace's
  History/original_paths section. Both new October 5 directories are included.
- [paths.json](paths.json) is the exact old-name → physical-owner map.
  Original top-level directories are gone; no compatibility symlinks are left
  there. Navigation links were retargeted inside the collections.
- The initial comparison found 1,417,160 equal original/archive file pairs,
  261,101,931,226 logical bytes (261.10 GB; 243.17 GiB). No differing matched
  files were found. Fourteen archive-only launchers and two new current-only
  directories are retained. Completion is established by the final journal and
  post-cleanup verification above, not merely by the initial comparison.

## Exact progress and safety record

- [comparison.json](comparison.json): initial complete SHA256 comparison,
  maximal duplicate candidates, differences and planning fingerprints.
- [hydro_relocation.json](hydro_relocation.json) and
  [retailers_relocation.json](retailers_relocation.json): journaled same-filesystem
  moves; all original inode/size/mode/mtime inventories matched after relocation.
- [deduplication.json](deduplication.json): authoritative live/final status and
  **completed** removed-file/byte counts. Failures remain explicit; no automatic
  resume, fallback, or replacement of distinct versions.
- [deduplication.log](deduplication.log): stdout/stderr. Persistent tmux session
  `organized_remove_duplicates_20261005`; controller command is the existing
  Python environment with `-B -u deduplicate.py`, working in this directory.
  Attach with `tmux attach -t organized_remove_duplicates_20261005`; check
  session presence with `tmux has-session -t organized_remove_duplicates_20261005`.

The initial read-only audit ran in session `organized_dedup_20261005` from this
directory using `/data/home/aviyac/ap_env/bin/python -B -u compare.py`, with
stdout/stderr in [comparison.log](comparison.log). Its monotonic elapsed time
was 2,913.453 seconds (48.558 minutes), one worker. Hydro rename/verification
took 3.194 seconds; Retailers rename/verification took 38.434 seconds. These
are storage-migration times, **not model runtimes**. Final removal time is
recorded separately in the completed deduplication journal.

Each candidate is first moved to temporary holding and replaced with an internal
relative alias to the retained canonical bytes. Immediately before irreversible
removal, both copies are freshly hashed again, including stable ctime, inode,
size, mode, mtime and directory membership checks. Subtrees containing symlinks
are rejected. Postorder removal unlinks only individually verified files; it
does not recursively delete unknown additions. A mismatch aborts with held
bytes and progress preserved for inspection. Edits and research jobs were to be
avoided during the now-completed maintenance operation. This is not protection
against a malicious same-UID writer racing the final system call.

**Archive aliases share the canonical bytes; they are not independent backup
copies.** Back up whole collections including workspace and hidden files,
preserving relative links. Do not back up only the archive subtree. Older
independent-copy verification reports remain historical records, not current
independence guarantees. Do not modify frozen canonical bytes through any alias.

## Work from the organized folders

Open the Hydro or Retailers workspace and use **Terminal → Run Task → Open
historical Hydro/Retailers shell**. [run_legacy.py](run_legacy.py) uses installed
Bubblewrap to expose original path names only inside a local mount namespace.
This preserves `Path.resolve()` manifest keys, imports and documented working
directories without modifying frozen sources. It is **path compatibility, not
a security sandbox**. The current host Python environment, license and system
tools remain prerequisites. Missing prerequisites fail rather than falling back.

The CLI takes `--cwd` (an old root/subdirectory or physical canonical path),
optional repeatable `--write` output subdirectories, then `--` and the command.
Whole-tree writes, parent traversal and symlinked write paths are rejected.
Output directories must already exist; choose fresh ones, not previous results.
Both historical and physical aliases are otherwise read-only inside the
namespace. Home-level files not backed by an explicit existing bind are tmpfs
and will not persist; keep outputs within designated physical directories.

Normal subprocesses inherit the mapping. Existing tmux servers, schedulers and
other external services do not. Launch tmux outside the runner, then invoke the
runner inside a pane. Numerical jobs still require their original timing/logging
protocols. Each wrapper invocation writes a separate sidecar under executions;
its elapsed time is invocation time, not a replacement for per-model timing.

## Validation and limits

- Nine scratch-only tests passed in 0.133 seconds: preserved virtual paths,
  child processes and output renames, frozen-file protection, no fallback,
  write-path traversal rejection, same-size unequal contents, archive-only
  history, unverified additions and symlink-subtree rejection.
- Hydro: all 17 virtual roots and 210 genuine frozen source hashes verified;
  the rescaled configuration imported without calling a data loader.
- Retailers: the original shipping `bound_file` function passed all 14 selected
  frozen source/README bindings. The October 5 published figure was present.
- [source_validation.json](source_validation.json) records the combined check
  of all 23 virtual roots, 210 Hydro hashes and 14 shipping bindings, configuration
  import and child cwd: latest post-cleanup check passed in 5.280 seconds without
  data loading or solves (the earlier check took 6.224 seconds).
- [navigation_validation.json](navigation_validation.json) records 48 internal
  Hydro navigation links, 64 internal Retailers navigation links, valid guide
  links/tasks, and unchanged inventories for all 23 canonical roots (34.279 s).
- Six existing synthetic archive-resolver tests passed (0.007 s); both shared
  green/black CLI protocol checks passed without creating output. The first
  bare `check` invocation was correctly rejected for missing required arguments;
  the successful checks supplied the explicit configuration.
- An initial configuration import failed because ordinary bind mounts blocked
  device access; explicit `/dev` binding corrected it, and checks passed again.
- No numerical solver, training, test/validation evaluation or plot generation
  was run. No model runtime was measured. Hydro raw observations were not read;
  Retailers storage comparison hashes bytes without parsing observations or
  using them for inference. Path/source tests are not numerical certification.