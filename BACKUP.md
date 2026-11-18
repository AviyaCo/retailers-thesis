# Single-folder backup

**October 5 physical consolidation supersedes the earlier navigation advice.**
[workspace/](workspace/START_HERE.md) now physically contains all six current
experiment trees, including the newer October 5 shipping work. **Do not exclude
it from backups.** The original top-level directories were relocated here.
Byte-identical archived copies are replaced with internal relative links only
after fresh verification; unique/changed/archive-only records remain retained.
Thus the archive alone is no longer an independent second copy. Back up this
whole collection, including workspace, archive, hidden dependencies and the
[migration/execution support](.consolidation_20261005/README.md). Preserve links
without dereferencing them. Earlier copy counts below are historical evidence,
not a claim that duplicate bytes are still independently stored.

**Historical September backup construction (before deduplication):**
It created independent copies of 1,205,012 archived files, 237,039,626,798 bytes
(237.04 GB / approximately 220.76 GiB). The 26 labelled method/comparison
shortcuts are relative symlinks whose targets are **inside this directory**.
Preserve relative symlinks when archiving; do not back up only the shortcuts.
No original source folder must accompany this backup to retain these files.

## Included

- Historical SRO-FA R3/R10 sweep scripts, result JSONs and available logs.
- Historical width-16 SCS R3/R10, N50/N200/N800 nominal and robust archives.
- Moreau green/black source, checkpoints, pools, data, logs, full continuation
  records, preflights and failed-attempt evidence.
- Original and black-overlay comparison figures, raw evaluation journals,
  frozen paired test arrays, manifests and runtime reports.
- Referenced shared source, supplemental timing evidence and Moreau 0.4.0 CPU
  dependency (including its existing package metadata/licenses).
- Current compact shared implementation, configuration and tests.

The original 28 September collection excluded shipping and other neural widths.
The shipping expansion requested on 30 September is recorded separately under
[.archive_migration/shipping_20260930](.archive_migration/shipping_20260930).
It completed: 205,290 added files / 23,950,539,995 bytes, all source/copy
SHA256 pairs verified, in 483.025048 seconds (8.050417 minutes).
The follow-up [.archive_migration/completion_20260930](.archive_migration/completion_20260930)
preserved all 7,184 additional uncovered files (including 38 plots), verified
their source/copy hashes, and completed in 57.345906 seconds. All six original
retailers source roots are now covered: 1,417,174 source files. Existing copies
were compared by size/mtime, hashing mismatches; no source differences remained
at the audit time. This is not a fresh full hash scan of all existing copies.
Neither operation overwrote archived evidence or deleted originals.
The original verification below applies to the original collection, not these
expansions. The path-map totals include archived dependency/support files too.

Shipping plotting dependency checks resolved all 4,205 manifest-bound source
and copy paths inside the backup. A saved-data-only local recreation took
2.830926 seconds and produced a PNG byte-identical to the published original.
See [shipping/README.md](shipping/README.md) for method scope and local plotting.

The code, documentation, figures and small result files are published in the public repository
[AviyaCo/retailers-thesis](https://github.com/AviyaCo/retailers-thesis) (since 18 November 2026; how to update it:
[REPRODUCE.md](REPRODUCE.md), "Updating this repository"). For what Git does and does not contain, and the
distinction from the full local evidence backup, see [GIT_BACKUP.md](GIT_BACKUP.md).

## Verification

- [Copy status](.archive_migration/status.json): all source/copy SHA256 pairs
  matched, with separate inodes (no hard links back to originals).
- [Per-file SHA256 inventory](.archive_migration/files.sha256.jsonl).
- [Final verification](.archive_migration/verification.json): exact source/copy
  file and directory inventories, unchanged size/mtime, all 26 links internal,
  and 932,481 frozen hash-binding checks across four primary manifests.
- Eight R10/N800 nominal timing-summary paths were **already absent** and
  explicitly marked missing in the historical manifest. Their checkpoints and
  metadata are copied. Missing timings remain unavailable; nothing was fabricated.
  The first diagnostic report is retained separately.
- Copy plus hashing: 2,063.321464 s (34.388691 min). Final inventory/dependency
  verification: 118.661410 s (1.977690 min). These are migration times, not model
  runtimes. No research experiment was rerun.
- Shared-code/path tests: 61 passed in 4.710 s. Unit fixtures do not prove
  historical numerical parity.
- A differently named temporary restore containing copied shared code, bundled
  Moreau and a real sealed summary passed local dependency import, historical
  input resolution and the shared CLI protocol check. All seven entry-point
  documents were checked to link only inside the backup. This was a bounded
  restore smoke test, not a second 221-GiB restore or a numerical experiment.

This is a verified point-in-time copy, not an atomic filesystem snapshot or a
promise that later changes to either directory will remain synchronized.

## Paths and provenance

[archive_paths.json](archive_paths.json) records the old-home-to-archive map.
[archive_paths.py](archive_paths.py) provides `resolve_archive_path(path)` and
`local_json(path)`. Resolution always uses the backup, even if the original
exists. Missing copies and paths escaping the backup fail explicitly. The CLI
accepts a path and optionally `--json-view`; that derived view lists unavailable
references and retains the original digest. It never overwrites or re-seals
historical records.

Current [run.py](run.py) remaps explicitly supplied historical CLI input paths;
the path map is required for commands reading inputs. Its Moreau default is
relative to this folder. Native source/environment identity checks remain
enforced. Historical checkpoint formats were not converted into native CCG runs.

**Historical scripts/manifests remain byte-identical.** Old path strings are
provenance. The current compatibility runner provides those virtual paths for
execution backed by the relocated files. Source/import checks do not certify
every numerical workflow. Use the resolver to inspect/reuse inputs. Blindly
replacing paths in those records invalidates hashes. This is a backup of the
selected evidence, not a portable VM/container or a newly certified executable
version of every historical campaign.

## Restoring an execution environment

The Python interpreter, OS, GPU drivers and Gurobi license are **not copied**.
Recreate the environment with `setup_env.sh`, which installs the exact versions in
[requirements.txt](requirements.txt) (see [REPRODUCE.md](REPRODUCE.md) section 2). Original Python was 3.11.13 on
Linux x86_64; bundled Moreau contains a platform-specific binary. Configure Gurobi licensing
separately; no license or credential was included.

The shared implementation remains unverified for full old/new numerical parity.
Keep archived evidence immutable and use separate outputs for future runs.