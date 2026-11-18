# Green

Edit shared code in [../src](../src), not a colour-specific implementation.
The green rule is the strict mean AP-incumbent/pool gap below one percent,
including candidate zero (zero MPs allowed). Return the stopping checkpoint,
after validating every assessment in the inclusive prefix; never choose an
earlier minimum AP-incumbent mean. Use the stopping checkpoint's own pool for
IS evaluation. Native v3 summaries record this return rule explicitly.
Green still publishes only on convergence; capped green runs remain nonconverged
without a published selection.

## Historical source and result directories

- [N50/R3 history](runs/R3_N50/) and [stopping-only results](runs/R3_N50/stopping_only_min_ap/results/RESULTS.md)
- [R3/N200/N800 campaign](runs/R3_N200_N800/)
- [R10/N50/N200/N800 campaign](runs/R10_N50_N200_N800/)
- [Shared figures](../comparison/README.md)

These are internal directory links to complete local archive copies, including run
scripts, checkpoints, results, logs and provenance. N50/R3 contains shared
green/black history; the stopping-only subdirectory identifies the green
selection evidence. All files were copied and hash-verified inside this backup.
Editing a linked file changes the local copy, not the external original; keep
archived evidence immutable. Back up the entire main directory.
Future parity-verified runs need a new directory, not an existing archive link.
No historical run is claimed to have used the new shared sources.
Historical min-AP selections and archived configurations remain immutable;
they are not relabelled as stopping-checkpoint results. Historical terminal-only
re-evaluation is a separate comparison, not a rewrite of these archives.