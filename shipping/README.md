# Shipping: local evidence and plotting workspace

Shipping is a separate formulation from the no-shipping black/green experiment.
Both are preserved in the consolidated parent directory without relabelling runs.

## Current stopping-checkpoint figure (5 October 2026)

The new [PNG](../../retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/published/comparison_shipping.png)
and [PDF](../../retailers_moreau_training_selected/shipping_stopping_20261005/production_v2/published/comparison_shipping.pdf)
use all270 green stopping checkpoints and their own pools. The480 historical
Clarabel and all benchmark curves are unchanged from September28. The replay
completed successfully in345.908s (5.765min), including108.726s parallel
evaluation with16 single-threaded workers; no training or CV was rerun.
See the [completion status and runtimes](../../retailers_moreau_training_selected/shipping_stopping_20261005/STATUS.md).
These links point to the original new experiment outside this consolidated
archive; they are navigation links, not a claim that the new evidence has been
copied into the September backup. The local restore entry point below still
recreates the historical minimum-AP figure, not this new stopping figure.

## What is here

- [shipping_live_green_20260926](shipping_live_green_20260926): original shipping
  green nominal/CCG campaign and all retained attempts/results.
- [shipping_advisory_continue_20260928](shipping_advisory_continue_20260928):
  advisory continuation evidence used by five policies in the published comparison.
- [retailers_with_shipping](retailers_with_shipping): historical **Clarabel**
  neural runs, SRO-FA, SAA-kNN caches and conditional-reference results for
  N50/N200/N800, including corrected N800 minibatch results.
- [shipping_comparison_20260928](shipping_comparison_20260928): comparison
  scripts, sealed manifest, frozen inputs, test arrays, all 750 evaluation
  records, plot statistics, runtime reports and published PNG/PDF.
- Other labelled shipping folders preserve diagnostics and failed attempts.

**No trained black shipping comparator exists in these records.** The shipping
comparison contains green and historical Clarabel, not black. The published
green cohort consists of 265 original policies and five advisory continuations.
Existing no-shipping black runs remain under the parent directory's black section.

The gray line is the saved finite one-million-scenario conditional MIP reference,
not an exact population lower bound. Its saved best bound and nonzero gap remain
in the original result JSONs. No solver run or new certificate was fabricated.

## Recreate the published figure here

[render_saved_plot.py](render_saved_plot.py) imports only the local archived
presentation module and reads its saved plot statistics. Supply `--output` with
a **new** directory outside the archive. It does not read the original home
folders, train, select policies, run solvers, or reevaluate the test set.

The [local restore check](plot_restore_check_20260930) successfully recreated
the PNG/PDF using that route in 2.830926 seconds; the recreated PNG is
byte-identical to the published original. Rendering provenance records input/renderer hashes
and wall time. [plot_dependency_check.json](plot_dependency_check.json) records
that all 4,205 manifest-bound sources and frozen copies resolve inside the
consolidated backup, with no missing paths. This is a dependency presence check,
not a fresh numerical evaluation.

Historical scripts and manifests retain original absolute paths and hashes.
Do **not** directly launch archived training/comparison controllers assuming
relocation makes them portable: some still access the old home. For inspecting
inputs use the parent's archive path resolver; for figure reproduction use the
local rendering entry point. Keep future working outputs outside the archive.
Full relocated training has not been parity-certified or launched.

## Storage limits

The full local folder includes results/checkpoints and plot inputs. The approved
remote Git backup is code/documentation/figures only; it is **not** a complete
results restore. See [../GIT_BACKUP.md](../GIT_BACKUP.md). No off-machine full
evidence destination has been selected. Original folders must not be deleted
on the assumption that the Git branch holds all evidence.