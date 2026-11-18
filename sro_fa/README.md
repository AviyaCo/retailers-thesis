# Historical SRO-FA runs and results

Original no-shipping sweep scripts, saved result JSONs and available logs for
the SRO-FA curves in the R3/R10 comparison. These are directory links to the
local archive copies, not new SRO-FA fits or reconstructed model artifacts.

| Configuration | Original scripts and results |
|---|---|
| R3, N50 | [R3/N50/](R3/N50/) |
| R3, N200 | [R3/N200/](R3/N200/) |
| R3, N800 | [R3/N800/](R3/N800/) |
| R10, N50/N200/N800 | [R10/](R10/) |

Each directory contains its original sweep script and result JSON files.
R10 uses one shared sweep script and three separate result files. R3 retains
its original per-sample-size directories. All targets were copied under this
backup's archive tree; each shortcut stays inside the main directory.

[Published comparisons](../comparison/README.md) reuse these saved values;
no radius-two SRO-FA fit was added for the black extension. Missing historical
phase timings remain unavailable rather than inferred. No model checkpoint or
full timing record is claimed where the historical archive did not save one.

Back up the whole main directory: all files are included locally. Editing
through a link changes the local copy, not the original; keep it immutable.
Historical source paths are preserved as provenance, not a claim that every
archived script executes unchanged after relocation.