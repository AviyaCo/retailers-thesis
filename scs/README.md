# Historical SCS runs and results

No-shipping, width-16 `net16_v2` evidence used by the existing comparisons.
Each configuration exposes its original nominal training and robust CCG sweep
directories, including scripts, checkpoints, saved pools, summaries, evaluation
results and available logs. No experiment was rerun or reclassified.

| R | N | Nominal training | Robust sweep and results |
|---|---|---|---|
| 3 | 50 | [nominal](R3/N50/nominal/) | [robust](R3/N50/robust/) |
| 3 | 200 | [nominal](R3/N200/nominal/) | [robust](R3/N200/robust/) |
| 3 | 800 | [nominal](R3/N800/nominal/) | [robust](R3/N800/robust/) |
| 10 | 50 | [nominal](R10/N50/nominal/) | [robust](R10/N50/robust/) |
| 10 | 200 | [nominal](R10/N200/nominal/) | [robust](R10/N200/robust/) |
| 10 | 800 | [nominal](R10/N800/nominal/) | [robust](R10/N800/robust/) |

`nominal` links to the corresponding original `saainit_net16_v2` directory;
`robust` links to `rho_sweep_net16_v2`, both under
the local archive's retailers-problem tree. Historical scripts are preserved
unchanged, including their recorded absolute paths; [the resolver](../archive_paths.py)
locates copied evidence without rewriting hashes. Archived scripts are not
certified portable executables.

The [comparison archive](../comparison/README.md) contains the later strict
Gurobi replay used in the published figures. Those evaluation records are
distinct from SCS training and older cached evaluation results. Existing
training-sample differences and nonconvergence qualifications are unchanged.
Missing historical runtimes remain unavailable, not newly estimated.

These internal links lead to independent copies under this backup's archive.
No external original directory is required to retain or browse the evidence.
Editing through a link changes the local copy; keep it immutable. Other widths
and shipping experiments are outside this collection.