# Retailers

**Public repository (since 18 November 2026):** [AviyaCo/retailers-thesis](https://github.com/AviyaCo/retailers-thesis).
After every Retailers change, publish it as described in [REPRODUCE.md](../REPRODUCE.md), "Updating this
repository". How to rerun everything and set up the environment: [REPRODUCE.md](../REPRODUCE.md).

Open [Retailers.code-workspace](../../Retailers.code-workspace) in VS Code using
**File → Open Workspace from File**. All six original Retailers trees now live
physically in this collection under History. The old top-level directories are
gone. Shortcuts expose the same files by purpose, not extra working copies.

| Section | Use it for |
|---|---|
| [01_code](01_code/) | Evaluation/figure modules used by the result pipelines (`shared` = `src/`), their tests, `requirements.txt` |
| [02_results](02_results/) | SRO-FA, SCS, Clarabel, Moreau, and LB/reference results |
| [03_figures](03_figures/) | Published figures, plot data, runtime reports, and plotting sources |
| [04_data_generation](04_data_generation/) | Original sampling/training scripts and benchmark generators |
| [05_history](05_history/) | Full archived evidence, original campaign folders, diagnostics, and backup records |

## FINAL paper figures (current, confirmed 2026-11-02)

These are the **final figures** for the paper. Use them, not the older figures below:

| Figure | PNG | PDF |
|---|---|---|
| R=3 | [comparison_R3.png](../comparison/paper_figures/published_straight_epsilon/comparison_R3.png) | [comparison_R3.pdf](../comparison/paper_figures/published_straight_epsilon/comparison_R3.pdf) |
| R=10 | [comparison_R10.png](../comparison/paper_figures/published_straight_epsilon/comparison_R10.png) | [comparison_R10.pdf](../comparison/paper_figures/published_straight_epsilon/comparison_R10.pdf) |
| Shipping | [comparison_shipping.png](../comparison/paper_figures/shipping_published_straight_epsilon/comparison_shipping.png) | [comparison_shipping.pdf](../comparison/paper_figures/shipping_published_straight_epsilon/comparison_shipping.pdf) |

Shortcuts: `03_figures/published/paper_final_R3_R10_straight_epsilon` and `03_figures/published/paper_final_shipping_straight_epsilon`.
Rendered 2026-11-02 by `comparison/paper_figures/render_epsilon.py`: identical to the 2026-10-26 render (`render.py`, `render_shipping.py`;
folders `published/`, `shipping_published/`, shortcuts `paper_final_R3_R10`, `paper_final_shipping`) except the x-axis label,
which is now the straight epsilon U+03F5 (supervisor request). Pixel diff confirms only the x-label strip changed. These are 17 x 5.5 cm figures
showing the blue NN policy (full-N Moreau stopping checkpoints), SRO-FA, SAA-kNN and the saved references, with the
x-axis epsilon_N. Display protocol, checksums and caveats: [paper_figures/README.md](../comparison/paper_figures/README.md)
and each folder's `provenance.json`. The `R3_R10_stopping` and `shipping_minAP_20260928` figures below are earlier,
superseded versions kept as history.

Statistical tests and R=10 adversary timing (2026-11-02): [statistical_tests_20261102/RESULTS.md](../comparison/paper_figures/statistical_tests_20261102/RESULTS.md),
produced by `comparison/paper_figures/paired_tests.py`. These are paired tests (between seeds) of the final NN policy against SRO-FA on
identical training data and test set; no training or evaluation was run.

Epsilon glyph (supervisor request 2026-11-02): DONE. matplotlib maps both `\epsilon` and `\varepsilon` to the curly
U+03B5, so the 2026-10-26 figures show a curly epsilon; the `*_straight_epsilon` folders use U+03F5 (straight). The old folders
are kept unchanged as the previous version.

## Which results to use

- **No shipping:** the published R3/R10 figures use stopping checkpoints for
  both Moreau colours. The original training summaries can still contain their
  historical minimum-AP selections; the stopping-checkpoint evaluation is a
  separate result, not a rewrite of that history.
- **Shipping:** the October 5 stopping-checkpoint replay is complete:
  [current PNG](02_results/Moreau/shipping_stopping_revision/production_v2/published/comparison_shipping.png)
  and [PDF](02_results/Moreau/shipping_stopping_revision/production_v2/published/comparison_shipping.pdf).
  All270 green policies were freshly replayed; historical Clarabel and the
  SRO-FA/kNN/reference curves are unchanged cached comparisons. The September28
  minimum-AP figure remains preserved as historical evidence. Corrected N800 Clarabel
  robust results have an explicit `N800_corrected_minibatch` shortcut.
  The October 5 stopping revision has its own current status under
  [02_results/Moreau/shipping_stopping_revision/STATUS.md](02_results/Moreau/shipping_stopping_revision/STATUS.md).
  There is no black shipping comparator in these records.
- **LB:** these are saved finite-sample reference computations, not guaranteed
  numerical population bounds. The shipping conditional MIP retains its
  incumbent, best bound, and nonzero solver-gap qualifications.

## Working here after physical consolidation

- Since 18 November 2026 the working code is the code that produced the results: the campaign scripts
  under History, plus `src/` evaluation/figure modules (shortcut `01_code/shared`). The unused training CLI
  was archived; see [REPRODUCE.md](../REPRODUCE.md) for how to rerun everything and set up the environment.
  Never create a second copy of a working script.
- For frozen historical code, use **Terminal → Run Task → Open historical
  Retailers shell**. It maps the physical collection to historical path names
  inside a local Bubblewrap mount namespace. Original manifest keys, sources
  and checksums are unchanged. No old host folders or numerical fallbacks are
  created. The [compatibility runner](../.consolidation_20261005/run_legacy.py)
  fails explicitly if a required directory or tool is missing.
- Historical trees are read-only by default in that shell. Approved runs need
  an explicit existing output subdirectory selected with `--write`; never reuse
  previous result directories. Ordinary subprocesses inherit the mapping. Start
  tmux outside the runner and invoke the runner inside its pane; external tmux
  servers/schedulers do not inherit a client's mount namespace.
- The two new October 5 folders are preserved. Archive-only launchers remain
  in the archive. Duplicate removal keeps the current physical owner and
  replaces **only freshly byte-verified archive duplicates** with internal
  relative links. Different/missing revisions are never overwritten or removed.
  See the [consolidation record](../.consolidation_20261005/README.md) for current
  progress, precise deletion counts and verification limits.
- The archive is consequently **not an independent second backup**. Back up
  the whole Retailers collection, not just its archive subtree. Preserve links
  without dereferencing them. Hydro's source collection and the existing Python
  environment/system tools remain dependencies of historical execution.
- Frozen historical/source/result aliases stay editor-read-only; this is not
  filesystem security. Shared code and tests remain editable. Search does not
  follow aliases; actual current experiment trees are under `05_history`.
  Data-generation entries remain the original scripts, not new generator APIs.
  No research training, evaluation, plotting or solver sweep was run here.

For protocol and backup details, see
[the experiment guide](../README.md), [comparison guide](../comparison/README.md),
[shipping guide](../shipping/README.md), [backup notes](../BACKUP.md), and
[Git backup scope](../GIT_BACKUP.md). Historical research documents retain their
original path strings; use the compatibility shell for those commands.