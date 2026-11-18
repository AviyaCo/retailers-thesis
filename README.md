# Retailers experiments: Neural CCG versus SRO-FA and SAA-kNN

This repository holds the code, configuration, final figures, statistical tests and timing reports of the
**two-stage multi-retailer inventory experiments** in the thesis of Aviya Strauss. It covers three
experiments:

- R=3 retailers
- R=10 retailers
- R=3 with a fixed shipping fee f=0.1

Each runs with N = 50/200/800 samples and ten data seeds, under two protocols:

- Protocol A: radius selected on a validation split.
- Protocol B: full-sample radius sweep.

| Method | What it is |
|---|---|
| Neural CCG | Feedforward neural decision rule with a differentiable feasibility projection (Moreau), trained by column-and-constraint generation against worst-case scenarios in ℓ∞ box uncertainty sets around each observation (sample robust optimization) |
| SRO-FA | Sample robust optimization with finite adaptability (Bertsimas, Shtern & Sturt), re-implemented in Python |
| SAA-kNN | Sample average approximation with k-nearest-neighbour recourse |
| Clairvoyant | Finite-sample reference values, not certified lower bounds |

**Start here**

1. [REPRODUCE.md](REPRODUCE.md): the environment, which script produced each result, exact commands for
   both protocols, figures, paired statistical tests, timing, and caveats.
2. Final figures: [comparison/paper_figures/published_straight_epsilon/](comparison/paper_figures/published_straight_epsilon/)
   (R=3, R=10) and [comparison/paper_figures/shipping_published_straight_epsilon/](comparison/paper_figures/shipping_published_straight_epsilon/).
3. Statistical tests: [statistical_tests_20261102/RESULTS.md](comparison/paper_figures/statistical_tests_20261102/RESULTS.md)
   (Protocol B) and [statistical_tests_validation_20261103/RESULTS.md](comparison/paper_figures/statistical_tests_validation_20261103/RESULTS.md)
   (Protocol A).

**Environment:** `bash setup_env.sh` installs the exact package versions ([requirements.txt](requirements.txt)).
Gurobi needs your own licence.

**Layout**

| Folder | Contents |
|---|---|
| `workspace/05_history/original_paths/` | The experiment campaigns that trained and evaluated the reported policies, each with its README, launch receipts and the source snapshot recorded at launch |
| `src/` | Shared evaluation (test scoring), data split and figure-validation modules called by those campaigns |
| `comparison/` | Test replay of the stopping checkpoints, final figure rendering, paired tests |
| `tests/` | Unit tests |
| `archive/hydro_rcr_rescaled/.moreau_diagnostic_deps/` | The exact Moreau 0.4.0 package used (Apache-2.0) |

**Not included:**

- Model checkpoints, per-run raw outputs and logs (hundreds of GB).
- The large figure-input files.
- The original Julia SRO-FA code by the SRO authors (no licence to redistribute).

Scripts contain the absolute paths of the original server (`/data/home/aviyac/...`). To rerun them unchanged,
clone into `/data/home/aviyac/retailers_experiment`.

The rest of this file is the historical organization log of the experiment folder, kept for provenance.

---

## Organization log

> **18 November 2026: one working version.** The code that produced the results is the working code: the
> historical campaign scripts under `workspace/05_history/original_paths/`, plus `src/evaluate.py`,
> `src/artifacts.py`, `src/config.py` and `src/plot.py`, which those pipelines call for test scoring, the
> Protocol A split and figure checks. The unused training half of the September rewrite (`run.py`,
> `src/ap.py`, `src/ccg.py`, `src/policy.py`, `src/select_lr.py`, `src/training.py`, the green/black
> `config.json` files and their tests) produced no result and was moved to a local archive (not published).
> Sections below that describe `run.py` or the shared trainer are historical.

**Organized VS Code view:** open [Retailers.code-workspace](../Retailers.code-workspace).
Its [start page](workspace/START_HERE.md) groups Code, Results, Figures, Data
generation, and History. Following the user's physical-consolidation approval,
all six original trees now live inside this collection. Old top-level folders
are gone; navigation links point internally. Frozen execution paths are supplied
by the explicit local compatibility runner, without rewriting research bytes.
See the [current consolidation record](.consolidation_20261005/README.md).

**Start here, not in the dated historical campaign folders.** Scope: no-shipping
R3/R10, total sample sizes 50/200/800, the existing 80/20 split and seed protocol.
Shipping is a different formulation and is preserved separately under
[shipping/](shipping/README.md), including its historical Clarabel, green,
SRO-FA and conditional-reference evidence. No trained shipping black comparator
exists in those records; the no-shipping black runs are not relabelled.

## Working-folder cleanup — 5 October 2026

Removed **14 obsolete working relaunch/boost/recovery launchers**: four from
the original no-shipping R10 folders and ten from the original shipping folders.
Their byte-identical copies remain in [archive/retailers_problem/](archive/retailers_problem/)
and [archive/retailers_with_shipping/](archive/retailers_with_shipping/).
No results, checkpoints, logs, data generators, training workers, plotting code,
or archived evidence were deleted or modified. No experiment was rerun.

Before deletion, all 14 original/archive pairs matched; 13,809 inspected
manifest/provenance/source/summary JSON records contained no exact original-path
references to these launchers. The retained 19,322 no-shipping and 6,534 shipping
files matched their pre-deletion inventory, size, modification time and mode
fingerprints afterward. This was a metadata preservation check, not a fresh
full-content hash of all results. Removed working bytes: 27,452; this is clutter
reduction, not substantial disk recovery.

The initial cleanup preserved all six top-level folders. The user subsequently
approved physical consolidation: they were renamed into
[workspace/05_history/original_paths/](workspace/05_history/original_paths/),
including the newer October 5 shipping work. A tested local filesystem namespace
preserves original resolved-path manifest keys for historical execution. Required
diagnostic/continuation modules remain present. Only exact duplicate archived
bytes are eligible for removal; unique history is retained.

## All methods and results in one place

**Back up the entire collection, including workspace and hidden files.** The
[workspace](workspace/START_HERE.md) now owns the current experiment trees; do
not exclude it. [archive/](archive/) retains historical paths, with verified
duplicate subtrees linked to their canonical current files. It is no longer an
independent second copy. The historical September verification figures below
describe that earlier copy operation, not the post-consolidation storage layout.
See [BACKUP.md](BACKUP.md) for current backup scope.

| Section | Contents |
|---|---|
| [sro_fa/](sro_fa/README.md) | Historical SRO-FA sweep scripts, saved results and available logs for R3/R10, N50/N200/N800 |
| [scs/](scs/README.md) | Historical width-16 SCS nominal and robust sweep scripts, checkpoints, pools, results and logs for all six R/N configurations |
| [green/runs/](green/runs/) | Original N50/R3 history, R3/N200/N800 campaign and R10 campaign, including sources and results |
| [black/runs/](black/runs/) | Original N50/R3 history and completed R3/R10 continuation archive, including sources and results |
| [comparison/black_green_figures/](comparison/black_green_figures/) | Historical minimum-AP R3/R10 figures, plot data, fixed-radius comparisons and runtime report |
| [comparison/](comparison/README.md) | Complete black/green evaluation workflow and original green comparison archives |
| [historical_support/moreau_R3_N50/](historical_support/moreau_R3_N50/) | Shared historical Moreau policy/AP/CCG and comparison support used by the frozen campaigns |

The original verified archive contained **1,205,012 files / 237.04 GB (220.76 GiB)**.
On 30 September, shipping and the remaining uncovered retailers files were
added: all six original retailers roots now have local copies, including
alternative-width evidence and previously uncovered plots. See
[BACKUP.md](BACKUP.md) for the separate verification records and current scope.
No original file was deleted or edited. N50/R3 appears under both colours but
is stored once locally. Editing through a shortcut now edits the local copy,
not the external original; keep archived bytes unchanged. Complete archives
retain diagnostics and failed attempts. Python/system packages and a Gurobi
license must be supplied separately when restoring an execution environment.

## The code you need to understand

Since 18 November 2026, only the evaluation/figure modules below remain in `src/`. The training CLI rows are
archived and were never used for a result; [REPRODUCE.md](REPRODUCE.md) lists the scripts that trained the
reported policies.

| Responsibility | Single owner |
|---|---|
| Commands and phase connections | [run.py](run.py) |
| Dimensions, seeds, tolerances, data and split | [src/config.py](src/config.py) |
| Policy network, native projection, cost, SAA initialization | [src/policy.py](src/policy.py) |
| Training: nominal objective or MP finite-pool objective | [src/training.py](src/training.py) |
| Three nominal LR arms, training replay, then validation selection | [src/select_lr.py](src/select_lr.py) |
| Fixed-policy adversarial problem | [src/ap.py](src/ap.py) |
| Pools, MP/AP loop, stopping, return choice, native continuation | [src/ccg.py](src/ccg.py) |
| Strict test/own-pool cost evaluation | [src/evaluate.py](src/evaluate.py) |
| Figures from supplied statistics | [src/plot.py](src/plot.py) |
| JSON, hashes, source/environment identity, timings | [src/artifacts.py](src/artifacts.py) |

There is **one trainer and one CCG loop**, not another copy for each colour,
sample size, radius or dimension. The MP is a training call with a finite-pool
objective; it does not need a second optimizer implementation. Python test files
are correctness checks, **not additional experiment stages**.

- [green/config.json](green/config.json): green stopping configuration.
- [black/config.json](black/config.json): black stopping configuration.
- [green/README.md](green/README.md) and [black/README.md](black/README.md): runs and historical navigation.
- [comparison/README.md](comparison/README.md): shared figure/evaluation location.

The runner has `nominal`, `ccg`, `evaluate`, `plot`, and `check` subcommands.
R, Ntotal, seed, radius, method and output are explicit. Numerical commands
require a tmux session; stdout/stderr must be retained. No subcommand launches
a sweep, retries a failure, invents test data or retrains a missing nominal model.
Method JSON files constrain the shared protocol; they are not arbitrary override
files. No CLI invocation is needed to inspect this directory.

## What changed, and what did not

Historical campaigns accumulated immutable versions, dimension adapters,
one-off diagnostics, launchers and comparison scripts in the same visible
namespace. Some separation was needed for reproducibility, but duplicating the
working pipeline was not a good permanent architecture.

This directory contains standalone shared implementations using NumPy, SciPy,
PyTorch, Moreau and Gurobi. It does **not** execute historical source via AST
extraction or import dated campaign controllers. Moreau 0.4.0 remains the
installed dependency; native projection is not replaced with a hand-written
solver. The historical discarded CVXPY-layer construction is temporarily
retained in model construction for initialization parity, not for inference.

**Migration status: implemented, not production-certified.** Unit tests and
source review are not evidence of numerical equivalence to every historical
run. No new training or held-out experiment is authorized or launched by this
reorganization. Actual old/new numerical parity remains required before using
this as a replacement production backend. Existing results are not relabelled
as outputs of this implementation.

The native continuation interface accepts only its own hash-bound assessed-stop
v3 format. Both colours return the **stopping checkpoint**, after validating
every assessment in the inclusive executed prefix, even when an earlier AP
mean is lower or tied. Candidate zero is eligible, requiring zero MPs when it
passes the method's stopping rule. Native CCG summaries explicitly set
`protocol.ccg_selection` to this return rule, consistently with the common
configuration, both method JSON files, and the CLI `check` output.

Continuation retains the method governing each inherited prefix segment, loads
the last assessed stop (not an earlier selected checkpoint), and includes
cached MPs in the forty-MP budget. It rejects old formats, including native v2;
old min-AP outputs are not silently relabelled. Capped black selection remains
explicit opt-in and returns fully assessed candidate 40 as **nonconverged**,
never an earlier AP minimum. Green still publishes only on convergence.

The generic plot renderer is not a substitute for the historical comparison's
570-policy integrity checks. A supplied JSON hash establishes identity, not
scientific validity. The historical comparison remains its own frozen workflow.
Its black overlays completed and passed verification on 28 September; see
[comparison/README.md](comparison/README.md) for both figures and measured runtimes.

## Historical evidence remains immutable

Original source, checkpoint, pool, log and manifest bytes remain unchanged.
The October 5 approved relocation changed physical ownership, not historical
provenance; duplicate copies are replaced only after fresh equality checks.
Absolute paths and source hashes remain recorded. [archive_paths.py](archive_paths.py)
resolves old absolute paths to byte-identical local copies and can return a
labelled, in-memory JSON view with local paths. Missing copies never fall back
to originals. Current shared code uses bundled Moreau by default. Historical
scripts retain original bytes. The explicit compatibility runner preserves their
historical paths; verified source/import checks are not full numerical
reproduction or certification of every historical entry point.

Each colour's `runs` directory now contains labelled links to historical
campaigns. Future parity-verified outputs must use a new, distinct run directory,
not write into those links. Paired figures and evaluation records are available
under `comparison`; no historical output is relabelled as a new shared-code run.
Historical min-AP selections, protocols and navigation-linked archive bytes
remain immutable. Terminal-only historical re-evaluation is a separate comparison.
The [stopping-checkpoint replay](comparison/stopping_checkpoint_20260928/README.md)
completed on 28 September: all 1,110 fresh evaluations passed, retaining all
11 nonconverged black caps. Both colours now use stopping checkpoints in the
[current R3 figure](comparison/stopping_checkpoint_20260928/production_v1/published/comparison_R3_stopping.png)
and [current R10 figure](comparison/stopping_checkpoint_20260928/production_v1/published/comparison_R10_stopping.png).
Whole replay/verification/publication time was 858.768170 seconds (14.312803
minutes), with no retraining. See [results and runtime details](comparison/README.md).

## Timing and validation

Per-seed training, nominal initialization, LR validation, each AP/MP iteration,
evaluation and total invocation timings remain separate. Resumed-prefix timing
is not current computation time; concurrent model durations are not sweep elapsed.
Unknown historical durations remain unavailable rather than estimated.

| Category | Reorganization measurement |
|---|---|
| Nominal neural training | Not run |
| Robust neural/CCG | Not run |
| SRO-FA | Not run; historical comparator only |
| Test/validation/CV experiment | Not run |
| Unit checks after path changes | 61 passed in 4.710 s; synthetic fixtures are not research results |
| Stopping-checkpoint v3 unit checks | Focused: 16 passed in 32.345 s; full suite: 63 passed in 31.940 s (28 September 2026) |

The v3 checks used the configured `ap_env` Python 3.11 interpreter with `-B`
on Linux. AP/MP controller fixtures are mocked, not numerical certification;
no training or OOS solver sweep was run. Both suites passed without failures.
Moreau emitted its optional CUDA-installation advisory; CLI rejection messages
were expected assertions in passing negative tests.

The first black plotting attempt passed its 57-test/six-probe preflight
(1,852.601 seconds whole-child), then stopped at the resource gate before any
policy worker: 196.909 seconds, insufficient reserved capacity for 16 workers.
The unchanged September 28 retry completed all 570 evaluations and published
both overlays in 1,561.301943 seconds (26.021699 minutes), with 16 single-thread
workers. These results belong to the frozen historical comparison, not to this
new shared implementation. No solver settings or resource guards were relaxed.