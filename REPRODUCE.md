# Reproducing the retailer results

This file explains how to regenerate the retailer results of the thesis and manuscript from the code in this
repository ([AviyaCo/retailers-thesis](https://github.com/AviyaCo/retailers-thesis)). It covers both protocols for
all three experiments (R=3, R=10, R=3 with fixed shipping fee f=0.1), N = 50/200/800, seeds 0–9:

- Neural CCG (Moreau "green" policy),
- SRO-FA,
- SAA-kNN,
- the clairvoyant ("lower bound") references,
- figures, paired statistical tests and timing reports.

## 1. What the repository contains, and what it does not

| In this repository | Not included (kept on the original server) |
|---|---|
| All source code (`.py`, `.sh`), including each campaign's launch-time source snapshots and tests | Model checkpoints (`.pth`), pools, raw evaluation arrays (`.npy/.npz`) |
| READMEs, STATUS/RESULTS reports, configs, launch/invocation receipts | Per-job JSON records (`jobs/`, `ccg_results/`), logs |
| Final paper figures (PNG/PDF) | The large figure inputs `plot_data_R3/R10.json` and the shipping `plot_data.json` (98–146 MB each) |
| Small final-result JSONs: Protocol A selections and per-seed test means, full-N plot data, paired-test outputs, runtime reports | The 810 full-N training summaries (12 GB) |
| `requirements.txt`, `setup_env.sh`, the exact Moreau 0.4.0 package | Gurobi licence (install your own) |
|  | The SRO authors' original Julia SRO-FA code (no licence to redistribute); unpublished manuscript drafts |

The small final-result JSONs are **reproduction targets**: compare a rerun with them.
Re-rendering the figures or rerunning the paired tests needs the large inputs listed above. The scripts verify
their SHA-256. Without them, regenerate the inputs by rerunning the pipeline (sections 4–6).

## 2. Environment

One command recreates everything except the Gurobi license (needs `python3.11`, `tmux`, internet):

```sh
cd /data/home/aviyac/retailers_experiment
bash setup_env.sh                      # creates /data/home/aviyac/ap_env (or: bash setup_env.sh /other/path)
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

`setup_env.sh` creates the virtual environment and installs [requirements.txt](requirements.txt). That file is
the exact `pip freeze` of the environment that produced every result: Python 3.11.13, Linux x86_64, numpy 2.4.2,
scipy 1.17.1, torch 2.10.0 (CUDA 12.8 build; all numerics ran on CPU), gurobipy 13.0.1, cvxpy 1.8.1,
cvxpylayers 1.0.3, matplotlib 3.10.8, scikit-learn 1.8.0. The script then runs the unit tests.

- **Moreau 0.4.0**, the differentiable QP projection layer, is loaded only from
  `archive/hydro_rcr_rescaled/.moreau_diagnostic_deps` (checked in `src/policy.py`). That folder is saved in Git.
  If it is missing, `setup_env.sh` reinstalls it from PyPI, and a PyPI rebuild was checked to be byte-identical
  (2026-11-18).
- **Gurobi 13** needs a valid license (`GRB_LICENSE_FILE`). It is used for the AP, the evaluation QPs, SRO-FA and the
  references. The license is never committed.
- The path-compatibility runner below also needs `bwrap` (Bubblewrap).

Clone the repository to `/data/home/aviyac/retailers_experiment`, because the historical scripts and their
manifests contain absolute paths:

```sh
git clone https://github.com/AviyaCo/retailers-thesis.git /data/home/aviyac/retailers_experiment
```

Each worker must stay single-threaded:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Each launcher already sets the Torch and Gurobi thread counts.

### Running historical campaign code

The campaigns were written for their original paths (e.g.
`/data/home/aviyac/retailers_moreau_training_selected/...`). They now live under
[workspace/05_history/original_paths/](workspace/05_history/original_paths/). Run
them through the compatibility runner, which maps the old paths read-only and
makes only one fresh output folder writable:

```sh
cd /data/home/aviyac/retailers_experiment
/data/home/aviyac/ap_env/bin/python -B .consolidation_20261005/run_legacy.py \
    --cwd <campaign folder> --write <new empty output folder> -- \
    /data/home/aviyac/ap_env/bin/python -B -u <script.py> <arguments>
```

Start `tmux` first, then run this command inside the pane; an existing tmux server
does not inherit the mapping. Never write into existing result folders. The
exact original command of each campaign is in its `*.invocation.json` /
`*.launch.json` file next to it.

## 3. Shared setup

- **Data:** `src/config.py::split_data`. Data seed s sets the global NumPy seed;
  ξ1, ξ2 are truncated normals. Protocol A permutes rows with `default_rng(100000+s)` and
  keeps the first 4N/5 for training (40/160/640) and the last N/5 for validation (10/40/160).
- **Test set:** 10,000 scenarios from seed 9999, common to all methods. It was inspected in
  earlier exploratory sweeps, so treat it as a reused held-out set.
- **Neural CCG radii:** 0, .001, .003, .01, .03, .2, .4, .7, 1.5; learning rates .005/.01/.05.
- **Stopping rule:** first candidate with relative AP–pool gap < 1%, cap 40 MPs. The method returns the
  stopping checkpoint with its own pool.

The campaigns below hold the code that produced the paper numbers. They call `src/evaluate.py` (test scoring),
`src/config.py` (Protocol A split) and `src/plot.py` (figure checks) from this folder.

## 4. Protocol A — validation-selected radius (paper Table)

All paths below are relative to
`workspace/05_history/original_paths/retailers_moreau_training_selected/`.

| Step | Method | Folder / entry point |
|---|---|---|
| A1 | Neural CCG, R3 N50 | `N50_R3_train40_val10/` (`launch.sh`, `ccg_training.py`) |
| A1 | Neural CCG, R3 N200/N800 | `green_R3_N200_N800_20260923/run_campaign.py` (see its README/STARTUP) |
| A1 | Neural CCG, R10 | `green_R10_N50_N200_N800_20260924/` |
| A1 | Neural CCG, shipping | `shipping_nominal_R3_N50_N200_N800_20260924/` → `shipping_green_launch_20260925/` → `shipping_live_green_20260926/` → `shipping_advisory_continue_20260928/` (5 advisory policies) |
| A2 | Stopping-checkpoint replay on the test set, R3/R10 | [comparison/stopping_checkpoint_20260928/](comparison/stopping_checkpoint_20260928/README.md): `launch.py preflight …`, then `launch.py production …` |
| A2 | Stopping-checkpoint replay, shipping | `shipping_stopping_20261005/` |
| A3 | Radius selection on validation, R3/R10 | `green_validation_R3_R10_20261015/run_validation.py` (command in its README) |
| A3 | Radius selection, shipping | `shipping_validation_qpositive_20261006/launch.py` |
| A4 | SRO-FA on the same split | `sro_nn_split_20261015/run.py --output runs/<new> --workers 28` (resumable) |
| — | SAA-kNN | Same policy in both protocols; see section 6 |

Selections are sealed (`selections.json`, `selection_seal.json`) before any test cost is read.
Targets in Git:

- `green_validation_R3_R10_20261015/production_v1/results.json` (+ `RESULTS.md`)
- `shipping_validation_qpositive_20261006/production_v1/selected_test_results.json`
- `sro_nn_split_20261015/runs/production_v1/test_results.json`

## 5. Protocol B — full-sample radius sweep (paper figures)

| Method | Entry point |
|---|---|
| Neural CCG, all 3 experiments | `green_fullN_cv_20261011/runner.py --output runs/<new> --workers 16`. Five-fold LR CV, full-N refit, 810 CCG configurations. Resume with `green_fullN_cv_20261011_resume/resume.py`; the logged single retry is `rerun_with_retry.py`. See README, STATUS, VALIDATION. |
| SRO-FA, R3 | `../retailers_problem/R3/N{50,200,800}/sro_fa/sro_fa_sweep.py` |
| SRO-FA, R10 | `../retailers_problem/R10/sro_fa/sro_fa_sweep.py` |
| SRO-FA, shipping | `../retailers_with_shipping/N{50,200,800}/sro_fa/sro_fa_shipping_sweep.py` (solver-status audit: `../retailers_with_shipping/run_sro_fa_solver_status_audit.py`) |
| SRO-FA + LDR (R3, supplementary) | `sro_nn_split_20261015/run_full_n.py` |

Targets in Git: `green_fullN_cv_20261011/runs/production_v1/published/{R3,R10,shipping_R3}/plot_data.json`
and `training_seal.json`, `RESULTS.md`, `runtime_report.json`.

## 6. SAA-kNN and clairvoyant references

Paths relative to `workspace/05_history/original_paths/`. These values are saved
historical comparators that the figure scripts reuse unchanged. Their numbers are
in the large local `plot_data` files.

- **SAA-kNN** (neighbor-level leave-one-out choice of k), implemented in the historical evaluation sweeps:
  - R3: `retailers_problem/R3/N*/rho_sweep_net16_v2/eval_sweep_net16_v2*_10k.py`
  - shipping: `retailers_with_shipping/N*/rho_sweep/eval_sweep_shipping_10k.py`
  - comparison check: `retailers_problem/test_neural_vs_knn.py`
  - R10: `retailers_problem/R10/N*/rho_sweep_net16_v2/eval_sweep_net16_v2_N*_R10*.py`; the joined
    comparator evaluation is in `retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/`
    (`evaluation.py`, `run_comparison.py`).

  These scripts were traced but not rerun when the repository was prepared. Before a rerun, confirm that the
  regenerated per-seed values equal the figure's saved values.
- **Conditional clairvoyant references (one million scenario pairs, seed 42):**
  - R3 LP 19.395: `retailers_problem/R3/saa_cond_lb.py` / `saa_cond_lb_large.py`, scored by `eval_cond_lb_seed9999.py`
  - R10 LP 58.184: `retailers_problem/R10/saa_cond_lb_1m.py` / `saa_cond_lb_large.py`
  - shipping MIP (incumbent 19.568, bound 19.488): `retailers_with_shipping/N50/saa_cond_mip_1m_shipping_exact.py`
- **Empirical clairvoyant on the 10,000 test scenarios:**
  - R3 19.385: `retailers_problem/R3/N50/rho_sweep_net16_v2/compute_lb_10k.py`
  - R10 57.870: `green_R10_N50_N200_N800_20260924/comparison_20260925/oracle.py`

None of these are certified population lower bounds.

## 7. Figures

From this folder (`retailers_experiment`):

```sh
/data/home/aviyac/ap_env/bin/python -m unittest comparison.paper_figures.test_render
/data/home/aviyac/ap_env/bin/python -m comparison.paper_figures.render_epsilon        # final R3, R10, shipping figures
/data/home/aviyac/ap_env/bin/python -m comparison.paper_figures.best_radius_shipping  # hindsight best radius, shipping
```

`render_epsilon.py` reuses `render.py` / `render_shipping.py` / `full_n.py`. It refuses to overwrite an existing
output folder and verifies the SHA-256 of every input. Final figures:
[published_straight_epsilon/](comparison/paper_figures/published_straight_epsilon/) and
[shipping_published_straight_epsilon/](comparison/paper_figures/shipping_published_straight_epsilon/)
(17 × 5.5 cm; bands are seed 20/80 percentiles).

## 8. Statistical tests

Paired across the ten seeds (same training data and test set): two-sided paired t-test, exact Wilcoxon
signed-rank, Holm correction within each family.

```sh
# Protocol B: matched radius and each method's hindsight-best radius; also R=10 adversary timing
/data/home/aviyac/ap_env/bin/python comparison/paper_figures/paired_tests.py
# Protocol A: NN vs SRO-FA at validation-selected radii
/data/home/aviyac/ap_env/bin/python comparison/paper_figures/statistical_tests_validation_20261103/paired_tests_validation.py
```

Saved outputs:

- [statistical_tests_20261102/RESULTS.md](comparison/paper_figures/statistical_tests_20261102/RESULTS.md):
  `paired_tests.json`, `best_radius_hindsight*.json`, `R10_adversary_timing_fullN.json`
- [statistical_tests_validation_20261103/RESULTS.md](comparison/paper_figures/statistical_tests_validation_20261103/RESULTS.md)

## 9. Timing

Each campaign records monotonic wall time per model and phase (nominal, CV, each AP/MP, evaluation) and
whole-sweep elapsed time separately; concurrent job times are never summed. Saved reports:

- Protocol B NN: `green_fullN_cv_20261011/runs/production_v1/runtime_report.json` (`RESULTS.md`)
- Protocol A NN replay: [comparison/stopping_checkpoint_20260928/production_v1/published/RESULTS.md](comparison/stopping_checkpoint_20260928/production_v1/published/RESULTS.md)
- Shipping: `shipping_stopping_20261005/production_v2/published/runtime_report.json`,
  `shipping_validation_qpositive_20261006/production_v1/runtime_report.json`, `shipping_runtime_comparison_20261005/RESULTS.md`
- SRO-FA split: `sro_nn_split_20261015/README.md` and `runs/production_v1/RESULTS.md`
- Earlier timing comparison: `TIMING_COMPARISON_20260924.md`
- R=10 adversary timing: `comparison/paper_figures/statistical_tests_20261102/R10_adversary_timing_fullN.json`

Timings were measured on gold-srv2 (72 logical CPUs) under varying load. A rerun elsewhere gives different times.

## 10. Code tests

```sh
cd /data/home/aviyac/retailers_experiment
/data/home/aviyac/ap_env/bin/python -B -m unittest discover -s tests          # src/ evaluation, split, plot
/data/home/aviyac/ap_env/bin/python -B -m unittest comparison.paper_figures.test_render
```

Each campaign folder also has its own `test_*.py` (run from that folder through the compatibility runner).
These are correctness checks, not experiment stages.

## 11. Caveats

- Numerical reproduction was not re-run as part of this backup. Exact equality also depends on package
  versions, Gurobi version and CPU. Compare reruns with the saved targets, allowing solver-tolerance differences.
- Shipping: advisory AP activation/duplicate checks (approved), one logged Gurobi retry (Protocol B, N200 s9 ρ1.5),
  and historical SRO-FA time-limited incumbents. See the manuscript and each campaign README.

## Updating this repository (after every Retailers change)

On the original server, from the home folder:

```sh
cd /data/home/aviyac
ap_env/bin/python -B retailers_experiment/.archive_migration/git_code_backup.py --dry-run   # review selection
ap_env/bin/python -B retailers_experiment/.archive_migration/git_code_backup.py --commit    # local commit
git push retailers-thesis retailers-thesis-main:main                                        # from a VS Code terminal
```

The script uses a private Git index in the home repository. The checked-out `WIP` branch and its index are
untouched, and `WIP` (which contains environments and a licence file) is never pushed. The earlier private backup
branch `backup/retailers-code-plots-20260930` of `AviyaCo/gold-srv2-homedir` is kept unchanged as history.

## One working version of the code

The working code is the code that produced the results (decision of 18 November 2026):

- the campaign scripts listed in sections 4–8 (under `workspace/05_history/original_paths/` and `comparison/`);
- the modules they call in `src/`: `evaluate.py`, `artifacts.py`, `config.py`, `plot.py`, with tests in `tests/`.

Edit these in place and never create a parallel copy or "v2". The per-launch source snapshots (`frozen/`,
`inputs/`, `source_snapshot/`, per-run copies) record exactly which code made which result: never edit them.
The unused training CLI of the September rewrite (`run.py` and `src/ap|ccg|policy|select_lr|training.py`)
produced no result. It was moved to a local archive and is not part of this repository.

Known pre-existing test failure: `comparison/stopping_checkpoint_20260928/test_selection.py::BoundedArchiveTest`
expects checkpoints under `archive/`. Since the 5 October consolidation they live under `workspace/`. The test
checks a path location only, not a numerical result.
