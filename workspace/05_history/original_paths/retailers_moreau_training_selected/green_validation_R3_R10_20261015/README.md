# GREEN STOP (no shipping) validation-selected radius, R3/R10 — 15 Oct 2026

> **18 November 2026 — radius selection rule changed.** Validation-based radius selection now uses the shared rule "largest radius within 1e-6 of the minimum validation score" (`src/config.py`), replacing "exact tie -> smaller radius". The sealed outputs of this run record the old rule and are unchanged; the current selections, which change no test cost, are in `comparison/paper_figures/radius_selection_20261118/`.

Analogue of `../shipping_validation_qpositive_20261006/` for the no-shipping
retailer cases. Evaluation of existing policies only: no retraining, no new
CCG, no solver substitution, no retry/fallback, no test-driven selection.

**Policies.** The 540 GREEN STOP checkpoints (R 3/10 × Ntotal 50/200/800 ×
seeds 0–9 × radii 0/.001/.003/.01/.03/.2/.4/.7/1.5) listed in the sealed
selection manifest of `comparison/stopping_checkpoint_20260928`
(`preflight_v1/selection_manifest.json`, bound via the sealed
`production_v1/execution_manifest.json`). Checkpoint and learned-state SHA256
are checked before replay.

**Validation rows.** The canonical split is rebuilt with `src.config.split_data`
(global NumPy seed s; truncnorm xi1/xi2; permutation
`default_rng(100000+s)`; first 4N/5 train, last N/5 held out: 10/40/160 rows).
Before use, the reconstructed training rows must equal each policy's saved
`training_data.json` (centers, sample IDs, raw-data SHA256, training-center
SHA256), and, where the campaign saved it, the full `split.json`. Mismatch stops
the run. **Caveat:** these held-out rows were already used for GREEN nominal LR
selection, so they are not an untouched validation set (same caveat as the
shipping run). The original campaigns also ran an in-campaign radius selection
with their own projection layer; it is not used here.

**Evaluator.** `src.evaluate.evaluate_policy` (reached through
`stopping_checkpoint_20260928/run.py`): float32 network targets, one fresh Gurobi
QP projection per row (1 thread), OPTIMAL required, float64 costs. This run
additionally requires physical residual ≤ 1e-6 (evaluator gate is 1e-4) and
re-reads every raw array and journal row.

**Selection.** Per (R, N, seed): minimum validation mean, exact ties → smaller
radius. All 540 arms must complete, otherwise no selection is made. The 60
choices are written to `selections.json` with a SHA256 seal before any test cost
is opened. Then the selected policies' test costs are looked up from the
existing seed-9999, 10,000-row STOP replay (`production_v1/jobs/<id>/oos`),
re-verified with `run.verify_replay` and against `plot_data_R{3,10}.json`. That
test set was historically inspected: results are descriptive.

**Execution.** `run_validation.py` is the single entry point; run in tmux:

```sh
cd /data/home/aviyac/retailers_experiment/workspace/05_history/original_paths/retailers_moreau_training_selected/green_validation_R3_R10_20261015
tmux new-session -d -s green_val_R3R10_20261015 "/data/home/aviyac/ap_env/bin/python -B run_validation.py --output production_v1 --workers 16 --session green_val_R3R10_20261015 --log $PWD/production_v1.log > production_v1.log 2>&1; echo \$? > production_v1.exit"
```

Spawned single-thread workers (Torch/BLAS/Gurobi = 1), one (R, N, seed) grid per
task, N800 grids first. Timing (time.perf_counter, seconds, UTC endpoints): per
arm (load + replay + raw verification), per grid, per phase and overall elapsed
(`summary.json`). Concurrent worker times are never summed. Test-evaluation and
training times are historical (not rerun) and labelled as such.

Outputs (`production_v1/`): `inputs.json` (sealed), `validation/<grid>/`
(raw arrays, solve journals, sealed `grid.json`), `selections.json` (sealed
before test lookup), `results.json`, `RESULTS.md`, `summary.json`.
