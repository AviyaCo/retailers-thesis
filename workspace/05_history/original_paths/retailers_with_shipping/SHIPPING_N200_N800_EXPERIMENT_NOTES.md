# Shipping Retailers N200/N800 Experiment Notes

Date: 2026-07-14

## Goal

Run the fixed-shipping retailer experiment for `R=3` at `N=200` and `N=800`, matching the existing `N=50` shipping setup while keeping train/validation/test roles separated.

## Directories

- `retailers_with_shipping/N200/`
- `retailers_with_shipping/N800/`

Both were initialized from the existing `retailers_with_shipping/N50/` code, then patched so scripts read and write only their scale-specific folders. Copied N50 outputs were removed before launch.

## Pipeline

Each scale uses a two-stage launcher:

1. Train per-seed nominal checkpoints with LR cross-validation:
   - Script: `run_cv_nominal.py`
   - Seeds: `0,...,9`
   - Folds: `5`
   - LR grid: `[0.005, 0.01, 0.05]`
   - Outputs: `seed_{seed}_nominal.pth`, `seed_{seed}_summary.json`, `cv_lr_results.json`

2. After all nominal checkpoints are available, launch the rho sweep:
   - Script: `rho_sweep/launch_sweep.sh`
   - Worker: `rho_sweep/run_rho_seed.py`
   - Outputs: `rho_sweep/rho_{tag}/seed_{seed}_summary.json` and intermediate AP/MP files

Top-level launchers:

- `N200/launch_pipeline.sh [cv_parallel] [sweep_parallel]`
- `N800/launch_pipeline.sh [cv_parallel] [sweep_parallel]`

Current intended tmux launch commands:

```bash
tmux new-session -d -s shipN200 'cd /data/home/aviyac; log=retailers_with_shipping/N200/pipeline_$(date +%Y%m%d_%H%M%S).log; bash retailers_with_shipping/N200/launch_pipeline.sh 3 10 2>&1 | tee "$log"'
tmux new-session -d -s shipN800 'cd /data/home/aviyac; log=retailers_with_shipping/N800/pipeline_$(date +%Y%m%d_%H%M%S).log; bash retailers_with_shipping/N800/launch_pipeline.sh 2 6 2>&1 | tee "$log"'
```

## Training Protocol

### N=200

- Optimizer: Adam
- Training uses the full fold-training set or full seed-training set in each epoch.
- LR selection is per seed by 5-fold validation.

### N=800

- Optimizer: SGD
- Batch size: `200`
- Each epoch shuffles the training rows and steps through mini-batches of size 200.
- LR selection is per seed by 5-fold validation under the same SGD mini-batch protocol.

The N800 SGD/batch-200 choice is a methodological change relative to the copied N50/N200 Adam/full-batch script, intended to reduce the cost of each N800 training epoch. It changes the optimization algorithm, not the model class, data generation, folds, LR grid, SAA initialization, or test protocol.

## Runtime Logging

Nominal CV logs now include:

- per-training-call start line with optimizer, sample count, LR, and batch size where applicable
- epoch progress lines with elapsed time for verbose final training
- early-stop elapsed time
- training-call completion line with best loss, epochs run, and elapsed seconds
- fold-level runtime
- SAA-MILP runtime
- final-training runtime
- seed total runtime in seconds and minutes

Runtime fields are also saved in nominal checkpoint/summary metadata:

- `optimizer`
- `batch_size` for N800
- `seed_time_s`

## Logs

Top-level pipeline logs are timestamped:

- `retailers_with_shipping/N200/pipeline_YYYYMMDD_HHMMSS.log`
- `retailers_with_shipping/N800/pipeline_YYYYMMDD_HHMMSS.log`

Per-seed CV logs:

- `retailers_with_shipping/N200/cv_logs/seed_{seed}.log`
- `retailers_with_shipping/N800/cv_logs/seed_{seed}.log`

Sweep worker logs:

- `retailers_with_shipping/N200/rho_sweep/logs/`
- `retailers_with_shipping/N800/rho_sweep/logs/`

## Reproducibility Notes

- Training data are generated per seed using the same truncated-normal generator as the N50 shipping experiment.
- Test data remain separate from LR selection.
- The rho sweep starts only after nominal checkpoints are available.
- If optimizer settings change again, remove partial nominal checkpoints and restart from seed 0 so all seeds within a scale use the same protocol.

## Useful Checks

```bash
tmux ls | grep -E 'shipN200|shipN800'
tmux capture-pane -pt shipN200 -S -80
tmux capture-pane -pt shipN800 -S -80
ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E 'retailers_with_shipping/(N200|N800)/(launch_pipeline|run_cv_nominal|rho_sweep/launch_sweep|rho_sweep/run_rho_seed)' | grep -v grep
```
