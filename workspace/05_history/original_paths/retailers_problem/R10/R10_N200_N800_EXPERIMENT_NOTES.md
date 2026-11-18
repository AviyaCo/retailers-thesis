# R10 N200/N800 retailer experiment notes

Date: 2026-07-14

## Scope

Run the net16-v2 SAA-initialized retailer experiments for `R=10` at `N=200` and `N=800` using the truncnorm data generator.

## Existing R10 N200 status

Directory: `/data/home/aviyac/retailers_problem/R10/N200`

The existing `N=200` R10 run is complete:

- Nominal checkpoints: 10/10 in `N200/saainit_net16_v2/seed_*_nominal.pth`.
- Rho sweep summaries: 170/170 completed in `N200/rho_sweep_net16_v2/rho_*/seed_*_summary.json`.
- Rho grid: `0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400, 0.450, 0.500, 0.600, 0.750, 0.800, 1.000, 1.500`.
- Seeds: `0..9`.
- Stored nominal LR: `0.005` for all seeds.

The old tmux sessions `r10n200v2` and `r10sweep200v2` remain attached as completed panes from the previous run.

## New R10 N800 setup

Directory: `/data/home/aviyac/retailers_problem/R10/N800`

New code files:

- `saainit_net16_v2/run_all_net16_v2_N800_R10.py`
- `rho_sweep_net16_v2/run_rho_seed_net16_v2_N800_R10.py`
- `rho_sweep_net16_v2/launch_net16_v2_N800_R10.sh`
- `rho_sweep_net16_v2/eval_sweep_net16_v2_N800_R10_10k.py`
- `rho_sweep_net16_v2/plot_N800_R10_net16v2.py`

Protocol:

- `R = 10`, `N_SAMPLES = 800`.
- LR grid: `[0.005, 0.01, 0.05, 0.1]`, matching the existing R10 runs.
- CCG cap: `15`, matching the existing R10 runs.
- Master-problem gradient retraining for N800 uses mini-batches of size `200`.
- Rho worker inherits the selected `lr_net` from the nominal checkpoint and records it as `lr_net_used`.
- Rho grid and seeds match N200.

Validation before launch:

- `python -m py_compile` passed for the new R10/N800 Python scripts.
- `thesis_modeling_notes.tex` was updated and compiled successfully to `/tmp/thesis_modeling_notes.pdf`.

## Current tmux launch

Launched at timestamp `20260714_135235`.

Nominal/SAA-init session:

```bash
tmux attach -t r10n800v2
```

Log:

```text
/data/home/aviyac/retailers_problem/R10/N800/saainit_net16_v2/run_log_20260714_135235.txt
```

Sweep watcher session:

```bash
tmux attach -t r10sweep800v2
```

Log:

```text
/data/home/aviyac/retailers_problem/R10/N800/rho_sweep_net16_v2/launch_log_20260714_135235.txt
```

The sweep watcher runs:

```bash
bash launch_net16_v2_N800_R10.sh 6
```

It waits for each nominal checkpoint and then launches the 17 rho jobs for that seed with at most 6 concurrent workers.

## Useful checks

```bash
tmux ls | grep -E 'r10n800v2|r10sweep800v2'

sed -n '1,80p' /data/home/aviyac/retailers_problem/R10/N800/saainit_net16_v2/run_log_20260714_135235.txt
sed -n '1,80p' /data/home/aviyac/retailers_problem/R10/N800/rho_sweep_net16_v2/launch_log_20260714_135235.txt

find /data/home/aviyac/retailers_problem/R10/N800/saainit_net16_v2 -name 'seed_*_nominal.pth' | wc -l
find /data/home/aviyac/retailers_problem/R10/N800/rho_sweep_net16_v2 -path '*/seed_*_summary.json' | wc -l
```
