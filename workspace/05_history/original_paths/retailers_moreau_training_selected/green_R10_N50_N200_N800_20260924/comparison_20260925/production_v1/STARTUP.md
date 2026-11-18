# R10 comparison production launch

The requested three-panel comparison is running in named tmux session
`r10_plot_20260925`, from the comparison_20260925 working directory.

Scope: 270 fresh green and 540 fresh SCS strict OOS/own-finite-pool IS replays,
10,000 shared R10 pairs, seed 9999, plus one freshly reconstructed common
test-fitted empirical clairvoyant LP. Cached SAA-kNN and SRO-FA remain descriptive
historical references. No training, policy replacement or test-based selection.

The common oracle is ONE LP, not eight missing independent seed evaluations.
Historical caches and training files remain unchanged. Saved nonconverged SCS
checkpoints and raw pool residuals are retained as explicitly approved.

Preflight passed 73 tests, all 810 policy identities, 7,047 bound files, 24
optimal projection QPs and one optimal training-only oracle LP. Internal time
339.872626 s; whole child 342.996289 s (5.717 min). Production time is pending.

## Exact invocation

```bash
/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/run_comparison.py --output /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1 --preflight /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/preflight_v1/summary.json --preflight-sha256 efdd81596a4c851750faa1049868daa251e0e5101e9749575c06829a8d5acd40
```

Persistent log: [../production_v1.launch/child.log](../production_v1.launch/child.log).
Exact launch metadata: [../production_v1.launch/invocation.json](../production_v1.launch/invocation.json).
Exit and whole-child runtime are written there at termination. Do not infer
success solely from a disappearing tmux session. The summary records actual
worker count and phase timing; at most 32 single-threaded evaluation workers.

```bash
tmux attach-session -t r10_plot_20260925
tmux has-session -t '=r10_plot_20260925'
cat ../production_v1.launch/child.exit
```

PNG and vector PDF are published under the production published subdirectory
only after complete evaluations, journal verification and source checks pass.
This startup report is outside the frozen computational source set.