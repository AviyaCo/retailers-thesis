# Prelaunch review, 16 September 2026

User authorized training-best checkpoint rerun after the saved-objective audit. Fixed protocol is in [README.md](README.md); no expanded radius/data/model sweep or test evaluation.

The initial development test import caught a controller indentation error; fixed before experiments. Subsequently23 trainer/controller tests passed in11.845s. Independent review required explicit failed-future timing and failed-evaluation records, plus an all90-evaluation barrier before radius selection. These were implemented and covered by two campaign tests. Follow-up static review approved launch conditional on runtime preflight.

Final [preflight summary](preflight/20260916T080510.808940Z/summary.json) completed in17.821456s. All25 tests passed (unittest13.156s; subprocess wall16.177318s). Real native two-epoch train40 check took0.997484s; one radius0.2 sample AP took0.058194s and passed unchanged strict gates. This is a correctness smoke check, not a performance pilot or a source of model selection. No validation/test evaluation occurred. Per-phase raw timings and test log are retained. Editor diagnostics were clear.

Resource check:72 CPU-affinity slots, host load11.38/10.99/10.78, approximately496092352KiB available RAM. Planned10 nominal seed workers then16 CCG workers, single numerical/solver threads each; no GPU switch. CPU Moreau remains as in previous runs.

Final experimental validation is post-training hyperparameter selection, not independent generalization evidence. The nominal LR winner is validated before CCG; the CCG trainer itself never receives validation observations. Training checkpoint/scheduler/early-stop semantics match the verified SCS rule, but the40/10 data split and Moreau numerical backend are intentionally retained, so this is not a solver-only comparison with historical SCS train50 results.

No silent fallback, retraining retry, relaxed gate, partial-grid selection, or best-looking-checkpoint substitution is authorized. Runtime/termination outcomes will be reported honestly. Launch through `bash launch.sh` in named tmux session `retailers_training_selected_20260916`; scripts and this protocol are frozen in the run's provenance manifest.