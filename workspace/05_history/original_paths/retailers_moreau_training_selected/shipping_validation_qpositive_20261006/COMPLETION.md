# Completed — 6 October 2026

The authorized green/blue validation-only radius selection completed, exit 0,
07:16:58.967628–07:17:47.748479 UTC. No retraining or test-based selection.
All720 candidates,60 per-seed grids and58350 validation projections completed;
all projections accepted as OPTIMAL with physical residual at most1e-6.
Six two-row training-only probes and14 regression tests passed.

## Results

Both validation and test costs charge0.1 per raw q>0 shipment. Mean ± sample SD
over ten seeds; each seed chooses its own radius using validation only:

| Ntotal | Blue | Green |
|---:|---:|---:|
| 50 | 20.604174 ± 0.181066 | 20.817281 ± 0.444580 |
| 200 | 20.091235 ± 0.204745 | 20.342430 ± 0.247809 |
| 800 | 19.857999 ± 0.023775 | 19.967017 ± 0.117240 |

Green training40/160/640 and existing holdout10/40/160 (already LR-used),
blue training50/200/800 and independent validation13/50/200; different radius
grids retained. Historical10k seed9999 test, previously inspected; descriptive,
not a fresh confirmatory or matched-training-protocol comparison. Green's mixed
265+5 cohort and historical capped/nonconverged blue policies are not relabelled.

[Detailed results and per-seed radii](production_v1/RESULTS.md).
[Structured scores](production_v1/score_summary.json).
[Full selected-test provenance](production_v1/selected_test_results.json).
[Runtime report](production_v1/runtime_report.json).

## Timings and verification

| Component | Seconds | Minutes |
|---|---:|---:|
| Whole child, including namespace wrapper/imports | 48.780893 | 0.813015 |
| Runner | 46.009300 | 0.766822 |
| Tests subprocess | 2.586410 | 0.043107 |
| Input freeze/split verification | 15.808766 | 0.263479 |
| Training-only preflight | 0.666030 | 0.011101 |
| Parallel validation | 9.087025 | 0.151450 |
| Raw validation verification/selection | 1.515873 | 0.025265 |
| Cached test lookup/raw verification | 10.580171 | 0.176336 |
| Final input integrity | 5.738568 | 0.095643 |
| Reporting | 0.017949 | 0.000299 |

Eight workers, one Torch/Gurobi thread each, Linux gold-srv2,72 affinity CPUs,
Python3.11.13/Torch2.10.0/NumPy2.4.2/Gurobi13.0.1. Nested phases are not additive;
concurrent job walls are not sweep elapsed. Nominal/robust CCG/SRO-FA/CV were
not rerun; saved prior training/evaluation timings are retained separately.
Cached test lookup is not fresh model evaluation. Per-configuration counts,
mean/median/min/max and all solver iteration counts are retained in raw records.

The pre-test selection SHA256 is
181ca2d475fe51f22417ee48ed09ae442d8faa02c5742588cbf5c51f89b7e2dd.
Independent post-run checks verified all60 validation argmins and post-seal
test-lookup timestamps, all720 arm-result hashes,5024 source/input bindings,
all60 selected test identities and five artifact hashes. All passed.
Historical sources/results and newly frozen experiment sources are unchanged.

## Execution record

Named tmux session: shipping_val_qpos_20261006 (finished).
Working directory: this physical experiment directory under
retailers_experiment/workspace/05_history/original_paths/retailers_moreau_training_selected.
Exact launcher from that directory:
`/data/home/aviyac/ap_env/bin/python -B launch.py --output production_v1 --session shipping_val_qpos_20261006`.
The exact wrapper/child argv and absolute cwd are preserved in
[launch receipt](production_v1.launch.json) and
[completed invocation receipt](production_v1.invocation.json).
Persistent stdout/stderr: [production_v1.log](production_v1.log).
During execution, attach with `tmux attach -t shipping_val_qpos_20261006`;
check session with `tmux has-session -t shipping_val_qpos_20261006`.
The completed receipt is now authoritative; do not relaunch into the existing
output or edit the frozen sources/results. Compatibility mapping is required
for replay because historical manifest paths remain immutable.