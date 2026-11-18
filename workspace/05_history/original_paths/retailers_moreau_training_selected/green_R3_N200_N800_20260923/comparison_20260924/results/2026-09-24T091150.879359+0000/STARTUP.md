# Matched comparison launch

Started2026-09-24 09:11:39.487634UTC, tmux
`retailers_comparison_R3_20260924`,32 spawn workers, one numerical/Gurobi
thread each. Startup verified parent running, full production freeze pending;
this is not a claim of completed evaluation or available figures.

Approved workload:180 new green N200/N800 and600 saved SCS N50/N200/N800
policies, each10000 fixed OOS observations plus its own historical finite
training pool. Cached90-policy green N50 and SAA-kNN/SRO-FA reference curves
are reused. Seed9999 is historically inspected: descriptive comparison only.
No training, checkpoint/radius reselection or test-based tuning.

Working directory:
`/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924`

Launcher command:
`bash launch.sh /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v2/summary.json`

Python command:
`/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/run_comparison.py --preflight /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v2/summary.json`

Attach: `tmux attach -t retailers_comparison_R3_20260924`.
Status: `tmux has-session -t retailers_comparison_R3_20260924`.
Progress is persisted in [summary.json](summary.json), not repeated in the log
after every job. Per-policy logs/journals remain in this output directory.

[Persistent stdout/stderr](../../campaign_20260924T091137.287453559.log).
An adjacent exit artifact is written by the shell when the process ends.
[Launch metadata](launch.json) gives the exact output, command and environment.

## Measured preflight

| Stage | Seconds |
|---|---:|
|Failed v1, incorrect historical source path; no numerical replay/test|61.903557|
|Successful v2 total|112.841658|
|v2 unit-test subprocess,43 passing tests|8.881 approx.|
|v2 full780-policy freeze|84.277 approx.|
|v2 training-only replay,20 strictOPTIMAL QPs|0.721 approx.|
|v2 final source verification|8.353 approx.|

Exact raw phases are in [preflight summary](../../preflight_v2/summary.json).
Observed preflight peakRSS1087930368bytes; copied frozen inputs4587026419bytes.
Production disk reservation estimate46211727334bytes, not an output cap.
Actual resource policy is stored in launch metadata.

Historical SCS diagnostic:348 saved-point observations exceed the green1e-9
membership threshold, maximum7.151707322877732e-6. Counts include repeated
use in multiple selected pools. The user explicitly approved evaluating
unchanged coordinates with disclosure, without claiming SCS box certification.
No projection acceptance tolerance or green membership gate was relaxed.

Full PNG/PDF publication requires all780 evaluations and source identities
to pass. Current full evaluation timing remains pending. Historical training
and CCG timings must not be replaced by this invocation's loading/evaluation
time. No new SRO-FA/kNN training or CV is run.