# Corrected production attempt

Started2026-09-24 09:19:34.011042UTC, tmux
`retailers_comparison_R3_v2_20260924`,32 single-threaded spawn workers.
At09:19:48UTC, parent was running its full freeze; no policy results yet.

Working directory:
`/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924`

Exact launcher:
`bash launch.sh /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v3/summary.json`

Python invocation:
`/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/run_comparison.py --preflight /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v3/summary.json`

Attach: `tmux attach -t retailers_comparison_R3_v2_20260924`.
Session status: `tmux has-session -t retailers_comparison_R3_v2_20260924`.
Progress: [summary.json](summary.json).
Persistent stdout/stderr: [campaign log](../../campaign_20260924T091931.770233601.log).
[Launch metadata](launch.json) records command, cwd, environment, log and exit paths.

Approved protocol is unchanged:780 fresh strict-Gurobi OOS/IS policies
(180 green N200/800 and600 SCS N50/200/800), same10000-pair seed9999 draw,
no training or reselection; cachedgreen N50 and historical SAA-kNN/SRO-FA.
Historical SCS pool coordinates remain unchanged with disclosed membership
residuals; green membership and strict physical projection gates unchanged.

## Attempt accounting

| Attempt | Outcome | Monotonic seconds |
|---|---|---:|
|Preflightv1|Failed source-path lookup, no numerical checks or test|61.903557|
|Preflightv2|Passed43tests and20training-rowQP checks|112.841658|
|Productionv1|Failed initializer serialization before any policy|92.853639|
|Focused real-spawn regression|Passed; both Torch thread counts1|2.426 (test-reported)|
|Preflightv3|Passed44tests,780freeze and20training-rowQP checks|114.946501|
|Productionv2 (this attempt)|Running at startup|Pending|

Prior attempts and their complete diagnostic artifacts are preserved in their
original output directories. The failed production did not evaluate any
policy, so its time is orchestration overhead, not model or test runtime.
The fix wraps the unchanged initializer in an importable module-level function.
The v3 preflight binds the repaired code and includes a real spawn test.

Current production logs per-model/phase timings, raw solve statuses and
iterations. PNG/PDF publication requires the complete grid and final input
verification. Full results are not claimed by this startup record.