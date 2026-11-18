# Saved-green continuation launch — 25 September 2026

Production launched at 12:33:28 UTC; completion is pending. Original results are immutable.

- tmux session: `black_cont_R3_R10_20260925`
- Working directory: `/data/home/aviyac/retailers_moreau_training_selected/black_continuation_R3_R10_20260925`
- Log: [production_v1.launch/child.log](production_v1.launch/child.log)
- Whole-child status and timing: [production_v1.launch/child_runtime.json](production_v1.launch/child_runtime.json)
- Exact invocation: [production_v1.launch/invocation.json](production_v1.launch/invocation.json)
- Production output: `production_v1/` (live summary appears after the launch gate verifies inputs).

Exact command from the working directory:

```sh
/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/black_continuation_R3_R10_20260925/campaign.py production /data/home/aviyac/retailers_moreau_training_selected/black_continuation_R3_R10_20260925/production_v1 --preflight /data/home/aviyac/retailers_moreau_training_selected/black_continuation_R3_R10_20260925/preflight_v2/summary.json --sha 83a6aa3bc14b169fc04dcf8113017c77691c7bc585dbc78909423ee29365dfd4
```

Attach: `tmux attach -t black_cont_R3_R10_20260925`.

Check status from the working directory: `cat production_v1.launch/child_runtime.json`.
Per-job progress: `production_v1/summary.json`; individual persistent logs and raw AP/MP records are beneath `production_v1/jobs/`.
Whole-child terminal status is authoritative once `production_v1.launch/child.exit` exists. Exit 2 means nonconvergence, not full convergence; exit 1 indicates failure. One failed job does not cancel the remaining configurations.

## Scope

R3 and R10, N50/200/800, seeds 0–9. R3 radii: 0, .001, .003, .01, .03, .2, .4, .7, 1.5; R10 adds 2.

- 450 existing green trajectories: 209 already satisfy every-center stopping; 241 require further iterations.
- 30 new R10/radius2 configurations, initialized independently from the saved nominal winners.
- 90 existing R3/N50 black results retained: 89 separated, one capped at 40 MPs.

Continue from the **last stopping checkpoint and its own pool**, not the earlier returned minimum-AP policy. Stop when every training center has bound-based relative residual at most 1%, or retain explicit cap/stagnation/failure status. The maximum is **40 total MPs including historical work**, not 40 additional MPs. The return rule remains earliest minimum AP mean in the whole executed prefix for converged continuations; selected and stopping checkpoint status remain separate.

No nominal retraining, validation/test evaluation, new radius selection, SCS training, or SRO-FA work occurs here. The earlier SRO-FA radius2 comparison request remains separate and unlaunched.

## Validation and timing

Final preflight: 39 unit tests plus six real training-only native-pool/AP/two-epoch-MP checks. All passed; 431542 input bindings and 84 source bindings verified. Peak smoke RSS 1.63 GB. Production uses one joint pool of up to16 single-thread processes, subject to the resource gate; no guaranteed memory bound is inferred from smoke RSS.

| Component | Measured wall time / status |
|---|---|
| Initial preflight, whole child | 412.879 s (6.881 min) |
| Final preflight, internal / whole child | 410.679 / 410.892 s (6.848 min whole child) |
| Nominal neural training | Cached original per-seed timings; no new training |
| Robust neural/CCG continuation totals | Pending; per-iteration AP/MP and per-job wall times persisted |
| Historical CCG prefix | Persisted original wall times, separate from continuation |
| SRO-FA | Not run by this campaign; no new timing |
| Validation/CV/final-test evaluation | Not run |
| Parallel production sweep | Pending; not a sum of concurrent job durations |

Host: Linux `gold-srv2.iem.technion.ac.il`, Xeon Gold 6354, Python 3.11.13, CPU execution. Cached-only, completed, nonconverged, and failed timing cohorts remain separate, including the historical capped black run.