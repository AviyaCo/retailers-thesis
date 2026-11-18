# First advisory-replay run (preserved)

All 21 tests passed; all 1,680 AP solves reached status 2. Controller outcome:
1,513 accepted, 162 signed bound/pool failures and five physical failures.
All six configurations failed readiness. Native support, cost and epsilon
disagreement alone no longer rejected cases; no full CCG or held-out work ran.

Audit found incomplete explicit cost-discrepancy coverage: the five physical
failures retained raw native/AP costs but did not reach final replay recording.
Thus this run reports 332 disagreements among 1,675 finalized cost replays;
raw evidence yields 337 disagreements among all 1,680 witnesses. This limitation
is not a success or a numerical change. A separately frozen logging revision
addresses it; see the
[final report](../shipping_replay_observed_complete_20260925/RESULTS.md).

Runtime: 57.494105 s (0.958235 min), body 52.260966 s, nested unit subprocess
3.189078 s. UTC 2026-09-25 12:44:51.201235–12:45:48.695336. One worker/thread,
gold-srv2 Linux, ap_env Python3.11.13. All rows/status/iteration/timing metadata
are retained. Completed-config count is zero; all reported case runtimes below
are failed-run values (one run each, so failed mean/median/min/max coincide).

| Ntotal | Radius | Failed case wall seconds | Minutes |
|---:|---:|---:|---:|
| 50 | 0 | 0.976532 | 0.016276 |
| 50 | 0.2 | 1.067137 | 0.017786 |
| 200 | 0 | 4.521432 | 0.075357 |
| 200 | 0.2 | 4.323038 | 0.072051 |
| 800 | 0 | 15.988687 | 0.266478 |
| 800 | 0.2 | 18.424040 | 0.307067 |

| Runtime scope | Seconds | Status |
|---|---:|---|
| Nominal training N50/N200/N800, seed0 LR0.01 | 79.761483 / 130.390117 / 165.776251 | Cached historical, not rerun |
| Robust neural/CCG and MP total | Unavailable | Not run |
| SRO-FA total | Unavailable | Not run |
| Training-center hard-pool evaluation | 0.050048 | Nested sequential sum |
| CV/validation/test evaluation | Unavailable | Not run |

Session `shipping_replay_observed_20260925` is finished. Exact command/cwd,
attach/status commands, log path and final timing:
[preflight_v1.launch.json](preflight_v1.launch.json),
[preflight_v1.invocation.json](preflight_v1.invocation.json).
Log: [preflight_v1.log](preflight_v1.log).