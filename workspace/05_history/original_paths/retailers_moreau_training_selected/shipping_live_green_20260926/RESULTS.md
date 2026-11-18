# Shipping campaign: launch and interim evidence

## State at 2026-09-26 18:57:05 UTC

Production is **running**, not complete. All **270 configurations were submitted**
with **62 CCG workers**, each using one Torch/Gurobi thread. The last persisted
snapshot contains **190 completed/converged, 2 failed, 78 not yet returned**.
It records 506.195495 s (8.436592 min) since the production body started; that is
the last persistence time, not a final sweep duration. Current live evidence is
[production_v1/summary.json](production_v1/summary.json) and
[production_v1/runtime_aggregates.json](production_v1/runtime_aggregates.json).

The complete 30-initialization/90-nominal-fit barrier passed. All 30 hard-cost
LR selectors completed: LR .005 selected 9 times, .01 selected 8 times, and
.05 selected 13 times. These are validation-tuned choices, not unbiased test
results. Nominal fits were reused, not retrained. No test, refit, SRO-FA,
benchmark, or CV was run. Final radius validation has not run and is **blocked
by the failed configurations** under the approved all-270-converged rule.
Other submitted jobs continue; no retries, substitutions, or guard changes.

## Native preflight and provenance

All **61 tests passed** (30 approved guards, 15 controller, 16 campaign).
The actual spawned production worker ran the two fixed train-only cases below:
Ntotal50/Ntrain40, data seed0, network seed0, split seed100000, nominal LR.01.
No validation data were evaluated in preflight. Its output never initializes
production.

| Radius | AP sweeps / samples | MPs | Selected candidate | Stop mean gap | Config wall s (min) | AP total s | MP total s | Evaluation s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 / 40 | 0 | 0 | 0.000747414047 | 2.656516 (0.044275) | 1.682622 | 0 | 0.014363 |
| .2 | 2 / 80 | 1 | 1 | 0.008356615018 | 27.568085 (0.459468) | 2.775315 | 22.319604 | 0.036736 |

Each configuration has one completed run; its mean, median, minimum and maximum
equal the listed wall time. All 120 APs reported Gurobi OPTIMAL, with maximum
reported MIP gap 6.163e-7. This is the epsilon-restricted AP, not an unrestricted
native-hard or globally optimal robust certificate.

The positive-radius MP used 79 points across 40 pools, evaluated 1764 epochs,
selected pre-update epoch1464 and stopped after 300 non-improving epochs. Smooth
loss changed from 20.72274850874844 to 20.35572776378363; returned/next-candidate
smooth replay matched. Hard stopping means were kept separate. The stopping
candidate still had 13/40 per-center separation residuals above 1%; the approved
criterion is an **approximate mean-gap stop**, not per-center separation.

Preflight elapsed scopes: tests 15.969436 s; spawned-controller stage 36.197026 s;
body 52.524999 s; whole child **59.990969 s (0.999849 min)**. These nested scopes
must not be added. Nominal training for the reused checkpoint was a historical
79.761483 s/2351 epochs, not current preflight training. Native parent/worker
maximum lifetime RSS was 828731392 bytes; this N50 evidence does not certify
growing N800 pool memory. Production reserves at least 4 GiB per worker and
16 GiB shared headroom, with four CPU slots plus observed load reserved.

The preflight summary SHA256 is
`fe469c476f681b06d0a54ae84310d270ebf859a3f479f62711d6703bffa32795`;
its bound manifest SHA256 is
`79d51a9daa8f9a37469c1872aa1c32cb518b045acd293701feedfa045cd460ff`.
The 5176-entry source/input map is identical in the production manifests;
preflight summary/manifest bindings are propagated into production worker checks.
Both launcher and production entrypoint enforce the successful integration gate.

Read-only review reconciled all AP records, selected checkpoint hashes, pool
identities, MP epochs, and timing scopes. One frozen intermediate assessment
has a stale convenience `mp_total_time_s=0`; its raw `mp_call.wall_time_s` and
final configuration/worker summaries correctly contain 22.319604 s. Reports use
the raw/final values; frozen computational files and artifacts remain unchanged.

## Preserved production failures

Both failures occurred at candidate1 after one MP. All 640 centers were attempted
in each sweep, including after the failing sample; neither configuration selects
a checkpoint. AP-internal epsilon/integrality/link checks and native physical
checks passed. The rejection was the **retained exact-duplicate/material-
separation guard**, not a removed replay or negative-bound/pool check.

| N | Seed | Radius | Sample (original ID) | AP bound | Native pool cost | Separation residual | Native physical residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 7 | .2 | 271 (465) | 12.593166339263046 | 12.393165702372789 | .015881679913 | 6.357828788e-7 |
| 800 | 9 | .4 | 176 (514) | 13.830171383412660 | 13.630171893537044 | .014461099890 | 9.536743200e-7 |
| 800 | 9 | .4 | 218 (701) | 11.951907575951589 | 11.792313498258590 | .013353021405 | 3.552713679e-15 |

All three APs reported OPTIMAL with zero MIP gap, and each witness exactly equals
its existing pool point at index1. Rejection threshold is 0.01 on the positive
bound/pool residual scaled by max(1,abs(bound),abs(pool)). Native physical
tolerance remains 1e-6; AP internal tolerance1e-9, epsilon1e-6, exact q>0 fees,
unmodified shipments and smooth gradient training remain unchanged.

For samples271 and218, separately evaluated native witness costs exceed their
batch-evaluated pool maxima despite identical points. This is a recorded
inconsistency; the cause was not established by this read-only inspection.
AP/native support disagreement alone does not explain every failure.

| Failed configuration | Config wall s (min) | AP total s | MP total s | Evaluation s |
|---|---:|---:|---:|---:|
| N800 seed7 radius.2 | 108.063431 (1.801057) | 68.736893 | 28.894191 | .186749 |
| N800 seed9 radius.4 | 98.224280 (1.637071) | 50.665937 | 30.268719 | .193398 |

Failure evidence: [seed7/radius.2](production_v1/N800/ccg/seed_7/radius_0.2/summary.json),
[seed9/radius.4](production_v1/N800/ccg/seed_9/radius_0.4/summary.json).
Failed times are excluded from completed-run aggregates, not discarded.

## Runtime tables

Environment: gold-srv2.iem.technion.ac.il, Linux, 72 logical CPUs; Python3.11.13,
Torch2.10.0, NumPy2.4.2, SciPy1.17.1, Gurobi13.0.1, Moreau CPU0.4.0.
Preflight used one CCG worker; production uses 30 validation workers and 62 CCG
workers. Times below are wall-clock seconds. Divide by60 for minutes, by3600
for hours. Concurrent model times must **not** be summed as sweep elapsed.

### Historical nominal training: persisted prior times, not cache-load times

| Ntotal | LR | Runs | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | .005 | 10 | 99.716380 | 98.776939 | 97.183827 | 105.290021 |
| 50 | .01 | 10 | 94.205906 | 97.729930 | 79.761483 | 102.302351 |
| 50 | .05 | 10 | 69.465364 | 68.056178 | 54.431093 | 89.939785 |
| 200 | .005 | 10 | 119.072872 | 120.512872 | 105.146575 | 129.575114 |
| 200 | .01 | 10 | 119.452738 | 125.218815 | 99.860643 | 130.390117 |
| 200 | .05 | 10 | 95.011041 | 93.833828 | 75.079683 | 108.295141 |
| 800 | .005 | 10 | 168.807134 | 170.851386 | 149.205489 | 179.149527 |
| 800 | .01 | 10 | 159.141227 | 166.261638 | 131.365634 | 176.840694 |
| 800 | .05 | 10 | 114.796211 | 117.460895 | 82.101536 | 129.501726 |

Current cached-grid verification: 5.615469 s. This is not nominal training time.
Historical nominal jobs used52 concurrent workers; initializations used30.

### Current hard-cost LR selection, including three arms and smooth replay

| Ntotal | Runs | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|
| 50 | 10 | .052755 | .050580 | .034673 | .088844 |
| 200 | 10 | .060275 | .050219 | .042506 | .098526 |
| 800 | 10 | .090151 | .091404 | .066060 | .114104 |

Overall LR-selection stage, including worker startup/shutdown: **12.316652 s
(0.205278 min)**. Arm and native-evaluation timings are persisted separately in
the per-worker records and live aggregates.

### Current CCG completed-only interim aggregates

This snapshot is **completion-biased**: later-finishing runs are absent. Do not
use it as a final runtime comparison, extrapolated total, or completed campaign.

| Ntotal | Radius | Completed runs | Mean s | Median s | Min s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 10 | 3.399647 | 3.401480 | 3.061479 | 3.562175 |
| 50 | .001 | 10 | 3.767170 | 3.763051 | 3.228611 | 4.282815 |
| 50 | .003 | 10 | 4.443661 | 4.376676 | 3.228873 | 5.673401 |
| 50 | .01 | 10 | 3.279932 | 3.260799 | 2.973633 | 3.575798 |
| 50 | .03 | 10 | 3.317359 | 3.190762 | 2.835792 | 3.951021 |
| 50 | .2 | 10 | 213.353159 | 246.338419 | 55.399526 | 422.313323 |
| 50 | .4 | 5 | 307.922202 | 359.838220 | 90.247038 | 412.280549 |
| 50 | .7 | 3 | 245.811188 | 232.322825 | 155.446925 | 349.663815 |
| 200 | 0 | 10 | 9.335824 | 9.213709 | 8.125651 | 11.059662 |
| 200 | .001 | 10 | 11.586208 | 11.603795 | 10.709534 | 12.209359 |
| 200 | .003 | 10 | 10.387608 | 10.885469 | 7.688797 | 11.748646 |
| 200 | .01 | 10 | 7.722364 | 7.649589 | 7.139407 | 8.346297 |
| 200 | .03 | 10 | 8.188500 | 8.231411 | 7.176501 | 8.833116 |
| 200 | .2 | 9 | 271.153378 | 197.752750 | 148.462092 | 395.984252 |
| 200 | .4 | 5 | 384.545132 | 393.065095 | 352.092230 | 405.867415 |
| 800 | 0 | 10 | 29.662859 | 30.279325 | 27.758865 | 31.372726 |
| 800 | .001 | 10 | 30.787872 | 30.864016 | 29.582324 | 31.723295 |
| 800 | .003 | 10 | 28.292375 | 28.634581 | 25.360236 | 30.403089 |
| 800 | .01 | 10 | 26.650552 | 26.765702 | 25.083315 | 27.921630 |
| 800 | .03 | 10 | 32.180743 | 31.503471 | 27.025295 | 38.934339 |
| 800 | .2 | 7 | 307.010753 | 317.115563 | 244.724732 | 353.507974 |
| 800 | .4 | 1 | 117.253977 | 117.253977 | 117.253977 | 117.253977 |

No completed runs yet for N50/radius1.5, N200/radii.7,1.5, or N800/radii.7,1.5;
their completed-run aggregates are unavailable. Ongoing AP/MP timing journals
remain under each configuration. Failed/interrupted/nonconverged cases remain
separate in the structured aggregate artifact.

| Other runtime scope | State |
|---|---|
| Full production CCG sweep/end-to-end total | In progress; final duration unavailable |
| Radius validation | Not run; complete/converged-grid barrier cannot pass with the recorded failures |
| SRO-FA total/evaluation | Not part of this authorized campaign; unavailable |
| Test, refit, benchmark, CV | Not run; unavailable |

## Launch and monitoring

Production body started **2026-09-26T18:47:53.478177+00:00**.
Working directory: `/data/home/aviyac/retailers_moreau_training_selected/shipping_live_green_20260926`.
Exact launch command from that directory (record only; do not rerun against
the existing output):

```sh
/data/home/aviyac/ap_env/bin/python -B launch.py production --output production_v1 --session shipping_live_green_20260926 --preflight preflight_v1/summary.json --preflight-sha256 fe469c476f681b06d0a54ae84310d270ebf859a3f479f62711d6703bffa32795
```

Exact child command and cwd are persisted in
[production_v1.launch.json](production_v1.launch.json).
Session: `shipping_live_green_20260926`.
Persistent merged stdout/stderr: [production_v1.log](production_v1.log).
Per-worker logs and raw records are under each N/worker directory.

```sh
tmux attach -t shipping_live_green_20260926
tmux has-session -t shipping_live_green_20260926
tail -n 30 /data/home/aviyac/retailers_moreau_training_selected/shipping_live_green_20260926/production_v1.log
```

The supervisor writes its final whole-child invocation record on termination;
it was not yet present at this snapshot. Detached execution is tmux-managed,
not nohup. Preflight used session `shipping_live_green_preflight_20260926` and
the same launcher with `preflight --output preflight_v1 --wait`.