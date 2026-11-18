# Launched: live green R3/no-shipping, N200 and N800

Production started **2026-09-23 09:22:53.404921 UTC**. Session:
`retailers_green_R3_200_800_20260923`. Parent shell PID at launch:2816070.
The first startup check confirmed **20/20 SAA initializations completed with
no reported failures**; the campaign was still running. This is not a claim
that training, CCG or final selection has completed.

Follow-up at **09:26:17.520138 UTC**: tmux session present, parent status
`running`, initialization20/20 completed (stage wall3.993812 seconds),
nominal59/60 completed with no collected failed jobs. The final nominal arm
was still pending; CCG had not started. No completion or quality claim follows
from this progress snapshot.

## Frozen workload

- N200:160 training /40 validation; N800:640 training /160 validation.
- R=3, no shipping, fresh initialization/training,10 data seeds each.
-60 nominal LR runs, then180 live-green CCG configurations across both sizes.
- Full-batch Adam at both sizes; unchanged3000-epoch/patience300,40-MP budgets.
- Green initial assessment eligible; first mean-gap pass stops computation;
  earliest minimum-AP checkpoint within the inclusive executed prefix returned.
- No test evaluation, SRO-FA/kNN retraining, shipping or R10 launch.

## Actual parallel allocation

| Stage | Maximum workers | Tasks |
|---|---:|---:|
| Training-only SAA initialization |20|20|
| Nominal LR training |59|60|
| Live green CCG |59|180|
| Each validation/selection stage |20|20|

One numerical/Gurobi thread per worker. Stages are not concurrent with each
other;59+59 is not118 simultaneous workers. N200/N800 jobs share the worker
pool. Host:gold-srv2.iem.technion.ac.il, Intel Xeon Gold6354,72 logical CPUs.
The frozen resource policy reserved9 slots for observed background load and4
for headroom;16GiB host memory reserve and at least4GiB/worker. The actual
resource snapshot and versions are in
[provenance.json](results/20260923T092253.404783+0000/provenance.json).
These reserves are not a proof of a bound on future MIP-tree memory.

## Preflight evidence and timings

Preflight completed, exit0: **36 tests passed**, both size subprocesses
completed actual SAA/native projection/AP/replay checks, and radius-zero live
green assessed all160/640 centers and stopped at candidate0 without an MP.
Maximum-pool fixtures used41 points per center:6560/26240 points. Those
constructed points and disposable two-epoch weights are test fixtures, never
production data or initialization. Production source hashes match preflight.

| Timing scope | Seconds | Interpretation |
|---|---:|---|
| Full preflight |45.502146|0.758369min; includes test processes/startup/checks |
| Solver-free test subprocess |18.009137|0.300152min; includes interpreter/import overhead |
| N200 numerical child |7.996728|0.133279min; peakRSS0.767155GiB |
| N800 numerical child |17.983101|0.299718min; peakRSS0.857792GiB |
| Production nominal training |Pending|Per-model timers/journals live; no final total yet |
| Production CCG/AP/MP |Pending|Not inferred from N50 or preflight |
| Validation/radius selection |Pending|No test/benchmark evaluation in this launch |
| SRO-FA partition/solver/evaluation/total |Not run|No new benchmark timing claimed |

Preflight records are in [summary.json](preflight_results_v1/summary.json),
with raw subprocess streams and each size's phases. The prelaunch review found
two accounting/provenance issues, both fixed and regression-tested before
production: preservation of nested failed-job evidence, and exact preflight
hash binding to the controller/launcher/tests. An indentation edit failed
before numerical execution and was repaired before tests passed. No solver
settings or optimization budgets were weakened to pass preflight.

## Exact reproduction and status

Working directory:
`/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923`

Exact production command:
`/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py --preflight /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/preflight_results_v1/summary.json`

The tmux session executes the [launcher](launch.sh) with that absolute
preflight-summary argument. Attach:
`tmux attach -t retailers_green_R3_200_800_20260923`.
Check presence: `tmux has-session -t retailers_green_R3_200_800_20260923`.
Check progress: `tail -n 20 campaign_20260923T092250.log` from the working directory.

- [Persistent campaign log](campaign_20260923T092250.log)
- [Live structured summary](results/20260923T092253.404783+0000/summary.json)
- [Launch metadata](launch.json)
- Per-job worker logs are in the output's N200/N800 initialization, nominal,
  selection and job_logs directories. Detailed AP/MP artifacts are under ccg.
- Exit artifact is written by the launcher when it terminates; absent at startup.
- Runtime aggregates are generated on parent completion, with failed/nonconverged
  jobs kept separate. Concurrent job walls are not summed as sweep elapsed.

Do not edit the frozen computation/protocol/test/launcher sources during this
run. This startup record was added after the freeze and is not an input to it.