# COMPLETED — 14 October 2026, 22:07 UTC

Resume receipt: completed, exit 0. Final counts: 90/90 nominal and CV jobs,
810/810 CCG configurations converged, 810/810 test evaluations, plots published
in `runs/production_v1/published/{R3,R10,shipping_R3}/`. Overall UTC elapsed time
was 84.2 h, including the 16-worker phase. One configuration used the logged
Gurobi retry (see thesis notes).

# Resumed with 56 workers — 14 October 2026

At 16:07 UTC the original 16-worker orchestrator (hard cap 16, no resume) was
stopped by user request to raise concurrency (receipt: failed, exit 143). Closing
its tmux session also SIGHUP-killed the 15 in-flight CCG workers (12 R10 at
radius 0.7/1.5 and 3 shipping N50). Their partial specs, logs and output folders
were moved unchanged to `runs/production_v1/_interrupted_20261014T1607Z/` and
those configurations rerun from scratch. The 585 completed CCG jobs and all 90
nominal/CV jobs are reused unchanged.

The remaining 225 CCG jobs, the training seal, the test evaluation and publication
run from [../green_fullN_cv_20261011_resume/resume.py](../green_fullN_cv_20261011_resume/resume.py)
(outside this folder because workers hash every local `.py`), through the frozen
`runner.invoke`, with at most 56 concurrent workers, each using one thread.
Session: `tmux attach -t green_fullN_cv_resume_20261014`. Log/receipt:
`../green_fullN_cv_20261011_resume/resume_v1.{log,invocation.json}`.
Runtime caveat: job-level `ccg_workers` metadata still says 16. In reality,
concurrency was 16 before the resume and up to 56 after it, and the interrupted
jobs' lost compute is not included in any reported time.

# Production launched — 11 October 2026

**Running, not completed.** Launched at 09:55:23 UTC (12:55 local server time).
Named tmux session: `green_fullN_cv_20261011`. Host: gold-srv2.iem.technion.ac.il.
16 workers, one numerical thread each; existing CPU implementations unchanged.

Verified startup: all three source preparation jobs completed; 16 actual CV
training epoch journals observed (sample epochs 1096, 1000, 926), no failed
worker invocation receipts, no completed nominal job yet, and no test seal.
This is a startup snapshot, not a live claim about later completion.

## Open status and logs

- [Production log](production_v1.log)
- [Launch command, cwd and completion receipt](production_v1.invocation.json)
- [Structured progress](runs/production_v1/summary.json)
- [Protocol and execution instructions](README.md)
- [Prelaunch validation](VALIDATION.md)

The tmux pane runs, from this physical directory:

    /data/home/aviyac/ap_env/bin/python -B -u launch.py --name production_v1 --workers 16

The full namespace-wrapped command and exact paths are recorded in the receipt.
Attach with `tmux attach -t green_fullN_cv_20261011`; check session presence with
`tmux has-session -t green_fullN_cv_20261011`. A disappeared session is not proof
of success: check the final receipt and summary. No implicit retry or resume.

## Scope and measured timing at startup

All three experiments, N50/200/800, seeds0–9; 1,350 nominal CV fits, 90 fresh
full-N refits, and 810 fixed-radius configurations. Previous practical validation
experiments are unchanged. Final test replay and plots are gated on completion
and convergence of the entire training grid. Shipping uses the explicitly
approved homogeneous advisory checks, not a mixture of original/continued runs.

| Phase | Measured duration / status |
|---|---|
| Final 39 regression tests | 3.976 s (0.066 min), passed |
| Three source-preparation jobs | 6.445 s (0.107 min), parallel elapsed |
| Nominal CV and full-N training | Running; totals not yet available |
| Robust green CCG | Pending; no measured production total yet |
| Strict test/own-pool evaluation | Pending; no test access at startup |
| SRO-FA / blue | Not run |

No completion time or result curve is inferred. On successful completion the
three plot folders will be under runs/production_v1/published. The final runtime
report separates CV, SAA initialization, nominal fit, every AP/MP and evaluation;
failed/nonconverged jobs retain their own records rather than being omitted.

Source Python files and README are frozen for this invocation. This status
document is not a computational source and may record subsequent verified status.