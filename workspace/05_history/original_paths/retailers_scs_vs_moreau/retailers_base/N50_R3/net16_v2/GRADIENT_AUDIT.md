# Matched-checkpoint gradient audit — 9 September 2026

**Completed:** 2026-09-09 10:32:22 UTC; total 884.885 s (14.75 min).
All six configurations and 81 central-difference probes completed. Final
checks confirm that source/input files, data and saved learned states remained
unchanged throughout the restart.

## Scope and provenance

This diagnostic uses only the original 50 training pairs (data seed 0), the
saved initial weights, and the two selected checkpoints. No optimizer steps,
validation/test access, hyperparameter selection or robust training occur.
The original projection capacity buffer of 1e-6 is retained. This is not the
prospective zero-buffer, train40/validation10 campaign.

Both backends receive identical learned parameters. Solver buffers belong to
their own backend and are not treated as learned checkpoints. Original neural
and input arithmetic is float32; projection outputs and the loss can be float64.
Supplemental float64 checks promote saved weights and float32-rounded inputs;
they do not reproduce the original training arithmetic exactly.

The three normalized random parameter directions (network, raw Q10, raw Q1r)
use seed 20260909. Original-neural-dtype finite differences cover Moreau at
h = .01, .001, .0001. Supplemental float64 checks cover both backends at
h = .001, .0001, .00001. Raw central/one-sided slopes, gradients, masks, solver
outcomes, timestamps and monotonic timings are in the structured artifact.

## Execution

- Working directory: this directory.
- Exact command: `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 /data/home/aviyac/ap_env/bin/python -u audit_gradients.py`
- tmux session: `retailers_gradient_audit_v2_20260909`.
- Attach: `tmux attach -t retailers_gradient_audit_v2_20260909`.
- Session status: `tmux has-session -t retailers_gradient_audit_v2_20260909`.
- Log: [gradient_audit_v2.log](gradient_audit_v2.log).
- Raw results: [gradient_audit_results_v2/audit.json](gradient_audit_results_v2/audit.json).
- Script: [audit_gradients.py](audit_gradients.py).
- Host: shared gold-srv2, Xeon Gold 6354. One model worker, Torch intra/inter-op
  1, BLAS/OpenMP/NumExpr/Rayon environment caps 1, Gurobi 1 thread. Legacy
  diffcp default batch pools remain; actual native thread counts unmeasured.

The first attempt failed in setup after 2.504 seconds because strict full-state
loading incorrectly demanded CVXPYLayers solver buffers absent from the saved
parameter-only checkpoints. No candidate was evaluated. Its
[artifact](gradient_audit_results/audit.json) and [log](gradient_audit.log) are
preserved. The correction consistently loads/restores/hashes named parameters.
A no-solve regression passed for both backends before the restart; three
existing checkpoint/projection tests also passed in 0.867 seconds.

## Verified findings from completed blocks

| Original-neural-dtype checkpoint | SCS loss | Moreau loss | SCS max projection error vs Gurobi | Moreau max projection error vs Gurobi |
|---|---:|---:|---:|---:|
| Initial | 26.1269812 | 26.1268981 | 9.89e-5 | 2.75e-12 |
| SCS-selected | 19.5555843 | 19.5646505 | 1.35813 | 1.03e-11 |
| Moreau-selected | 19.9942205 | 19.9891774 | 0.359383 | 5.42e-9 |

At initial weights, network gradients differ by about 1.30e-5 relative norm.
At SCS-selected weights, network gradient relative difference is 45.85%
(cosine .89045), versus 7.33% for the inventory group. These are comparisons
of whole numerical stacks, not backward algorithms at identical projections.
At Moreau-selected weights, the differences are 3.68% and 22.21%, respectively.

At SCS-selected weights, SCS returns minimum projected inventory -0.506305 and
maximum aggregate-capacity violation 0.547785, despite every observed base solve
reporting solved. Moreau's corresponding violations are at roundoff scale.
The network targets reach magnitude 68.49, compared with 1.52 initially.
SCS eps=1e-4 is not a promise of 1e-4 accuracy for every physical decision.
[SCS termination criteria](https://www.cvxgrp.org/scs/algorithm/index.html#termination-criteria)
use absolute-plus-relative bounds on canonical residuals, including the norms
of Ax, slack and b. The audit does not measure each problem's termination
threshold or establish scaling as the sole cause of the discrepancy.

For the initial and SCS-selected checkpoints, the completed smooth-looking
Moreau float64 probes agree closely with finite differences (roughly 1e-9 or
better at the smaller steps). SCS-selected float64 finite differences are
unstable across step sizes; forward errors prevent isolating a backward defect.
In original float32 arithmetic, some coarse perturbations cross cost branches
and fine perturbations show rounding sensitivity. A close central slope can
hide disagreement between the one-sided slopes.

The completed Moreau-selected float64 probes show the same distinction:
network and Q10 central derivative errors at h=1e-4 are 2.19e-11 and 1.41e-11
with no observed ReLU/cost switch. Q1r crosses a cost branch at h=1e-4, but at
h=1e-5 no such switch is observed and its error is 1.37e-10. Coarse-step
disagreement is therefore not evidence of a faulty local derivative. This
supports the tested directions, not an exhaustive certification.

## Interpretation limits

- No obvious Moreau derivative defect is established by the completed probes.
  Three directions at three checkpoints are not an exhaustive gradient proof.
- Binding projection constraints alone do not imply nondifferentiability.
  Exact-zero masks are sensitive to roundoff; endpoint masks cannot rule out
  interior crossings. Inactive inner maximum switches may not affect total cost.
- SCS finite differences were not evaluated in original float32 neural
  arithmetic; both-backend comparisons there include autograd and forward
  solutions, not a complete same-dtype FD validation.
- The selected SCS policy still costs 19.564650 under accurate projection,
  lower than the selected Moreau policy at 19.989177. SCS forward inaccuracy
  therefore does not explain away the entire 2.17% policy-quality gap.
- Local diagnostics do not reconstruct Adam moments, clipping, scheduler
  history or intermediate active-set transitions. The cause of the final
  training difference remains unresolved; no optimizer or solver settings have
  been changed to improve the ranking.

## Runtime accounting

| Checkpoint | Original float32 neural/input diagnostic | Supplemental float64 diagnostic |
|---|---:|---:|
| Initial | 12.824 s | 84.323 s |
| SCS-selected | 75.920 s | 205.657 s |
| Moreau-selected | 155.668 s | 337.941 s |

Each cell is one completed run (count=1; mean=median=min=max=the displayed
duration); configurations differ and are not pooled. Total invocation time is
884.885 s (14.75 min), including setup and final persistence. The float32 arm
has fewer FD evaluations, so these columns are not a precision-speed benchmark.
Candidate totals include both backends, Gurobi references, finite
differences, diagnostics and frequent JSON persistence; they are not solver-only
benchmarks. Nested phase durations overlap and must not be summed.
Nominal training, robust CCG, SRO-FA, CV and final-test evaluation were not run
in this audit and have no new timings. Failed setup time remains separate.