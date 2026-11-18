# Replay rejection removed; retained safeguards still block readiness

## Outcome

The user-selected change is implemented: **AP/native support, cost and native
epsilon disagreement no longer reject a witness**. Each finite native witness
has an explicit `recorded_only` cost observation, including witnesses rejected
for physical infeasibility. Exact `q > 0` fees and raw shipments remain unchanged.

The complete rerun tested the same 1,680 training centers: seed 0, LR 0.01,
Ntotal 50/200/800 (Ntrain 40/160/640), radii 0/0.2. All 1,680 Gurobi APs reached
status 2. The controller accepted **1,513**, rejected **167**, and rejected all
six configurations. Remaining failures: **162 signed bound/pool gaps and five
native physical-feasibility violations**. No full CCG was launched.

| Ntotal | Radius | APs | Accepted | Signed-gap failures | Physical failures | Configuration status |
|---:|---:|---:|---:|---:|---:|---|
| 50 | 0 | 40 | 35 | 5 | 0 | Failed |
| 50 | 0.2 | 40 | 39 | 1 | 0 | Failed |
| 200 | 0 | 160 | 100 | 60 | 0 | Failed |
| 200 | 0.2 | 160 | 158 | 1 | 1 | Failed |
| 800 | 0 | 640 | 550 | 90 | 0 | Failed |
| 800 | 0.2 | 640 | 631 | 5 | 4 | Failed |

There are **337 support disagreements**, **32 native-epsilon disagreements**,
and **337 cost differences above the former 1e-4 replay threshold** among all
1,680 recorded observations. These are advisory counts, not failures or claims
of agreement. A completed observation means the observation was recorded; it
does not mean the AP, physical check or controller passed.

The five physical failures retain negative native allocations approximately
−6.36e-7 to −3.18e-7, beyond the unchanged 1e-9 witness tolerance. A Moreau
solved status does not override this physical check. AP-internal binary/link,
rational folded-policy support and epsilon checks remain required. Signed
bound/incumbent and bound/pool ordering, membership, duplicate/collision,
unresolved-bound and finite/evidence-integrity checks are unchanged.

## Verification and interpretation

- **23 regression tests passed**, including one-call replay, advisory genuine
  support/cost/epsilon disagreement, physical-failure cost logging, retained
  physical/solver/AP-internal/signed failures, and separate hard/smooth costs.
- Independent read-only review verified all 1,680 row/case/top-level mirrors,
  raw observation arithmetic and numerical identity against the first advisory
  run. All incumbents, points, bounds, native shipments/costs and pools match.
- The first advisory run had already matched all raw numerical results from
  the original readiness run. Its 32 early replay failures became 27 accepted
  cases and five newly exposed physical failures, not changed numerical outputs.
- All **3,469 source/comparison SHA-256 bindings plus six overlapping input
  bindings** verified; both prior comparison trees and the regression fixture
  are included. Historical code and outputs were not overwritten.
- No training, MP, LR/radius selection, validation/test evaluation or SRO-FA ran.
  Removing replay rejection is not a certificate that the epsilon-restricted
  folded AP bounds the native hard-cost objective.

## Measured runtime

Host: gold-srv2.iem.technion.ac.il, Linux, Intel Xeon Gold 6354 (previously
identified), 72 logical CPUs visible. Python 3.11.13 in ap_env; native Moreau
CPU active-set and Gurobi; **one sequential worker, one Torch/Gurobi thread**.
Shared-host load was not controlled. Peak process RSS: **692,514,816 bytes**.

Final named-tmux invocation: **2026-09-25 12:50:53.969058–12:51:51.960988 UTC**,
exit **1**, full child **57.991928 s (0.966532 min)**. Readiness body:
**53.645575 s**. Regression-test subprocess: **3.562350 s**. These are nested,
not additive. The prior advisory run took **57.494105 s (0.958235 min)** and
also failed readiness; its 332/1,675 finalized cost-disagreement count omitted
five physical failures. Both invocations are retained, not selectively replaced.

All configurations below are **failed readiness runs**. Completed-configuration
aggregate count is zero for each N/radius; completed mean/median/min/max are
therefore unavailable. Each failed configuration has one measured run, so its
failed-run mean, median, minimum and maximum equal the single value shown.

| Ntotal | Radius | Failed runs | Failed configuration wall seconds (mean/median/min/max) | Minutes | Nested hard-pool evaluation seconds | Nested sequential AP sum seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 1 | 1.057085 | 0.017618 | 0.005475 | 0.952161 |
| 50 | 0.2 | 1 | 1.103647 | 0.018394 | 0.007899 | 1.004476 |
| 200 | 0 | 1 | 3.676779 | 0.061280 | 0.006066 | 3.256127 |
| 200 | 0.2 | 1 | 5.226674 | 0.087111 | 0.009608 | 4.214750 |
| 800 | 0 | 1 | 16.311087 | 0.271851 | 0.014206 | 14.112776 |
| 800 | 0.2 | 1 | 18.796483 | 0.313275 | 0.022738 | 16.804210 |

Sequential AP sum: **40.344500 s**; nested solver wall sum **5.735043 s**;
controller-row sum **41.404359 s**. None is additional to the whole-child time.
Solver iterations, barrier iterations, nodes, statuses, parameters, UTC
timestamps and monotonic durations are preserved per sample. There are no MP
iterations in this readiness-only run.

| Required method/runtime scope | Measured seconds | Interpretation |
|---|---:|---|
| Nominal neural training, N50 seed0 LR0.01 | 79.761483 (1.329358 min) | Previously persisted training, not retrained |
| Nominal neural training, N200 seed0 LR0.01 | 130.390117 (2.173169 min) | Previously persisted training, not retrained |
| Nominal neural training, N800 seed0 LR0.01 | 165.776251 (2.762938 min) | Previously persisted training, not retrained |
| Current initial-policy AP/controller readiness | 57.991928 (0.966532 min) | Failed whole-child run; not a robust CCG total |
| Robust neural/CCG total and MP training | Unavailable | Not run; full experiment required to measure |
| SRO-FA partition/solver/OOS/total | Unavailable | Not run |
| Current training-center hard-pool evaluation | Per configuration above | Separate nested evaluation timings |
| Validation/CV/selection/final test evaluation | Unavailable | Not run; no held-out data accessed |

Repeated radii reuse the same nominal checkpoint; cached training is not counted
again for each radius. No cached loading time is represented as model training.

## Reproduction and artifacts

Intended working directory: this revision directory. Exact supervisor command:
`/data/home/aviyac/ap_env/bin/python -B launch.py --output preflight_v1 --session shipping_replay_observed_complete_20260925`.
The exact absolute child command and cwd are preserved in
[preflight_v1.launch.json](preflight_v1.launch.json) and
[preflight_v1.invocation.json](preflight_v1.invocation.json).

Session: `shipping_replay_observed_complete_20260925` (finished).
While active: `tmux attach -t shipping_replay_observed_complete_20260925`;
status: `tmux has-session -t shipping_replay_observed_complete_20260925`.
After termination, the invocation artifact provides final status and timing.

- Persistent stdout/stderr: [preflight_v1.log](preflight_v1.log).
- Structured results: [preflight_v1/summary.json](preflight_v1/summary.json).
- Frozen bindings: [preflight_v1/provenance.json](preflight_v1/provenance.json).
- Regression log: [preflight_v1/unit.log](preflight_v1/unit.log).

**Conclusion: the requested replay rejection is removed. Full CCG remains
blocked by deliberately retained signed-gap and physical safeguards.**