# Green protocol review and cross-experiment launch gate

**Decision: HOLD new launches pending protocol alignment.**

Requested scope: first N=50 with fixed shipping (R=3, f=0.1) and with ten
retailers (R=10, f=0), then consistency with N=200 and N=800 and the base
R=3, f=0 experiment. This is not authorization to silently substitute the
historical variant workers or change the physical shipping model.

The completed base-R3/N50 green figure is credible as a **descriptive,
retrospectively reconstructed heuristic**. The extension is not currently a
ready-to-run, consistently implemented experiment. No new training, AP, MP,
held-out evaluation, or benchmark campaign was launched during this review.
Historical experiment inputs and the supervisor-review manuscript were not
modified. This review makes no claim that the identified legacy error paths
were exercised in every historical run.

## 1. What passes in the current green implementation

- Training and validation are separated by one seeded 40/10 split of a total
  50 observations. This is **not five-fold CV** and has no all-50 refit.
- SAA initialization uses only training observations. The policy observes
  first-period demand and fixed inventories, not future second-period demand.
- Nominal/MP checkpointing, scheduler and patience use training loss. Each MP
  has fresh Adam state and an eligible incoming-policy checkpoint. The weights
  saved correspond to the evaluated preupdate objective.
- Validation selects a nominal LR, then later a radius. This reuse is tuning,
  not test leakage; selected validation costs are not unbiased performance
  estimates.
- Green tests its mean-gap condition at the initial AP. It returns the earliest
  minimum mean-AP incumbent **through and including** its first passing AP.
  No later candidate or later scenario pool is used in that choice.
- All centers must pass AP status, membership and numerical replay gates.
  Failed APs are not removed before averaging. The current projection has no
  capacity buffer or zero-allocation fallback.
- The new IS curve is the average of per-center maxima on the returned
  checkpoint's own saved pools. It is not the AP mean or the terminal pool.
- Fixed policy/radius decisions precede current test lookup, and cached scores
  require matching checkpoint, learned state and evaluation/data contracts.

Evidence: [train-only trainer](N50_R3_train40_val10/training.py#L94-L186),
[current reconstruction](N50_R3_train40_val10/stopping_only_min_ap/reconstruct.py#L10-L38),
[fixed-selection evaluation ordering](N50_R3_train40_val10/stopping_only_min_ap/run_update.py#L128-L240),
[selected-pool figure evaluation](N50_R3_train40_val10/stopping_only_min_ap/paper_figure/make_figure.py#L167-L244).
The independent read-only reviewer also checked the saved trajectories,
checkpoint/pool chains, selection decisions, hashes and projection journals;
this is an audit of saved evidence, not fresh numerical reproduction.

## 2. Launch blockers

### A. There is no prospective general-purpose green controller yet

The current [campaign launcher](N50_R3_train40_val10/run_campaign.py#L280-L299)
calls the older every-center controller. Its
[termination and publication](N50_R3_train40_val10/ccg_training.py#L296-L401)
stop on every-center closure and return the terminal policy. Green is a
subsequent reconstruction of a completed black trajectory. Reusing the launcher
would do extra work after green should have stopped and would not measure a
fresh green end-to-end runtime.

The retrospective reader requires a completed source trajectory. A future
green policy must not depend on a later black continuation succeeding.
It also [rejects a missing green stop](N50_R3_train40_val10/stopping_only_min_ap/reconstruct.py#L27-L31):
broader cap/stagnation cases must have an explicit treatment, not an invented
successful return. Preserve failures/nonconvergence and do not omit them.

The [current core](../retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/core.py#L137-L224),
AP folding, pool checks and evaluator hardcode R=3, Ntrain=40 or six-dimensional
centers. Network layers elsewhere parameterized by R do not make these
components automatically dimension-independent.

**Required:** an isolated, tested live-green implementation with explicit
dimensions/data budget, immediate green stopping and prefix-only return.
Existing tested Moreau/Gurobi components should be reused rather than importing
legacy experiment scripts that execute on import. Preserve frozen sources.

### B. Shipping has an objective and activation-convention mismatch

The [legacy shipping MP](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L116-L132)
charges the smooth surrogate

\[
0.1\sum_r (1-\exp(-5q_{2r})),
\]

whereas its [AP](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L248-L268)
charges a binary fee and enforces either q2=0 or q2>=1e-6. The
[evaluation](../retailers_with_shipping/N50/rho_sweep/eval_sweep_shipping_10k.py#L73-L81)
charges when q2>1e-6, while the projection itself allows intermediate positive
allocations. The threshold equality and excluded interval are not identical
models. Historical SRO-FA additionally uses q2>1e-10.

The old stopping diagnostic subtracts this smooth MP objective from the hard
AP objective. That difference combines surrogate error with pool separation;
it is not the same-cost green diagnostic. Tightening a tolerance or silently
clipping q2 would not fix the definition.

**Decision needed:** define the intended shipping cost and activation semantics
once for AP, AP replay, pool assessment, validation, IS and OOS. A smooth
gradient-training surrogate can be retained as an explicitly separate
approximation if approved, but must not silently become the hard-cost gap
denominator. Its training-loss improvement does not guarantee hard-cost
improvement. Exact threshold behavior and solver-tolerance edge cases require
tests and an approved formulation before launching shipping.

### C. Legacy variant workers would reintroduce known bugs/failure policies

Verified examples, not claims that all historical outputs failed:

| Risk | Source evidence |
|---|---|
| Loss evaluated before Adam update but checkpoint copied afterwards | [Shipping N50](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L142-L157), [R10 N50](../retailers_problem/R10/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2_N50_R10.py#L116-L131) |
| Failed AP centers can be skipped, then the remaining costs averaged | [Shipping N50](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L270-L292), [R10 N50](../retailers_problem/R10/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2_N50_R10.py#L204-L227) |
| Failed projection can return a zero allocation | [Shipping evaluator](../retailers_with_shipping/N50/rho_sweep/eval_sweep_shipping_10k.py#L98-L110) |
| Nominal shipping CV initializes each fold from an SAA fit on all 50, including the fold's validation outcomes, then refits on all 50 | [Nominal shipping](../retailers_with_shipping/N50/run_cv_nominal.py#L270-L323) |
| Solver exception can restore and return a previous best checkpoint instead of failing the configuration | [R10 N800 MP](../retailers_problem/R10/N800/rho_sweep_net16_v2/run_rho_seed_net16_v2_N800_R10.py#L129-L153) |
| Buffered legacy training projection differs from unbuffered AP/evaluation | [Shipping projection](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L110-L124), [R10 projection](../retailers_problem/R10/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2_N50_R10.py#L88-L104) |

These are reasons not to reuse the old workers/checkpoints as the new protocol.
The full-SAA fold initialization is validation leakage in that legacy CV; it
does not occur in the audited training-only SAA initialization for new green.

### D. Larger-N optimizer/checkpoint protocols are not already consistent

The historical [R3 N800](../retailers_problem/R3/N800/rho_sweep_net16_v2/run_rho_seed_net16_v2_N800.py#L116-L156)
and [R10 N800](../retailers_problem/R10/N800/rho_sweep_net16_v2/run_rho_seed_net16_v2_N800_R10.py#L117-L166)
MP trainers use **minibatch Adam with batch size 200**, unlike current green's
full-batch Adam. Their mean batch losses are evaluated at changing weights,
then used to select an end-of-epoch checkpoint. That is not the current
fixed-checkpoint full-pool criterion. These MP functions are not SGD; the
optimizer of a different nominal/CV routine must not be attributed to them.

Shipping N800 also has explicitly
[superseded full-batch artifacts](../retailers_with_shipping/N800/rho_sweep/STALE_README.md).
The old launcher points at a worker currently in that stale directory; it
must not be trusted to regenerate the corrected historical minibatch campaign.
No old shipping N800 launch command was executed during this review.

## 3. Qualifications, not reasons to discard the current green figure

1. **Stopping is not a certificate for the returned policy.** At rho=0.2,
   seed 1 returns candidate 1 but stops at 3; seed 7 returns candidate 1 but
   stops at 2. The former checkpoint need not satisfy the latter's gap.
   Mean closure also does not imply closure at every center or global neural
   optimality. Keep these identities/diagnostics distinct.
2. **Numerical incumbents are not exact robust optima.** AP bounds concern
   its folded numerical formulation; native replay checks returned points,
   not a uniform floating-point error bound over the entire uncertainty set.
   Keep existing strict acceptance gates and avoid overstating certification.
3. **Epoch caps are budget completion, not optimizer convergence.** Do not
   relabel a capped training call or capped CCG trajectory as converged.
4. **The test and historical grid have been inspected.** The current curve
   remains descriptive. Extending it consistently does not retroactively
   make it confirmatory. A fresh test is a separate protocol decision; any
   benchmark comparison would need evaluation on the same new draw.
5. **Historical red/yellow references are not matched retraining baselines.**
   Their training/selection budgets and status evidence differ. Conditional
   kNN leave-one-out interpolation is not leave-one-out refitting of the full
   SAA pipeline. Do not select radii to win plotted test comparisons.
6. **Reporting correction:** cached SRO-FA partition/LP/evaluation/total time
   fields do exist in the [historical cache](../retailers_problem/R3/N50/sro_fa/sro_fa_results_N50_R3.json#L1-L17).
   Earlier blanket wording that SRO-FA timing was unavailable was too broad.
   Fresh matched timing and complete historical solver certification remain
   unavailable; cached reported durations must not be represented as newly
   measured, comparable campaign time. Frozen reports are preserved.

## 4. Proposed consistent specification — not yet implemented or approved

Treat N as the **total** per-seed neural data budget everywhere:

| Total N | Training | Validation | Split |
|---:|---:|---:|---|
| 50 | 40 | 10 | One fixed paired 80/20 split |
| 200 | 160 | 40 | Same rule |
| 800 | 640 | 160 | Same rule |

Apply this to R3/no-shipping, R10/no-shipping, and R3/fixed-shipping, while
preserving the legitimately different physical cost functions. Proposed common
choices follow current green rather than inheriting historical variant defaults:

- Data seeds 0–9, network seed 0, split seed 100000+data seed; train-only fresh
  SAA initialization appropriate to each physical model; no final refit.
- Two hidden layers of width 16, input/output dimensions appropriate to R;
  native Moreau projection everywhere, zero capacity buffer, no post-solve
  clipping or silent solver substitutions.
- Fresh **full-batch Adam at all N**, train-selected preupdate checkpoints,
  maximum 3,000 evaluated epochs, patience 300, same threshold/scheduler/clipping
  and inventory LR as the current trainer. If this is too expensive at N800,
  stop and discuss a batch protocol rather than switch silently.
- Nominal LR candidates 0.005/0.01/0.05, fixed-checkpoint validation selection;
  selected LR inherited by MPs. Validation is not used to choose MP epochs.
- Same raw per-coordinate nonnegative box-radius grid
  0, 0.001, 0.003, 0.01, 0.03, 0.2, 0.4, 0.7, 1.5; no R-dependent rescaling or
  new radii based on test curves. This fixes the uncertainty-set definition,
  not a claim of equal statistical coverage in dimensions 6 and 20.
- Assess the initial AP, stop at first mean-gap <1%, return earliest minimum
  mean-AP candidate through that assessment. At most 40 MPs plus final AP.
  **Proposed conservative cap policy:** preserve a nonconverged/failure record
  and withhold an authoritative full-grid comparison if no valid green stop
  occurs; no replacement checkpoint or continuation beyond the cap.
- Require all center APs and strict replay/status/membership checks. Measure
  AP and pool costs under the same declared physical objective, with any
  shipping training surrogate explicitly distinguished.
- Freeze returned policies before per-seed validation radius selection; freeze
  radius choices before test evaluation. IS remains returned-checkpoint
  own-pool cost; strict Gurobi OOS evaluation has no fallback.
- Initially retain historical benchmark lines only as descriptive references,
  unless a matched benchmark/test campaign is separately approved. No automatic
  new kNN/SRO-FA training or fresh-test generation is part of this proposal.
- Preserve monotonic per-model, per-iteration AP/MP, nominal initialization,
  LR selection, evaluation and total timings. Record worker/thread counts,
  seeds, timestamps, statuses and completed vs capped/failed cohorts. Use
  named tmux sessions and persistent logs for actual launches.

Before the first new experiment: dimensional R=3/R=10 tests, shipping
cost/activation agreement tests, same-state pool/checkpoint replay, initial
stop and min-AP-return tests, failure/cap tests, train/validation boundary
checks, and timing/provenance checks. These are implementation-validation
requirements, not permission to tune on evaluation outcomes.

## 5. Runtime and review scope

This was a read-only source/artifact audit followed by documentation. Audit
wall time was **not instrumented and is unavailable**; no experiment duration
is inferred from it. New nominal, CCG, CV, SRO-FA and evaluation runtimes are
all **not run**, not zero-second completed experiments. Existing green runtime
evidence remains in the [completed comparison](N50_R3_train40_val10/stopping_only_min_ap/results/COMPLETION.md)
and [finite-pool evaluation report](N50_R3_train40_val10/stopping_only_min_ap/paper_figure/results/RESULTS.md).

| Requested future work | Fresh measured runtime/status |
|---|---|
| Nominal training / LR selection for shipping and R10 | Not run; blocked pending alignment |
| Prospective green CCG for shipping and R10 | Not run; live controller required |
| N200/N800 shared-protocol extensions | Not run; specification not yet approved |
| IS/OOS evaluation / CV | Not run during this review |
| SRO-FA partition / solver / evaluation / total | No new run; historical reported timing fields exist, not reaudited |

**Next decision:** approve the common specification and resolve shipping's
cost/activation convention before implementation and any long run. The
current figure need not be changed merely because these extension blockers
were identified.

## 6. User clarification: smooth MP, hard AP

The user subsequently agreed with the shared-protocol direction and explicitly
requested **smooth shipping cost for MP training and hard shipping cost for
the AP**. This is an intentional surrogate-training method, not an instruction
to replace the differentiable MP objective with a binary charge. Preserve the
existing smooth form, f=0.1 and gamma=5; training checkpointing/scheduling use
that smooth training objective.

For a meaningful physical-cost stopping diagnostic, the planned pool assessment
reevaluates the same fixed policy on its current pools with the **hard** cost.
The mean-gap comparison is hard AP versus hard finite-pool mean, not hard AP
versus the smooth training loss. AP replay, minimum-AP return, validation,
reported finite-pool IS and OOS must follow the same hard-cost convention.
This adds no hard-cost gradient to the MP. Smooth-objective improvement need
not improve hard cost; no hard robust optimum or convergence guarantee follows.

Full-batch Adam remains the recommendation for consistency at total N=800
(640 training centers), **not a measured claim of better speed or solution
quality**. The augmented pools, not just the number of centers, determine MP
work and memory. An implementation/resource preflight is required before a
large run; do not silently switch to minibatching if it is infeasible.

The user subsequently chose **any positive allocation**, f times1{q2>0},
with no threshold or positive minimum shipment. This settles the physical
cost, not its exact AP implementation. Removing the AP lower link alone lets
a maximizing AP charge for zero shipments. The strict-positive graph is not
closed and a worst-case maximum need not be attained; shipping remains paused
pending a sound certification/witness contract. No epsilon approximation or
paid-zero relaxation has been authorized. This mathematical issue has not
been demonstrated on the actual saved policies.

## 7. Authorized launch order

The user requested **N200 and N800, base R3/no shipping first**, with as much
safe parallelism as possible. This supersedes the earlier proposed N50-first
order. The isolated implementation and launch protocol are in
[green_R3_N200_N800_20260923/README.md](green_R3_N200_N800_20260923/README.md).
Only that no-shipping launch is proceeding; shipping's formulation remains
paused. N200/N800 retain full-batch Adam and the common 80/20 splits. Actual
preflight status and resource-selected worker limits are recorded alongside
the new campaign rather than inferred from old experiment runtimes.