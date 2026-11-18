# Prelaunch methodological review — 2026-09-09

Status: **controller and parallel launcher implemented and tested; launch authorized**.
The user approved the five controller rules and full-batch Adam on 9 September
2026, then explicitly confirmed that the 40-MP budget is acceptable after
clarification of a request for a limit of 16. The approved budget remains
40 MP retrainings and at most 41 AP sweeps; the epoch cap remains 3000.
The controller and launcher now exist. This is a source-based
review, not a literature survey or a numerical reproduction of old results.

## Approved prospective design

- No-shipping R3, total development budget50, paired split40train/10validation,
  data seeds0--9, network seed0; split permutation seed100000+data seed.
- SAA initialization, gradients, AP centers and scenario pools must use train40
  only. Validation is shared across selection steps, never gradient fitting.
- Per-seed nominal network LR grid .005/.01/.05; inventory initial LR .01.
  Nominal selected LR reused for robust runs. Validation drives checkpoint,
  early stopping and scheduler. No refit on50 and no final test yet.
- Radius grid0,.001,.003,.01,.03,.2,.4,.7,1.5. Remove the capacity buffer
  everywhere in the new implementation; preserve all historical artifacts.
- Reuse only the completed new-protocol nominal LR-search winners, as explicitly
  approved after that search. Verify their saved splits, checkpoint hashes and
  source provenance. No historical-protocol weights or cached AP results.
  Report previously persisted nominal timing separately from this invocation.

## Findings requiring controller changes before launch

1. Canonical sweep `run_mp` appends paired AP outputs by ball ID without
   membership tests. An exact duplicate leaves the per-ball maximum objective
   unchanged, but adds computation; numerical/tie gradients can differ.
2. The historical sweep checkpoint may be postupdate while its stored loss is
   preupdate. Minibatch epoch losses also mix parameter states. The corrected
   new trainer avoids this, but the controller must reevaluate the complete
   restricted pool at each immutable AP candidate, not reuse a best-loss label.
3. AP minus an achieved neural MP objective is **not a global optimality gap**.
   Pool inclusion orders objectives at fixed parameters; Adam's achieved value
   is not a lower bound on the globally minimized robust problem. Across
   different approximate checkpoints, master losses need not be monotone.
4. Legacy AP aggregation skips nonoptimal samples. The new controller must
   require all40 valid results or record an incomplete/failed iteration, never
   compute a successful mean over the remaining samples.
5. Trajectory stop, selected candidate, and selected candidate's AP/separation
   status are distinct. The legacy minimum-AP incumbent must not silently
   override the approved validation selection. The final generated MP needs
   its own AP assessment if included in a certified candidate list.
6. Validation can restore an MP's incoming checkpoint. Repeating that candidate
   must be recorded as stagnation, not global convergence.

## Approved CCG contract — implemented and tested

- Maintain a pool per training observation containing full paired6-vectors,
  initially its nominal pair. Check exact membership before insertion; never
  merge across observation IDs. Log near duplicates separately; retain them
  unless a compression rule and tolerance are separately justified. Preserve
  raw float64 AP points and disclose float32 training collisions.
- For frozen candidate theta and the **pre-insertion** pool, compute
  m_i = max_{z in J_i} C_theta(z). Record AP incumbent a_i and upper bound b_i,
  candidate hash, pool hash, status, replay mismatch and times separately.
- After validating numerical ordering, require every ball to satisfy
  max(0,b_i-m_i)/max(1,abs(b_i),abs(m_i)) <= .01 for separation closure.
  Keep solver/replay consistency tolerances distinct from this threshold.
  This is stricter than the old mean criterion and is not global optimality.
- A material negative ordering violation is an integrity failure, not evidence
  of convergence. Raw signed discrepancies are never overwritten by clipping.
- Only novel, violating scenarios are needed as additions. A duplicated or
  float32-colliding adversary with a material residual requires diagnosis;
  duplicates alone do not certify optimization or separation closure.
- Distinguish separation_satisfied, stagnation, iteration_cap, and failure.
  Approved budget: at most40 MP retrainings plus their final AP assessment
  (at most41 AP sweeps); no unassessed terminal policy in selection.
- Select the validation-best assessed candidate within each radius (nominal
  reference included), then validation-best radius. Deterministic ties favor
  smaller LR/radius and earlier candidate. Report minimum-AP candidate only as
  a separate diagnostic. Do not certify a selected candidate based on another
  candidate's stop status. Do not silently shrink the grid after failures.

## Cross-experiment issues (do not silently fold into the R3 pilot)

- Different optimizers are not intrinsically invalid. Historical protocols
  differ and must remain described as executed. A new common full-batch Adam
  rule for train40, same per-seed LR grid and same selection budget is a
  reasonable prospective design, not a proven optimum. Larger sample sizes
  need a separately predeclared full-/minibatch rule. Equal epochs are not
  equal optimizer updates, processed examples, or elapsed compute.
- LR_Q=.01 is an initial rate; the plateau scheduler can reduce both groups.
- Shipping five-fold nominal scripts initialize all folds from full-dataset
  SAA, so folds are not clean for the whole pipeline. Train40-only SAA fixes
  this for the new split; a retained CV design would need fold-local SAA.
- Existing kNN LOO excludes the observation as a neighbor but reuses globally
  fitted SAA/targets. It is not full-pipeline LOO. New comparator policies must
  fit entirely on train40 and tune on the same validation10.
- The old R3/N50 `cv_rho_selection.py` calls the last10 held-out although models
  trained on all50. This is not genuine held-out validation. Later canonical
  selectors use separate independent validation; do not conflate the versions.
- Shipping uses smooth training costs, binary AP costs and threshold-based
  evaluation with different feasible conventions. This needs its own explicit
  policy/objective decision before shipping migration; the no-shipping pilot
  does not resolve it.
- Current shipping N800 launcher/worker do not by themselves reproduce the
  corrected minibatch artifact. Freeze source/config hashes for new campaigns.
- Validation10 reused for many choices produces a noisy tuning score, not an
  unbiased performance estimate. A fresh final test may be used only after
  freezing the whole comparison, with no tuning after viewing its outcomes.

## Prelaunch regression gates

Paired split and train-only SAA; exact zero-buffer projection and derivatives
through target/capacity; radius-zero AND positive-radius AP replay; complete
AP coverage; exact/near duplicate cases; candidate/pool loss alignment;
negative-gap failure; validation restoration stagnation; cap/final AP mapping;
failure persistence and no test imports; reproducible seed/thread handling.

The controller, duplicate rule, stopping and terminal-candidate mapping now have
regression coverage: 34 primitive/controller tests passed in20.558s, including
real radius-zero and positive-radius AP replay. Thirteen launcher tests passed
in4.901s. A reviewer found and the implementation corrected a publication bug:
radius selections remain provisional until final shared provenance verification
succeeds. One-bad-ball separation, failed-MP timing and batch lifecycle checks
also passed. Tests do not establish convergence of the actual campaign.

Launch scope:90 configurations (ten seeds, nine radii), at most16 spawned
workers, one numerical-library/Torch/Gurobi thread per worker, sequential APs
within a job. No hydro work, test evaluation, refitting or automatic next-stage
experiment. Wait for the full N50/R3 sweep before deciding what follows.

## Source evidence

- Historical R3 pool/stop: ../../../../retailers_problem/R3/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2.py
- Old training-subset selection: ../../../../retailers_problem/R3/N50/rho_sweep_net16_v2/cv_rho_selection.py
- Shipping CV: ../../../../retailers_with_shipping/N50/run_cv_nominal.py
- New primitives: core.py and adversary.py; fixtures test_core.py and test_adversary.py.