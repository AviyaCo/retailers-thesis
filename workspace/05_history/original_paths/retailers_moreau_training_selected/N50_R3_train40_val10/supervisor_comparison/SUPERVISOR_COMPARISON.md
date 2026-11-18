# Supervisor comparison: four neural retailer methods

**16 September 2026 — R=3, no shipping charge, nominal sample budget N=50.**
This is a retrospective, descriptive comparison. No new training or test-set
optimization was performed for this report. Green's fixed-policy validation
selection was newly computed; the other methods' saved selections were retained.

## Main result: radius selected by validation, not by test

Each seed chooses its own radius; there is no single globally selected radius.
The following means use the same archived 10,000 scenarios, seed9999, and the
same unbuffered Gurobi projection evaluator. Lower cost is better.

| Curve | Method | Training / validation observations per seed | Mean test cost |
|---|---|---:|---:|
| Blue | Historical SCS / training-selected epochs / historical CCG | 50 / 13 separately generated | 20.412406972 |
| Purple | Native Moreau / validation-selected epochs and CCG candidate | 40 / 10 | 20.819380984 |
| Green | Native Moreau / training-selected epochs / mean-gap stop, minimum-AP return | 40 / 10 | 20.547568019 |
| Black | Native Moreau / training-selected epochs / every-center stop, terminal return | 40 / 10 | 20.405601829 |

Black minus blue is **−0.006805143 cost units (−0.03334%)**. This is a small
observed difference, not evidence of statistically established superiority.
Black being below blue at all seven shared fixed radii is a different statement
from comparing per-seed validation-selected policies.

| Seed | Blue radius | Purple radius | Green radius | Black radius |
|---:|---:|---:|---:|---:|
| 0 | .003 | .2 | .4 | .4 |
| 1 | .2 | 0 | 1.5 | 1.5 |
| 2 | .2 | .7 | .4 | .03 |
| 3 | .45 | 0 | .2 | 0 |
| 4 | .45 | 0 | .001 | .4 |
| 5 | .4 | .03 | 0 | .03 |
| 6 | .001 | .2 | 0 | 0 |
| 7 | .15 | 0 | 0 | .2 |
| 8 | .15 | 0 | .4 | .4 |
| 9 | .2 | .2 | .2 | 0 |

Blue selects among23 positive radii (nominal is not a separate radius candidate).
The Moreau grid is0,.001,.003,.01,.03,.2,.4,.7,1.5, with nominal eligible.
All methods use data seeds0–9 and nominal network seed0. Moreau uses the same
disjoint40/10 split, permuted with seed100000+data seed. Blue validation seed is
88880+data seed. Blue thus has a larger data and radius-selection budget.

## Formulation and numerical solvers

All four target the same continuous no-shipping inventory problem:
7→16→16→3 ReLU allocation network, softplus first-stage inventories,
purchase/holding/shortage costs0.25/0.05/0.5. Only first-period demand enters
the recourse network; second-period demand enters the eventual cost, not the
allocation decision. First-stage inventory is initialized by a training-only
SAA LP (50 observations for blue,40 for Moreau).

For each training pair, the uncertainty set is a nonnegative six-dimensional
box of radiusρ. The MP minimizes the mean across training centers of the
maximum scenario cost in each center's accumulated pool. The AP fixes the
learned policy and maximizes its cost over the full local box, encoding ReLUs
with indicator constraints, projection KKT/complementarity with indicators,
and the three-term cost maximum with Gurobi general maximum constraints.

The intended projection is

$$q=\arg\min_{x\ge0,\,\sum_r x_r\le Q_{10}}\tfrac12\|x-z\|_2^2.$$

| Component | Blue | Purple / green / black |
|---|---|---|
| Differentiable projection | CVXPYLayers → conic canonicalization → SCS/diffcp; eps1e−4, max200000 solver iterations | Native Moreau0.4.0 CPU active-set QP, exact active-set differentiation, no warm-start/fallback |
| Numerical QP form | Squared-distance objective canonicalized through CVXPY | Direct QP P=I, linear term−z, constraints−x≤0 and sum(x)≤Q10; constants omitted |
| Training capacity | Q10+1e−6 | Q10 exactly |
| Arithmetic | Float32 network; numerical conic stack | Float32 network/softplus, float64 native projection; AP folded coefficients float64 |
| AP | Historical Gurobi MILP, one thread; inspected worker omits non-OPTIMAL centers rather than failing the sweep | Strict OPTIMAL for all40 centers, one thread, MIPGap1e−4, feasibility/optimality/integrality1e−9, NumericFocus3, pointwise native replay |
| Test evaluator used here | Common fresh unbuffered Gurobi QP per scenario, OPTIMAL only, no fallback | Same |

This is primarily a **different numerical representation and solution of the
same intended projection**, not a different economic model. The removed buffer,
arithmetic, checkpoint rules and AP gates are real protocol differences.
Current historical worker source does not establish every original run's exact
source revision. In particular, nominal checkpoint alignment was subsequently
corrected in one canonical trainer, not retroactively in all saved runs.

## Explicit training, CCG stopping and returned-policy rules

All use full-batch Adam, fresh optimizer per nominal arm/MP, inventory LR.01,
gradient norm cap1, at most3000 evaluated epochs per training call, and
patience300. ReduceLROnPlateau uses factor.7, patience150 and relative threshold
.001; checkpoint improvement threshold is absolute.0001. These training
limits are distinct from the number of CCG/AP iterations.

| Rule | Blue | Purple | Green | Black |
|---|---|---|---|---|
| Epoch checkpoint / patience / scheduler metric | Training objective; historical pre-update-loss/post-update-state alignment limitation | Validation cost on10 rows; aligned pre-update checkpoint | Training objective; aligned pre-update checkpoint including incoming state | Same as green |
| Initial network LR | .05; historical training-based LR choice | Per-seed validation winner among.005,.01,.05 | Same LR grid; validation only after training each fixed checkpoint | Same as green |
| Validation during MP/CCG | Not used for training | Every epoch and every assessed CCG candidate | No validation inside training or CCG | No validation inside training or CCG |
| CCG stop | Historical mean AP-vs-recorded-MP discrepancy<1%, first AP ineligible; current worker absolute, dedicated timing benchmark signed | Every-center bound/pool residual≤1%; initial candidate eligible | Absolute aligned mean AP/pool discrepancy<1%, only after at least1 MP | Every-center bound/pool residual≤1%; initial candidate eligible |
| Returned CCG checkpoint | Minimum-AP incumbent under saved selector | Minimum validation cost among assessed candidates | Earliest minimum AP cost among candidates up to the stopping point | Terminal fully assessed candidate |

For green, with A_i the AP incumbent cost and P_i the **same fixed policy's**
current-pool maximum, stop after a complete post-initial AP sweep when

$$\frac{|\bar A-\bar P|}{\max(|\bar P|,10^{-9})}<0.01.$$

For purple/black, with B_i the maximization solver's upper bound, stop when

$$\frac{\max(0,B_i-P_i)}{\max(1,|B_i|,|P_i|)}\le0.01\quad\text{for every center }i.$$

Both are fixed-policy consistency/separation diagnostics, **not global neural
optimization guarantees**. Green is similar to the historical stopping idea,
not a byte-for-byte reproduction of SCS. Its returned minimum-AP candidate
need not itself meet the stopping candidate's criterion.

Purple/black insert numerically violating, exact-nonduplicate incumbents and
also stop with an explicit stagnation status on repeated policies or no new
violating points. Both have a40-MP cap plus terminal assessment. Green retains
that maximum budget but applies the mean-gap rule; radius0 is nominal-only.
No historical all-experiment uniform cap40 claim is made.

**Observed status matters:** purple has90/90 separated terminal trajectories,
but47/90 returned validation-best candidates are not every-center separated;
3 of its10 validation-radius winners have that status. Black has89/90 separated
trajectories and one capped case (seed9,ρ=.7,40 MPs/41 AP sweeps); that capped
case is not among its10 validation-radius winners. Green completes every
positive-radius mean-gap trajectory;6/10 validation-radius winners are not
every-center separated. This is not the same as physical infeasibility.

## Agreement with Gurobi: what is actually established

1. **Common test evaluator:** blue230, purple90, black90, green90 saved policy
   evaluations,10,000 Gurobi QPs each; all5,000,000 recorded solves accepted as
   OPTIMAL. This makes the reported evaluator common; it does **not** prove
   the original differentiable backends produce identical allocations.
2. **Legacy matched nominal audit:** seed0,50 training rows, three checkpoints,
   buffer1e−6. At SCS-selected weights, maximum SCS/Gurobi allocation-coordinate
   error is1.3581295; Moreau/Gurobi error at those same weights is1.0289e−11.
   SCS own mean cost19.55558434 versus Gurobi19.56465047. Thus “solved” status
   did not ensure high physical accuracy in this audited SCS case. This is not
   an exhaustive audit of every blue checkpoint.
3. **Unbuffered nominal Moreau projection audit:** ten prior nominal checkpoints,
  40 rows each, detached float64 inputs; largest forward error against Gurobi
  8.8058e−8. Projection directional-FD errors reached4.1936e−7 for targets and
  1.8448e−6 for capacity. This is projection-only evidence, not a universal
   full-network gradient guarantee.
4. **AP/native cost replay:** largest normalized discrepancy
   |AP−native|/max(1,|AP|,|native|): purple7.7556e−7 over6280 AP solves;
   black1.7255e−6 over27120; green completion5.1315e−7 over1240 **new** solves.
   All below the1e−4 replay gate. Reused green prefixes and copied initial APs
   are not counted as new solves. Pointwise replay is not uniform equivalence.

## Runtime table to attach

### Nominal training, separate from CCG

Seconds; n10 winning/fixed-LR runs. Full LR-search elapsed includes rejected
arms and parallel scheduling, unlike winning training time.

| Method | Mean | Median | Min–max | Mean epochs | LR-search campaign elapsed |
|---|---:|---:|---:|---:|---:|
| SCS separate nominal timing rerun | 347.42 (5.79min) | 369.97 | 243.75–405.27 | 2525.5 | Separate historical selection; not included in this mean |
| Purple | 14.54 | 12.78 | 6.78–25.60 | 1123.0 | 56.55s,30 trainings,10 workers |
| Black / green shared initialization | 37.62 | 37.24 | 29.63–45.88 | 2903.2 | 121.80s,30 trainings,10 workers |

### Per-radius robust work, excluding nominal training and OOS

**Mean seconds.** SCS is the dedicated timing experiment (n3, seeds10–12),
CCG AP+MP sums. Purple/black are measured per-job CCG wall time (n10 unless
stated). Green is a reconstructed sum of measured prefix phases, **not fresh
whole-run elapsed**. These columns must not be turned into a solver-only speedup.

| Radius | SCS AP+MP | Purple wall | Green prefix phases | Black wall |
|---:|---:|---:|---:|---:|
| .001 | 922.87 (15.38min) | 1.37 | 45.95 | 2.03 |
| .003 | Not measured in dedicated grid | 1.41 | 42.65 | 2.03 |
| .01 | Not measured in dedicated grid | 1.39 | 45.62 | 1.95 |
| .03 | Not measured in dedicated grid | 9.64 | 42.63 | 98.54 |
| .2 | Not measured in dedicated grid | 13.38 | 160.83 | 715.09 |
| .3 | 4791.73 (79.86min) | Not in grid | Not in grid | Not in grid |
| .4 | Not measured in dedicated grid | 10.37 | 392.62 | 1412.57 |
| .7 | Not measured in dedicated grid | 10.14 | 604.88 | 4017.67 separated n9;11411.63 capped n1 |
| 1.5 | 9171.80 (152.86min) | 9.41 | 837.72 (13.96min) | 3859.72 (64.33min) |

| Radius | Purple MP mean (range) | Green MP mean (range) | Black MP mean (range) |
|---:|---:|---:|---:|
| .001/.003/.01 | 0 | 1 (1–1) | 0 |
| .03 | 1.4 (1–3) | 1 (1–1) | 1.8 (0–3) |
| .2 | 1.6 (1–3) | 2.2 (2–3) | 6.6 (4–12) |
| .4 | 1.4 (1–3) | 4.1 (3–7) | 10.2 (8–15) |
| .7 | 1.2 (1–2) | 5.6 (4–9) | 18.67 (9–28), separated n9;40 capped n1 |
| 1.5 | 1.1 (1–2) | 7.4 (4–17) | 19.4 (9–35) |

AP sweeps are MP count+1 for these completed Moreau paths. Selected iteration
is not generally the amount of work executed. Purple's fast run made only67
MP calls overall;47 lasted301 evaluated epochs. Its90-job CCG sweep elapsed
56.066s with16 concurrent workers; black's elapsed14942.580s (4.15h),588 MPs.
Green's31 missing-point continuations elapsed147.204s with16 workers; the
complete current invocation including evaluation/reporting took435.583s.

**SCS timing exists.** The dedicated n3/radius run gives mean SAA+nominal+CCG
totals21.36,85.90,158.86min at.001,.3,1.5. It uses time.time rounded.01s,
nine concurrent jobs and a signed stopping diagnostic. Separately, ten
selected-radius CCG reruns average2646.286s (44.10min), median2128.967s,
range449.094–6244.696s, nominal excluded. Those launched with30 jobs across
three sample sizes. Exact durations of all230 originally plotted SCS policies
remain unavailable; zero placeholder fields are not runtimes.

### Evaluation, selection, other methods

| Component | Measured duration / availability |
|---|---|
| Common OOS per policy,10k QPs | Blue4.926s; purple4.920s; green4.924s; black4.945s mean;8 evaluation workers |
| Blue+purple320-policy evaluation invocation | 203.398s elapsed |
| Black+initial-green149-policy evaluation invocation | 106.857s elapsed |
| Green completion31-policy evaluation parent phase | 231.606s elapsed; includes substantial controller/executor overhead, cause not fully instrumented |
| Black terminal90-policy validation | 1.659s elapsed |
| New green90-policy val10 phase | 1.679s elapsed, one worker,900 native solves; selection.002035s separately |
| Current completed evidence builder | 5.448s elapsed; no training/AP/test solves |
| First failed evidence assembly | 5.311s, preserved separately; missing historical timestamp key, not a model failure; identical costs/selections |
| SRO-FA partition/solver/evaluation/total | Not measured in these neural campaigns; unavailable, not zero |
| CV | No new CV; fixed splits. Blue historical LR selection and radius-validation timing not reconstructed here |

Moreau runs: shared Linux host gold-srv2, Xeon Gold6354,72 logical CPUs,
ap_env Python3.11.13, PyTorch2.10.0, Moreau CPU0.4.0, Gurobi13.0.1;
one numerical/solver thread per worker. Historical SCS execution metadata
are incomplete. Never sum concurrent model wall times as sweep elapsed.
Full n/mean/median/min/max statistics by radius and status, raw per-model
durations, timestamps where available, hashes and selected costs are preserved
in the linked structured evidence. Green whole-run times remain null.

## Decision for the meeting

- The poorer purple result is **consistent with** heavy reuse of a small
  validation set, but this comparison does not isolate or prove that cause.
  Validation checkpointing is not inherently wrong; here it differs from the
  intended training-only checkpoint protocol and also affects CCG candidates.
- Black gives the lowest observed validation-selected test mean, essentially
  tied with historical blue at this precision of comparison; it costs much
  more robust training work than green. Green's selected mean is higher but
  it saves measured work at moderate/large radii. At tiny radii it can be slower
  than black because green requires one MP and black can stop immediately.
- Do not rerun every experiment solely on the basis of this plot. A defensible
  next decision is which algorithm is intended: stricter per-center separation
  with terminal return (black), or average consistency with minimum-AP return
  (green). Approve that choice and equal training/validation budgets first.
- If the goal is a solver comparison, hold the split, initialization, LR/epoch
  selection, stopping, returned-policy rule, buffer and numerical acceptance
  criteria fixed. If the goal is replacing the full method, report it as such.
- Existing numerical evidence motivates a controlled extension, not automatic
  equivalence for other architectures, sample sizes, shipping or hydro.
  Predeclare treatment of capped runs and a new unseen evaluation draw if a
  confirmatory claim is required. No additional experiment is authorized here.

## Evidence and attachments

- [Completed graph](../short_rule_completion/matched_oos_completed.png)
- [Structured evidence, all runtime statistics](evidence_complete/summary.json)
- [40 selected policy identities, validation and test costs](evidence_complete/selected_policies.csv)
- [Green choices saved before test lookup](evidence_complete/green_selections_before_test_lookup.json)
- [Reproducible evidence builder](build_evidence.py)
- [Original detailed black/green/SCS timing report](../short_rule_completion/REPORT.md)
- [Dedicated N50 SCS timing correction](../short_rule_completion/SCS_TIMING_CORRECTION.md)
- [Purple training/controller](../../../retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/ccg.py)
- [Blue validation metadata](../../../retailers_problem/R3/rho_validation_results_net16_v2.json)
- [Legacy forward/gradient audit](../../../retailers_scs_vs_moreau/retailers_base/N50_R3/net16_v2/gradient_audit_results_v2/audit.json)
- [Unbuffered projection audit](../../../retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/scs_qp_feasibility/campaign_20260909/summary.json)

The active supervisor manuscript, original graph values and frozen experiment
sources are unchanged. Radius-selected test results are descriptive: the
seed9999 draw was viewed before this comparison was finalized.