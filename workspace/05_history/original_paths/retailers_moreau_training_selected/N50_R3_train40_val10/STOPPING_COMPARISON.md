# Why SCS stopped sooner than this Moreau campaign

## Main explanation: different termination requirements

This is **not a solver-only comparison**. Matching the training-best checkpoint
rule did not match the entire SCS CCG algorithm. The new controller retained the
previous Moreau controller's every-center bound-based stopping criterion; the
historical SCS loop used one mean-cost diagnostic. This distinction should have
been made explicit when describing the training-selected rerun.

Historical SCS stopping, after the first AP sweep:

$$\frac{|\overline A-L_{\mathrm{stored}}|}{\max(|L_{\mathrm{stored}}|,10^{-9})}<0.01.$$

Current Moreau stopping requires **every one of the 40 centers** to satisfy

$$\frac{\max(0,B_i-P_i)}{\max(1,|B_i|,|P_i|)}\leq0.01.$$

Here A is an AP incumbent cost, B an AP upper bound, and P the current
checkpoint's maximum cost on a center's pool. A small average discrepancy can
hide larger discrepancies at a few centers. The normalizations also differ;
the tests are not mathematically identical or strictly nested in every case.
Neither condition establishes global optimality of the nonconvex neural MP.

### Direct evidence from the same Moreau trajectory

Seed 9, radius 0.7; all figures below use aligned, saved training/AP assessments,
not validation scores. Candidate k has completed k MPs and is assessed by AP
sweep k+1. The mean diagnostic was recomputed from each candidate's current
pool and AP mean, not a potentially misaligned historical loss.

| MPs completed | Mean AP–pool discrepancy | Worst-center discrepancy | Centers above 1% | Actual decision |
|---:|---:|---:|---:|---|
| 8 | 0.534682% | 9.240720% | 9 | Continue |
| 39 | 0.176325% | 4.296220% | 3 | Continue |
| 40 | 0.175097% | 2.845885% | 2 | Stop at budget, **not separated** |

The first eligible mean-diagnostic pass on the observed path was **after 8 MPs**.
At that point the pool/AP means were 18.419788731 and 18.518276006. The worst
center had pool cost 14.320359290 and AP bound 15.778396704. Its bound equalled
its incumbent to floating-point roundoff; the discrepancy was not an unresolved
solver bound. The worst center after 39 and 40 MPs likewise had zero recorded
bound–incumbent gap. These are genuine residuals under the every-center rule,
not evidence that Moreau intrinsically requires 40 iterations.

This is a retrospective diagnostic on the observed trajectory. It does not
reconstruct how SCS would train from the same initial policy, nor justify
reselecting candidate 8 using held-out scores. No stopping rule or selected
checkpoint has been changed. An average-objective stopping protocol may be a
reasonable future choice for the mean robust objective, but needs explicit
approval and must retain complete AP coverage, replay/membership gates and
same-checkpoint loss alignment. Numerical validity checks are distinct from
the outer optimization stopping target.

## Counts: "under 8" and "40" are not universal

The historical N50 **selected-radius timing** cohort has 2–6 AP sweeps and
1–5 actual MP trainings, not the full radius grid. In seed order 0–9 its AP
counts are 2,6,4,6,5,5,3,2,2,3. The full-grid SCS seed9/radius0.7 record has
**10 AP sweeps and 9 MPs**, so even that SCS configuration was not under 8.

Some historical `ccg_iters` fields denote the selected incumbent's MP index,
not total work. For example, selected-timing seed6/radius0.001 reports index0
but executed 2 MPs and 3 APs before returning the nominal incumbent. Historical
schemas differ; count trajectory entries and terminal reasons rather than
assuming a uniform meaning for that field.

The new campaign completed all 90 jobs: **89 separated and one capped**.
Only seed9/radius0.7 exhausted 40 MPs; its final AP was sweep 41.

| Radius | Seeds | MP minimum | MP median | MP maximum | Separated / capped |
|---:|---:|---:|---:|---:|---|
| 0 | 10 | 0 | 0 | 0 | 10 / 0 |
| 0.001 | 10 | 0 | 0 | 0 | 10 / 0 |
| 0.003 | 10 | 0 | 0 | 0 | 10 / 0 |
| 0.01 | 10 | 0 | 0 | 0 | 10 / 0 |
| 0.03 | 10 | 0 | 2 | 3 | 10 / 0 |
| 0.2 | 10 | 4 | 6 | 12 | 10 / 0 |
| 0.4 | 10 | 8 | 10 | 15 | 10 / 0 |
| 0.7 | 10 | 9 | 20.5 | 40 | 9 / 1 |
| 1.5 | 10 | 9 | 18.5 | 35 | 10 / 0 |

## Other differences prevent a solver-only causal claim

| Component | Historical SCS | Current Moreau |
|---|---|---|
| Training sample | All 50 observations | 40 training, 10 reserved for selection |
| Initialization/LR | Full50 SAA; historical network LR0.05 | Fresh train40 SAA; per-seed LR from three completed trainings |
| Projection | CVXPYLayers/SCS, eps1e-4; capacity buffer1e-6 | Native Moreau active-set, exact derivative mode; zero buffer |
| Scenario insertion | Adds all successful AP points | Adds numerically violating, nonduplicate points |
| MP checkpoint association | Inspected sweep source stores post-update state for pre-update loss | Evaluated pre-update state, incoming eligible, replay alignment checked |
| Returned policy | AP-mean incumbent | Terminal fully AP-assessed policy |
| AP integrity | Inspected source can omit nonoptimal centers | All40 required, explicit status/replay/membership checks |

The inserted-point threshold is the numerical 1e-4 gate, not exclusively the
outer 1% separation threshold. Both trainers use the same broad Adam/patience/
epoch-cap schedule, so that alone does not explain the gap. New selected LRs
are 0.005 for five seeds, 0.01 for four and 0.05 for one; seed9 uses 0.01.

Historical checkpoint misalignment and omitted AP-center risk are established
in inspected source, not proven to explain any specific historical result;
the exact source revision for each old run was not recovered. The current
nominal trainer and historical sweep worker must not be conflated. Moreau's
different numerical derivatives can change optimization, but no matched
solver-only experiment here quantifies their effect on outer iterations.

## Runtime and evaluation disposition

| Component | Wall time | Scope/status |
|---|---|---|
| New nominal LR-search phase | 121.796206 s (2.029937 min) | 30 fresh trainings, SAA/setup and LR selection, 10 workers |
| New selected nominal training | Mean37.620406 s; median37.241888; min29.626559; max45.877120; n10 | Winners only; rejected LRs excluded |
| New robust CCG sweep | 14942.580206 s (4.150717 h) | 90 jobs, 16 single-thread workers; 89 separated, 1 capped |
| Capped seed9/radius0.7 | 11411.629000 s (3.169897 h) | Current-job wall; 40 MPs, 41 APs, not convergence |
| Final fixed-policy validation | 1.658737 s | All90 frozen policies; 1 worker |
| New campaign overall | 15092.965248 s (4.192490 h) | 08:06:36–12:18:09 UTC, 16 September 2026; exit0 |
| Historical selected-radius N50 CCG | Mean2646.285778 s (44.104763 min); median2128.967273; min449.094197; max6244.696002; n10 | Different radii/protocol; not a matched speed test |
| Historical SCS seed9/radius0.7 total | Unavailable | Artifact's 0.0 is not a credible measured total; rerun required |
| This read-only count/mean-gap extraction | 1.428930 s | Snapshot through assessed candidate39; later terminal lookup not included |
| New SRO-FA / CV / independent test | Not run | No missing timings inferred |

Host gold-srv2, Intel Xeon Gold6354 Linux CPU. Historical selected-path timing
had up to 30 overlapping workers across sample sizes. Concurrent model walls
are not additive sweep elapsed; AP timings here include checks and repeated
persistence, not just solver time. Full per-radius timing mean/median/min/max
and separated-versus-capped cohorts are in [RESULTS.md](RESULTS.md).

All90 fixed-policy validations completed automatically after training. The
nominal validation mean is 18.948014549; per-seed radius-selected mean is
18.700114478 (1.3083% lower on selection data). This is not independent test
evidence: the validation set selected LR and radius. No positive fixed radius
improves the aggregate validation mean over nominal in this run; a clear
aggregate U-shape was not obtained. These final scores were not used for the
training-only stopping comparison above. Historical thesis results are unchanged.

## Evidence and reproduction

Compute the mean diagnostic from each saved assessed candidate's
`ap_incumbent_mean` and `restricted_mean`; compute the maximum and count above
1% from its `samples[*].residual`. Scan candidate IDs in order, requiring ID>=1
to match SCS's no-stop-on-first-AP convention. Count actual MP/AP trajectory
entries separately from the selected-policy index. No model replay is needed.

- [Exact comparison values and source hashes](STOPPING_COMPARISON.json)
- [Historical SCS loop](../../retailers_problem/R3/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2.py)
- [Historical selected nominal-index example](../../retailers_problem/R3/selected_timing_net16_v2_run2/N50/rho_0p001/seed_6_summary.json)
- [Historical SCS seed9/radius0.7](../../retailers_problem/R3/N50/rho_sweep_net16_v2/rho_0p700/seed_9_summary.json)
- [New CCG controller](ccg_training.py)
- [Per-center residual implementation](../../retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/ccg.py)
- [Completed campaign and all runtimes](RESULTS.md)