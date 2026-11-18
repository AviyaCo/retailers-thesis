# Epsilon AP launch gate — failed; full CCG not launched

25 September 2026. The user authorized a shipping run with the previous
epsilon AP lower link. The new AP was implemented and tested before launching
the full sweep. **The epsilon MIP solved, but one saved-policy witness failed
the unchanged native hard-fee replay rule.** No failed case was discarded.

## Setup and result

R3, total sizes50/200/800, saved seed0 nominal LR.01 (predeclared, not selected
using validation), training center0, radii0/.2. All six cases were attempted.
No training, validation, test evaluation, LR/radius selection, MP or CCG ran.
Native policy and its returned shipments were not modified. The hard cost
remains0.1 per native q>0. The AP adds1e-6*z<=q<=Q*z and0.1*sum(z).

- First attempt: failed a synthetic fixture assertion before any AP solve.
  The fixture multiplied an integer count into a float32 fee; the expected
  float64 value0.1 differed from0.10000000149011612. Only the fixture was
  corrected to use the production-like shipment-dtype conversion. The failed
  attempt is preserved, and no solver tolerance changed.
- Second attempt: **7/7 unit tests passed; 6/6 APs reported Gurobi OPTIMAL
  with zero reported MIP gap; 5/6 returned points passed acceptance.**
- All52 recorded source/provenance entries and12 protected saved-input entries
  were independently rechecked unchanged; standalone case records match the
  summary. No full shipping controller was integrated or launched after failure.

## Actual mismatch

N800, seed0, LR.01, original sample445 (training index0), radius.2:

| Quantity for retailer3 | Value |
|---|---:|
| AP shipment |0|
| Rational folded-policy projection |exactly0|
| AP shipping binary |0|
| Unchanged native shipment |1.7763568394002505e-15|
| Native q>0 fee |0.1|

AP and exact point-replay support are `[true,true,false]`; native support is
`[true,true,true]`. AP cost is13.12403051721862; native hard cost at the returned
point is13.224030014779418, a difference0.09999949756079829. The fee-component
difference is0.1; the remaining cost arithmetic differs by about-5.02e-7.
This is a witness-level comparison, not a computed worst-case native objective.
Recorded activation links, integrality and KKT residuals are zero. Thus the
observed issue is not a ghost fee in this epsilon MIP. The precise native
backend origin of the positive residual was not separately diagnosed.

The failed activation call preserved shipments and fees but did not persist
its projection iteration telemetry before raising. The later ordinary cost
replay block was not executed for that case; its duration is unavailable, not
zero. This diagnostic limitation does not remove the observed support mismatch.

## Measured runtimes

Linux gold-srv2, one worker and one numerical/solver thread. Raw monotonic
seconds, timestamps, solver iterations/nodes, seed and radius are in the JSON
artifacts. This is a short launch check, not a performance benchmark.

| Scope | Status | Wall seconds | Human time |
|---|---|---:|---|
| AP gate v1, whole child |Failed fixture; no APs|6.497982|6.50 s|
| AP gate v2, whole child |Failed acceptance|6.782098|6.78 s|
| v2 internal gate body |Failed acceptance|3.763859|3.76 s|
| v2 unit-test subprocess |Passed|2.582767|2.58 s|
| Nominal/SAA campaign, previously persisted |Cached; not rerun|6016.852093|1 h40 min16.85 s|
| Fresh nominal training |Not run|—|—|
| Robust neural/CCG total, including MP |Not launched|—|—|
| SRO-FA partition/optimization/evaluation |Not run|—|—|
| Validation/test/CV |Not run|—|—|

Nested gate/test/AP/solver/replay times must not be added to whole-child time.
The cached nominal/SAA total includes startup/teardown and its own sequential
stages (30 initialization then52 nominal workers); it is not current runtime.
Full cached per-model training statistics remain in the original
[runtime aggregates](../shipping_nominal_R3_N50_N200_N800_20260924/production_v1/runtime_aggregates.json).

| Ntotal | Radius | Acceptance | AP wall s | Solver wall s | Activation replay s |
|---:|---:|---|---:|---:|---:|
|50|0|Passed|0.491230|0.002368|0.015638|
|50|.2|Passed|0.055861|0.005434|0.003804|
|200|0|Passed|0.080125|0.002128|0.004449|
|200|.2|Passed|0.068881|0.003852|0.003726|
|800|0|Passed|0.018558|0.001585|0.003599|
|800|.2|**Failed**|0.066232|0.003587|0.003465|

Each passed method/size/radius group contains one observation: mean, median,
minimum and maximum equal its listed AP wall time. The failed N800/.2 case is
not included in any completed-run aggregate. No representative sweep timing
or expected convergence is inferred from these six checks.

## Remaining decision

The epsilon AP authorization did not change the native hard-fee comparison
q>0. A common numerical hard-fee cutoff q>=1e-6 would remove the extra fee for
the observed1.78e-15 residual, while charging at equality; this would be an
explicit change of the assessment convention, not a solver fix. It would not
remove the restricted-domain limitation of the AP or guarantee all other
boundary cases pass. Smooth MP and native shipment values could remain unchanged.
No such cutoff, clipping, tolerance relaxation or alternative policy was applied.
User approval is required before changing that convention and testing anew.

**Final user decision (25 September): retain exact native q>0 fees and leave
the run paused.** The proposed hard-fee cutoff was not approved. No further
preflight, full CCG launch, or change of shipment/fee semantics is authorized
by this turn.

Before full controller integration, preserve signed negative-gap rejection
before the absolute green stopping calculation, distinct smooth-training and
hard-pool accounting, complete-grid barriers, and failure-path telemetry.

## Reproduction / evidence

Working directory: this experiment directory.

- Session `shipping_epsilon_gate_v2_20260925` completed with child exit1; it is
  not an active full CCG session.
- Exact child command: `/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/shipping_epsilon_green_R3_20260925/ap_gate.py --output /data/home/aviyac/retailers_moreau_training_selected/shipping_epsilon_green_R3_20260925/ap_gate_v2`.
- Attach while running: `tmux attach -t shipping_epsilon_gate_v2_20260925`.
- Session status: `tmux has-session -t shipping_epsilon_gate_v2_20260925`.
- [Persistent merged log](ap_gate_v2.log), [launch record](ap_gate_v2.launch.json),
  [whole invocation](ap_gate_v2.invocation.json), [gate summary](ap_gate_v2/summary.json),
  [failed case](ap_gate_v2/N800_radius0.2.json), [unit tests](ap_gate_v2/unit.log).
- [Preserved first attempt](ap_gate_v1/summary.json).

Sources used in the gate are frozen in its provenance. Any revision requires
a new output/session and a fresh source-bound gate; do not overwrite these runs.