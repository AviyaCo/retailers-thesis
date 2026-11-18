# Shipping projection diagnosis — reproduced, not repaired

25 September 2026. **The diagnostic completed; full shipping CCG remains
blocked.** No fee, solver setting, tolerance, checkpoint, training rule or
acceptance condition was changed. Reference projections are diagnostic only.

## Setup and coverage

- Saved seed-0, LR-0.01 nominal policies; Ntotal 50/200/800, with 40/160/640
  training centers, and saved AP witnesses at radii 0 and 0.2.
- All 1,680 saved witnesses, not a selected failure subset, replayed twice
  from fresh model state: 12 configuration executions.
- Each execution: one original-center batch, all saved AP singleton calls in
  their historical order, then separate fixed-target singleton projection
  controls. No control was interleaved into the historical sequence.
- Passive hooks captured actual float32 network targets/capacities and returned
  float64 shipments. They neither replaced outputs nor changed numerical inputs.
- Exact rational sorting projection of the represented native target/capacity
  separates target disagreement from projection error. It is not substituted
  for any production shipment or cost.
- 8 regression tests passed, including hook equivalence, state preservation,
  reference boundary cases, unchanged raw values and exact dtype promotion.

Both repetitions reproduced **every saved native shipment vector and hard
cost**, and all six saved hard-pool arrays exactly. Repeated batch, singleton
and fixed-target outputs also matched. These repetitions establish local
repeatability, not independent statistical samples or global robustness.

## 1. Batch-dependent network targets, not sampled projection batching

At radius zero, raw AP points equal their original centers. Capacities also
match; comparison is therefore at identical input points and fixed weights.

| Ntotal | Centers | Different batch/singleton targets | Different native supports | Different exact-reference supports |
|---:|---:|---:|---:|---:|
|50|40|40|7|0|
|200|160|160|69|0|
|800|640|638|117|0|
|**Total per repetition**|**840**|**838**|**193**|**0**|

Maximum target difference was 4.57763671875e-5. This originates upstream of the
projection, in the float32 network evaluation path. The precise matrix-kernel
or operation-order mechanism was not traced. **Finite float32-to-float64
promotion is exact** and does not itself cause these changes.

All **3,360 fixed-target singleton controls** returned shipments bit-identical
to the corresponding batch shipments. Thus changing projection batch size
alone, with captured target/capacity fixed, did not reproduce a difference in
these controls. This is not proof about every possible backend workload.

The exact projections at radius zero retain identical support across batch
and singleton targets, but the floating projection sometimes leaves tiny
positive residues at exact zeros. Maximum original-center batch projection
error against its own exact reference was 5.684341886080802e-14. With strict
q > 0 fees, even such small residues trigger an entire additional 0.1 fee.
Batch/singleton hard-cost differences reached 0.20000000000000995; after
subtracting the observed fee-count difference diagnostically, maximum residual
was 1.4686584467682451e-5. No raw fee was actually removed.

**Example: N50, radius 0, training index 13.**

| Quantity | Original-center batch | Same-point singleton |
|---|---:|---:|
| Second network target |3.825476884841919|3.8254780769348145|
| Exact projected second shipment |0|0|
| Native second shipment |1.7763568394002505e-15|0|
| Raw shipping fee |0.3|0.2|
| Native hard cost |10.631364274024964|10.531364274024963|

Solving the captured batch target singly reproduces the tiny-positive batch
shipment exactly. This implicates target-dependent projection residuals and
the discontinuous fee, rather than an intrinsic projection-batch difference
in the tested path. The batch and singleton exact projected positive amounts
also differ slightly; their support remains the same.

## 2. Active-epsilon failures include genuine target disagreement

The native float32 evaluation and the AP's folded real-arithmetic policy need
not produce the same target. At positive radius, center and AP input points
also differ; their target difference must not be described as a batching effect.

**N800, radius 0.2, training index 60:**

- AP third shipment: 1e-6.
- Exact projection of the captured native target: **1/1572864**, approximately
  **6.357828776041666e-7**.
- Native returned shipment: **6.357828770120477e-7**.
- Maximum native projection error: **5.921189464667501e-16**.

The solver is already accurate here. The exact native projection is below
epsilon − tolerance = 9.99e-7. **Tightening projection tolerance cannot fix
this active-epsilon mismatch for the unchanged native target.**
[Captured witness](diagnostic_v1/N800_radius0.2_repeat0/sample_060.json).

**N800, radius 0.2, training index 74:**

- AP third shipment: 1e-6; folded exact reference is positive.
- Exact projection of the captured native target: **0**.
- Native returned shipment: **−6.357828770120477e-7**.

Two effects coexist: target disagreement removes the positive AP support,
and projection error returns a negative value instead of the native target's
exact zero. Correcting only the projection would still not recover the AP's
positive shipment for this target.
[Captured witness](diagnostic_v1/N800_radius0.2_repeat0/sample_074.json).

## 3. Moreau's physical allowance is looser than AP replay

Captured settings confirm Moreau 0.4.0 **active-set**, with
`primal_tol=1e-6`, `dual_tol=1e-12`, `zero_tol=1e-11`, `pivot_tol=1e-6` and
`diff_method=exact`. The native wrapper separately permits physical residuals
up to 1e-6. AP witness checks use 1e-9. None was changed.

Five unique positive-radius witnesses have native physical residual above
1e-9, up to 6.357828787884046e-7 in the full projection-error comparison.
These remain within the unchanged native wrapper allowance and are returned
with solved status. A negative shipment is nevertheless infeasible in exact
arithmetic. `diff_method=exact` is not an exact-arithmetic guarantee.

The AP is solved by **Gurobi**, not SCS. The historical `SCS` argument dictionary
is checked for compatibility by the native wrapper but does not select the
Moreau algorithm or its tolerances.

### Classification of the previous gate's first failures

Counts below use one repetition; the second reproduces them exactly. Columns
describe distinct comparisons and may overlap.

| Original first outcome | Rows | Native support differs from own exact projection | Native-target exact support differs from folded-AP exact support | Native physical residual >1e-9 | Exact native active shipment below epsilon−tolerance |
|---|---:|---:|---:|---:|---:|
|Accepted|1,486|241|0|0|0|
|Negative bound/pool|162|74|0|0|0|
|Active-epsilon failure|14|4|3|0|14|
|Unsupported support mismatch|18|0|18|5|18|

All 14 active-epsilon failures persist at the exact native-target projection.
All 18 unsupported support failures reflect a native-target versus folded-AP
support difference; five also show material native physical error. These are
not covered by the approved inactive-exact-zero to tiny-native-positive exception.

## Implication for the next decision

The current mismatch is not fixed by a larger consistency threshold or a
tighter Moreau tolerance alone. The numerical policy and AP must have a
consistent intended interpretation before a trustworthy launch.

- A projection improvement could address physical residuals, but cannot by
  itself fix float32/native-target versus folded-AP support or epsilon mismatch.
- Any exact-projection replacement, precision/execution change, numerical fee
  cutoff or broadened acceptance contract is a separate methodological change
  requiring approval and fresh validation. No such change is implemented here.
- No silent clipping, dropped cases, corrected-fee substitution or signed-gap
  waiver has been applied. Full CCG remains unimplemented/unlaunched in this work.

## Runtime

Linux gold-srv2, Python 3.11.13 in ap_env, Torch 2.10.0, Moreau 0.4.0; one
worker, one Torch/Gurobi-configured thread. Execution was CPU-only. Peak process
RSS: 748,879,872 bytes. UTC whole-child interval:
12:21:06.622357–12:21:33.328585 on 25 September 2026.

| Stage | Outcome | Seconds | Human time |
|---|---|---:|---|
|Whole diagnostic child, imports/teardown included|Completed|26.706225|26.71 s|
|Diagnostic body|Completed|22.295439|22.30 s|
|Unit-test subprocess|8 passed|2.391769|2.39 s|
|New nominal neural training / SAA|Not run|—|—|
|Robust neural/CCG total / MP|Not run|—|—|
|New AP optimization|Not run; saved witnesses reused|—|—|
|SRO-FA partition/solver/evaluation/total|Not run|—|—|
|Validation/test/CV|Not run|—|—|

Per-configuration wall time includes initialization, unchanged inference,
reference computation, fixed-target controls and row persistence. Every
configuration completed; no failure or cached-only configuration is pooled
with these timings. Both runs have the same saved seed and policy.

| Ntotal | Radius | Runs | Mean s | Median s | Minimum s | Maximum s |
|---:|---:|---:|---:|---:|---:|---:|
|50|0|2|0.150571|0.150571|0.138718|0.162424|
|50|0.2|2|0.136050|0.136050|0.135945|0.136155|
|200|0|2|0.524714|0.524714|0.476067|0.573361|
|200|0.2|2|0.470453|0.470453|0.469666|0.471240|
|800|0|2|1.999441|1.999441|1.930550|2.068332|
|800|0.2|2|1.942980|1.942980|1.875180|2.010779|

Summed measured substage work across sequential configurations: initialization
0.468377 s; native batch calls 0.064029 s; native singleton calls 2.825219 s;
fixed-target calls 1.656834 s; exact-reference computation 1.011426 s.
These are not additional time beyond the containing configuration/body/child
durations. Hooked calls include observation overhead and are not pure-solver
benchmarks. Singleton-sequence and fixed-control totals in JSON also include
reference work and must not be added to their nested call timings.

Cached nominal training per seed-0/LR-0.01 model: N50 79.761483 s, N200
130.390117 s, N800 165.776251 s. Those historical timings are not current
initialization/replay time. Raw per-call settings, solver iterations, status,
UTC endpoints and monotonic seconds persist in every configuration/row artifact.

## Reproducibility and independent review

- Completed tmux session: `shipping_projection_diagnostic_20260925` (exit 0).
- Working directory: this experiment directory.
- Exact child command: `/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/shipping_projection_diagnostic_20260925/diagnose.py --output /data/home/aviyac/retailers_moreau_training_selected/shipping_projection_diagnostic_20260925/diagnostic_v1`.
- Attach while active: `tmux attach -t shipping_projection_diagnostic_20260925`.
- Status: `tmux has-session -t shipping_projection_diagnostic_20260925`.
- [Persistent log](diagnostic_v1.log), [launch](diagnostic_v1.launch.json),
  [whole-child timing](diagnostic_v1.invocation.json), [summary](diagnostic_v1/summary.json),
  [tests](diagnostic_v1/unit.log), [provenance](diagnostic_v1/provenance.json).
- All 1,772 frozen source/input/evidence bindings rechecked; learned state and
  prior artifacts unchanged. Independent review parsed all 3,374 diagnostic
  JSONs and matched all 3,372 case/row mirrors. Review found no blocker for the
  diagnostic conclusions, but explicitly did not clear full CCG launch.
- The exact-reference experiment diagnoses represented values and these fixed
  execution paths. It does not certify a mathematical global robust objective,
  identify a particular BLAS kernel, or establish out-of-sample performance.