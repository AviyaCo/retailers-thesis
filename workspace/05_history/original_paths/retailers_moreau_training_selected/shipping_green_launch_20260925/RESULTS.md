# Shipping launch blocked by full-center readiness

25 September 2026. The requested full launch was **not started** because the
initial-policy integration gate failed. No CCG loop, MP training, validation,
test evaluation, selection or new nominal training was performed. All previous
sources, checkpoints and results are unchanged.

## Executed setup and decision

- Fixed saved shipping policies: seed 0, LR 0.01, Ntotal 50/200/800, with
  40/160/640 training centers; radii 0 and 0.2. All 1,680 planned APs executed.
- The previous 21 replay tests and 7 new integration tests passed.
- All 1,680 Gurobi solves reported OPTIMAL (status 2).
- 1,486 rows passed controller validation; **194 failed**. All six size/radius
  configurations therefore failed readiness. No failing scenario was omitted.
- Hard pool costs use unchanged native fees for every returned q > 0; smooth
  MP objective remains separate. AP epsilon is 1e-6; replay-only zero-residue
  tolerance is 1e-9; other cost and signed-gap thresholds remain unchanged.

The generic validator now explicitly revalidates the approved exception from
logged support and cost evidence, rather than rejecting every raw cost
difference. Its subsequent signed bound/pool, bound/incumbent, membership,
duplicate/collision and unresolved-bound checks are unchanged. The new tests
confirm that an eligible replay exception cannot waive a negative pool gap.

| Ntotal | Radius | AP count | Accepted | Negative bound/pool | Support mismatch | Active-epsilon failure |
|---:|---:|---:|---:|---:|---:|---:|
|50|0|40|35|5|0|0|
|50|0.2|40|36|1|0|3|
|200|0|160|100|60|0|0|
|200|0.2|160|150|1|5|4|
|800|0|640|550|90|0|0|
|800|0.2|640|615|5|13|7|
|**Total**||**1,680**|**1,486**|**162**|**18**|**14**|

These are first-failure labels, not disjoint physical causes. The AP wrapper
completed 1,648 rows and rejected 32 during activation replay; the controller
then rejected another 162. Solver success does not imply replay or controller
acceptance. Logged replay exceptions can coexist with later controller failure.

## Concrete blockers

1. **An accepted replay exception does not repair an invalid AP/pool ordering.**
   At N800, radius 0, training index 8 (original sample 284), AP incumbent and
   bound are 47.46027167146894, while pool and native replay cost are
   47.560274553298946. The native inactive-component shipment is
   1.7763568394002505e-15. The extra 0.1 fee is explained in replay, but the
   signed bound-minus-pool difference remains −0.10000288183000805.
   [Raw record](preflight_v1/N800_radius0/sample_008.json).
2. **Some mismatches are the opposite direction and outside the exception.**
   At N800, radius 0.2, training index 74 (original sample 184), AP shipment
   is 1e-6 with positive exact folded-policy projection, but the native
   shipment is −6.357828770120477e-7. This is not an inactive exact zero with
   a tiny positive residue. The later physical check is not reached because
   support classification already fails.
   [Raw record](preflight_v1/N800_radius0.2/sample_074.json).
3. **Positive support alone does not satisfy the active-epsilon check.**
   At N800, radius 0.2, training index 60 (original sample 766), AP shipment
   is 1e-6 and native shipment is +6.357828770120477e-7, below
   epsilon − feasibility tolerance = 9.99e-7.
   [Raw record](preflight_v1/N800_radius0.2/sample_060.json).

Do not attribute all 162 negative gaps to the first example: 74 have a replay
exception and 88 do not. Batched hard-pool evaluation and singleton native
replay also differ in some recorded costs at identical raw inputs. For
example, N50/radius0/index13 has pool 10.631364274024964 versus singleton
native 10.531364274024963. Pool shipment/support arrays were not saved, so the
underlying mechanism is unresolved. Moreau's solved status does not establish
exact arithmetic or physical accuracy at the required threshold.

These observations do not establish an incorrect Gurobi optimum: the AP
optimizes its epsilon-restricted folded policy, not a certified upper bound
on the unrestricted native floating-point hard-cost program.

## Runtime record

Linux host gold-srv2, Python 3.11.13 in ap_env, one worker, one Torch and Gurobi
thread. Whole child ran from 12:03:04.871026 to 12:03:57.645155 UTC. Peak
process RSS was 691,712,000 bytes. All current timings use perf_counter,
seconds, with UTC endpoints; raw AP solver iterations/statuses are persisted.

| Stage | Outcome | Seconds | Human time |
|---|---|---:|---|
| Entire readiness child, imports and teardown included |Failed, exit 1|52.774123|52.77 s|
| Readiness body |Failed|49.073605|49.07 s|
| Both unit-test subprocesses |28 tests passed|4.603615|4.60 s|
| New nominal training / initialization |Not run|—|—|
| Robust neural/CCG total, MP phases |Not launched|—|—|
| SRO-FA partition/solver/evaluation/total |Not run|—|—|
| Validation/test/CV |Not run|—|—|

The test and configuration times are nested inside the child duration, not
additional elapsed time. No concurrent models were run in this gate.

| Ntotal | Radius | Readiness status | Configuration wall s | Hard-pool evaluation s |
|---:|---:|---|---:|---:|
|50|0|Failed|1.736397|0.005261|
|50|0.2|Failed|0.827583|0.004214|
|200|0|Failed|3.173925|0.004652|
|200|0.2|Failed|4.272126|0.005772|
|800|0|Failed|14.204181|0.012916|
|800|0.2|Failed|16.678840|0.012774|

Each method/size/radius failed-run cohort has n=1, so its mean, median, minimum
and maximum equal the listed wall time. There are no completed-configuration
aggregates; accepted individual APs are not substituted for complete cases.

Previously persisted nominal training for these seed-0/LR-0.01 policies was
79.761483 s (N50), 130.390117 s (N200) and 165.776251 s (N800). These are cached
completed training times, not current loading or gate times. No fresh nominal
evaluation outside the training-only readiness scope was performed.

## Reproducibility and review

- Completed tmux session: `shipping_green_readiness_20260925`.
- Working directory: this experiment directory.
- Exact child command: `/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/shipping_green_launch_20260925/preflight.py --output /data/home/aviyac/retailers_moreau_training_selected/shipping_green_launch_20260925/preflight_v1`.
- Attach while active: `tmux attach -t shipping_green_readiness_20260925`.
- Status: `tmux has-session -t shipping_green_readiness_20260925`.
- [Persistent log](preflight_v1.log), [launch record](preflight_v1.launch.json),
  [whole-child timing](preflight_v1.invocation.json), [summary](preflight_v1/summary.json),
  [unit results](preflight_v1/unit.log), [provenance](preflight_v1/provenance.json).
- Independent read-only review reconciled every individual sample with both
  summaries, all input IDs/centers, pool hashes and failure arithmetic. All
  79 source/evidence-map and 6 input-map bindings matched; the latter six are
  included in the former, so there are 79 unique artifacts, not 85.
- No integration bug was found that overturns the blocked decision. Generated
  AST definitions can produce misleading historical traceback line locations;
  predicate reconstruction, not the displayed historical source line, was
  used to verify the checks. Frozen sources were not edited to correct this.

**Next decision:** investigate native projection and batched-versus-singleton
execution without changing the cost convention, or explicitly authorize a
revised numerical/modeling contract. Neither a general fee cutoff, a broader
exception nor a signed-gap waiver has been applied. The full CCG controller
still needs implementation and validation after these blockers are resolved.