# Consistency-only tolerance — verified

25 September 2026. Implemented the user-authorized numerical tolerance only
in AP/native replay acceptance. **21/21 regression tests passed; all six saved
policy APs were accepted: five ordinary passes and one explicit tolerated
exception.** `all_replays_consistent=false` is intentional: the raw costs of
the exceptional case are not equal.

## Scope and unchanged quantities

Same saved seed0/LR.01 nominal policies, Ntotal50/200/800, training center0,
radii0/.2. Fixed tests and settings; no new model fitting, validation/test,
selection, MP or full CCG. The previous failed artifacts and all historical
sources remain unchanged. Smooth MP, hard fee q>0, returned shipment values,
AP epsilon1e-6 and its solver settings are unchanged. No fee cutoff, clipping,
scenario omission or AP/pool stopping-rule waiver was introduced.

The consistency-only threshold is1e-9 (existing numerical feasibility
tolerance). A component can qualify only if AP binary inactive, AP shipment
exactly0, independent rational folded-policy projection exactly0, and native
shipment in(0,1e-9]. Every other mismatch still fails. All physical and active
epsilon checks still run. Same-call raw native cost is retained. The known
extra fee is subtracted only to explain the discrepancy; the unexplained
residual must satisfy the existing1e-4 cost tolerance on fee-free scaling.
Smooth training cost is not used as fee-free cost.

## Previously failing case

N800, radius.2, original training sample445 (training index0), retailer3:

| Evidence | Recorded value |
|---|---:|
| AP and independent folded-policy shipment |exactly0|
| Native shipment |1.7763568394002505e-15|
| AP cost |13.12403051721862|
| Unchanged native hard cost |13.224030014779418|
| Raw native minus AP cost |0.09999949756079829|
| Explained extra shipping fee |0.1|
| Unexplained cost difference |−5.024392017161095e-7|
| Unexplained fee-free scaled difference |3.887635525517463e-8|
| Acceptance |accepted_with_replay_tolerance|

Raw cost scaled discrepancy remains0.007561953311436606, above the ordinary
1e-4 threshold. The exception is not relabelled an ordinary pass. AP/native
raw costs, shipments, rational projection and AP scenario are exactly identical
to the preserved failed case. Native projection telemetry now records one
solved call and three iterations for this case. This identifies the permitted
discrepancy, not the backend mechanism producing the residue.

## Tests and independent review

The21 tests cover the saved failure, zero/no exception, tolerance equality and
just-above, epsilon-sized differences, reverse/swapped support, genuine exact
positives smaller than floating range, nonzero AP residues, nonfinite values,
multiple residues, unexplained costs, unchanged matching-support behavior,
single native invocation, active epsilon/physical failures, and ghost fees.

Independent read-only review found no numerical blocker in this scope. It
reconstructed the AP adapter and confirmed that only one native invocation
supplies support and cost. The inherited original solver docstring describes
the stricter old rule; the authoritative revised acceptance contract is this
directory's README and explicit consistency metadata, not that old docstring.

All72 source/evidence-map entries and12 input-map entries were independently
rechecked. These maps overlap; they are not84 distinct files. All six case
files equal their entries in the gate summary. The old failed case remains
failed and unchanged, not rewritten as a success.

## Runtime

Linux gold-srv2, one worker, one Torch/solver thread. Whole-child interval:
11:12:57.081663–11:13:02.954425UTC,25September2026. Stored units are seconds
from perf_counter; UTC endpoints and AP solver iterations/nodes persist.

| Stage | Status | Wall seconds | Human time |
|---|---|---:|---|
| Current whole gate child incl. imports/teardown |Completed with one exception|5.872763|5.87 s|
| Gate body |Completed with one exception|3.381738|3.38 s|
| Unit-test subprocess |21 passed|2.263713|2.26 s|
| Fresh nominal training / SAA |Not run; saved fits reused|—|—|
| Robust neural/CCG total |Not launched|—|—|
| SRO-FA partition/solver/evaluation/total |Not run|—|—|
| Validation/test/CV |Not run|—|—|

Tests, gate body and AP times are nested in the child wall time, not additive.
Each case retains its previously measured nominal training time separately;
loading it is not retraining. The original nominal/SAA whole-child runtime
was6016.852093s (1h40m16.85s), under its own30-initialization/52-nominal-worker
configuration; it is not part of current runtime.

| Ntotal | Radius | Acceptance | AP wall s | Solver wall s | Solver iterations / nodes |
|---:|---:|---|---:|---:|---:|
|50|0|Ordinary pass|0.599772|0.001692|0 / 0|
|50|.2|Ordinary pass|0.065030|0.005896|14 / 3|
|200|0|Ordinary pass|0.018162|0.001437|0 / 0|
|200|.2|Ordinary pass|0.077146|0.004548|10 / 1|
|800|0|Ordinary pass|0.018113|0.001570|0 / 0|
|800|.2|Tolerated exception|0.029439|0.002956|8 / 1|

Each method/size/radius/acceptance cohort has n=1, so mean, median, minimum
and maximum equal the listed AP wall seconds. Exception and ordinary cohorts
remain separate; no speed or full-sweep convergence claim follows.

## Execution and artifacts

- Working directory: this experiment directory.
- Completed tmux session: `shipping_replay_tolerance_20260925`, child exit0.
- Exact child command: `/data/home/aviyac/ap_env/bin/python -B -u /data/home/aviyac/retailers_moreau_training_selected/shipping_replay_tolerance_20260925/ap_gate.py --output /data/home/aviyac/retailers_moreau_training_selected/shipping_replay_tolerance_20260925/ap_gate_v1`.
- Attach while active: `tmux attach -t shipping_replay_tolerance_20260925`.
- Session status: `tmux has-session -t shipping_replay_tolerance_20260925`.
- [Persistent merged log](ap_gate_v1.log), [launch record](ap_gate_v1.launch.json),
  [whole child timing](ap_gate_v1.invocation.json), [summary](ap_gate_v1/summary.json),
  [unit results](ap_gate_v1/unit.log), [exception case](ap_gate_v1/N800_radius0.2.json).
- Provenance SHA256:18a21bf16d3e9e81093087caef54e548d824ec6bf9b2bfb9a7c12ad686d29042.

This resolves the narrow replay blocker. It does not implement or launch the
full shipping green controller, waive signed AP/pool ordering checks, or
certify the restricted AP as a bound for native or unrestricted hard costs.