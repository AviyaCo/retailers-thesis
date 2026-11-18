# Shipping: stopping-checkpoint return — 5 October 2026

User-authorized revision: return the **first passing stopping assessment** for
all 270 green shipping configurations, including candidate zero, rather than
the earliest minimum-mean-AP assessment. Reuse the saved trajectories and the
stopping checkpoint's own pre-insertion pool. Do not retrain, change stopping
tolerances, select radii, refit, or modify original evidence.

The cohort remains 265 original trajectories plus five approved advisory
continuations, N=50/200/800 (training40/160/640), seeds0–9 and radii
0/.001/.003/.01/.03/.2/.4/.7/1.5. Passing the approximate aggregate green gap
does not certify every-center separation or global robust optimality. The
smooth training objective and epsilon-restricted AP remain unchanged.

Selection validates the historical minimum-AP record as provenance, reconstructs
the first strict mean-gap pass, verifies the corresponding checkpoint state and
pool, then seals all 270 choices before accessing test arrays. No fallback to a
nonpassing terminal candidate. Original frozen inputs are hash-verified and
referenced in place; new stopping inputs and revision sources are copied.
This output therefore depends on the retained parent archive; it is not a
standalone backup. The parent manifest and completed summary are hash-pinned.

All270 green policies are freshly evaluated using the **unchanged shipping
evaluator**: full-set float32 targets, fresh strict Gurobi projection per row,
raw0.1 per q>0 shipment, OPTIMAL required, physical tolerance1e-6, no clipping,
cutoff, retry or omission. OOS uses the exact existing10000 seed9999 arrays.
IS is the mean per-center maximum on the returned stopping policy's own pool,
not a full robust supremum. Before publication, raw journals, shipments,
scenario identity, costs and aggregates are reverified for every fresh policy.

Blue (480 historical Clarabel policies), kNN, SRO-FA, reference levels, radius
grids and bands remain exact copies of the completed September28 plot data.
These are explicitly cached, not newly evaluated. Cached kNN/SRO use fee
thresholds1e-6/1e-10; retain their solver-status/feasibility caveats. Gray remains
a finite1M-scenario conditional MIP estimate, not an exact population bound.
Bands remain pointwise20–80% over ten training seeds, not confidence intervals.
The inspected test and historically test-informed design are descriptive, not
new confirmatory evidence. Nothing is selected based on the revised OOS curve.

Execution:16 spawned workers, one Torch/Gurobi thread each, unchanged resource
gate. Three spawned two-center training-only probes cover all sizes, including
two advisory stopping checkpoints. Tests and probes precede held-out replay.
Every model/configuration has perf_counter wall time, UTC boundaries, status,
seed, radius, sample size, thread counts, solver iterations and separate OOS/IS
times. Phase and whole-child timings are stored separately, not sums of
concurrent model times. Existing nominal and AP/MP timings remain historical;
SRO partition/optimization/evaluation/total timings remain in parent provenance.
Unknown timings remain unavailable (rerun required); CV/validation are not run.

Run from this directory:

    /data/home/aviyac/ap_env/bin/python -B launch.py --output production_v1 --session shipping_stop_20261005

The launcher uses named tmux and persists stdout/stderr to production_v1.log,
command/cwd to production_v1.launch.json, and whole-child monotonic runtime to
production_v1.invocation.json. Attach with `tmux attach -t shipping_stop_20261005`;
check session existence with `tmux has-session -t shipping_stop_20261005`.
Failed attempts are retained; a rerun requires a new output directory.

Publication is atomic after all gates pass: production_v1/published contains
comparison_shipping.png/.pdf, plot_data.json, RESULTS.md and runtime_report.json.
The new manifest is the returned-policy index. Historical training summaries
keep their original minimum-AP fields and are never rewritten or relabelled.
Sources and this protocol are frozen by each invocation; subsequent completion
documentation must be a new file, not an edit to this protocol.