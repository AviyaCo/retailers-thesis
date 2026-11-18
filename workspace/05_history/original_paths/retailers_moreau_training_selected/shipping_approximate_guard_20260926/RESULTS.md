# Authorized approximate guards: readiness passed

## Change and outcome

The user explicitly approved two changes on 26 September:

1. Remove rejection solely because the encoded AP bound is below the native
   hard-pool cost. Preserve the unmodified signed gap as a diagnostic.
2. Use a **1e-6 native physical-feasibility guard**, rather than 1e-9.

Both changes are implemented in this isolated revision. There is no ReLU,
shipment clipping, hard-fee cutoff, changed cost or projection/solver parameter.
AP epsilon stays 1e-6, AP feasibility/integrality tolerances stay 1e-9, and
bound/incumbent ordering, solver validity, finite evidence, membership,
duplicate/collision and unresolved-bound checks remain enforced. Native
support/cost/epsilon disagreement remains advisory, with raw evidence retained.

**30 tests passed. All 1,680 AP solves were optimal and all 1,680 controller
validations passed. All six fixed configurations passed readiness.**

Same setup as the parent: seed0, LR0.01, Ntotal 50/200/800, all 40/160/640
training centers, radii0/0.2, cached checkpoints, one sequential worker and one
Torch/Gurobi thread. No new training, MP, full CCG, hyperparameter selection,
held-out evaluation or SRO-FA was performed.

| Ntotal | Radius | Accepted / tested | Material negative bound/pool gaps (advisory) | Native residuals above old 1e-9 guard | Aggregate `green_gap` |
|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 40 / 40 | 5 | 0 | 0.000747414 |
| 50 | 0.2 | 40 / 40 | 1 | 0 | 0.033219232 |
| 200 | 0 | 160 / 160 | 60 | 0 | 0.002487459 |
| 200 | 0.2 | 160 / 160 | 2 | 1 | 0.034429725 |
| 800 | 0 | 640 / 640 | 90 | 0 | 0.000765407 |
| 800 | 0.2 | 640 / 640 | 5 | 4 | 0.025189119 |

The five physical residuals previously rejected are unchanged; the maximum is
**6.357828787884046e-7**, below 1e-6. The new count of material negative gaps is
**163**, not 162: one previously physical-rejected witness now reaches the
signed-gap diagnostic. Thus 163 negative gaps plus five old-tolerance physical
violations, with one overlap, account for all 167 previously rejected cases.

All **337 support disagreements, 32 native-epsilon disagreements and 337 raw
cost discrepancies above the former 1e-4 replay threshold** remain recorded.
Every witness has a cost observation. No discrepancy is labeled agreement.

**Interpretation:** this is passing numerical readiness under the authorized
approximate convention, not CCG convergence and not an exact native hard-cost
certificate. The aggregate gap remains the absolute difference of the AP and
pool means, not the mean of absolute sample differences; it is not the readiness
acceptance criterion. In particular all three radius0.2 gaps still exceed the
prospective 1% stopping threshold. Full CCG has not been launched.

## Independent verification

- All 1,680 raw AP incumbents, points, bounds, native shipments, hard/smooth
  costs and pool costs match the parent bit-for-bit, including signed zero.
- All sample/case/global mirrors and training identities verified; no row was
  dropped. Raw Gurobi status remains 2 for every case. Wrapper/controller
  statuses change only through the explicitly authorized acceptance rules.
- All **5,164 source/comparison bindings plus six overlapping input bindings**
  verified. The previous 1,688 JSON comparison artifacts are hash-bound.
- Bound/incumbent checks remain enforced. The minimum signed difference is
  −1.4210854715202004e-14; none exceeds its unchanged rejection tolerance.
- Boundary regressions accept equality at the native guard and reject values
  immediately beyond it. Failures beyond 1e-6 still retain raw cost observations.
- Prior failed runs and source files were not overwritten.

## Runtime accounting

Host **gold-srv2.iem.technion.ac.il**, Linux, Python3.11.13 in ap_env, 72 visible
logical CPUs; one sequential worker and one Torch/Gurobi thread. The host is
shared, not a controlled runtime benchmark. Peak process RSS **713,433,088 B**.
Native projection uses Moreau CPU active-set; AP uses Gurobi.

UTC start/end: **2026-09-26 18:06:25.391251–18:07:17.022058**.
Whole-child elapsed **51.630800 s (0.860513 min)**, exit0. Readiness body
**47.013174 s**; regression-test subprocess **3.132548 s**. These scopes are
nested, not additive. The 30-test runner itself reports 0.017 s when run before
launch; the persisted subprocess scope includes startup/import overhead.

Each configuration has one completed readiness run: its completed mean, median,
minimum and maximum are therefore equal to the single wall-time value below.
There are zero failed configurations in this invocation; earlier failed runs
use different acceptance rules and are not pooled into these aggregates.

| Ntotal | Radius | Completed runs | Wall seconds: mean = median = min = max | Minutes | Nested hard-pool evaluation seconds | Nested sequential AP sum seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 1 | 1.883249 | 0.031387 | 0.006142 | 1.779208 |
| 50 | 0.2 | 1 | 1.098204 | 0.018303 | 0.004841 | 1.014063 |
| 200 | 0 | 1 | 3.369980 | 0.056166 | 0.008571 | 3.084438 |
| 200 | 0.2 | 1 | 3.365440 | 0.056091 | 0.005608 | 3.062987 |
| 800 | 0 | 1 | 13.074355 | 0.217906 | 0.009538 | 11.896216 |
| 800 | 0.2 | 1 | 16.946611 | 0.282444 | 0.014215 | 15.640145 |

Sequential AP sum **36.477056 s**, nested solver wall sum **4.384626 s**,
controller-row sum **37.125591 s**. None is additional to overall elapsed time.
Per-row solver iterations (range0–66), nodes/status, UTC boundaries, parallel
settings and monotonic timings remain in structured JSON; no MP iterations ran.

| Required method/runtime scope | Seconds | Meaning |
|---|---:|---|
| Nominal neural training N50 seed0 LR0.01 | 79.761483 (1.329358 min) | Cached historical training; not rerun |
| Nominal neural training N200 seed0 LR0.01 | 130.390117 (2.173169 min) | Cached historical training; not rerun |
| Nominal neural training N800 seed0 LR0.01 | 165.776251 (2.762938 min) | Cached historical training; not rerun |
| Current initial-policy AP/controller readiness | 51.630800 (0.860513 min) | Whole child; not a full robust-model runtime |
| Full robust neural/CCG total, MP training | Unavailable | Not run; a full run is required to measure |
| SRO-FA partition/solver/OOS/total | Unavailable | Not run |
| Current training-center evaluation | Per configuration above | Nested hard-pool evaluation |
| CV/selection/validation/final-test evaluation | Unavailable | Not run; no held-out data accessed |

Cached training is counted once per checkpoint, not once per radius. Loading or
readiness work is not represented as a nominal or robust training time.

## Reproduction and persistent artifacts

Run from this revision directory. Exact launch command:
`/data/home/aviyac/ap_env/bin/python -B launch.py --output preflight_v1 --session shipping_approximate_guard_20260926`.
The absolute child command and intended cwd are recorded in
[preflight_v1.launch.json](preflight_v1.launch.json).

Named tmux session **shipping_approximate_guard_20260926** is now finished.
While active: `tmux attach -t shipping_approximate_guard_20260926`;
status: `tmux has-session -t shipping_approximate_guard_20260926`.
The invocation artifact records completion after the session exits.

- Final status and whole-child timing: [preflight_v1.invocation.json](preflight_v1.invocation.json).
- Persistent stdout/stderr: [preflight_v1.log](preflight_v1.log).
- Structured results: [preflight_v1/summary.json](preflight_v1/summary.json).
- Hash bindings: [preflight_v1/provenance.json](preflight_v1/provenance.json).
- Tests: [preflight_v1/unit.log](preflight_v1/unit.log).

**The two identified readiness blockers are cleared under the agreed
approximation. Full CCG remains a separate, not-yet-launched stage.**