# Bounded direct-QP SCS gradient + speed feasibility

Isolated, **not launched** on dataset leaves. No training, optimizer, test/OOS
evaluation, LR reselection, dependency installation, or frozen-core edits.
Only the existing ten LR-selected nominal checkpoints are eligible (seeds 0–9,
radius 0). The saved selection is consumed, never recomputed from costs.

## Predeclared protocol

- Freeze all ten selections before the first projection benchmark: checkpoint,
  selection-summary, split, raw-pair and training-pair hashes, exact original
  train40 indices, source/binary hashes, versions, float32 features and actual
  double leaves `z,C`, independent upstreams and target directions. Re-read the
  frozen manifest before benchmarking and verify bindings before every config.
- Historical split JSON contains all 50 pairs. Its bytes and original paired-data
  digest authenticate provenance only; the ten held-out rows never enter
  normalization, model features, gradients, references, costs or diagnostics.
  Training uses the saved original permutation (`100000+seed`), not first40.
- Neural input is the frozen baseline's float32 `[xi1,softplus(Q1r),softplus(Q10)]`;
  no fitted normalization. Projection leaves are detached doubles for every arm.
  Second-stage demand is bound for provenance, not used to construct `z`.
- Per-row Gaussian upstreams have unit L2 norm, seed `20260909+seed`; independent
  unit target directions use `20260909+100000+seed`. The scalar is `g @ x` per
  row, not a training cost. Capacity derivatives are independent row leaves,
  not a shared-parameter sum or softplus-chain gradient.
- All arms solve `min .5*x'x-z'x`, `x>=0`, `sum(x)<=C`: `P=I`, `c=-z`,
  `A=[-I;ones]`, `b=[0,0,0,C]`. No capacity buffer.
- Native arm is unchanged `core.NativeProjection`, exact active-set backward,
  batch40, construction/matrix cache retained within configuration, fresh solves
  with `warm_start=None`. All native status checks remain in force; observed
  records are drained after every pass.
- SCS: installed diffcp 1.1.8 supported `solve_and_derivative`, `P=P`,
  `solve_method='SCS'`, `mode='lpgd'`, `derivative_kwargs={'tau':tau,'rho':0}`.
  Six fixed settings: `eps_abs=eps_rel` in `{1e-4,1e-8}` crossed with `tau` in
  `{1e-2,1e-3,1e-4}`, `max_iters=200000`, `acceleration_lookback=10`.
  `DT(g,zeros(4),zeros(4))` supplies `dA,db,dc`; `grad_z=-dc`, `grad_C=db[-1]`.
  No replacement derivative, manual Jacobian, regularization or retries.
- SCS observer transparently intercepts **every** base and both backward
  perturbation calls; stores full returned x/y/s and info, call settings,
  warm-start presence, b/c, active sets, status and monotonic duration.
  Only `status_val==1` is accepted; outputs, adjoints and numerical info must
  be finite. The actual diagnostics `res_pri`, `res_dual`, `gap`, `pobj`, `dobj`
  must exist as finite real numbers. Only after status 1, certificate diagnostics
  `res_infeas`, `res_unbdd_a`, `res_unbdd_p` may be finite or NaN, never +/-infinity.
  SCS initializes these certificates to NaN when their normalization is
  inapplicable, including valid solved results. This is certificate semantics,
  not a relaxed solve tolerance. Full raw info is retained without substitutions;
  NaN is serialized as the string `"NaN"`. All other nonfinite diagnostics fail.
- **LPGD is approximate**, a finite-perturbation adjoint, not exact implicit
  differentiation at finite tau. Installed diffcp warm-starts its two perturbed
  solves from the base primal/dual/slack solution. Installed `scs.solve`
  constructs a new solver for every base/perturbed call; no cached SCS workspace.
  Base calls receive no warm-start arrays. This asymmetry and the 40 serial
  Python QPs versus one native batch are integration differences, not a pure
  backend speed comparison. No n_jobs argument is passed to the single-instance
  API: serial loops enforce effective `n_jobs=1`.

## Accuracy, finite differences and failure semantics

Strict fresh Gurobi P1 references use Threads=1, FeasibilityTol=OptimalityTol=
BarConvTol=1e-9; require OPTIMAL, finite solution/objective and physical residual
<=1e-6. All40 training rows receive forward references and gradient comparisons
against Moreau. Separately, positions 0,19,39 in each seed's *training order*
receive Gurobi central differences at steps 1e-2,1e-3,1e-4, target direction and
capacity separately. Negative minus-capacity is flagged invalid, never replaced
by a one-sided estimate. No additional training samples are generated.

Raw per-row metrics: forward max-absolute error versus Gurobi, nonnegativity and
capacity violation; target-gradient L2 difference divided by `max(1,||g_ref||)`,
cosine (null when undefined), max-absolute error; capacity absolute and
floor-scaled errors; FD directional absolute and floor-scaled errors with
denominator `max(1,abs(FD))`. Predeclared diagnostic gates are 1e-6 forward and
physical error, 1e-4 floor-scaled gradient error. These do **not** select settings,
alter tolerances or discard configurations. Raw results for every arm remain.

Active sets use solution threshold 1e-7 and capacity slack; store base and
perturbed signatures, branch changes and raw slack. At kinks, a unique classical
gradient need not exist. Branch-crossing discrepancies do not establish a bug.
A failed solve stops that configuration (unattempted passes counted), then
independent configurations continue without fallback. Partial solutions and
nonfinite diagnostics are retained as explicit JSON strings. Failed references
make corresponding accuracy metrics unavailable, not substituted. Campaign
success requires every case and reference to complete, not every diagnostic gate
to pass; the gates must be inspected before claiming comparable accuracy/speed.

## Runtime accounting

One serial worker, Torch/interop/BLAS/OpenMP/Rayon threads=1. Each seed runs Moreau
then all six SCS configs; **each configuration** has a first forward+backward pass
followed by five repeats. First-call timing never enters repeat statistics.
Construction, leaf setup, forward including derivative-closure construction,
and backward/differentiation are separated. The SCS API does not expose an
independent closure-setup timer: it belongs to forward, explicitly, not zero.
SCS native setup/solve info is also stored in seconds (SCS reports milliseconds).
Moreau lazy native setup is in first forward; construction cache is retained.
Backward timings include solver observation and output conversion consistently
within each arm; no attempt is made to subtract observer overhead.

Per-pass and per-case UTC + monotonic start/end, durations, statuses and solver
iterations are persisted. Cases bind seed, LR, radius0, sample size40, workers and
manifest hash. Reference construction/optimization/FD durations are separate;
every fresh reference QP stores its total plus solver time. Freeze/preparation,
diagnostics/persistence and serial end-to-end wall time are not disguised as
solver time. Raw first/repeat samples and completed-repeat n/mean/median/min/max
are in case JSON, while REPORT.md summarizes completed-case totals across seeds.
Failures/interruption are separate. No resumed/cached-model time is claimed.
Training, robust/CCG, SRO-FA, CV and held-out evaluation were not run; no historical
timing is inferred from loading. Shared-host hardware/library details are in the
manifest. Repeated timings are descriptive, not independent-seed estimates.

## Bounded preflight

From this directory:

```bash
/data/home/aviyac/ap_env/bin/python -B test_study.py --output-name preflight_REVIEW_UNIQUE
```

Small actual SCS/Moreau/Gurobi fixtures, all six settings, chain-rule sign,
capacity gradient, strict base/plus/minus rejection without retries,
required finite actual diagnostics, all three certificate NaNs accepted and
both infinities rejected, raw nonfinite preservation, first/repeat separation,
native observer drain, checkpoint/train/leaf hash binding, invalid FD capacities,
zero-gradient metrics and independent failure continuation. Fabricated arrays
exist only in file-format/control-flow unit tests, never campaign fixtures.
Each preflight creates its own structured timings/results and refuses overwrite.
Both preflight and study provenance include the test-source SHA-256.

Measured implementation preflight: initial 9 tests passed in 1.257361 s;
expanded 11 tests passed in 1.203466 s (0.020058 min), on Linux host gold-srv2,
one process/thread. Raw per-arm fixture timings and solver info are in the
respective preflight JSONs. Two intervening import attempts failed on indentation
before the timer started; they ran zero tests, timings are unavailable, and the
failure artifact is retained. The indentation was corrected before the final
successful rerun. Dataset-campaign timings remain unmeasured. Pylance reports
`reportMissingImports` for `moreau_cpu` because the frozen core adds its isolated
dependency directory at runtime; actual imports and solver tests passed. No
environment, dependency, or editor-path setting was changed to hide that warning.

## Main-owned launch

The study is user-approved; Main owns its launch. This correction runs only the
bounded preflight and does not launch the dataset study.

Exact cwd:
`/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/scs_qp_feasibility`

```bash
tmux new-session -d -s retailers_scs_qp_feasibility_20260909 -c /data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/scs_qp_feasibility '/data/home/aviyac/ap_env/bin/python -B study.py --output-name campaign_20260909 > campaign_20260909.log 2>&1'
```

Persistent stdout/stderr: `campaign_20260909.log` in the cwd above. Exact inner
command is the Python invocation shown inside the tmux command. Study refuses an
existing output directory; use a new output/session/log name for a genuinely new
invocation. Never relaunch over a prior log. The tmux session exits when done;
summary JSON and persistent log remain. Attach/check:

```bash
tmux attach -t retailers_scs_qp_feasibility_20260909
tmux has-session -t retailers_scs_qp_feasibility_20260909
tail -n 30 campaign_20260909.log
cat campaign_20260909/summary.json
```

No dataset campaign, historical OOS result inspection, or performance conclusion
is part of implementation/preflight.