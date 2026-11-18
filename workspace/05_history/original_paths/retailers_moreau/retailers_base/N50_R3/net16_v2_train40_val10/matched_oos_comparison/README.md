# Frozen R3/N50 matched OOS comparison

Separate evaluation only. Original figure, old scripts, neural checkpoints,
training primitives, and CCG runner are not modified. No fitting, checkpoint
search, radius selection, optimizer update, or fallback occurs here.

## Frozen protocol

- Old: every **positive originally plotted radius <= 1.5**, all seeds 0–9;
  width16, 50 training observations, historical SCS training.
- New: radii **0, .001, .003, .01, .03, .2, .4, .7, 1.5**, all seeds 0–9;
  width16, 40 train / 10 validation, Moreau training. Read saved per-radius
  selected checkpoints, never terminal checkpoints or test-selected incumbents.
- Old discovery/selection exactly reproduces the original evaluator: a real
  radius directory takes precedence over the special .4 nominal-directory map;
  its summary's `final_mp_iter` (default zero) selects the MP, with zero meaning
  the nominal checkpoint. The .4 hardcoded map is used only without that directory.
- The original cache has 23 positive plotted radii and 10 seeds each: 230 old
  plus 90 new configurations = **320 jobs / 3,200,000 projections**. No repeated
  state caching: repeated nominal states are genuinely re-evaluated and timed.
- Historical identity verification checks the full forced completed-cache run,
  evaluator version, all seed counts, exact source-rule replay for all 230 jobs,
  and that source summaries/checkpoints predate cache completion. **The old cache
  contains no checkpoint hashes. Historical byte identity cannot be proved.**
  Newly frozen file/state hashes establish identity from this freeze onward.
  No numerical agreement with old test values is used to select a checkpoint.
- Freeze copies and hashes source files, reference JSON, selector summaries,
  checkpoints and saved new selections before creating or reading test scenarios.
  Copies and manifest are read-only; original/copy hashes and package versions
  are rechecked before generation and after evaluation. Existing output/freeze
  directories cause errors, not overwrites or silent resumes.
- Exactly legacy `np.random.seed(9999)` and SciPy `truncnorm` generate 10,000
  observations. Float64 raw demands and QP outputs feed the legacy NumPy cost
  expression; neural features/weights and softplus Q1r remain float32, including
  the legacy float32 `Q1r.sum()` convention. The network observes only xi1,
  concatenated with first-stage inventory. xi2 enters cost only.
- Learned schema is exact: eight float32 finite tensors with explicit shapes.
  Old `model_state_dict` additionally contains only the two known non-learned
  projection buffers; new selected files contain learned tensors directly.
- Each scenario builds a **fresh standard Gurobi QP**, the exact original squared
  Euclidean objective, Q2 >= 0 and sum Q2 <= Q10, without a +1e-6 cap. Only the
  environment is shared. No solver solution warm start/model reuse. Threads=1;
  other Gurobi numerical defaults are unchanged and recorded. Require OPTIMAL,
  finite solution/objective, and maximum absolute nonnegativity/budget violation
  <= 1e-4. No clipping, zero replacement, SUBOPTIMAL acceptance, relaxed tolerance,
  or retry. First failed projection fails that configuration; others retain their
  full seed/radius identities and continue. Full chart requires every job succeed.
- Fixed spawn concurrency: 8 CPU workers, one Gurobi/Torch/BLAS thread each.
  Shared-host load (including the user's hydro workers) is not controlled.

## Descriptive plot, not a new selection exercise

Blue old50train and purple new40train10val curves show means and 20th/80th
percentiles across the same ten seeds, **not confidence intervals**. A symlog
radius axis (linear through .001) displays zero explicitly. No minima/winners
are labeled. Training and validation protocols differ; this is not an isolated
SCS-versus-Moreau training comparison or a confirmatory superiority test.

Saved final new radius choices for seeds **2, 6, 9** have `violating` selected
separation status, even though terminal trajectories report separation satisfied.
The saved selection remains unchanged; it is **not a separation guarantee**.

Cached SRO-FA, seed-wise LOO-selected SAA-kNN, and lower bounds remain references,
not matched re-solves or newly status-audited results. SAA-kNN is historical10k;
the SAA-LP bound in that cache was copied from **2k**, not recomputed on10k.
The conditional bound uses separate **5,000 x 200 = 1M** scenarios, seed42.
These bounds need not be lower bounds on this particular empirical10k draw.

## Commands (do not launch the campaign during preflight)

Working directory for every command below:
`/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison`

```bash
/data/home/aviyac/ap_env/bin/python preflight.py solver_free
/data/home/aviyac/ap_env/bin/python compare.py freeze
/data/home/aviyac/ap_env/bin/python preflight.py gurobi
```

Preflight status, timestamps and monotonic timing are saved separately in
timestamped `preflight_solver_free_*.json` and `preflight_gurobi_*.json`;
failed invocations are preserved rather than overwritten.
Freeze performs no model forward pass or test generation. Gurobi preflight
uses five hand-chosen target/capacity fixtures and five original-reference QPs,
never held-out scenarios. Tests execute only AST-selected legacy definitions
and the selector loop with numerical operations replaced by fixtures.

**For reviewer-approved launch later**, from the working directory above:

```bash
tmux new-session -d -s retailers_matched_oos_20260909 -c /data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison '/data/home/aviyac/ap_env/bin/python -u compare.py run > campaign.log 2>&1; rc=$?; printf "\nEXIT_CODE=%s\n" "$rc" >> campaign.log; exit "$rc"'
tmux attach -t retailers_matched_oos_20260909
tmux has-session -t retailers_matched_oos_20260909
tail -n 30 campaign.log
```

The exact experiment command inside tmux is
`/data/home/aviyac/ap_env/bin/python -u compare.py run`.
The log is `campaign.log` in this directory; no long run is launched by freeze
or tests. No package installation is needed (configured ap_env).

## Outputs and runtime interpretation

- `frozen/manifest.json`, detached SHA256, read-only input copies: all identities.
- `results/xi1.npy`, `xi2.npy`: raw arrays, file and raw-byte hashes in summary.
- `results/<method_radius_seed>/result.json`: status, seed, radius, train/val/test
  sizes, selected iteration, checkpoint/source-manifest/test hashes, timestamps,
  per-phase and end-to-end monotonic seconds, exact status counts/iteration totals.
- Adjacent `solves.jsonl`: every attempted solve's status, iterations, barrier
  iterations, physical check, time, and failure if any. Completed rows are flushed
  immediately; an externally killed process may leave an incomplete running job.
- `results/summary.json`: all collected jobs, failures, environment, actual parallel
  elapsed time, shared generation timing, plotting timing, completed-only runtime
  count/mean/median/min/max by method/sample size/radius. No concurrent-time sum is
  presented as elapsed wall time. Worker crashes have unavailable timing, not zero.
- `results/matched_oos.png`: emitted only for the complete strict320-job grid.
- `results/runtime_report.md`: seconds/minutes/hours and runtime table. Nominal
  training, robust/AP/MP totals, SRO-FA construction/optimization and CV are **not
  run here**; historical times are not inferred or relabeled as current evaluation.
  Where missing, reruns are required. New training artifacts retain their own
  historical per-iteration times. Plot-only regeneration is separately timed.

No OOS results or campaign runtimes exist until the explicit later launch.