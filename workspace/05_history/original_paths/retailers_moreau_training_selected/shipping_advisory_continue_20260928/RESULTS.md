# Five shipping advisory continuations — completed 28 September 2026

## Outcome and scope

All five previously failed configurations completed under the approved approximate acceptance protocol and reached the **first mean-gap stopping rule, strictly below 1%**. The original five failures and all historical artifacts are unchanged. These results are a **mixed-protocol continuation**, not a homogeneous rerun of the 270 configurations, a per-center separation certificate, or proof of robust optimality.

The 15 completed historical MPs were retained. Four new MPs ran: one for N50 and three for N200. Each failed AP sweep was redone in full. All three N800 cases passed the mean-gap rule immediately at their restored repeated state; no extra training or reset of stagnation history was needed. The original 40-total-MP budget remained enforced.

| Total/train N | Seed | Radius | Stop → selected candidate | Stop gap | Selected own-pool gap | Cached + new MPs |
|---|---:|---:|---:|---:|---:|---:|
| 50/40 | 7 | 1.5 | 10 → 10 | 0.478966% | 0.478966% | 9 + 1 |
| 200/160 | 5 | 1.5 | 4 → 3 | 0.776428% | **1.147191%** | 1 + 3 |
| 800/640 | 6 | 0.7 | 3 → 2 | 0.073602% | **1.391443%** | 3 + 0 |
| 800/640 | 7 | 0.2 | 1 → 0 | 0.129707% | **3.661752%** | 1 + 0 |
| 800/640 | 9 | 0.4 | 1 → 0 | 0.103586% | **7.469242%** | 1 + 0 |

Candidate IDs are zero-based. Selection is the earliest minimum AP incumbent mean across the full assessed prefix, including cached candidates and the stopping assessment. **Four selected assessments do not themselves pass the mean-gap rule.** For all three N800 cases, selected and terminal learned states are identical; their different own gaps arise from different saved pools. N800 seeds 7 and 9 return the nominal policy. N200 returns an earlier, different policy.

All five terminal assessments still contain centers with separation residual above 1%: respectively 10, 43, 2, 1 and 2 centers. Mean-gap stopping must not be described as every-center closure. The terminal mean-absolute incumbent discrepancies also remain below 1%; negative-gap cancellation is not necessary for these five passes.

## What changed, and what did not

Activation-link, exact folded-support and positive/epsilon consistency discrepancies, plus exact-duplicate material-separation discrepancies, are diagnostic-only as authorized. Raw AP solutions and costs are not repaired. Exact duplicates are not inserted again. Native fees remain `0.1 * sum(q > 0)` on raw shipments; smooth MP training and the AP epsilon of 1e-6 are unchanged. No clipping or fee cutoff was added.

Finite values, binary integrality/range, solver OPTIMAL/incumbent requirements, AP physical/box checks, native physical tolerance 1e-6, bound/incumbent ordering, unresolved-bound checks, and material float32-collision rejection remain enforced. These guards do not certify the waived activation links. The inherited protocol string `ap_internal_validation` still describes the old checks as unchanged: this wording is obsolete for rejection behavior; the explicit advisory flags, executable adaptations, and this report describe the actual protocol.

Current-sweep diagnostics, excluding cached rows and preflight probes:

| N / seed / radius | Rows with activation-related discrepancy | Exact duplicates with material separation | Native support / cost discrepancy rows | Material negative bound–pool gap rows |
|---|---:|---:|---:|---:|
| 50 / 7 / 1.5 | 1 | 0 | 23 / 23 | 17 |
| 200 / 5 / 1.5 | 1 | 0 | 111 / 110 | 41 |
| 800 / 6 / 0.7 | 0 | 2 | 123 / 119 | 100 |
| 800 / 7 / 0.2 | 0 | 1 | 186 / 183 | 175 |
| 800 / 9 / 0.4 | 0 | 2 | 181 / 180 | 131 |

Columns can overlap. The two activation-affected rows and five material exact-duplicate rows were retained explicitly, not silently declared consistent.

## Runtime accounting

Wall-clock seconds are measured with `time.perf_counter`. Host: gold-srv2.iem.technion.ac.il, Linux, Intel Xeon Gold 6354, 72 logical CPUs. Python 3.11.13, Torch 2.10.0, NumPy 2.4.2, Gurobi 13.0.1, Moreau CPU 0.4.0. Five concurrent configuration workers; one Torch intra/inter-op thread and one Gurobi thread per worker. Historical CCG used 62 workers, so old/new timings are **not a controlled speed comparison**.

| Configuration | Current AP | Current MP | Current training-pool evaluation | Current job wall | Human-readable job wall |
|---|---:|---:|---:|---:|---:|
| N50 / s7 / r1.5 | 3.797287 | 8.227059 | 0.042378 | 35.891175 | 35.89 s |
| N200 / s5 / r1.5 | 16.616411 | 133.601320 | 0.127979 | 176.184010 | 2 min 56.18 s |
| N800 / s6 / r0.7 | 19.121043 | 0 | 0.051554 | 31.640754 | 31.64 s |
| N800 / s7 / r0.2 | 14.290320 | 0 | 0.031177 | 20.836988 | 20.84 s |
| N800 / s9 / r0.4 | 16.178213 | 0 | 0.035639 | 22.945317 | 22.95 s |

Job wall includes boundary verification, persistence and other overhead, not just these phases. For each method/N/radius group, completed count is **1**, so mean = median = minimum = maximum = the corresponding job wall. No failed, interrupted, cached-only or nonconverged configuration is included in these five completed-run groups. Raw full-precision statistics are in [continuation_v1/summary.json](continuation_v1/summary.json).

| Current campaign component | Seconds | Human-readable |
|---|---:|---:|
| Whole child invocation, including imports | **216.913857** | **3 min 36.91 s** |
| Main runner body | 211.764434 | 3 min 31.76 s |
| Five-worker continuation stage | 183.055829 | 3 min 3.06 s |
| Regression-test subprocess | 10.855937 | 10.86 s |
| Sequential native-probe stage | 2.073581 | 2.07 s |

These phases are nested, not additive. The five concurrent job walls sum to 287.498 s; that sum is **not** elapsed campaign time. Worker-wrapper wall times (38.781, 178.013, 34.490, 21.801, 24.088 s in table order) include additional loading/verification overhead outside the per-configuration controller timer.

Historical work, not charged to this continuation:

| Configuration | Cached valid AP / MP / evaluation seconds | Prior failed-job wall seconds | Selected nominal training seconds (epochs evaluated / best pre-update epoch) |
|---|---:|---:|---:|
| N50 / s7 / r1.5 | 16.220 / 487.295 / 0.430 | 592.895 | 102.302 (3000 / 2976) |
| N200 / s5 / r1.5 | 7.758 / 146.717 / 0.096 | 166.076 | 91.614 (3000 / 2933) |
| N800 / s6 / r0.7 | 76.540 / 523.553 / 0.490 | 670.238 | 111.431 (3000 / 2963) |
| N800 / s7 / r0.2 | 36.061 / 28.894 / 0.149 | 108.063 | 100.712 (2514 / 2214) |
| N800 / s9 / r0.4 | 26.319 / 30.269 / 0.166 | 98.224 | 82.102 (2596 / 2296) |

The historical failed-job walls include the failed sweeps and other overhead; do not add them to cached-prefix phase sums as distinct work. Selected nominal timings cover only the chosen LR arm, not all LR-search arms. Selected LRs are .01 for N50 and .05 for the other four cases. No nominal training occurred now.

New MP evaluated/best epochs: N50 output candidate 10, 606/306; N200 outputs 2, 3 and 4, respectively 3000/2797, 2940/2640 and 1947/1647. All per-iteration AP/MP timing, start/end UTC, status and epoch journals are retained in the configuration artifacts. There are 8,493 newly evaluated MP epochs and 30,420 cached MP epochs; an epoch-budget stop is not itself optimizer convergence.

| Other model/evaluation category | Current runtime/status |
|---|---|
| Nominal neural training | Not run; cached per-seed values above |
| Nominal evaluation / hyperparameter selection | Not run; no new LR selection |
| Robust neural/CCG | Five completed continuations, current/cached scopes above |
| Training-pool evaluation | Separately measured above; not held-out evaluation |
| SRO-FA partition / optimization / OOS / total | Not run in this trial; comparable timing unavailable here, not inferred |
| Cross-validation / final radius validation / test / benchmark / refit | Not run |

## Verification and reproducibility

- 42 regression tests passed before native execution.
- Five predeclared failed-AP probes passed, exactly reproducing each historical AP incumbent.
- Nine current AP sweeps contain 2,640/2,640 completed rows. Cached plus current valid prefixes contain 24 sweeps and 6,360 rows.
- Independent read-only review checked 5,184 source bindings and 6,028 historical-input bindings, all five output-summary hashes, and selected-checkpoint byte hashes: no mismatch.
- All cached prefixes, numerical pools, MP-output chains and sample mirrors agree. Selection and the original iteration budget were verified.
- At the resumed boundaries, 188,664 shared raw numerical/exact-value fields across 2,120 rows matched historical values exactly. Timings and deliberately changed acceptance statuses were excluded; historically absent fields were not imputed.
- Independent review made no model/solver calls. Checkpoint bytes and recorded state chains were checked; tensor-state hashes were not independently rebuilt by that review. The run itself verifies learned-state hashes during restoration.
- Final campaign-summary SHA-256: `1984d5ac9182b222b213d08bbc53a418d4c5dd87433c8b7e41faafa719c820cb`.

Execution: 2026-09-28 10:17:53.544648–10:21:30.458510 UTC; child exit 0. Named tmux session: `shipping_advisory_continue_20260928` (finished). Working directory: `/data/home/aviyac/retailers_moreau_training_selected/shipping_advisory_continue_20260928`.

Exact launch command, from that working directory: `/data/home/aviyac/ap_env/bin/python -B launch.py --output continuation_v1 --session shipping_advisory_continue_20260928`. The actual child argv and working directory are persisted in [continuation_v1.launch.json](continuation_v1.launch.json) and [continuation_v1.invocation.json](continuation_v1.invocation.json). Persistent stdout/stderr: [continuation_v1.log](continuation_v1.log).

During execution, attachment was `tmux attach -t shipping_advisory_continue_20260928` and live-session status was `tmux has-session -t shipping_advisory_continue_20260928`; the session is no longer expected after completion. Final status is the persisted invocation and [continuation_v1/summary.json](continuation_v1/summary.json), not a stale launch marker. Do not relaunch into this existing output or edit its frozen computational sources/README.

**No radius selection or held-out evaluation was performed.** The 265 historical successes plus these five continuations are not relabeled as an unchanged original campaign. This report records the approved trial only.