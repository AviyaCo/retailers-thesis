# Black overlay validation status

The requested R3/R10 black overlays were published on 28 September 2026.
All original figures and completed training artifacts remain unchanged.

- [R3 PNG](production_v2_20260928/published/comparison_R3_with_black.png) / [PDF](production_v2_20260928/published/comparison_R3_with_black.pdf)
- [R10 PNG](production_v2_20260928/published/comparison_R10_with_black.png) / [PDF](production_v2_20260928/published/comparison_R10_with_black.pdf)
- [Results and per-configuration runtime tables](production_v2_20260928/published/RESULTS.md)

## Protocol

570 fresh fixed-policy evaluations, all ten seeds per radius/size, on the
original paired 10,000-row seed9999 tests. All eleven training caps remain
labelled nonconverged. The ten new cap selections use the explicitly approved
earliest minimum mean AP incumbent over the full executed prefix, including
terminal40. IS uses the selected checkpoint's own pool. No training, AP, new
radius selection, SRO-FA or oracle run is included.

## Preserved first preflight

- Session: `black_plot_pref_v1_20260926` (finished).
- Result: **failed during training-evidence collection**, before numerical probes.
- Whole-child wall time: **26.940099 seconds / 0.449002 minutes**.
- 56 unit tests passed. The adapter's box-bound arithmetic was reassociated;
  restoring the original operation grouping removes the spurious 4.8e-15
  residual without changing tolerances. A regression test was added.
- [Summary](preflight_v1/summary.json), [log](preflight_v1.launch/child.log),
  [runtime](preflight_v1.launch/child_runtime.json), and
  [exact invocation/attach/status commands](preflight_v1.launch/invocation.json).

| Runtime category | Status |
|---|---|
| Nominal neural training | Not run; persisted historical timings separate |
| Robust neural/CCG | Not run; historical continuation whole-child 62,823.446786 s (17.451 h), nonconverged, 16 single-thread workers |
| SRO-FA | Not run; original figure retained |
| Test/own-pool replay | Completed September 28; 1,083.598741 s parallel evaluation stage, 1,561.301943 s whole invocation |
| CV/radius selection | Not run |
| Initial preflight | Failed, 26.940099 s; no numerical probes |

Independent read-only review found no additional numerical blocker but requires
a successful final-source preflight and six real training-only probes before
production. Historical OOS-containing manifests/plot schemas are read for
provenance/display tests; their values do not enter policy selection.

The standalone, unlogged full-collection diagnostic was interrupted deliberately
while still checking training/AP rows, not because a numerical solver failed.
Its full monotonic duration is unavailable. It produced no research result or
test evaluation. The replacement preflight is tmux-managed with persistent
per-policy progress and structured whole-invocation timing.

## Corrected preflight

Started 2026-09-26 at 18:28:31 UTC in `black_plot_pref_v2_20260926` and completed
at 18:59:23 UTC. All 57 unit tests, all 570 training-only policy selections,
six two-center training probes (12 accepted projections) and final hash checks
passed. Whole-child time: **1,852.601043 s / 30.876684 min**. No test evaluation
or updated plot was performed by the preflight.

- [Persistent log](preflight_v2.launch/child.log)
- [Whole-child runtime/status](preflight_v2.launch/child_runtime.json)
- [Exact command and attachment/status commands](preflight_v2.launch/invocation.json)
- Attach: `tmux attach -t black_plot_pref_v2_20260926`.

Source files are frozen during this preflight. The independent review's
production approval remains conditional on all validation gates passing.

## Production attempt: resource gate failure

The queued production launched in `black_test_plots_20260926` at 18:59:24 UTC
and stopped at 19:02:41 UTC after passing complete preflight revalidation.
The resource gate could not reserve the required capacity for 16 workers.
**Zero policy workers started; no new test evaluation or plots completed.**
No reduced-worker fallback, tolerance change or automatic retry occurred.

- Whole-child time: **196.908665 s / 3.281811 min**; exit 1.
- [Failure summary](production_v1/summary.json)
- [Whole-child timing](production_v1.launch/child_runtime.json)
- [Command and log references](production_v1.launch/invocation.json)

The sealed successful preflight remains preserved. Restarting with lower
concurrency would require an explicit resource-plan change, not a silent
fallback. The new [shared experiment entry point](../../../retailers_experiment/README.md)
does not replace this frozen comparison or change its saved inputs.

## Authorized retry: 28 September 2026

Current resource inspection passed the unchanged 16-worker gate: 72 available
CPU affinity slots, load allowance 3, and approximately 468 GiB available memory.
The user-authorized retry started at 08:54:00 UTC in
`black_test_plots_20260928`, output `production_v2_20260928`.
It uses the same preflight SHA256
`f3e3a983c6824dc6d13966b70c5bf1da2dafad6e24382665ac0c3bdf68666e87`,
all 570 fixed policies and the original scenario arrays. No numerical source,
tolerance, concurrency, selection or evaluation protocol was changed.

The invocation completed at 09:20:02 UTC with exit 0. All 570 policy evaluations
and global journal, source and artifact checks passed. There were 6,451,480
attempted and accepted strict optimal projections: 5,700,000 test projections
and 751,480 own-pool projections. All eleven capped training policies remain
labelled nonconverged. Both figures were published together; their images were
visually inspected, and all nine published artifact hashes were independently
rechecked. The earlier failed attempts remain preserved.

- [Persistent log](production_v2_20260928.launch/child.log)
- [Whole-child runtime/status](production_v2_20260928.launch/child_runtime.json)
- [Exact command and status instructions](production_v2_20260928.launch/invocation.json)
- Attach: `tmux attach -t black_test_plots_20260928`.

Whole-child time was **1,561.301943 s / 26.021699 min** on
`gold-srv2.iem.technion.ac.il`, with 16 single-thread workers. Parallel policy
evaluation took 1,083.598741 s (18.059979 min), journal verification 114.748420 s,
rendering 42.068550 s, and final verification 152.188252 s. Per-policy and
per-configuration timings remain in the structured runtime report; concurrent
durations are not summed as sweep elapsed time. No new nominal training,
CCG/AP/MP, SRO-FA, validation/CV or oracle reconstruction was performed.