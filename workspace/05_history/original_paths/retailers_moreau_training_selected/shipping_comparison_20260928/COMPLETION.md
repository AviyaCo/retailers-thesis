# Shipping comparison completed

- [PNG figure](production_v1/published/comparison_shipping.png)
- [Vector PDF](production_v1/published/comparison_shipping.pdf)
- [Results and per-configuration runtime tables](production_v1/published/RESULTS.md)
- [Structured plot data](production_v1/published/plot_data.json)
- [Structured runtime report](production_v1/published/runtime_report.json)

Completed 28 September 2026, 15:59:38 UTC, exit 0. All 750 policies completed:
270 green and 480 historical Clarabel. All 8,165,942 projections were accepted
as OPTIMAL with finite values and physical residual at most 1e-6. This includes
7,500,000 OOS rows and 665,942 selected-own-pool rows. No model was omitted.
The frozen source, README, manifest and five published artifacts must remain
unchanged. This completion note is additional documentation, not a rerun.

## Runtime

Host gold-srv2.iem.technion.ac.il, Linux, 72 logical CPUs; 16 workers with one
Torch/Gurobi thread each. Seconds are measured monotonic wall time, not sums
of concurrent jobs.

| Phase | Seconds | Minutes |
|---|---:|---:|
| Whole child invocation | 666.259365 | 11.104323 |
| Runner body | 662.730 (rounded) | 11.046 |
| Freeze | 142.282068 | 2.371368 |
| Tests and post-test verification | 24.660168 | 0.411003 |
| Six training-only probes and verification | 15.203220 | 0.253387 |
| Test generation and manifest sealing | 9.209429 | 0.153490 |
| Parallel model evaluation | 283.267780 | 4.721130 |
| Raw-journal verification | 168.555040 | 2.809251 |
| Final input verification | 13.236601 | 0.220610 |
| Rendering and plot-data writing | 6.097350 | 0.101623 |

Phase times are nested in the whole invocation. Nominal neural training,
robust CCG, SRO-FA fitting, CV, radius validation and refitting were **not run**.
Cached nominal/CCG timing records and SRO partition/optimization/evaluation/
total timings remain separately available in the published runtime and plot
data. Missing historical timings remain unavailable; rerunning would be needed
to measure them. Per-method/N/radius completed-only evaluation summaries each
contain ten runs with mean, median, minimum and maximum; OOS and IS are separate.

## Independent checks and interpretation

The independent read-only audit checked 4,205 original/frozen file pairs and
11,547 historical dependency bindings with zero mismatches. All five published
hashes matched the completed summary. The canonical R3 test arrays and current
test arrays are elementwise and entire-file-byte identical; no test regeneration
was used for that audit. The figure was visually inspected after publication.

All 480 historical pools were independently reconstructed from AP records up
to the selected MP. Above-1e-9 membership residual counts were 133, 516 and 962
for N50, N200 and N800 respectively, with maxima 6.898892333e-6,
4.974046747e-6 and 7.448611326e-6. These exact saved coordinates were retained
under explicit user approval, not clipped or certified as inside the boxes.
No green pool exceeded 1e-9.

Twelve of 270 green selections precede the stopping assessment (0/3/9 by N).
All twelve selected assessments have green_pass=false; the later terminal
assessment passes. The plotted IS correctly uses the selected policy's own
pool, not that later pool. This is finite-pool cost, not a full robust supremum.

The independent raw audit covered a fixed N50/radius0.2/seed0 pair, one policy
per neural family; both OOS/IS inputs, fees, journals, feasibility and aggregates
matched. Production verification covered every journal. No independent model
or solver rerun was performed, so the audit does not independently certify
network targets or QP optimality.

Strict-positive fees materially count small positive numerical shipments.
The cached kNN/SRO-FA references use thresholds 1e-6/1e-10 and therefore are
not identical-fee evaluations. They are retained only as labeled descriptive
references. N50/N200 SRO records lack solver status; N800 includes 80 recorded
time-limit incumbents. The gray finite-scenario conditional reference has a
saved relative MIP gap of about 0.40625%, not exact population-bound status.
No historical checkpoint-byte identity is inferred retroactively from today's
freeze. No test-based radius winner is selected or annotated.

## Execution record

Session: `shipping_plot_20260928` (finished).

Working directory: `/data/home/aviyac/retailers_moreau_training_selected/shipping_comparison_20260928`.

Exact launch: `/data/home/aviyac/ap_env/bin/python -B launch.py --output production_v1 --session shipping_plot_20260928`.

During execution, attach with `tmux attach -t shipping_plot_20260928` and check
live-session existence with `tmux has-session -t shipping_plot_20260928`.
The completed session is no longer expected; final status is in the persisted
[invocation](production_v1.invocation.json) and [summary](production_v1/summary.json).
Persistent stdout/stderr is in [production_v1.log](production_v1.log).

Manifest SHA-256: `57ca39132e8be101a4c4a58095de9593fe401f562c437c3e56919480b5545d77`.