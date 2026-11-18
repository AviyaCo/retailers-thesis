# Shipping fixed-policy comparison — 28 September 2026

Authorized: reproduce the R3/R10 three-panel comparison style for shipping,
N=50/200/800. The user explicitly chose fresh common Gurobi projection replay
of both neural families, with raw q>0 fees and the historical 10,000-row test
draw (seed 9999). This is not native Moreau evaluation. No clipping, fee cutoff,
projection capacity buffer, solver retry, failed-model omission or changed
checkpoint choice is permitted. Each row requires Gurobi OPTIMAL, finite
values, and physical residual at most the shipping acceptance tolerance 1e-6.
Gurobi numerical defaults and the prior R3 fresh per-row QP formulation remain
unchanged. All network targets are computed in one float32 call per evaluation
set, then promoted exactly to float64; no tuning of batch shape from outcomes.

Freeze before held-out evaluation: 270 green policies (265 completed original
plus exactly five advisory continuations), 480 historical policies (nominal
plus fifteen positive radii through 1.5, ten seeds, three sizes). Historical
shipping training used Clarabel, not SCS; N800 uses corrected minibatch runs.
Green returns its existing earliest-minimum-mean-AP checkpoint through the first
mean-gap stop. It is a mixed acceptance-protocol cohort, not a newly homogeneous
270-case training experiment. Four continued selected own-pool assessments do
not themselves pass green; no selected-policy/per-center certificate is implied.
No trained shipping black comparator exists and none is manufactured here.

IS is mean per-training-center maximum on the returned checkpoint own saved
finite pool, not AP mean, terminal/future pool, or full-box robust cost. Green
membership remains strict at 1e-9. The user separately approved retaining exact
historical pool coordinates with ALL positive box residuals disclosed, as in
R3: do not clip/delete points or label their pools box-certified. Test costs
never determine policy choices, radius grids or feature/batch configuration.
No radius validation or best-test-radius annotation is added.

Blue/green neural IS/OOS are newly replayed with identical raw q>0 costs.
Historical SRO-FA and SAA-kNN values remain cached descriptive references with
their distinct q>1e-10 and q>1e-6 fees. Saved SRO time-limit incumbents and
unknown solver statuses retain those labels. Gray is the saved finite
one-million-scenario conditional MIP estimate, not an exact population bound.
The test seed was viewed previously and historical grids were test-informed;
this comparison is descriptive, not fresh confirmatory evidence. Training
budgets differ: green 40/160/640, historical 50/200/800. Bands are pointwise
20th/80th percentiles over ten seeds, not confidence intervals.

Sequence: immutable input copies and hash-only historical dependency bindings;
analytic tests; six spawned training-only policy probes (two original centers
per method/size); reverify; fixed 16-worker resource gate; seal test and final
manifest; all 750 evaluations; full raw-journal/cost verification; final hashes;
only then PNG/PDF and plot-data publication. Probe/freeze/replay failures persist
and abort publication. No training, CV, refitting, new SRO/kNN/oracle fitting or
test-based selection runs. Sources/README must not change after freeze.

One Torch/Gurobi thread per worker; sampled affinity, cgroup, load and memory
headroom must support all 16 workers, without an automatic degraded retry.
Structured perf_counter timing stores current OOS, IS, verification, freeze,
tests, preflight, rendering and whole invocation separately. Historical
nominal/AP/MP/SRO records are preserved separately; missing timings stay
unavailable (rerun needed). Concurrent model walls are not elapsed sweep time.
Raw solver status/iteration/time records and q/sign/cost arrays are preserved.
The launcher uses named tmux, persists exact command/cwd/stdout+stderr and UTC
boundaries, and never overwrites an output. Final artifacts live under the
chosen output's published directory only after all gates pass.