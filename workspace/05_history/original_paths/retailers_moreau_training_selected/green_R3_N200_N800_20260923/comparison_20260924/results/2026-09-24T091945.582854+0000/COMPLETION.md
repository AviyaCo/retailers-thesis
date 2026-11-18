# Completed and independently verified comparison

Production v2 completed2026-09-24 at09:25:15UTC, exit0. All780 policy
evaluations completed; no failed, incomplete or substituted policies.

- [Three-panel PNG,300dpi](comparison_R3.png)
- [One-page vector PDF](comparison_R3.pdf)
- [Raw figure data and per-seed provenance](plot_data.json)
- [Full runtime report](RESULTS.md)
- [Structured runtime aggregates](runtime_aggregates.json)
- [Completed summary](summary.json)

The figure includes green Moreau OOS/IS, SCS-trained OOS/IS, SRO-FA OOS,
SAA-kNN OOS and the existing gray references at N50,N200,N800. Green N50
is reused from the completed strict evaluation; all600 SCS and180 newgreen
policies were freshly evaluated. The optional `--green-comparison` entrypoint
in the existing combined plot script reproduces the completed figure.

## Validation

Independent read-only verification reconciled all1,560 current journals:
7,800,000 OOS and718,113 finite-pool IS rows, totaling8,518,113. Every row
was recorded `OPTIMAL` and accepted; maximum recorded physical residual0.
All4,369 original/frozen inputs matched their hashes. Independently reconstructed
learned-state hashes matched all747 distinct checkpoint paths, and every
selected pool and all ten-seed plotting cohorts matched their frozen identities.
The two green earlier-return cases used their own pools, not later stopping pools.
The PNG matched an in-memory reconstruction; the PDF is one page with vector
drawings and equivalent page content. No fresh solves were needed for the audit.

## Timings

Host gold-srv2.iem.technion.ac.il, Xeon Gold6354;32 actual worker PIDs,
one numerical/solver thread per worker. Current timings are monotonic seconds;
historical training and comparator timings remain separate.

| Current pipeline phase | Seconds | Human-readable |
|---|---:|---|
|Preflight binding validation|11.568639|11.57s|
|Full production freeze|76.882013|1min16.88s|
|Matched test generation|0.013969|0.014s|
|Manifest sealing|7.786995|7.79s|
|780-policy OOS/IS evaluation sweep|226.575250|3min46.58s|
|Final verification|11.158709|11.16s|
|Plot data and PNG/PDF generation|5.237533|5.24s|
|Reporting|0.032308|0.032s|
|**Total successful production invocation**|**341.443545**|**5min41.44s**|

Total includes reporting, but excludes final timing-footer/JSON serialization.
Phase sums omit intervening orchestration overhead. Per-policy OOS/IS
mean/median/min/max statistics by method,size,radius are preserved in the report
and JSON; concurrent model walls are not summed as elapsed time.

| Previously measured / not rerun | Seconds | Scope |
|---|---:|---|
|Green N200/N800 nominal training|206.209766|60-model parallel training stage, previous campaign|
|Green N200/N800 CCG|1045.852028|180-model parallel robust stage, previous campaign|
|Green nominal LR validation|5.899489|Previous campaign|
|Green final radius validation|6.362978|Previous campaign|
|SRO-FA total/partition/solver/evaluation|Historical only|Per-radius saved timings in RESULTS; not newly measured|
|SCS full training/CCG|Unavailable as certified full runtime here|Saved fields retained; rerun needed to measure missing scope|
|CV|Not run|Single paired split; no new benchmark selection|

Attempt history is separate: preflightv1 failed61.903557s; v2 passed112.841658s;
productionv1 failed92.853639s before any policy solve; preflightv3 passed
114.946501s with44tests and20training-row QPs. These are not added to or
misreported as per-model evaluation times. All earlier artifacts remain preserved.

## Example fixed-radius comparison (rho=0.4)

Arithmetic means over all ten seeds, lower OOS cost is better; these are not
test-selected optima or validation-selected deployment summaries.

| Ntotal | Green OOS | Green finite-pool IS | SCS OOS | SCS finite-pool IS | SAA-kNN OOS | SRO-FA OOS |
|---:|---:|---:|---:|---:|---:|---:|
|50|20.529497|20.022256|20.604119|19.952444|20.797716|21.285986|
|200|19.873891|20.366686|19.789189|20.512046|20.268573|20.454089|
|800|19.753921|20.545928|19.619379|20.430028|19.998804|19.960933|

No superiority or backend-only causal conclusion is established. The complete
actual radius grids are shown, with no invented evaluations at missing radii.

## Publication caption and qualifications

Fixed-radius policy comparison for three retailers without fixed shipping cost,
at total sample budgets50,200,800. Green and blue solid curves show mean OOS
cost for green-stopping Moreau-trained and historical SCS-trained policies;
their dashed curves show the mean of per-center maxima on each returned
checkpoint's own saved finite training pool. Red and yellow curves are
historical SRO-FA and SAA-kNN OOS references. Gray lines retain the historical
SAA-LP and conditional-clairvoyant reference estimates, not certified numerical
population bounds. Shading denotes pointwise20th/80th percentiles across ten
training seeds, not confidence intervals. OOS uses the same10000-pair seed9999
draw; this historically inspected test supports descriptive comparison only.
No checkpoint or radius was selected using these test scores. New green
N200/N800 and all SCS policies use fresh strict-OPTIMAL Gurobi replay; green
N50 and baseline curves are cached with provenance, not new baseline solves.
Green trains on40/160/640 observations and reserves10/40/160 for validation,
whereas the historical baselines train on50/200/800; backend effects are not
isolated. IS is a finite-pool diagnostic, not an adversarial supremum or a
comparison of identical scenario pools. User-approved historical SCS pool
coordinates remain unchanged despite348 saved-point observations above the
1e-9 membership reporting threshold (maximum7.151707322877732e-6); this does
not certify those pools' exact box membership. Green membership checks and
the strict physical projection acceptance contract remain unchanged. Historical
baseline solver certification is not inferred from the fresh neural replay.

## Published file identities

- PNG SHA256: `2aef9ab4e55cf1006c8d961cd2488f07c186c7b68f1a1faa31fefb2f1c030447`
- PDF SHA256: `5df9da00a84334485a7e9ac6d838b8fea82e559a262662456a3d7cab042bf4d2`
- Manifest SHA256: `482f729123d52ad4c547817f309372c8d3ad00cc18bd63f4d7171e7925ffebcd`

This post-run report does not change the frozen experiment or original manuscript.