# Paper comparison: Moreau NN, SRO-FA and references

> **Radius selection (2026-11-18):** Protocol A now selects the largest radius within 1e-6 of the minimum validation score. Re-selection, per-seed changes and the paste-ready thesis text: `radius_selection_20261118/`. No figure and no test cost changes. The Protocol A paired tests (`statistical_tests_validation_20261103/`) were regenerated from the new selections, with identical statistics.

> **FINAL (2026-11-02):** `published_straight_epsilon/comparison_R3.*`, `published_straight_epsilon/comparison_R10.*` and
> `shipping_published_straight_epsilon/comparison_shipping.*` are the final paper figures (`render_epsilon.py`). They are
> identical to `published/` and `shipping_published/` (2026-10-26, kept unchanged) except the x-axis label: straight
> epsilon U+03F5 instead of the curly U+03B5 that matplotlib draws for both `\epsilon` and `\varepsilon`.
> Statistical tests (NN vs SRO-FA, paired between seeds) and the R=10 adversary timing: `statistical_tests_20261102/RESULTS.md`.
> Paired NN vs SRO-FA tests at validation-selected radii (shared 80/20 split, Protocol A; 2026-11-03):
> `statistical_tests_validation_20261103/RESULTS.md`.

Display-only comparison using the completed October 11–14 full-N NN campaign
and the original saved SRO-FA, SAA-kNN and reference results.
No fitting, training, policy selection or scenario evaluation is performed.
Original publications and saved data remain unchanged.

- [R=3 PDF](published/comparison_R3.pdf) / [PNG](published/comparison_R3.png)
- [R=10 PDF](published/comparison_R10.pdf) / [PNG](published/comparison_R10.png)
- [Shipping PDF](shipping_published/comparison_shipping.pdf) / [PNG](shipping_published/comparison_shipping.png)
- [Source checksums and display protocol](published/provenance.json)

All figures are exactly **17 cm wide by 5.5 cm high**, including the legend,
with no overall title. Panel headings identify N=50/200/800. The horizontal
axis is epsilon_N. Both NN curves use the green Moreau stopping checkpoints:
solid **NN policy (out-of-sample)** and dashed **NN policy (in-sample)**.
Both NN curves and their out-of-sample band are displayed in blue; "green"
still identifies the unchanged underlying Moreau experiment, not its display
color. SRO-FA is red; SAA-kNN is a gold horizontal out-of-sample mean with
its saved pointwise seed 20/80 percentile band.
All saved horizontal references are retained. No black or SCS curves are displayed.

Bands are pointwise 20th/80th percentiles across ten training seeds, not
confidence intervals. Lines connect actual saved radii only; they do not
represent interpolated evaluations. In-sample values are means of per-center
maxima on the policy's own finite pool, not full adversarial suprema.
Neural training counts are now **50/200/800 for total N=50/200/800**.
These replace the earlier 80/20 policies (40/160/640). The existing full-N
campaign selected nominal learning rate with train-only five-fold CV, then
performed fresh all-N refits and stopping-checkpoint CCG. All three cases have
270 converged configurations, ten seeds and nine radii. Rendering verifies
pinned plot hashes, the training seal, all 270 training-summary hashes per case,
Ntrain=Ntotal/Nval=0, and identical comparator test arrays/seeds.
No new training or evaluation is performed by this revision.

The R=3 SAA-LP and conditional references and the R=10 empirical clairvoyant
reference retain their original values and historical qualifications.
They are **not newly certified population lower bounds**. The already-known
10,000-scenario test (seed 9999) is descriptive, not a fresh holdout.
Historical SRO-FA strict solver-status certification remains unavailable.
Use these caveats in the manuscript caption rather than crowding the figure.

From the experiment root, using the existing environment:

```bash
../ap_env/bin/python -m unittest comparison.paper_figures.test_render
../ap_env/bin/python -m comparison.paper_figures.render
```

Rendering verifies the original JSON checksum sidecars and recomputes the
saved per-seed statistics using the shared plot validator. It refuses to
overwrite an existing `published` directory. The PDFs preserve the exact page
size; do not use tight cropping when exporting them.

## Shipping

The shipping figure uses the completed full-N NN campaign with the October 5
`production_v2` comparators: the same blue NN, red SRO-FA, gold SAA-kNN and gray conditional
reference presentation, without the historical Clarabel curves. The saved shipping
source is SHA-256 pinned; values, seed bands and reference level are unchanged.
Run `../ap_env/bin/python -m comparison.paper_figures.render_shipping` from
the experiment root. Output is exclusive to the new `shipping_published`
directory; [shipping provenance](shipping_published/provenance.json) preserves
the original caption and reference metadata.

Shipping-specific caption qualifications: neural evaluations use common strict
Gurobi projection with fixed charges for raw q2 > 0, not native Moreau inference.
Full-N shipping training used the approved homogeneous advisory activation and
duplicate checks, not exact hard-fee robust certification; a logged Gurobi retry
occurred for N200/seed9/radius1.5. Original campaign evidence remains unchanged.
Cached SRO-FA uses q2 > 1e-10 and retains time-limited incumbents and missing
statuses; it was not freshly evaluated or certified. Cached shipping SAA-kNN
uses q2 > 1e-6; its saved evaluations are unchanged. The conditional reference
19.56794332142972 is a finite one-million-scenario MIP incumbent, with saved
best bound 19.488449380294433 and relative solver gap 0.004062457654823084
(about 0.406%). It is not an exact population lower bound.
