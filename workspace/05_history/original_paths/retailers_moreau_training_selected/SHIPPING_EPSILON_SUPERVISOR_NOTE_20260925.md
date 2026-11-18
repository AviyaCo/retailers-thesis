# Shipping activation: provisional epsilon formulation

25 September 2026

## Email draft

**Subject: Fixed shipping fees in the adversarial problem — epsilon formulation**

Hi,

I would like your advice on the fixed shipping fee in our robust neural-policy
model. The intended cost is $f\mathbf{1}\{q_r>0\}$, with $f=0.1$: zero shipment
has no fee, and every positive shipment incurs the full fee.

In the adversarial problem (AP), a binary variable $z_r$ represents activation.
The constraint $0\le q_r\le Qz_r$ ensures that a positive shipment incurs the
fee, but also permits $q_r=0,z_r=1$. Since the AP maximizes cost, it can exploit
this to charge for a shipment that does not occur. The missing implication is
$z_r=1\Rightarrow q_r>0$, which involves a strict inequality that an ordinary
mixed-integer linear formulation cannot directly express.

For now, I would like to use the previous epsilon formulation:

$$
\varepsilon z_r\le q_r\le Qz_r,\qquad
z_r\in\{0,1\},\qquad \varepsilon=10^{-6},
$$

and include $f\sum_r z_r$ in the AP objective. In exact arithmetic, this
prevents a fee at zero shipment and allows either no shipment or a shipment
of at least epsilon. Numerical solver tolerances still require checking.
The MP would retain its differentiable training surrogate
$0.1\sum_r(1-e^{-5q_r})$; hard-cost assessments would remain separate.

There is an important qualification: in the AP, the shipment is determined by
the fixed neural policy and its continuous projection, not freely chosen.
Consequently, adding the epsilon constraint only to the AP excludes demand
scenarios where that policy produces $0<q_r<\varepsilon$. It is therefore a
restricted adversarial search, not automatically an exact reformulation of
the original worst-case problem or a minimum-shipment policy implemented
everywhere. A small epsilon alone does not bound the resulting cost error,
because the shipping fee jumps by the full amount at zero.

Would you recommend treating this as an explicitly documented numerical
approximation for the preliminary experiments, or defining a genuine
minimum-shipment policy consistently across the model? If we retain the exact
$q_r>0$ convention, how would you recommend formulating the AP? In particular,
the discontinuous objective can have a supremum without an attaining scenario,
so an exact approach may need to distinguish worst-case bounds from feasible
adversarial scenarios.

Best,

## Recorded scope and implementation boundary

- The user approved using an epsilon formulation provisionally. The working
  value above is the previous AP value, $10^{-6}$, not a tuned value.
- This records the intended AP approximation; no new shipping AP/CCG code or
  production run is introduced by this note. The existing new shipping code
  implements only the completed nominal stage.
- Keep the smooth MP objective and all saved nominal artifacts unchanged.
  Approval of the AP lower link is not approval to clip shipments, change the
  projection, or silently replace the hard diagnostic $q_r>0$ by a fee cutoff.
- On epsilon-admissible shipments, the intended hard rule charges at
  $q_r=\varepsilon$ as well as above it. The legacy evaluation's strict
  $q_r>\varepsilon$ rule differs at equality and does not enforce a minimum
  shipment. It must not be presented as an identical convention.
- Before a robust launch, specify and test AP replay, fixed-policy arithmetic,
  hard pool evaluation, solver tolerances and any excluded tiny-positive
  shipments. Retain failures or empty restricted AP domains. Do not claim a
  restricted-AP gap certifies separation over the original uncertainty box;
  do not change pools or tolerances merely to make the gap pass.
- This is a formulation/discussion note, not an experiment or results report.
  No new runtime or performance claim is made.

## Repository evidence (not part of the email)

- [Legacy AP lower/upper activation links](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L248-L253).
- [Legacy smooth MP cost and Clarabel projection](../retailers_with_shipping/N50/rho_sweep/run_rho_seed.py#L116-L132).
- [Legacy hard evaluation cutoff](../retailers_with_shipping/N50/rho_sweep/eval_sweep_shipping_10k.py#L71-L80).
- [New nominal smooth and strict-positive diagnostic costs](shipping_nominal_R3_N50_N200_N800_20260924/shipping_model.py#L6-L8).
- [Synthetic ghost-fee and nonattainment diagnostic](shipping_exact_activation_diagnostic_20260924/README.md).

The diagnostic demonstrates possible formulation failures, not that the saved
trained policies exhibit them. This issue is not specific to SCS, Clarabel or
Moreau: changing the projection solver does not implement the missing logical
implication.