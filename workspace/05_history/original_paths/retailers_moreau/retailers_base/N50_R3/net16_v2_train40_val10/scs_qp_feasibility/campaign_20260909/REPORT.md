# Direct-QP gradient and speed feasibility

Training/test/optimizer steps: none. No automatic selection.

## Runtime table (seconds; completed configurations only)
| Method/configuration | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| moreau | 10 | 0.211739 | 0.121988 | 0.106421 | 1.056295 |
| scs_eps0.0001_tau0.01 | 10 | 0.907937 | 0.916046 | 0.784610 | 1.026579 |
| scs_eps0.0001_tau0.001 | 10 | 0.872629 | 0.911227 | 0.778308 | 1.028722 |
| scs_eps0.0001_tau0.0001 | 10 | 0.845694 | 0.794783 | 0.776630 | 1.039949 |
| scs_eps1e-08_tau0.01 | 10 | 0.853321 | 0.832022 | 0.795718 | 0.938959 |
| scs_eps1e-08_tau0.001 | 10 | 0.830485 | 0.802320 | 0.792258 | 0.924326 |
| scs_eps1e-08_tau0.0001 | 10 | 0.837448 | 0.797560 | 0.793429 | 0.912065 |
| Gurobi reference + FD | 10 | 1.337523 | 1.293787 | 1.198819 | 1.726032 |
| freeze inputs | 1 | 0.170541 | 0.170541 | 0.170541 | 0.170541 |
| Nominal neural training | 0 | not run | not applicable | — | — |
| Robust neural/CCG total | 0 | not run | not applicable | — | — |
| SRO-FA total | 0 | not run | not applicable | — | — |
| CV / held-out evaluation | 0 | not run | not applicable | — | — |

Serial campaign elapsed: 78.764970 s (1.3127 min; 0.021879 h).

First-call and five-repeat setup/forward/backward distributions are separate in each case JSON.
Case totals include diagnostics/persistence; solver-stage timings exclude JSON writes.
Failed/interrupted cases are excluded from completed aggregates and retained in summary JSON.
LPGD is approximate; perturbation warm starts and serial SCS versus batched native integration differ.
No speed claim is valid without checking raw forward/physical and gradient diagnostics.
Historical training times are not remeasured or inferred from checkpoint loading.
