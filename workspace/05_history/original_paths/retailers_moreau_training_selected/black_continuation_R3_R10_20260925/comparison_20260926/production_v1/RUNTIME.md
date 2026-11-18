# Invocation runtime (not historical training)

Status: failed; mode: production.

| Category | Measured time / status |
|---|---|
| Nominal neural training | Not run; persisted history separate; missing fields unavailable, rerun required |
| Robust neural/CCG AP / MP / total | Not run; persisted history separate |
| SRO-FA partition / optimization / evaluation / total | Not run; original references unchanged |
| Hyperparameter selection / CV / benchmark construction | Not run |
| Final neural evaluation | Production raw per-model results only; preflight probes are not model training |
| preflight_validation (completed) | 196.644047 s / 3.277401 min / 0.054623 h |
| resources (failed) | 0.019243 s / 0.000321 min / 0.000005 h |

| Policy | Process status | Process wall time | Backend model wall time |
|---|---|---|---|

These per-process statuses do not replace the global journal/publication barrier. No failed or unverified result is a certified completed-model aggregate. Raw worker records retain separate OOS/IS phases and iterations. Final elapsed time and this reporting phase are in the sealed summary; whole-child time including serialization is in the launcher runtime artifact. Concurrent wall times are not summed. Missing unreturned timings are unavailable, not zero.

Workers: 16; numerical threads per worker: 1. Environment: null
