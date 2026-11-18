# Shipping validation-selected radii: strict positive shipping fees

Authorized 6 October 2026. Evaluate existing green STOP and historical blue
policies only. No retraining, solver substitution, altered stopping criteria,
test-driven radius selection, refitting or silent omission of failures.

- Green: Ntotal 50/200/800, training 40/160/640, saved holdout 10/40/160,
  seeds 0–9, original split seed 100000+s. The holdout was used for LR selection.
  Nine radii 0/.001/.003/.01/.03/.2/.4/.7/1.5; 265 original + five advisory policies.
- Blue: training 50/200/800, independent validation 13/50/200, seed 88880+s;
  original 15 positive radii through1.5, not nominal radius zero. Historical
  return rules, caps and corrected N800 minibatch policies retained unchanged.
- Both: unchanged common Gurobi projection, float32 full-set network targets,
  float64 projection/cost arithmetic, raw 0.1 per q>0, OPTIMAL required and
  physical residual at most1e-6. No clipping, threshold, retry or fallback.
- Select separately for each method/N/seed by minimum validation mean, breaking
  exact ties toward smaller radius. All720 arms must complete. All60 choices are
  SHA256 sealed before any test costs are opened. Cached test evaluation uses
  the identical selected learned state and already-established strict-q>0
  10000-row seed9999 replay; verify raw costs, journals and result hashes.
- The historically inspected test and test-informed grids remain descriptive,
  not fresh confirmatory evidence. Different training/validation protocols and
  reused green LR holdout preclude a matched-protocol comparison claim.

Execution: eight spawned workers, one Torch/Gurobi thread each, in named tmux
session shipping_val_qpos_20261006. The physical consolidation wrapper exposes
original paths read-only; only a fresh output directory is writable. Launch from
this directory using the existing ap_env Python and launch.py with output
production_v1 and the named session. launch.py writes the exact command, cwd,
session, persistent log and whole-child monotonic runtime receipt beside output.

Per-arm and per-seed-grid timing, UTC boundaries, status, seed, N, radius,
solver iterations and threads are persisted. Failed runs remain explicit.
Prior nominal/CCG and test timings are historical, not current invocation work.
No SRO-FA/CV runs; unavailable timings are not inferred. Current cached-test
lookup/raw verification is timed separately from original test evaluation.
No sums of concurrent worker wall times are presented as sweep elapsed.

Historical sources/results remain immutable. New sources become frozen at
launch. Results and raw timing are written under the fresh output directory.