# Retailer Moreau rerun: training-selected checkpoints

Authorized on 16 September 2026. This isolated rerun leaves all old sources/results untouched and reuses the frozen tested model, native Moreau projection and strict AP helpers. It does not switch to SCS; it adopts SCS's training-loss checkpoint/scheduler/early-stopping rule.

## Fixed protocol

- R3, network16–16, total50 paired observations per data seed0–9, network seed0. Exact previous40/10 split (split seed100000+seed), verified against frozen split artifacts.
- Fresh train40 SAA initialization per seed; identical initial state for each nominal LR .005/.01/.05. Fresh Adam, inventory LR.01, clip1.
- Nominal and MP checkpoint: evaluated **pre-update training best**, improvement strictly greater than1e-4, including incoming policy. Scheduler uses training loss (factor.7, patience150, threshold1e-3); early stopping uses training nonimprovement300, epoch cap3000. Last post-update state and optimizer/RNG preserved separately. No validation inputs in trainer API.
- Only after all three nominal training configurations finish per seed: evaluate their fixed checkpoints on validation10 and select LR by minimum mean, smaller LR on ties. This is ordinary hyperparameter selection, not validation-independent end-to-end training.
- Fresh independent CCG at radii0,.001,.003,.01,.03,.2,.4,.7,1.5. At most40 MPs plus the terminal AP, not silently reduced for runtime. All40 APs must pass; unchanged original1% bound-based separation and1e-4 replay gate. Exact raw duplicate policy unchanged; zero capacity buffer. No fallback or failed-AP averaging.
- MP uses mean of forty per-center pool maxima. Terminal fully AP-assessed policy is selected, **not an intermediate validation-best candidate**. Repeated-state closure, stagnation and iteration cap are recorded distinctly; pool separation is not global optimizer convergence. Stagnated/capped but fully assessed terminal policies remain eligible and are explicitly labelled.
- CCG receives only a hashed training-data JSON, never validation arrays. After all90 configurations complete and terminal identities are fixed, evaluate validation and choose radius per seed, ties smaller radius. All90 successful evaluations and final integrity checks are required before authoritative radius selections.
- No test evaluation, new distribution, larger grid, refit, relaxed solver tolerance or changed physical model. Previously viewed test seed9999 is not used. The ten validation observations are reused for LR and radius selection, so selected validation scores are not independent generalization evidence.

## Runtime and failures

Named tmux launch through [launch.sh](launch.sh); one persistent log and exit code per launch; exclusive timestamped structured output per invocation, no resume/overwrite. Nominal10 seed workers (three LRs sequential per worker); CCG16 jobs, one Gurobi/Torch thread per job; later validation one worker. Raw per-model/MP/AP/evaluation wall durations use perf_counter with seed/radius, timestamps, statuses, epochs/solver iterations. Cached nominal timing is separate from CCG time. Parallel model walls are not summed as elapsed. Failures persist; escaped worker timing is unavailable rather than invented, with parent-observed queue-inclusive duration labelled separately. No further phase or final selection after an incomplete phase.

Run command: `/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py`, cwd this folder. Session/log/actual results are recorded after launch. The first development test invocation caught an indentation error before experiments; corrected before the passing test suite.