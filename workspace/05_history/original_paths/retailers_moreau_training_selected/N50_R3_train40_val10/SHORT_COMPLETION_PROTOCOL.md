# Complete the missing green points; preserve historical experiments

Authorized 16 September 2026 by the request to add the missing green points.
Continue only 31 absent positive-radius cases: seeds 0–9 at .001/.003/.01 and
seed 0 at .03. Reuse each own saved nominal policy, forty training centers,
initial pool and fully validated initial AP. No nominal retraining, changed
learning rate, validation selection, new radius or test-based selection.

Use the frozen training-only trainer and AP assessor. Insert the same
numerically violating nonduplicate AP points. Fresh Adam per MP, full batch,
3000 maximum epochs, training patience300, training-best preupdate selection,
original per-seed learning rate, unbuffered Moreau, all strict AP gates unchanged.
After each complete assessment k>=1, stop at the first absolute mean AP/pool
gap below 1%, then retain the earliest minimum-AP checkpoint among 0…k.
Every-center closure is still recorded, but is not this experiment's stopping
rule. Retain the original 40-MP safety cap and explicit stagnation/failure
outcomes; no retry or degraded fallback. Any incomplete case blocks a full
green curve. This completes the aligned-loss Moreau comparison, not the full
historical SCS algorithm.

Freeze all31 selected identities before evaluating any. Evaluate31 policies
using the unchanged strict Gurobi evaluator and byte-identical archived
10000-row seed9999 arrays. Reuse59 previous green/reference evaluations
explicitly; do not claim90 fresh evaluations. Join only complete ten-seed
cohorts. Preserve previous curves and plot files; write a new completed plot
under short_rule_completion. This previously inspected test is descriptive,
not a sealed confirmatory set. The supervisor manuscript is not modified.

## Timing and comparison

Record per-iteration AP/MP, per-worker continuation invocation, and the whole
16-worker continuation sweep using perf_counter; one numerical/solver thread.
Distinguish saved initial AP/loading/nominal time from new work. Comparable
phase sums include original loading and all pool/AP/MP phases through stopping,
excluding following MP, nominal training and selection/uninstrumented overhead.
They are NOT fresh shorter end-to-end or sweep times. Evaluation uses8 workers,
is measured separately, and all31 policies actually solve10000 new QPs each.

Compare full Moreau end-to-end and phase sums against short phase sums, with
mean/median/min/max,n and capped/incomplete statuses separate. Historical
plotted SCS fixed-radius totals are zero placeholders: report unavailable,
never zero work. Show measured selected-radius SCS timing reruns separately as
UNMATCHED references (50train, different trajectories/radii, 30 original
concurrent jobs); no numerical solver-only speedup. Nominal, LR search,
evaluation and cached SRO-FA/unrun CV timing scopes remain separate.

## Launch and artifacts

Working directory: /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10

Freeze: /data/home/aviyac/ap_env/bin/python -B fill_short_completion.py freeze

Exact experiment: /data/home/aviyac/ap_env/bin/python -B -u fill_short_completion.py run

tmux session: retailers_short_completion_20260916

Launcher: bash launch_short_completion.sh

Log: short_rule_completion/campaign.log; exit: short_rule_completion/campaign.exit

Attach: tmux attach -t retailers_short_completion_20260916

Status: tmux has-session -t retailers_short_completion_20260916

Progress: tail -n 15 short_rule_completion/campaign.log

Outputs: frozen manifest, per-seed/radius continuation summaries/checkpoints,
per-iteration training/AP journals, fresh evaluation journals, completed plot,
plot_data.json, runtime_comparison.json, REPORT.md and final summary.json.
Existing output refuses overwrite/resume. Source hashes verified before/after.