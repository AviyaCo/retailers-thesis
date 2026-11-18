# Standalone paper figure: initial-eligible green, IS and OOS

Requested23September2026. Only the current green Neural CCG remains; historical
SCS, purple and black are omitted. Red SRO-FA, yellow kNN and both reference
levels are retained numerically, with neutral-gray reference lines for clarity.
Standalone vector PDF and300dpi PNG; the manuscript is not edited.

As in the SCS figure, **IS means the augmented finite-pool objective**, not the
AP objective and not ordinary empirical loss. For each returned checkpoint,
take its own recorded40 training pools, evaluate its fixed policy using the
existing strict OPTIMAL-only Gurobi projection evaluator, take each center's
maximum, and average over40. Use the pool that trained/assessed the selected
checkpoint, not later pools (two green returns precede stopping). At selected
candidate0 the singleton pools reduce to empirical training cost. Green inserts
only qualifying nonduplicate points, unlike the historical SCS all-AP-point
rule; this redraw does not alter either training protocol. No fresh AP/MP,
training, validation, test or benchmark solves. Saved native pool values are
retained alongside strict replays to disclose arithmetic differences.

All90 policies/pools are hash-bound to the completed minimum-AP comparison,
frozen before solves and verified afterwards. Gurobi/float32 target and float64
cost arithmetic are unchanged from the strict OOS evaluator. No clipping,
retry, missing-seed replacement, or changed solver tolerance. Stop publication
on failure. All OOS and reference numbers are reused unchanged. Mean curves
contain ten seeds; OOS shading is20th--80th seed percentiles, not confidence
intervals. Differences in historical training budgets and previously viewed
test9999 remain; no new statistical superiority claim.

All new90 configuration evaluation wall times, phases, solver counts/iterations,
settings, UTC boundaries, seed/radius/selection IDs, raw IS per-center costs,
source hashes and overall time are written to results/summary.json. Single
CPU worker and one numerical/Gurobi thread. Cached nominal and CCG timings are
retained separately; completed per-radius runtime statistics include
n/mean/median/min/max. No SRO-FA or CV rerun; no missing timing is invented.
Invocation time starts after imports and excludes final-summary serialization.

Working directory: this folder. Exact command:
`/data/home/aviyac/ap_env/bin/python -B -u make_figure.py`.
Launch via `bash launch.sh` in tmux session `retailers_green_paper_20260923`.
Persistent log `campaign.log`, exit `campaign.exit`.
Attach: `tmux attach -t retailers_green_paper_20260923`.
Status: `tmux has-session -t retailers_green_paper_20260923`;
progress: `tail -n 10 campaign.log`.
Outputs refuse overwrite. Caption source: caption.tex (standalone text only).