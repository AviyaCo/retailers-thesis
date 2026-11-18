# Authorized update of the existing retailer comparison

Approved 16 September 2026: evaluate fixed terminal policies and recover saved
shorter-rule checkpoints, then add both curves to the previous figure. No new
training, AP search, validation/radius selection, refit, retry or fallback.

## Training-only selection, before evaluation

Source: completed run `results/20260916T080636.766867Z`; data seeds 0–9,
network seed 0, 40 train / 10 validation, unchanged nine radii. Freeze all 90
terminal identities and source hashes. For each stored trajectory, scan
chronologically for the first candidate k>=1 with
`abs(AP_incumbent_mean - current_pool_mean) / max(abs(current_pool_mean),1e-9) < .01`.
Return the earliest minimum-AP-mean candidate among 0…k, including nominal.
Do not use later candidates for stopping or selection. The aligned pool mean
matches the preceding selected MP training loss, not the historical buggy
preupdate/postupdate mismatch. This changes stopping/return rules on the saved
Moreau path, **not** all historical SCS training/data/cut rules.

31 positive-radius jobs have only the initial AP and no historical-rule stop:
all seeds at .001/.003/.01, and one seed at .03. No hypothetical later iterate
is fabricated, repeated, or replaced with nominal. Recover the available 49
positive-radius prefixes plus 10 nominal references. The shorter-rule graph
uses only complete ten-seed radius cohorts (.2/.4/.7/1.5); zero is an isolated
nominal marker, not joined across missing radii. The .03 nine-seed cohort is
retained separately in the report/raw artifacts. This is missing evidence,
not non-convergence under a freshly executed historical rule.

## Matched fixed-policy evaluation

Reuse the previous `matched_oos_comparison/compare.py` loader, network targets,
cost arithmetic, fresh unbuffered standard Gurobi QP per scenario, strict
OPTIMAL status and physical tolerance 1e-4. Eight spawned workers; one numerical
thread each. Copy the exact archived 10,000-row float64 demand arrays (seed9999)
and verify file hashes. All 149 new evaluations are actually performed, even
for identical checkpoints. xi1 alone feeds the policy; xi2 only enters cost.
This previously inspected draw is descriptive, not a sealed confirmatory test.
The old numerical evaluator and experiment artifacts are not edited.

Retain every existing line's numeric values and 20th/80th seed percentiles
(not confidence intervals): SCS, previous Moreau, cached SRO-FA and SAA-kNN,
cached 2k clairvoyant SAA-LP and separate 1M conditional reference. Training
and selection protocols differ. The new terminal line includes capped seed9
at .7. Do not label a recovered min-AP checkpoint every-center converged.

## Runtime and execution

Persist nonoverlapping prefix phases (load; AP and pool evaluation through k;
incoming/MP/returned evaluations only through k-1), original timestamps and
parallelism (16 original CCG workers, one thread). These are saved monotonic
phase sums, **not a measured fresh shorter-run end-to-end total**: controller,
persistence, hypothetical selection overhead and new sweep scheduling are
unavailable. No speedup is inferred from an unequal-scope comparison. Record
actual new extraction/evaluation/plot wall time separately; old nominal and
full CCG times stay historical. Each evaluation retains all solve diagnostics,
status, iteration count, phases, seed/radius and checkpoint/input hashes.

From this directory, freeze with
`/data/home/aviyac/ap_env/bin/python -B update_comparison.py freeze`.
Run in tmux `retailers_graph_update_20260916`, launched with
`tmux new-session -d -s retailers_graph_update_20260916 -c /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 bash launch_comparison.sh`.
Exact experiment command: `/data/home/aviyac/ap_env/bin/python -B -u update_comparison.py run`.
Log: `matched_oos_update/campaign.log`; exit: `matched_oos_update/campaign.exit`.
Attach: `tmux attach -t retailers_graph_update_20260916`.
Status: `tmux has-session -t retailers_graph_update_20260916` and
`tail -n 10 matched_oos_update/campaign.log`.

Outputs are isolated under `matched_oos_update/`, including the augmented
`matched_oos_updated.png`, plotted arrays, raw evaluations, frozen identities,
recovery details and runtime report. Existing output is an error, not a resume.