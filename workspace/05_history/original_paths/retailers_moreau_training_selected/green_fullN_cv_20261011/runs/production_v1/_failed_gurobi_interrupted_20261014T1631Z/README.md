# Archived failed attempt: shipping_R3 N200 seed 9 radius 1.5

Failed 2026-10-14T16:31:54Z (resume_v1, 56 workers): after 4 MPs, the fixed-policy
shipping AP for sample 89 of candidate_04 returned Gurobi status 11 (INTERRUPTED)
after 0.004 s with 0 nodes and no incumbent (no time limit or callback is set), so
reading ObjBound raised GurobiError. No other AP solve in the run returned status 11.
The configuration was treated as a transient infrastructure failure and, with user
approval, rerun from scratch with identical inputs and settings via runner.invoke.

Attempt 2 (16:48-16:53Z, identical settings, `attempt2/`) failed identically: same
candidate hash, same sample 89, status 11. An isolated diagnostic showed Gurobi
13.0.1 finds incumbents (best 12.10772, gap 0.83%) then reports "Solve interrupted
(error code 10005)" with solution count 0; with Heuristics=0 or Presolve=0 the same
model is OPTIMAL at 12.10772. With user approval the configuration is rerun via
../../../green_fullN_cv_20261011_resume/rerun_with_retry.py: a single Heuristics=0
retry only for AP solves that return status 11, logged per row.
