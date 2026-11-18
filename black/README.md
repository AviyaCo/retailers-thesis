# Black

Edit shared code in [../src](../src), not a colour-specific implementation.
The black rule requires each center's bound/pool residual to be at most one
percent. The forty-MP budget includes retained green work when continuing.
Return the stopping checkpoint after validating its inclusive assessed prefix,
never an earlier minimum-AP checkpoint. Candidate zero may stop with zero MPs.
At the forty-MP cap, `allow_capped_black_selection=True` explicitly permits
returning the last fully assessed checkpoint (candidate 40); the result remains
nonconverged. Without that opt-in, no capped checkpoint is published. Native v3
summaries record the stopping-checkpoint return rule explicitly.

## Historical source and result directories

- [R3/R10 continuation archive](runs/R3_R10_continuation/)
- [Completed continuation runtimes](runs/R3_R10_continuation/production_v1/RUNTIMES.md)
- [Historical N50/R3 source and results](runs/R3_N50/)
- [Completed test figures](../comparison/black_green_figures/)

These internal links lead to verified local copies of the complete historical
scripts, checkpoints, results, logs and provenance, not newly executed
shared-code runs. N50/R3 is the same local archive linked under green. Editing
through a link changes the local copy; keep archived bytes unchanged and back
up the entire main directory.

These results comprise continued green trajectories, fresh R10 radius-two
trajectories, and historical N50/R3 black trajectories. They are not relabelled
as a homogeneous fresh black training sweep.
Historical min-AP selections and archived configurations remain immutable;
they are not relabelled as stopping-checkpoint results. Historical terminal-only
re-evaluation is a separate comparison, not a rewrite of these archives.