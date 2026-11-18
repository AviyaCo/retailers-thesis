# Retailers Moreau trial — N50 R3 net16_v2

## Current status

The first launch failed before nominal training during fixed-weight check
report persistence: Moreau's default unlimited time_limit was represented as
floating infinity, which strict JSON rejects. Both periodic and final writes
failed, leaving only the initial trial.json snapshot. That snapshot is not
evidence of a live run. The log and results/attempt_failure.json preserve the
failure; attempted-run total timing is unavailable, not reconstructed.
Metadata serialization is repaired without changing actual solver settings,
and a regression test covers it. No automatic rerun has been launched.

Separate no-shipping nominal comparison, approved 2026-09-09. Existing result
directories and checkpoints are not overwritten. This is not an all-run migration.

## Fixed protocol

- Canonical definitions: ../retailers_problem/R3/N50/saainit_net16_v2/run_all_net16_v2.py.
  Its CLI and pipeline have import-time side effects, so baseline.py compiles
  only the explicitly named class/function definitions from its AST. Model,
  data generation, SAA initialization, and corrected trainer are reused, not
  reimplemented. The source hash is persisted.
- Data seed0, network seed0, N50/R3, width16--16. One fresh SAA solve supplies
  identical inventory initialization to both newly constructed models. No old
  neural checkpoint is loaded. No held-out/validation/test draws or LR search.
- Historical LR .05 network/.01 inventory, Adam, clip1, plateau .7/patience150,
  improvement threshold1e-4, early stop300, maximum3000epochs.
- Approved checkpoint correction: save evaluated pre-update weights before
  backward/Adam; no extra rollout and no change to optimizer/scheduler steps.
  Both new arms use it. Historical selected checkpoints have the previous
  mismatch and remain historical references. Other script copies not fixed.
- Old arm uses actual SCS eps1e-4/max_iters200000 through CVXPYLayers/diffcp.
  New arm uses Moreau0.4 CPU active-set exact derivatives, unchanged defaults,
  native P=I, q=-z, inequalities -x<=0 and sum(x)<=Q10+1e-6. Gradients through
  both z and Q10 are retained; no smoothing, decision cache, or custom derivative.
- Separate synthetic derivative fixtures test active/inactive capacity with
  h1e-5 central differences. They are not experiment observations. Both fixture
  and checkpoint regression passed (2 tests,0.917s); the checkpoint test first
  reproduced the original error before the correction.
- Before training, compare native outputs on the real initial50 targets to
  strict-OPTIMAL Gurobi QPs (coordinate threshold1e-5). Four alternating
  full-batch forward/backward passes per arm, first separate from3repeat
  statistics. No optimizer updates in that bounded check.
- Then train SCS followed by Moreau in one process. Data/initial weights are
  identical, but numerical gradients may alter learning and stopping epochs.
  Compare total training time AND time per completed epoch; do not claim total
  time differences measure only per-update cost.
- Measure each selected checkpoint's training loss separately with its own
  backend and common strict Gurobi projection (capacity buffer preserved).
  No generalization or derivative-equivalence claim follows.
- Abort and preserve exceptions/nonfinite gradients/native solver failures.
  No trial-level retry, fallback, shorter rescued run, or tolerance relaxation.
  Legacy diffcp SCS retry on inaccurate results and inaccurate-result acceptance
  are retained in the comparator; they are not claimed to be absent.
  SCS status/iteration observations are recorded without altering its legacy
  adapter acceptance behavior. A returned SCS output is not automatically a
  numerical-accuracy certificate.

## Timing and execution

The top-level timer excludes imports, initial metadata loading and final JSON
persistence; it is not full process lifetime. All raw durations are monotonic seconds with UTC start/end, status, seed,
configuration, epochs and thread context in results/trial.json. Epoch records
contain forward time only (explicitly named), while training timers encompass
backward, updates, scheduler, selection and instrumentation. Initialization,
SAA, fixed-weight passes, own-backend evaluation and Gurobi evaluation are
separate. Trial elapsed is sequential elapsed, not a parallel sweep.
Torch/BLAS caps1 do not control diffcp's separate pools or establish native
thread count; shared-host conditions and instrumentation limit causal claims.
Each arm is one seed: count1 training aggregate has mean=median=min=max.

Historical seed0 timing reference: nominal_timing_net16_v2/seed_0_summary.json,
243.749316s training /1716epochs,243.862838s seed workflow. This is not a new
measurement or cached model runtime. Broader historical reports use other
seeds and cannot substitute for this paired trial.

| Category | Trial scope |
|---|---|
| Nominal training | New SCS and Moreau arms, timed separately |
| SAA initialization | One shared fresh solve, separate timing |
| Training-data evaluation | Own backend and Gurobi, separate timing |
| Robust neural AP/MP/CCG total | Not run |
| SRO-FA | Not run |
| CV/LR search/validation/final test | Not run |

Named tmux session: retailers_moreau_N50_R3_seed0_20260909. From this directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/home/aviyac/ap_env/bin/python -B -u run_trial.py > trial.log 2>&1
```

Attach: `tmux attach -t retailers_moreau_N50_R3_seed0_20260909`.
Status: `tmux has-session -t retailers_moreau_N50_R3_seed0_20260909`.
After exit check results/trial.json and trial.log process_exit_code.
Results directory creation is exclusive; never rerun over these artifacts.