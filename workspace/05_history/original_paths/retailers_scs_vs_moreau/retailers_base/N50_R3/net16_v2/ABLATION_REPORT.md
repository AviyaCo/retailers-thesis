# Formulation / solver / integration ablation — 9 September 2026

Completed at 11:14:46 UTC in **26.591607 seconds**. Three saved checkpoints,
50 original training observations each, first pass plus five subsequent passes
per arm. No training, LR selection, validation/test evaluation or robust runs.
All source, input, model and prepared-QP hashes remained unchanged. First-pass
A/B projections matched exactly at every checkpoint. Six regression tests
passed in 0.907 s before launch.

## Arms and controls

- A: CVXPYLayers/diffcp/SCS, original squared-distance SOC formulation.
- B: native SCS using A's captured canonical matrices and compiler recovery,
  retaining solver setup between passes.
- C: native SCS direct QP, retaining solver setup between passes.
- D: native Moreau active-set direct QP with its differentiable native batch.

Original neural operations remain float32; identical projected inputs are then
promoted to float64. All arms retain the historical capacity buffer 1e-6.
C/D use P=2I and c=-2z to preserve the original sum_squares objective scale,
dropping only the constant z'z. Historical Moreau used P=I: this is a controlled
ablation, not an exact replay of that timing trial. Captured SOC A is 9x4;
direct-QP A is 4x3. SCS eps_abs=eps_rel=1e-4 and max_iters=200000 are unchanged.
Moreau uses its native exact active-set settings, not accuracy-matched SCS
tolerances. Outputs are not clipped or replaced to repair feasibility.

One benchmark worker on shared gold-srv2, Xeon Gold 6354; Torch and diffcp
forward/backward workers explicitly 1, BLAS/OpenMP/NumExpr/Rayon caps 1, Gurobi
reference threads 1. Actual native thread counts are not independently measured.
No solution warm starts are supplied. Reused SCS objects may retain adaptive
scaling state even with warm_start=False: B outputs change across repetitions.

## Repeated forward timing

Milliseconds per complete 50-projection batch; each cell is
**mean / median / min / max**, n=5 completed subsequent passes. First passes
and constructor setup are excluded here and persisted separately below.

| Checkpoint | A: framework SCS | B: cached SOC SCS | C: cached QP SCS | D: native QP Moreau |
|---|---:|---:|---:|---:|
| Initial | 63.780 / 64.804 / 60.871 / 66.259 | 2.885 / 2.894 / 2.458 / 3.403 | 1.945 / 1.927 / 1.709 / 2.270 | 4.181 / 3.364 / 3.117 / 7.244 |
| SCS-selected | 89.739 / 88.138 / 87.349 / 97.219 | 5.103 / 5.289 / 4.265 / 6.182 | 2.073 / 2.086 / 1.874 / 2.283 | 3.890 / 3.062 / 2.883 / 7.423 |
| Moreau-selected | 78.050 / 77.972 / 77.687 / 78.495 | 6.880 / 6.104 / 5.849 / 9.834 | 1.953 / 1.976 / 1.855 / 2.019 | 2.898 / 2.909 / 2.687 / 3.146 |

## Repeated backward timing

Same fixed random upstream gradient, seed20260909. Gradients are with respect
to target and capacity leaves, not full neural parameters. Mean / median /
min / max milliseconds, n=5; different differentiation algorithms.

| Checkpoint | A: diffcp/SCS | D: Moreau |
|---|---:|---:|
| Initial | 9.367 / 9.375 / 8.848 / 9.822 | 3.063 / 2.645 / 2.164 / 5.100 |
| SCS-selected | 9.295 / 9.150 / 9.065 / 9.922 | 2.718 / 2.208 / 2.162 / 4.749 |
| Moreau-selected | 9.199 / 9.198 / 9.133 / 9.291 | 2.181 / 2.186 / 2.108 / 2.245 |

B/C have **no measured backward implementation**, not zero backward cost.
Installed diffcp's usual implicit derivative modes do not accept a direct P;
substituting LPGD would change differentiation and was not done. A forward
includes derivative preparation; C is forward-only, D builds its autograd
state. Their timings are not equal-work forward/backward comparisons.

## Setup and first pass

Each entry: constructor setup / first-forward milliseconds. A setup includes
model/layer construction, B/C native solver construction for all50 samples,
D wrapper construction. D native batch/matrix setup is lazy inside first
forward. A first forward also captures canonical matrices. These are not
identical setup boundaries, nor process-cold measurements.

| Checkpoint | A | B | C | D |
|---|---:|---:|---:|---:|
| Initial | 9.672 / 77.424 | 1.899 / 2.889 | 6.897 / 1.852 | 0.263 / 4.224 |
| SCS-selected | 12.071 / 103.954 | 1.945 / 38.381 | 8.955 / 2.965 | 0.393 / 3.581 |
| Moreau-selected | 5.856 / 82.913 | 1.699 / 18.287 | 6.671 / 2.295 | 0.261 / 3.223 |

Within subsequent A forwards, mean measured native SCS setup/solve batch
times are 2.027/1.848 ms initially, 1.948/30.093 ms at SCS-selected weights,
and 1.913/19.369 ms at Moreau-selected weights. The remaining A time includes
canonicalization, conversion, derivative preparation, recovery, instrumented
warning handling and other overhead; it must not all be called wrapper cost.

## Accuracy alongside timing

Maximum coordinate error versus Gurobi across the five subsequent passes:

| Checkpoint | A | B | C | D |
|---|---:|---:|---:|---:|
| Initial | 9.89e-5 | 2.77e-4 | 2.11e-7 | 2.75e-12 |
| SCS-selected | 1.35813 | 1.33964 | 1.12e-3 | 1.03e-11 |
| Moreau-selected | 0.359383 | 0.943934 | 9.17e-4 | 5.42e-9 |

At SCS-selected weights, worst capacity excess over those passes is
0.547785 (A), 0.542147 (B), 0.000361347 (C), and 4.58e-15 (D).
At Moreau-selected weights, C's worst excess is 0.00161388 versus 2.80e-15
for D. Direct-QP SCS greatly improves accuracy over the lifted formulation,
but is still not accuracy-equivalent to Moreau. Solved status alone is not a
physical-feasibility certificate at a specified absolute tolerance.

## Conclusions

1. The observed speedup is **not attributable solely to Moreau**. Native
   direct-QP SCS has the fastest repeated forward times in this test.
2. Changing cached native SCS from SOC to QP improves mean forward time by
   approximately 1.48x, 2.46x and 3.52x across these checkpoints, while greatly
   improving trained-checkpoint accuracy at the unchanged SCS settings.
3. The much larger A/B difference combines derivative preparation, integration,
   setup reuse, instrumentation and changed cached-solver trajectories. It is
   not an isolated percentage contribution of Python/framework overhead.
4. C is about 1.48--2.15x faster than D for repeated forward solves here, but
   has no backward implementation and returns substantially less accurate
   trained-checkpoint projections. This does not establish faster accurate
   differentiable SCS training.
5. Moreau backward is about 3.06--4.22x faster than A's diffcp backward here.
   No equivalent direct-QP SCS backward was measured; gradients need not match
   when numerical forward points differ.
6. Equal-accuracy and fresh-solver-state controls would be required for stronger
   causal attribution. No tighter-tolerance runs, new derivative method or
   training changes were silently added.

## Runtime, files and reproduction

Total benchmark wall time: 26.591607 s (0.4432 min), including setup, reference
evaluation and persistence. Checkpoint totals: 4.379020, 8.078140, 11.995805 s.
Nested timers overlap; do not add them as independent elapsed phases.
Nominal training, robust CCG, SRO-FA, CV and final-test evaluation: **not run**.

- Script: [benchmark_ablation.py](benchmark_ablation.py).
- Tests: [test_ablation.py](test_ablation.py).
- Raw phases, solver outcomes and reference evaluation timings:
  [ablation_results/ablation.json](ablation_results/ablation.json).
- Log: [ablation.log](ablation.log).
- tmux session at launch: `retailers_ablation_20260909` (now completed).
- Working directory: this directory.
- Command: `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 /data/home/aviyac/ap_env/bin/python -u benchmark_ablation.py`.
- The script refuses to overwrite the existing output directory.