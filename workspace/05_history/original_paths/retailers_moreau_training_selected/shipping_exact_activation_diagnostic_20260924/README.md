# Isolated exact-activation shipping diagnostic

## Boundaries and interpretation

Only deterministic synthetic R3 fixtures are constructed. N=50/200/800 are prospective campaign labels, not sample sizes used here. No production imports, source edits, checkpoint loading, training, optimization of neural weights, held-out/test data, CV, or OOS evaluation. No production policy is selected or changed. No nominal/CCG/SRO-FA experiment is launched. Their runtimes are **not applicable (not run)**; historical runtimes are not inferred. This diagnostic does not change the repository experimental method or protocol; modeling-note integration is left to the parent task.

One process/worker; Torch intra/inter-op, BLAS and Gurobi threads are one. Seed 0 fixes otherwise overwritten neural initialization; every coefficient is set explicitly. No data split or temporal boundary applies because there are no sampled or loaded data. Existing concurrent tmux work is neither inspected for results nor modified.

The fee is exactly 1/10 per strictly positive component. Economic fractions are explicitly rational constants. Binary numerical inputs and network coefficients use `Fraction.from_float(float(value))`, without denominator limits, epsilon thresholds, clipping policy changes or fallback arithmetic. The standard sorting/simplex projection formula is evaluated over `Fraction` solely as a three-dimensional synthetic oracle: standard floating NumPy/Torch projection implementations do not provide exact rational support certification. Dense affine/ReLU Fraction operations likewise establish the exact real interpretation of stored network coefficients, not an alternative production network.

## Seven checks

1. Target (2,1,-1), capacity 1 gives (1,0,0). The feasible perturbation (1-d,d,0), d=2^-30, has zero **primal feasibility** residual, squared-distance objective gap 2d², and one extra fee. No claim of zero KKT/optimality residual is made.
2. A positive 2^-21 component incurs the fee.
3. Zero capacity produces exactly zero and no fee.
4. A deterministic Torch `nn` 7→16→16→3 network with two hidden ReLUs embeds ReLU(x+I-1); x=2^-25, I=Q=1. Native float32 forward is observed, not assumed. The same stored coefficients and inputs are evaluated unfolded as exact rationals. A backend that does not reproduce cancellation is explicitly labeled NOT_REPRODUCED, not silently adjusted.
5. At capacity 2^-26, exact target (1,1-2^-25,-1) has one positive projection component. Casting that target to float32 creates (1,1,-1), with two positives. Both projections are exact rational projections of their respective targets.
6. An actual Gurobi indicator-complementarity KKT MIP for target (-1,-1,-1), Q=1, maximizes sum(zship), with q<=Q*zship and **no lower link**. Settings: Threads=1, FeasibilityTol=IntFeasTol=OptimalityTol=1e-9, NumericFocus=3, MIPGap=1e-4. OPTIMAL is required. This is explicitly a **closed shipping relaxation, not an AP solution**. Independent exact feasible witness q=0, mu=1, lambda=0, slack=1, both complementarity flags=0, zship=1 attains the universal upper bound sum(zship)<=3; true projected shipments are identically zero. Ghost fee is 3/10. The demonstration is not dependent on numerical tolerance admitting tiny positive q.
7. Same-radius boxes: radius 1/2, centers x1=(1/2)^3 and x2=(5/2)^3, Q=1, first-stage inventory (3,4,4), target (t,0,0), t=x1[0]. Exact costs at decreasing positive t reproduce C(t)=33/10-t/20+(1/10)1{t>0}. Other coordinates are checked at all 32 corners for each t.

### Mathematical nonattainment argument (not inferred from a finite chain)

For retailer 1, the three recourse affine terms simplify to x2/2-3/2-t/20, (3-t-x2)/20, and t/2-3/2-x2/20. Over 2<=x2<=3 and 0<=t<=1 their maximum over x2 is (1-t)/20, attained at x2=2. For each of retailers 2 and 3, maximize max((a+b-4)/2,(4-a-b)/20,(a-4)/2-b/20) over 0<=a<=1, 2<=b<=3: the maximum is 1/10, at a=0,b=2. These are convex maxima of affine functions, so endpoints/corners suffice. Adding first-stage 61/20 and fees yields the stated C(t). For t=0, C=33/10. Every t>0 gives C=17/5-t/20<17/5, but the limit as t↓0 is 17/5. Thus the supremum 3.4 exists and is not attained. A finite sequence checks implementation; the inequalities and limit establish the claim.

## Reproduction and artifacts

Run from this directory in named tmux session `shipping_exact_activation_20260924`:

    /data/home/aviyac/ap_env/bin/python -B launch.py

The supervisor invokes `/data/home/aviyac/ap_env/bin/python -B diagnostic.py` with single-thread environment variables and PYTHONHASHSEED=0. It writes diagnostic.log, diagnostic.exit, invocation.json (full child invocation monotonic wall time, timestamps, status, exact command and source hashes). The child writes summary.json with each case/phase monotonic wall time, raw exact fractions, solver evidence, statuses, environment and source SHA256s. Its total excludes final serialization; invocation.json includes process startup, all imports, serialization and interpreter teardown. Parent and subphase timings overlap and must not be summed. All cases run sequentially; nothing here is a sum of concurrent timings.

Attach: `tmux attach -t shipping_exact_activation_20260924`. Status: `tmux has-session -t shipping_exact_activation_20260924`; after completion inspect diagnostic.exit, invocation.json and diagnostic.log. The launch command also writes launcher.log and launcher.exit and signals a `tmux wait-for` completion channel; no sleep/poll is required. Preserve existing artifacts; reproduce in a fresh copy rather than overwriting diagnostic.log.

## Remaining policy decision

Native float32 program evaluation and real mathematical evaluation of its stored coefficients are different candidate policies. A real-arithmetic AP formulation is not automatically a valid upper bound for the native float32 policy, especially under exact activation fees. Conversely, an exact rational diagnostic does not certify finite-tolerance solver bounds. The closed shipping relaxation can have ghost fees, while the actual discontinuous objective may have only a supremum. These seven fixtures demonstrate blockers; they neither validate a full AP nor select a remedy. Any authoritative policy choice, exact/supremum AP formulation, numerical policy change, or production launch requires separate approval.