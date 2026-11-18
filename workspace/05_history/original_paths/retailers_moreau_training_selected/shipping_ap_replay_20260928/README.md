# Fixed shipping AP replay (28 September)

User requested rerunning the AP solve to diagnose returned shipping indicators
and shipment quantities. This isolated diagnostic preserves all historical code
and results. Cases are fixed before execution: N200/seed5/radius1.5/candidate1/
sample6; N50/seed7/radius1.5/candidate9/sample33; N800/seed7/radius.2/candidate1/
sample271. Only saved training policies/centers/pools are used. No validation,
test, training, campaign continuation, radius selection, or acceptance change.

Each AP is called once unchanged, then once with logging and original-model/
solution/quality capture. Logging-only settings are OutputFlag1, LogToConsole0,
and a persistent LogFile; all numerical settings, epsilon1e-6, native physical
guard1e-6, and AP-internal tolerance1e-9 are unchanged. The baseline is not a
fallback and instrumented failure remains failure. Raw returned variables and
original linear-row violations are captured before the existing postcheck.
LP/MPS/solution export uses Gurobi, not a reimplemented optimizer. Solver logs
and quality attributes can establish a numerical residual, not necessarily its
internal cause. Missing quality attributes are explicitly recorded.

For sample271 only, evaluate its unchanged full saved pool before the AP call;
passive projection hooks capture targets/shipments and permit comparison of the
identical stored point inside the pool versus the AP singleton. Pool maxima,
AP values, and native replay are compared against saved production evidence.
Fresh models are used; unrelated preceding MP/AP calls are not reproduced.
Thus reproduction is tested, not assumed. No alternate solver, presolve change,
epsilon change, fee cutoff, clipping, or degraded retry is performed.

One worker, one Torch/Gurobi thread; original seed0/default Gurobi Seed is
captured. Per-case, AP/solver, pool evaluation, capture and whole invocation
monotonic seconds/UTC are persisted. Baseline vs logged timing scopes differ:
the logged AP includes model export/capture overhead. They are not independent
performance samples. Nominal/MP/SRO-FA/CV runtimes are not remeasured.

Run through the named tmux supervisor; launch and invocation JSONs record exact
command, working directory, timestamps and persistent merged stdout/stderr.