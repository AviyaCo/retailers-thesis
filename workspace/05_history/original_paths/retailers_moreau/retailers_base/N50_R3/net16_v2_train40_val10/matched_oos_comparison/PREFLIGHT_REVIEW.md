# Preflight review — no campaign launched

Final frozen manifest SHA256:
`d537cfc944d31c3ca6cdbfbc42b962254f5da26a9a2165bda6c8bc442f74d1d5`

730 captured input/source/reference files, 320 configurations (230 old,90 new).
All original/copy hashes reverified after tests. No results directory,10k demand
arrays, campaign log or completed comparison chart was produced. No long run
was launched. The plotting implementation is gated on all320 strict successes;
its reference key/seed coverage was checked, but final visual output remains
unverified until actual results exist. No synthetic result curve was created.

| Check | Outcome | Measured wall time |
|---|---|---:|
| Final solver-free suite | 10 passed; all230 original selector paths replayed | 2.636467 s (0.043941 min) |
| Manifest capture | All320 checkpoints strict-loaded;730 files captured | 1.329519 s (0.022159 min) |
| Actual Gurobi preflight, including provenance check | 1 test passed;5 strict QPs and5 original reference QPs | 0.225196 s (0.003753 min) |
| Nominal neural training | Not run; no current measurement | unavailable |
| Robust neural/CCG training total/AP/MP | Not run; historical artifacts separate | unavailable |
| SRO-FA construction/solver/total | Not run; historical reference only | unavailable |
| Full neural OOS evaluation / plotting | Not launched | unavailable |
| Hyperparameter selection / CV | Not performed; saved choices fixed | not applicable |

The five strict fixture projections returned status2 (`OPTIMAL`), passed the
declared physical threshold, matched the original QP outputs bit-for-bit, and
agreed with the hand-computed fixture projections within1e-4. These are bounded
target/capacity fixtures, not held-out demand observations. Per-fixture statuses,
iterations, residuals and timing are persisted in the Gurobi preflight JSON.
Editor diagnostics report no errors in the three Python modules.

An initial solver-free test invocation failed because its mocked Gurobi sum
expression did not support `<=`; this was a test fixture error, not a solver
failure. The initial2.392148-second failure artifact is retained. A subsequent
nine-test pass and the final expanded ten-test pass are separately retained.

Environment: configured ap_env Python3.11.13 on gold-srv2, Linux, Intel Xeon
Gold6354 host (72 logical CPUs observed). Bounded checks used one process and
one solver/Torch thread. The later campaign is fixed at8 workers x1 thread.
Historical runtime fields are not guessed from current verification time.

## Limitations for reviewer

No launch blocker was found in the implemented preflight. **Historical old
cache checkpoint byte identity remains unprovable**, because that cache never
stored hashes. The exact original selection-rule replay, full forced-cache
metadata and pre-cache file mtimes all agree; newly captured hashes certify
only the current frozen inputs. This limitation must stay in reported provenance.

New saved final selections2,6,9 retain `violating` status. Neither these labels,
training-size differences, nor historical benchmark sample mismatches are
removed to improve the chart. The full320-job throughput, actual shared-host
8-worker behavior, and final chart appearance are not established by this
bounded preflight. Use the exact tmux launch/cwd/log instructions in README
only after reviewer approval.