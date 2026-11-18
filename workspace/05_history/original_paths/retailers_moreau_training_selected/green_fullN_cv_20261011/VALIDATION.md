# Prelaunch validation — 11 October 2026

This records implementation checks, not production performance or test scores.

- Original/adapted model, projection, SAA, cost and two-epoch trainer checks:
  seven tests per kind, 21 total, passed. Raw records remain in backend_checks_v3
  and the later reporting_checks_20261011T084749Z backend recheck.
- CV/orchestration and reporting regression suite: 39 tests passed in 2.955 s
  after fixing test expectations for preserved nonconvergence and output scope.
  An earlier run failed due to outdated assertions; a subsequent test-only
  indentation error was corrected. No numerical tolerances were changed.
- Native strict evaluation preflight: two training rows for each case, six
  Gurobi QPs, all OPTIMAL with zero physical residual. No test rows were read.
- Actual CV orchestration integration: all 15 fold/LR fits plus fresh full-N50
  initialization/refit, **explicitly limited to two epochs for testing only**,
  then original zero-radius CCG. All three passed, candidate0 STOP, zero MPs.
  Structured records: integration_v1/{R3,R10,shipping_R3}/preflight.json.
  Measured integration seconds: R3 5.275464; R10 5.126038; shipping 7.329863.
- All new Python sources parsed; editor diagnostics clear. Methodology notes
  compiled successfully (71 pages).

Production uses the original 3000-epoch / patience300 protocol, not the short
integration-test override. The test script cannot supply a production checkpoint.
No positive-radius production convergence or full-N200/800 outcome is claimed
by these checks. Full source freeze and each worker's provenance checks remain
mandatory. Original validation-based experiments were not modified.