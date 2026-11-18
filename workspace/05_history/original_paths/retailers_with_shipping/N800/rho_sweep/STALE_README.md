# Stale Full-Batch Shipping N=800 Results

This directory contains the superseded full-batch shipping N=800 sweep. Its
neural-policy fields and figures must not be used for current comparisons. The
canonical corrected neural results are in `../rho_sweep_minibatch/`.

Keep `sweep_eval_shipping_10k.json` in place. The corrected evaluator imports
the benchmark block from that file and records its SHA-256 hash for provenance.
Removing or modifying it breaks exact reconstruction of the corrected result.

The remaining files document the original memory-error run and recovery
history. They are retained as stale provenance, not canonical evidence.