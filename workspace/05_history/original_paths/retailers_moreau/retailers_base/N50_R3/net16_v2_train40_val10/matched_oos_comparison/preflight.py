"""Persist bounded test/preflight runtimes; never launch held-out evaluation."""
import argparse
import platform
import time
import unittest

import compare as c
import test_compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('solver_free', 'gurobi'))
    args = parser.parse_args()
    c.initialize_worker()
    start = time.perf_counter()
    report = dict(experiment='matched_OOS_preflight', method=args.suite, seed=None,
                  radius=None, n_test=0, fixtures=5 if args.suite == 'gurobi' else None,
                  start_utc=c.utc(), status='running', workers=1, threads=1,
                  environment=dict(host=platform.node(), platform=platform.platform()),
                  sources={p.name: c.sha(p) for p in (c.HERE / 'compare.py',
                           c.HERE / 'test_compare.py', c.HERE / 'preflight.py')})
    path = c.HERE / f'preflight_{args.suite}_{time.time_ns()}.json'
    try:
        if args.suite == 'gurobi':
            c.verify_manifest()
            report['manifest_sha256'] = c.sha(c.FROZEN / 'manifest.json')
        cls = test_compare.SolverFreeTests if args.suite == 'solver_free' else test_compare.GurobiPreflight
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(tests_run=result.testsRun, failures=[(str(t), error) for t, error in result.failures],
                      errors=[(str(t), error) for t, error in result.errors],
                      status='completed' if result.wasSuccessful() else 'failed')
        if not result.wasSuccessful():
            raise SystemExit(1)
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        if args.suite == 'gurobi':
            report['projection_fixtures'] = test_compare.GurobiPreflight.diagnostics
        report.update(end_utc=c.utc(), wall_time_s=time.perf_counter() - start)
        report['wall_time_human'] = c.human(report['wall_time_s'])
        c.write_json(path, report)


if __name__ == '__main__':
    main()