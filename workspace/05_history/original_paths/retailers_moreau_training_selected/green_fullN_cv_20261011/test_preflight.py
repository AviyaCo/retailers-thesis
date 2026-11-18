"""Explicit TWO-EPOCH TRAIN-ONLY integration test, never production evidence.

Run each kind in a fresh legacy-namespace process. No test generator or reporting.
"""
import argparse
from pathlib import Path
import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=runner.KINDS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    b = runner.backend_for(args.kind)
    record = dict(disposable=True, purpose=__doc__, kind=args.kind, seed=0, Ntotal=50,
                  epoch_limit=2, production=False, test_accessed=False)
    args.out.mkdir(exist_ok=False, parents=True)
    try:
        with runner.timed(record):
            prepared = runner.prepare(args.kind, args.out / 'prepared', 1)
            original_train = b.train_model

            def short_train(model, loss, lr, folder, meta):
                meta.update(disposable=True, production=False, explicitly_short_test=True)
                return original_train(model, loss, lr, folder, meta, epochs=2, patience=300)

            b.train_model = short_train
            result = runner.nominal(args.kind, 50, 0, args.out / 'nominal', prepared['path'])
            record['nominal'] = result
            saved = runner.read(result['summary_path'])
            if (len(saved['cv_rows']) != 15 or len(runner.read(args.out / 'nominal/training_data.json')['centers']) != 50):
                raise ValueError('Preflight did not cover CV and full-N refit')
            record['zero_radius'] = b.run_configuration(50, 0, 0., args.out / 'ccg',
                                                        result['nominal'], prepared['path'])
            if record['zero_radius']['status'] != 'completed':
                raise ValueError('Zero-radius controller failed')
    finally:
        runner.write(args.out / 'preflight.json', record)
    print(args.kind, record['status'], record['wall_time_s'], flush=True)


if __name__ == '__main__':
    main()