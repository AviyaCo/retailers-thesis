"""Short, deterministic train-only checks; invoke separately for each kind.

python -B test_backend.py --kind R3 --scratch /absolute/fresh/writable/subdir
The scratch directory must already exist; original paths remain read-only.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import platform
import time
import unittest

from backend import load_backend, TRAIN_SIZES

B = None
OUT = None
TIMINGS = []


class BackendTests(unittest.TestCase):
    def test_01_identity_protocol_and_generators(self):
        self.assertIs(load_backend(B.kind), B)
        with self.assertRaises(RuntimeError):
            load_backend('R10' if B.kind != 'R10' else 'R3')
        B.configure_worker()
        B.configure_worker()
        self.assertEqual(B.torch.get_num_threads(), 1)
        self.assertEqual(B.torch.get_num_interop_threads(), 1)
        self.assertEqual(B.protocol['ccg_selection'], 'stopping_checkpoint')
        self.assertTrue(B.protocol['refit'])
        self.assertFalse(B.protocol['resume'])
        self.assertEqual(B.protocol['cv_folds'], 5)
        for n in (50, 200, 800):
            row = B.metadata(n, 0)
            self.assertEqual((row['Ntrain'], row['Nval'], row['split_seed']), (n, 0, None))
            data = B.generate(n, 0)
            self.assertEqual(set(data), {'xi1', 'xi2'})
            if B.kind != 'R3' or n != 50:
                old = B._original.common.split_data(n, 0)
                for key in data:
                    B.np.testing.assert_array_equal(data[key], old[key])
            self.assertEqual(data['xi1'].shape, (n, row['R']))
        if B.kind == 'shipping_R3':
            scope = B._controller
            self.assertEqual(scope['adversary'].__name__, 'advisory')
            self.assertIs(scope['validate_ap'], scope['adversary'].validate_ap)
            self.assertFalse(B.protocol['activation_link_rejection_enabled'])
            self.assertFalse(B.protocol['exact_duplicate_separation_rejection_enabled'])
            self.assertTrue(B.protocol['float32_collision_rejection_enabled'])
            self.assertNotIn('continuation', scope['run_configuration'].__module__)

    def test_02_stop_not_min_ap_and_prefix_validation(self):
        candidates = [dict(candidate_id=0, assessed=True, ap_incumbent_mean=8., restricted_mean=10.),
                      dict(candidate_id=1, assessed=True, ap_incumbent_mean=10.01, restricted_mean=10.)]
        for row in candidates:
            row.update(B._controller['terminal_decision'](row))
        self.assertEqual(B._original.controller.choose_candidate(candidates, 1)['candidate_id'], 0)
        self.assertIs(B.choose_candidate(candidates, 1), candidates[1])
        for change in ({'assessed': False}, {'candidate_id': 9},
                       {'ap_incumbent_mean': float('nan')}, {'green_pass': True}):
            bad = deepcopy(candidates)
            bad[0].update(change)
            with self.assertRaises(ValueError):
                B.choose_candidate(bad, 1)
        with self.assertRaises(ValueError):
            B.choose_candidate(candidates, 0)
        # Future rows are irrelevant; malformed future data must never be consulted.
        self.assertIs(B.choose_candidate(candidates + [{}], 1), candidates[1])

    def test_03_native_model_cost_projection_and_training_equivalence(self):
        n = 200 if B.kind == 'R3' else 50
        old = B._original.common.split_data(n, 0)
        x1, x2 = old['train_x1'], old['train_x2']
        size = len(x1)
        saa_a, saa_b = {}, {}
        q10, q1 = B._original.common.solve_saa(x1.numpy(), x2.numpy(), saa_a)
        other_q10, other_q1 = B.solve_saa(x1.numpy(), x2.numpy(), saa_b)
        self.assertEqual(q10, other_q10)
        B.np.testing.assert_array_equal(q1, other_q1)
        for label, row in [('original_saa', saa_a), ('adapted_saa', saa_b)]:
            TIMINGS.append(dict(label=label, kind=B.kind, seed=0, Ntrain=size, timing=row))
        make = B.core.make_model
        a, b = make(q10, q1), B.make_model(q10, q1)
        self.assertEqual(B.state_hash(a), B.state_hash(b))
        self.assertEqual(B.state_hash(B.model_from_state(B.weights(a))), B.state_hash(a))
        with B.torch.no_grad():
            va, vb = a.per_sample_cost(x1, x2), b.per_sample_cost(x1, x2)
            self.assertTrue(B.torch.equal(va, vb))
            z = B.torch.arange(2 * len(q1), dtype=B.torch.float32).reshape(2, -1)
            cap = B.torch.full((2, 1), 4.)
            pa = a.proj_layer(z, cap, B._original.training.core._SOURCE_SOLVER_ARGS)[0]
            pb = b.proj_layer(z, cap, B._original.training.core._SOURCE_SOLVER_ARGS)[0]
            self.assertTrue(B.torch.equal(pa, pb))
        record = {}
        measured = B.evaluate(b, x1, x2, record)
        with B.torch.no_grad():
            values = b.hard_shipping_cost(x1, x2) if B.kind == 'shipping_R3' else b.per_sample_cost(x1, x2)
        self.assertEqual(measured, float(values.mean()))
        TIMINGS.append(dict(label='native_train_only_evaluation', timing=record))
        records = []
        for label, model, trainer in [('original', a, B._original.training.train_model),
                                       ('adapted', b, B.train_model)]:
            row = B.metadata(n, 0, method='two_epoch_equivalence')
            row.update(Ntrain=size, Nval=0)
            trainer(model, lambda m: m.per_sample_cost(x1, x2).mean(), .01,
                    OUT / label, row, epochs=2, patience=300)
            records.append(row)
            TIMINGS.append(dict(label=label, kind=B.kind, seed=0, Ntrain=size, timing=row))
        self.assertEqual(B.state_hash(a), B.state_hash(b))
        for key in ('best_train', 'first_train', 'last_train', 'best_epoch',
                    'selected_training_loss', 'optimizer_steps', 'epochs_evaluated'):
            self.assertEqual(records[0][key], records[1][key])
        self.assertEqual([r['train_loss'] for r in records[0]['epoch_history']],
                         [r['train_loss'] for r in records[1]['epoch_history']])

    def test_04_full_n_guards(self):
        r = B.metadata(50)['R']
        for n in TRAIN_SIZES:
            # NaN rejection is after the size guard and before any solver creation.
            x = B.np.full((n, r), B.np.nan)
            with self.assertRaises((ValueError, FloatingPointError)) as caught:
                B.solve_saa(x, x, {})
            self.assertIn('SAA', str(caught.exception))
            self.assertNotIn('shape', str(caught.exception))
            self.assertNotIn('requires approved', str(caught.exception))
            model = B.make_model(20., B.np.full(r, 5.))
            row = B.metadata(50, 0, method='guard_only_parameter_loss')
            row['Ntrain'] = n
            # Guard-only check, explicitly not a measured demand-training experiment.
            B.train_model(model, lambda m: sum(p.square().mean() for p in m.parameters()),
                          .01, OUT / f'guard_{n}', row, epochs=1)
            self.assertEqual(row['status'], 'completed')
            TIMINGS.append(dict(label='guard_only', kind=B.kind, Ntrain=n, timing=row))
        with self.assertRaises(ValueError):
            B.train_model(model, lambda m: m.Q10.sum(), .01, OUT / 'invalid', {'Ntrain': 51})

    def test_05_zero_radius_assessment(self):
        data = B.generate(50, 0)
        centers = B.np.concatenate([data['xi1'][:2], data['xi2'][:2]], axis=1)
        model = B.make_model(20., B.np.full(B.metadata(50)['R'], 5.))
        pools = [x[None, :].copy() for x in centers]
        scope = B._controller
        folder = OUT / 'zero_assess'
        folder.mkdir()
        checkpoint = folder / 'candidate.pth'
        B.torch.save(B.weights(model), checkpoint)
        row = dict(candidate_id=0, Ntotal=50, Ntrain=2, seed=0, radius=0., ccg_workers=1,
            state_sha256=B.state_hash(model), checkpoint=str(checkpoint),
            checkpoint_sha256=B.core.digest(checkpoint), pool_sha256=scope['pool_hash'](pools))
        row['pool_file_sha256'] = scope['_save_pool'](folder / 'pool.json', pools, [0, 1])
        scope['assess_candidate'](model, pools, centers, [0, 1], 0., row, folder, lambda: None)
        self.assertTrue(row['assessed'])
        self.assertEqual(len(row['samples']), 2)
        row.update(scope['terminal_decision'](row))
        self.assertIs(B.choose_candidate([row], 0), row)
        TIMINGS.append(dict(label='two_training_center_zero_radius_AP_probe', timing=row['ap']))

    def test_06_source_bindings(self):
        manifest = dict(source_sha256=B.sources(), protocol=B.protocol, workers={'ccg': 16})
        B.verify_sources(manifest)
        key = str(Path(__file__).with_name('backend.py').resolve())
        manifest['source_sha256'][key] = '0' * 64
        with self.assertRaises(RuntimeError):
            B.verify_sources(manifest)

    def test_07_fresh_full50_controller_contract(self):
        data = B.generate(50, 0)
        x1 = B.torch.tensor(data['xi1'], dtype=B.torch.float32)
        x2 = B.torch.tensor(data['xi2'], dtype=B.torch.float32)
        saa = B.metadata(50, 0, method='full50_SAA_contract_probe')
        q10, q1 = B.solve_saa(data['xi1'], data['xi2'], saa)
        model = B.make_model(q10, q1)
        training = B.metadata(50, 0, method='one_epoch_full50_contract_probe')
        folder = OUT / 'full50'
        B.train_model(model, lambda m: m.per_sample_cost(x1, x2).mean(), .01,
                      folder, training, epochs=1)
        raw_hash = __import__('hashlib').sha256()
        for key in ('xi1', 'xi2'):
            raw_hash.update(B.np.asarray(data[key], dtype='<f8').tobytes())
        checkpoint = folder / 'selected.pth'
        train_path = folder / 'training_data.json'
        B.core.write_json(train_path, dict(
            centers=B.np.concatenate([data['xi1'], data['xi2']], axis=1),
            original_sample_ids=list(range(50)), data_sha256=raw_hash.hexdigest()))
        record = dict(Ntotal=50, seed=0, lr_net=.01, checkpoint=str(checkpoint),
            checkpoint_sha256=B.core.digest(checkpoint), state_sha256=B.state_hash(model),
            train_mean=training['selected_training_loss'], training_timing=training,
            training_data_path=str(train_path), training_data_sha256=B.core.digest(train_path),
            data_sha256=raw_hash.hexdigest())
        manifest = folder / 'provenance.json'
        B.core.write_json(manifest, dict(source_sha256=B.sources(), protocol=B.protocol,
                                      workers={'ccg': 16}))
        result = B.run_configuration(50, 0, 0., folder / 'ccg', record, manifest)
        TIMINGS.extend([dict(label='full50_SAA_probe', timing=saa),
                        dict(label='full50_one_epoch_probe', timing=training),
                        dict(label='full50_zero_radius_CCG_probe', timing=result)])
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual((result['Ntrain'], result['Nval']), (50, 0))
        self.assertEqual(result['mp_count'], 0)
        self.assertEqual(result['selected_candidate_id'], result['stopping_candidate_id'])
        self.assertEqual(result['selected_state_sha256'], record['state_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('R3', 'R10', 'shipping_R3'))
    parser.add_argument('--scratch', required=True, type=Path)
    args = parser.parse_args()
    OUT = args.scratch.resolve()
    if not OUT.is_dir() or any(OUT.iterdir()):
        raise ValueError('Provide an existing empty scratch directory')
    start = time.perf_counter()
    B = load_backend(args.kind)
    B.configure_worker()
    summary = dict(kind=args.kind, seed=0, started_at_utc=B.core.utc_now(),
        environment=platform.platform(), hostname=platform.node(), workers=1,
        torch_threads=1, gurobi_threads=1, scope='short deterministic train-only tests; no OOS')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    summary.update(status='completed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        ended_at_utc=B.core.utc_now(), wall_time_s=time.perf_counter() - start, timings=TIMINGS,
        sources_unchanged=False)
    B.verify_sources(dict(source_sha256=B.sources(), protocol=B.protocol, workers={'ccg': 16}))
    summary['sources_unchanged'] = True
    B.core.write_json(OUT / 'test_summary.json', summary)
    raise SystemExit(0 if result.wasSuccessful() else 1)