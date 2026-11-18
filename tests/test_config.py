"""Solver-free unit data only; never reads test arrays or campaign artifacts."""
from dataclasses import FrozenInstanceError
import hashlib
import unittest

import numpy as np
from scipy.stats import truncnorm
import torch

from src.config import Config, LR_GRID, RADII, split_data


class ConfigTests(unittest.TestCase):
    def test_approved_grid_and_frozen_fields(self):
        for R in (3, 10):
            for n in (50, 200, 800):
                for seed in range(10):
                    for method in ('black', 'green'):
                        for radius in RADII + ((2.,) if R == 10 else ()):
                            config = Config(R, n, seed, radius, method)
                            self.assertEqual((config.Ntrain, config.Nval), (4 * n // 5, n // 5))
        with self.assertRaises(FrozenInstanceError):
            Config(3, 50, 0).seed = 1

    def test_rejects_unapproved_inputs(self):
        for kwargs in ({'R': 4}, {'R': 3.}, {'R': True}, {'Ntotal': 40},
                       {'Ntotal': 50.}, {'seed': -1}, {'seed': 10}, {'seed': True},
                       {'radius': 2.}, {'radius': .1}, {'radius': float('nan')},
                       {'radius': float('inf')}, {'radius': True}, {'method': 'other'}):
            args = dict(R=3, Ntotal=50, seed=0)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**args)

    def test_protocol_is_independent_and_method_explicit(self):
        green = Config(10, 200, 7, 2., 'green').protocol()
        black = Config(10, 200, 7, 2., 'black').protocol()
        self.assertEqual(green['lr_grid'], list(LR_GRID))
        self.assertEqual((green['max_mp'], green['max_epochs'], green['patience']), (40, 3000, 300))
        self.assertEqual(green['split_seed'], 100007)
        self.assertEqual(green['scheduler']['metric'], 'training_loss')
        self.assertNotEqual(green['ccg_stop'], black['ccg_stop'])
        self.assertEqual(green['ccg_selection'], black['ccg_selection'])
        self.assertEqual(green['ccg_selection'],
                 'stopping checkpoint after inclusive assessed-prefix validation; no earlier AP minimum')
        self.assertEqual(green['test'], 'not accessed')
        self.assertFalse(green['solver_fallback'])
        green['lr_grid'].clear()
        green['scheduler']['metric'] = 'changed'
        self.assertEqual(Config(3, 50, 0).protocol()['lr_grid'], list(LR_GRID))
        self.assertEqual(Config(3, 50, 0).protocol()['scheduler']['metric'], 'training_loss')

    def test_exact_generator_split_hash_and_global_rng(self):
        """Reference is canonical arithmetic on unit-generated training data.

        It does not load historical campaigns, saved policies or OOS data.
        Exact array equality detects altered SciPy size/broadcast/RNG semantics.
        """
        saved_numpy = np.random.get_state()
        saved_torch = torch.get_rng_state().clone()
        try:
            for R in (3, 10):
                for n in (50, 200, 800):
                    for seed in (0, 9):
                        with self.subTest(R=R, n=n, seed=seed):
                            np.random.seed(seed)
                            expected1 = truncnorm.rvs((0.0 - 6.0) / 2.5, np.inf,
                                loc=6.0, scale=2.5, size=(n, R))
                            mu2 = 2 * (6.0 - expected1) ** 2
                            expected2 = truncnorm.rvs((0.0 - mu2) / 2.5, np.inf,
                                loc=mu2, scale=2.5)
                            np.random.seed(724)
                            before = np.random.get_state()
                            data = split_data(Config(R, n, seed))
                            after = np.random.get_state()
                            self.assertEqual(before[0], after[0])
                            np.testing.assert_array_equal(before[1], after[1])
                            self.assertEqual(before[2:], after[2:])
                            self.assertTrue(torch.equal(saved_torch, torch.get_rng_state()))
                            np.testing.assert_array_equal(data['xi1'], expected1)
                            np.testing.assert_array_equal(data['xi2'], expected2)
                            self.assertEqual(data['xi1'].dtype, np.float64)
                            self.assertEqual(data['xi2'].dtype, np.float64)
                            indices = np.random.default_rng(100000 + seed).permutation(n)
                            np.testing.assert_array_equal(data['train_indices'], indices[:4 * n // 5])
                            np.testing.assert_array_equal(data['val_indices'], indices[4 * n // 5:])
                            self.assertEqual(len(set(data['train_indices']) & set(data['val_indices'])), 0)
                            for split in ('train', 'val'):
                                for part in (1, 2):
                                    tensor = data[f'{split}_x{part}']
                                    self.assertEqual(tensor.dtype, torch.float32)
                                    self.assertTrue(torch.equal(tensor, torch.tensor(
                                        data[f'xi{part}'][data[f'{split}_indices']], dtype=torch.float32)))
                            hasher = hashlib.sha256()
                            hasher.update(np.asarray(expected1, dtype='<f8').tobytes())
                            hasher.update(np.asarray(expected2, dtype='<f8').tobytes())
                            self.assertEqual(data['data_sha256'], hasher.hexdigest())
                            again = split_data(Config(R, n, seed, method='black'))
                            self.assertEqual(data['data_sha256'], again['data_sha256'])
                            np.testing.assert_array_equal(data['train_indices'], again['train_indices'])
        finally:
            np.random.set_state(saved_numpy)
            torch.set_rng_state(saved_torch)


class RadiusSelectionTests(unittest.TestCase):
    def test_largest_radius_among_ties_within_tolerance(self):
        from src.config import RADIUS_SELECTION_RULE, select_radius
        self.assertEqual(RADIUS_SELECTION_RULE, 'largest radius within 1e-6 of the minimum validation score')
        # Exact tie of identical (nominal) policies at the minimum: the largest tied radius wins.
        self.assertEqual(select_radius({0.: 19.5, .001: 19.5, .03: 19.5, .2: 19.6}), .03)
        # A gap of 5e-7 counts as tied; a gap of 2e-6 does not.
        self.assertEqual(select_radius({.2: 19., .4: 19. + 5e-7}), .4)
        self.assertEqual(select_radius({.2: 19., .4: 19. + 2e-6}), .2)
        # A unique minimum is unaffected; string keys are returned as given.
        self.assertEqual(select_radius({'0.0': 20., '0.4': 19., '1.5': 21.}), '0.4')
        with self.assertRaises(ValueError):
            select_radius({.1: float('nan')})


if __name__ == '__main__':
    unittest.main()