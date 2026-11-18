"""Synthetic common-format display fixtures; not experiment results."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.plot import render, validate_plot_data


def statistic(values):
    a = np.asarray(values, dtype=float)
    return dict(per_seed=a.tolist(), mean=a.mean(axis=-1).tolist(),
                p20=np.percentile(a, 20, axis=-1).tolist(), p80=np.percentile(a, 80, axis=-1).tolist())


def fixture(R=3):
    panel = dict(Ntotal=50, label='UNIT FIXTURE—not measured', horizontal=[
        dict(value=1.5, label='Supplied synthetic reference', kind='unit')])
    for method, radii in (('green', [0., .2]), ('black', [0., .2, .7]), ('sro_fa', [.2, .4])):
        panel[method] = dict(label=method, seeds=[0, 1], radius=radii,
                             oos=statistic([[2., 4.] for _ in radii]))
        if method != 'sro_fa':
            panel[method]['finite_pool'] = statistic([[1., 2.] for _ in radii])
    panel['saa_knn'] = dict(seeds=[0, 1], oos=statistic([2., 3.]))
    return dict(status='completed', panels=[panel], seeds=[0, 1],
                test={'arrays': {k: {'shape': [2, R]} for k in ('xi1', 'xi2')}},
                footnote='Synthetic UNIT fixture only', caption='Preserve me',
                provenance={'timings': 'unmeasured unit fixture'})


class PlotTests(unittest.TestCase):
    def test_genuine_grids_and_percentiles(self):
        data = fixture()
        before = deepcopy(data)
        validate_plot_data(data, 3)
        self.assertEqual(data, before)
        self.assertEqual(data['panels'][0]['green']['oos']['p20'], [2.4, 2.4])
        self.assertNotEqual(data['panels'][0]['green']['radius'], data['panels'][0]['black']['radius'])
        data['panels'][0]['black']['oos']['mean'][0] += .01
        with self.assertRaises(ValueError):
            validate_plot_data(data, 3)

    def test_rejects_false_dimension_duplicate_grid_and_missing_values(self):
        with self.assertRaises(ValueError):
            validate_plot_data(fixture(), 10)
        for change in ('grid', 'values', 'status'):
            data = fixture()
            if change == 'grid':
                data['panels'][0]['green']['radius'] = [.2, .2]
            elif change == 'values':
                data['panels'][0]['green']['oos']['per_seed'][0][0] = float('nan')
            else:
                data['status'] = 'running'
            with self.assertRaises(ValueError):
                validate_plot_data(data, 3)

    def test_render_is_exclusive_and_preserves_supplied_evidence(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'plots'
            data = fixture(10)
            paths = render(data, output, 10)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(json.loads((output / 'plot_data.json').read_text()), data)
            with self.assertRaises(FileExistsError):
                render(data, output, 10)


if __name__ == '__main__':
    unittest.main()