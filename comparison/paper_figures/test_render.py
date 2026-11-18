"""Check the requested paper layout against the sealed saved evaluations."""
from copy import deepcopy
import unittest

import matplotlib.pyplot as plt
import numpy as np

from .render import KNN_COLOR, NN_COLOR, SIZE_CM, build_figure, load_data
from .render_shipping import build_shipping_figure, load_shipping_data
from .full_n import HASHES, RUN, load_full_n_data


class PaperFigureTests(unittest.TestCase):
    def test_full_n_curves_replace_only_nn_and_match_verified_sources(self):
        import json

        for kind in HASHES:
            r = 10 if kind == 'R10' else 3
            old = load_shipping_data() if kind == 'shipping_R3' else load_data(r)[0]
            data = load_full_n_data(kind)
            full = json.loads((RUN / f'published/{kind}/plot_data.json').read_text())
            fig = build_figure(data, r)
            try:
                for ax, panel, original, replacement in zip(
                        fig.axes, data['panels'], old['panels'], full['panels']):
                    self.assertEqual(panel['green']['Ntrain'], panel['Ntotal'])
                    for method in ('sro_fa', 'saa_knn', 'horizontal'):
                        self.assertEqual(panel[method], original[method])
                    for line, field in ((ax.lines[0], 'oos'), (ax.lines[2], 'finite_pool')):
                        self.assertEqual(panel['green'][field], replacement['green'][field])
                        np.testing.assert_array_equal(
                            line.get_ydata(), replacement['green'][field]['mean'])
            finally:
                plt.close(fig)

    def test_exact_layout_and_unchanged_values(self):
        for case in (3, 10, 'shipping'):
            data = load_shipping_data() if case == 'shipping' else load_data(case)[0]
            before = deepcopy(data)
            fig = (build_shipping_figure(data) if case == 'shipping'
                   else build_figure(data, case))
            try:
                np.testing.assert_allclose(fig.get_size_inches() * 2.54, SIZE_CM)
                self.assertIsNone(fig._suptitle)
                self.assertEqual(len(fig.axes), 3)
                for ax, panel in zip(fig.axes, data['panels']):
                    self.assertEqual(ax.get_xlabel(), r'$\epsilon_N$')
                    self.assertEqual(len(ax.lines), 4 + len(panel['horizontal']))
                    for line, expected in zip(ax.lines[:3], (
                        panel['green']['oos']['mean'],
                        panel['sro_fa']['oos']['mean'],
                        panel['green']['finite_pool']['mean'],
                    )):
                        np.testing.assert_array_equal(line.get_ydata(), expected)
                    self.assertEqual(ax.lines[0].get_color(), '#2465aa')
                    self.assertEqual(ax.lines[2].get_color(), NN_COLOR)
                    np.testing.assert_allclose(
                        ax.collections[0].get_facecolor()[0],
                        [36 / 255, 101 / 255, 170 / 255, .12],
                    )
                    self.assertEqual(ax.lines[2].get_linestyle(), '--')
                    np.testing.assert_array_equal(
                        ax.lines[3].get_ydata(), [panel['saa_knn']['oos']['mean']] * 2)
                    self.assertEqual(ax.lines[3].get_color(), KNN_COLOR)
                    np.testing.assert_allclose(
                        ax.patches[0].get_facecolor(),
                        [201 / 255, 155 / 255, 8 / 255, .08],
                    )
                    for line, ref in zip(ax.lines[4:], panel['horizontal']):
                        np.testing.assert_array_equal(line.get_ydata(), [ref['value']] * 2)
                labels = [text.get_text() for text in fig.legends[0].get_texts()]
                self.assertIn('NN policy (out-of-sample)', labels)
                self.assertIn('NN policy (in-sample)', labels)
                self.assertIn('SAA-kNN (out-of-sample)', labels)
                if case == 'shipping':
                    self.assertIn('Conditional reference (1M)', labels)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for artist in [fig.legends[0]] + [
                    item for ax in fig.axes for item in (
                        ax.xaxis.label, ax.yaxis.label, ax.title,
                        *(label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
                          if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]),
                        *(label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                          if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]),
                    ) if item.get_visible() and item.get_text()
                ]:
                    box = artist.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, 0)
                    self.assertGreaterEqual(box.y0, 0)
                    self.assertLessEqual(box.x1, fig.bbox.width)
                    self.assertLessEqual(box.y1, fig.bbox.height)
                self.assertEqual(data, before)
            finally:
                plt.close(fig)

    def test_rejects_changed_aggregate_or_missing_reference(self):
        for change in ('aggregate', 'reference', 'selection', 'knn'):
            data, _ = load_data(3)
            if change == 'aggregate':
                data['panels'][0]['green']['oos']['mean'][0] += 1
            elif change == 'reference':
                data['panels'][0]['horizontal'] = []
            elif change == 'knn':
                del data['panels'][0]['saa_knn']
            else:
                data['selection_rule'] = 'minimum_ap'
            with self.assertRaises(ValueError):
                build_figure(data, 3)


if __name__ == '__main__':
    unittest.main()
