"""Re-render the FINAL figures with a straight lunate epsilon (U+03F5) on the x-axis.

matplotlib maps both \\epsilon and \\varepsilon to U+03B5 (the curly form), so the final figures show
the curly epsilon. This wrapper builds the identical figures (same data loaders, same build functions)
and only replaces the x-axis label with U+03F5. Writes NEW folders; `published/` and
`shipping_published/` are left untouched.
Run from the experiment root: python -m comparison.paper_figures.render_epsilon
"""
import json

import matplotlib.pyplot as plt

from .full_n import load_full_n_data
from .render import ROOT, build_figure
from .render_shipping import build_shipping_figure

XLABEL = '$\u03f5_N$'
OUTPUTS = {
    'R3': ('published_straight_epsilon', 'comparison_R3', lambda d: build_figure(d, 3)),
    'R10': ('published_straight_epsilon', 'comparison_R10', lambda d: build_figure(d, 10)),
    'shipping_R3': ('shipping_published_straight_epsilon', 'comparison_shipping',
                    build_shipping_figure),
}


def main():
    for folder in {spec[0] for spec in OUTPUTS.values()}:
        (ROOT / folder).mkdir(exist_ok=False)
    for kind, (folder, stem, build) in OUTPUTS.items():
        data = load_full_n_data(kind)
        fig = build(data)
        try:
            for ax in fig.axes:
                if ax.get_xlabel() != r'$\epsilon_N$':
                    raise ValueError(f'Unexpected x label: {ax.get_xlabel()!r}')
                ax.set_xlabel(XLABEL, labelpad=0)
            for ext in ('pdf', 'png'):
                fig.savefig(ROOT / folder / f'{stem}.{ext}', dpi=300)
        finally:
            plt.close(fig)
        print(f'Wrote {folder}/{stem}.pdf/.png')
    for folder in {spec[0] for spec in OUTPUTS.values()}:
        record = dict(
            derived_from=folder.replace('_straight_epsilon', ''),
            change='x-axis label U+03B5 (curly epsilon) -> U+03F5 (straight lunate epsilon); '
                   'everything else identical (same loaders, same build functions)',
            script='render_epsilon.py', new_training_or_evaluation=False)
        (ROOT / folder / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
