"""Paper-sized Moreau/SRO-FA comparison from sealed STOP publication data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from src.plot import validate_plot_data


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'stopping_checkpoint_20260928/production_v1/published'
SIZE_CM = (17., 5.5)
NN_COLOR = '#2465aa'
RED = '#c43d3d'
KNN_COLOR = '#c99b08'
TICKS = [0., .001, .01, .1, .4, 1.5]
REFERENCE_STYLES = {
    'saa_lp': ('SAA-LP reference', '#555555', ':'),
    'conditional': ('Conditional reference', '#888888', '-.'),
    'empirical_clairvoyant_reference': ('Empirical clairvoyant reference', '#555555', ':'),
}


def load_data(r):
    path = SOURCE / f'plot_data_R{r}.json'
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'Published data checksum mismatch: {path}')
    return json.loads(payload), digest


def build_figure(data, r):
    validate_plot_data(data, r)
    if (data['selection_rule'] != 'stopping_checkpoint'
            or data['seeds'] != list(range(10))
            or [p['Ntotal'] for p in data['panels']] != [50, 200, 800]):
        raise ValueError('Require the complete ten-seed STOP publication')
    for panel in data['panels']:
        if 'saa_knn' not in panel:
            raise ValueError('Missing SAA-kNN comparator')
        if not panel['horizontal']:
            raise ValueError('Missing lower-bound/reference data')
        for ref in panel['horizontal']:
            if ref['kind'] not in REFERENCE_STYLES:
                raise ValueError(f"Unsupported reference: {ref['kind']}")
        for method in ('green', 'sro_fa'):
            if panel[method]['radius'][-1] != 1.5:
                raise ValueError('Unexpected comparison radius endpoint')

    with plt.rc_context({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 7,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    }):
        fig, axes = plt.subplots(1, 3, figsize=np.asarray(SIZE_CM) / 2.54)
        handles = {}
        for ax, panel in zip(axes, data['panels']):
            for method, color, label in (
                ('green', NN_COLOR, 'NN policy (out-of-sample)'),
                ('sro_fa', RED, 'SRO-FA (out-of-sample)'),
            ):
                curve = panel[method]
                x, stats = curve['radius'], curve['oos']
                line, = ax.plot(x, stats['mean'], color=color, lw=1.15,
                                marker='o', ms=1.6, label=label, zorder=4)
                handles.setdefault(label, line)
                ax.fill_between(x, stats['p20'], stats['p80'], color=color,
                                alpha=.12, lw=0)
            curve = panel['green']
            label = 'NN policy (in-sample)'
            line, = ax.plot(curve['radius'], curve['finite_pool']['mean'],
                            '--', color=NN_COLOR, lw=1.1, label=label, zorder=3)
            handles.setdefault(label, line)
            stats = panel['saa_knn']['oos']
            label = 'SAA-kNN (out-of-sample)'
            line = ax.axhline(stats['mean'], color=KNN_COLOR, lw=1.1, label=label)
            ax.axhspan(stats['p20'], stats['p80'], color=KNN_COLOR, alpha=.08, lw=0)
            handles.setdefault(label, line)
            for ref in panel['horizontal']:
                _, color, style = REFERENCE_STYLES[ref['kind']]
                label = ref['label']
                line = ax.axhline(ref['value'], color=color, ls=style, lw=.9,
                                 label=label)
                handles.setdefault(label, line)
            ax.set_xscale('symlog', linthresh=.001, linscale=.8)
            ax.set_xlim(0, 1.5)
            ax.set_xticks(TICKS)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='both', labelsize=6, pad=2)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel(r'$\epsilon_N$', labelpad=0)
            ax.set_title(f"$N={panel['Ntotal']}$", fontsize=8, pad=3)
            ax.grid(axis='y', color='#dddddd', lw=.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Cost', labelpad=3)
        # Legend column order puts the two NN curves together in the first column.
        order = ['NN policy (out-of-sample)', 'NN policy (in-sample)',
                 'SRO-FA (out-of-sample)']
        order.extend(label for label in handles if label not in order)
        fig.legend([handles[label] for label in order], order, loc='lower center',
                   bbox_to_anchor=(.51, .005), ncol=3, frameon=False, fontsize=6,
                   handlelength=2.4, columnspacing=1.3, labelspacing=.45)
        fig.subplots_adjust(left=.075, right=.975, top=.91, bottom=.36, wspace=.30)
    return fig


def main():
    from .full_n import load_full_n_data

    output = ROOT / 'published'
    output.mkdir(exist_ok=False)
    sources = {}
    for r in (3, 10):
        _, digest = load_data(r)
        data = load_full_n_data(f'R{r}')
        fig = build_figure(data, r)
        try:
            for ext in ('pdf', 'png'):
                fig.savefig(output / f'comparison_R{r}.{ext}', dpi=300)
        finally:
            plt.close(fig)
        sources[f'R{r}'] = {
            'path': str((SOURCE / f'plot_data_R{r}.json').relative_to(ROOT.parent)),
            'sha256': digest,
            'full_n_nn': data['full_n_provenance'],
        }
    record = {
        'sources_relative_to_comparison': sources,
        'size_cm': SIZE_CM,
        'selection_rule': 'stopping_checkpoint',
        'methods': ['green_moreau', 'sro_fa', 'saa_knn'],
        'nn_training_counts': {str(p['Ntotal']): p['green']['Ntrain']
                               for p in data['panels']},
        'references': 'All saved horizontal references retained with original values.',
        'bands': 'Pointwise seed 20/80 percentiles; not confidence intervals.',
        'in_sample': 'Own finite-pool maxima mean; not a full adversarial supremum.',
        'caveat': 'Historical descriptive test; references are not certified population bounds.',
        'new_training_or_evaluation': False,
    }
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote R3/R10 PDF and PNG figures to {output}')


if __name__ == '__main__':
    main()
