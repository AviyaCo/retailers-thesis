"""Display supplied, already-verified common-format plot_data only.

Accepts original comparison panel schema including optional black overlay:
panels[*].{green,black,scs,sro_fa}.{radius,seeds,oos,finite_pool}; each statistic
contains per_seed,mean,p20,p80. saa_knn is horizontal; horizontal contains
provided labeled scalar references. Nothing is imported from old renderers.
Hash/solver/provenance verification belongs to the producer; recomputing seed
statistics here is NOT independent certification of the producer's experiments.
"""
from pathlib import Path

import numpy as np

from .artifacts import timed, write_json

COLORS = dict(green='#228833', black='#000000', scs='#7755aa', sro_fa='#cc3311',
              saa_knn='#ccaa00')


def _statistics(stats, shape):
    values = np.asarray(stats['per_seed'], dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('Invalid per-seed statistic shape/values')
    actual = dict(mean=values.mean(axis=-1), p20=np.percentile(values, 20, axis=-1),
                  p80=np.percentile(values, 80, axis=-1))
    for key, value in actual.items():
        if not np.array_equal(np.asarray(stats[key]), value):
            raise ValueError(f'Incorrect pointwise statistic: {key}')


def validate_plot_data(data, R):
    if R not in (3, 10) or data['status'] != 'completed' or not data['panels']:
        raise ValueError('Require completed supplied R3/R10 plot data')
    # Both original formats bind retailer dimension in the common test arrays.
    for key in ('xi1', 'xi2'):
        if data['test']['arrays'][key]['shape'][1] != R:
            raise ValueError('Plot/test retailer dimension mismatch')
    seen = set()
    for panel in data['panels']:
        if panel['Ntotal'] in seen:
            raise ValueError('Duplicate sample-size panel')
        seen.add(panel['Ntotal'])
        if not any(method in panel for method in ('green', 'black')):
            raise ValueError('Require a supplied green or black curve')
        for method in ('green', 'black', 'scs', 'sro_fa', 'saa_knn'):
            if method not in panel:
                continue
            curve = panel[method]
            seeds = curve['seeds']
            if not seeds or len(set(seeds)) != len(seeds) or seeds != data['seeds']:
                raise ValueError('Inconsistent/duplicate seed ordering')
            shape = (len(seeds),)
            if method != 'saa_knn':
                radii = np.asarray(curve['radius'], dtype=float)
                if (radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all()
                        or (radii < 0).any() or (np.diff(radii) <= 0).any()):
                    raise ValueError('Require genuine increasing finite radius grid')
                shape = (len(radii), len(seeds))
            _statistics(curve['oos'], shape)
            if 'finite_pool' in curve:
                _statistics(curve['finite_pool'], shape)
        for reference in panel.get('horizontal', []):
            if not np.isfinite(reference['value']) or not reference['label']:
                raise ValueError('Invalid supplied horizontal reference')


def render(data, output, R):
    """Write exclusive consolidated PNG/PDF/JSON; preserve all supplied evidence."""
    validate_plot_data(data, R)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(task='plot', R=R, note='Display only; input verification is producer-owned')
    try:
        with timed(record):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter, NullLocator

            with plt.rc_context({'font.family': 'serif', 'font.size': 10,
                                 'pdf.fonttype': 42, 'axes.spines.top': False,
                                 'axes.spines.right': False}):
                fig, axes = plt.subplots(1, len(data['panels']), squeeze=False,
                                         figsize=(4.9 * len(data['panels']), 5.5))
                try:
                    handles = {}
                    for ax, panel in zip(axes[0], data['panels']):
                        ticks = set()
                        for method in ('green', 'black', 'scs', 'sro_fa'):
                            if method not in panel:
                                continue
                            curve = panel[method]
                            x, stat = curve['radius'], curve['oos']
                            ticks.update(x)
                            label = curve.get('label', method) + ' OOS'
                            line, = ax.plot(x, stat['mean'], color=COLORS[method],
                                            lw=1.9, marker='o', ms=3, label=label)
                            handles.setdefault(label, line)
                            ax.fill_between(x, stat['p20'], stat['p80'], color=COLORS[method], alpha=.12)
                            if 'finite_pool' in curve:
                                label = curve.get('label', method) + ' IS'
                                line, = ax.plot(x, curve['finite_pool']['mean'], '--',
                                                color=COLORS[method], alpha=.6, label=label)
                                handles.setdefault(label, line)
                        if 'saa_knn' in panel:
                            stat = panel['saa_knn']['oos']
                            label = panel['saa_knn'].get('label', 'SAA-kNN') + ' OOS'
                            line = ax.axhline(stat['mean'], color=COLORS['saa_knn'], label=label)
                            ax.axhspan(stat['p20'], stat['p80'], color=COLORS['saa_knn'], alpha=.08)
                            handles.setdefault(label, line)
                        for reference in panel.get('horizontal', []):
                            line = ax.axhline(reference['value'], color='#777777',
                                             ls='-.', lw=1.2, label=reference['label'])
                            handles.setdefault(reference['label'], line)
                        ax.set_xscale('symlog', linthresh=.001, linscale=.8)
                        ax.set_xticks(sorted(ticks))
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                        ax.xaxis.set_minor_locator(NullLocator())
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.set_title(panel.get('label', f"N={panel['Ntotal']}"))
                        ax.set_xlabel(r'Radius $\rho$')
                        ax.grid(axis='y', alpha=.25)
                    axes[0, 0].set_ylabel('Cost')
                    fig.suptitle(f'Fixed-radius policy comparison · R={R}')
                    fig.legend(list(handles.values()), list(handles), loc='lower center',
                               bbox_to_anchor=(.5, .08), ncol=4, frameon=False, fontsize=8)
                    fig.text(.5, .02, data.get('footnote', '') + '\nPointwise seed 20–80%; not confidence intervals.',
                             ha='center', fontsize=8)
                    fig.subplots_adjust(top=.85, bottom=.34, wspace=.25)
                    fig.savefig(output / f'comparison_R{R}.png', dpi=300)
                    fig.savefig(output / f'comparison_R{R}.pdf')
                finally:
                    plt.close(fig)
            write_json(output / 'plot_data.json', data)
    finally:
        write_json(output / 'runtime.json', record)
    return [output / f'comparison_R{R}.png', output / f'comparison_R{R}.pdf', output / 'plot_data.json']