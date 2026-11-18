"""Full-N green OOS and IS curves plus 80%-train green, SRO-FA / SRO-FA+LDR / SAA-kNN OOS curves.

Requested 15 October 2026: the comparators were fitted on all N = 50/200/800 rows,
like the full-N green refit, so they are the comparable references. Nothing is
retrained or re-evaluated: every curve is read from an already-published
plot_data.json and drawn with the shared, hash-locked renderer src/plot.py. The
test arrays must be byte-identical (same SHA-256) between green and comparators.
Green keeps its own-pool IS curve; comparators are OOS only. The neural SCS / Clarabel
comparator was dropped on 15 October 2026 at the user's request. The x-axis starts at
radius 0; the R3 y-axis is capped at 22. R3 also shows the full-N SRO-FA+LDR curve from
sro_nn_split_20261015/run_full_n.py (all N rows, same test set), drawn in the renderer's
otherwise unused 'scs' slot with its own label. Every case also shows the earlier green
STOP policies trained on 80% of the rows (40/160/640; same seeds, radii and test set) as an OOS
curve in the renderer's 'black' slot, to show the effect of the full-N refit.

Output: green_fullN_cv_20261011/runs/production_v1/published_with_comparators/
"""
import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from textwrap import fill

PROJECT = Path('/data/home/aviyac/retailers_experiment')
CAMPAIGN = PROJECT / 'workspace/05_history/original_paths/retailers_moreau_training_selected'
RUN = CAMPAIGN / 'green_fullN_cv_20261011/runs/production_v1'
OUTPUT = RUN / 'published_with_comparators'
STOPPING = PROJECT / 'comparison/stopping_checkpoint_20260928/production_v1/published'
LDR_JOBS = CAMPAIGN / 'sro_nn_split_20261015/runs/full_n_v1/jobs'
LDR_CASES = ('R3',)
# kind -> (R, comparator plot_data, comparator key -> renderer slot, y-axis cap)
CASES = {
    'R3': (3, STOPPING / 'plot_data_R3.json', {'sro_fa': 'sro_fa', 'saa_knn': 'saa_knn'}, 22.),
    'R10': (10, STOPPING / 'plot_data_R10.json', {'sro_fa': 'sro_fa', 'saa_knn': 'saa_knn'}, None),
    'shipping_R3': (3, CAMPAIGN / 'shipping_stopping_20261005/production_v2/published/plot_data.json',
                    {'sro_fa': 'sro_fa', 'saa_knn': 'saa_knn'}, None),
}
FOOTNOTE = {
    'R3': 'SRO-FA/kNN: Sept 28 results; SRO-FA+LDR: Oct 15 full-N sweep.',
    'R10': 'SRO-FA/kNN: Sept 28 results.',
    'shipping_R3': 'SRO-FA/kNN: Oct 5 results; their fee if q2 > 1e-6 / 1e-10, green q2 > 0.',
}
GREEN_RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)

sys.path.insert(0, str(PROJECT))
from src.plot import render  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def merge(kind):
    """Green full-N curves + comparator OOS curves; checks that all share one test set and seeds."""
    R, comparator_path, slots, _ = CASES[kind]
    green = read(RUN / f'published/{kind}/plot_data.json')
    old = read(comparator_path)
    for name in ('xi1', 'xi2'):
        if green['test']['arrays'][name]['array_sha256'] != old['test']['arrays'][name]['array_sha256']:
            raise ValueError(f'{kind}: comparator test set differs from green test set')
    if green['seeds'] != old['seeds']:
        raise ValueError(f'{kind}: seed lists differ')
    comparators = {p['Ntotal']: p for p in old['panels']}
    panels = []
    for panel in green['panels']:
        n = panel['Ntotal']
        curve = dict(panel['green'])
        curve['label'] = 'Green full-N (CV LR)'
        merged = dict(Ntotal=n, label=f'{kind} · N={n}', green=curve)
        for source, slot in slots.items():
            reference = copy.deepcopy(comparators[n][source])
            if reference['Ntrain'] != n:
                raise ValueError(f'{kind} N={n}: {source} was not fitted on all N rows')
            merged[slot] = reference
        earlier = copy.deepcopy(comparators[n]['green'])  # 80%-train green from the same comparison file
        if earlier['Ntrain'] != int(.8 * n) or earlier['radius'] != curve['radius']:
            raise ValueError(f'{kind} N={n}: unexpected earlier green curve')
        earlier.pop('finite_pool', None)  # OOS only, to keep the figure readable
        earlier['label'] = 'Green 80% train (40/160/640)'
        merged['black'] = earlier
        if kind in LDR_CASES:
            merged['scs'] = ldr_curve(kind, n, green['seeds'])
        panels.append(merged)
    note = (f'Same 10k test (seed 9999, already seen) for all methods; all except black fit on all N rows. Green: full-N '
            f'refit after train-only 5-fold LR CV; black: earlier green on 40/160/640 rows; green rho <= .03 = nominal. {FOOTNOTE[kind]} '
            f'Dashed: green own-pool IS. Minima are hindsight, not radius selection.')
    data = dict(status='completed', seeds=green['seeds'], panels=panels, test=green['test'],
                footnote=fill(note, width=200), sources=dict(green=str(RUN / f'published/{kind}/plot_data.json'),
                                                            comparators=str(comparator_path)))
    return data, R


def ldr_curve(kind, n, seeds):
    """Full-N SRO-FA+LDR test curve; every radius must be complete for every seed."""
    import numpy as np
    rows = [json.loads(p.read_text()) for p in LDR_JOBS.glob(f'fullN_FA_LDR_{kind}_N{n}_s*_r*.json')]
    if any(r['status'] != 'completed' for r in rows):
        raise ValueError(f'{kind} N={n}: failed SRO-FA+LDR jobs')
    radii = sorted({r['rho'] for r in rows})
    lookup = {(r['rho'], r['seed']): r['test']['mean'] for r in rows}
    if len(lookup) != len(radii) * len(seeds) or len(rows) != len(lookup):
        raise ValueError(f'{kind} N={n}: incomplete SRO-FA+LDR grid')
    values = np.array([[lookup[rho, s] for s in seeds] for rho in radii])
    return dict(method='sro_fa_ldr', label='SRO-FA+LDR', Ntotal=n, Ntrain=n, seeds=seeds, radius=radii,
                oos=dict(per_seed=values.tolist(), mean=values.mean(axis=-1).tolist(),
                         p20=np.percentile(values, 20, axis=-1).tolist(),
                         p80=np.percentile(values, 80, axis=-1).tolist()))


@contextmanager
def display_adjustments(top):
    """Just before the shared renderer saves: label x-ticks on the green grid only (the union with
    the dense comparator grids is unreadable) and apply the optional upper y-limit."""
    from matplotlib.figure import Figure
    original = Figure.savefig

    def savefig(fig, *args, **kwargs):
        for ax in fig.axes:
            ax.set_xticks(GREEN_RADII)
            ax.set_xlim(left=0.)
            if top is not None:
                ax.set_ylim(top=top)
        return original(fig, *args, **kwargs)

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = original


def main():
    OUTPUT.mkdir(exist_ok=False)
    for kind, (_, _, _, top) in CASES.items():
        data, R = merge(kind)
        data['y_axis_upper_limit'] = top
        with display_adjustments(top):
            render(data, OUTPUT / kind, R)
        print(kind, 'written to', OUTPUT / kind)


if __name__ == '__main__':
    main()
