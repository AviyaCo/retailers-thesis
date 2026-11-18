"""Join verified existing full-N NN curves to the unchanged comparators."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .render import ROOT, load_data, validate_plot_data
from .render_shipping import load_shipping_data


HISTORY = ROOT.parents[1] / 'workspace/05_history/original_paths'
RUN = HISTORY / (
    'retailers_moreau_training_selected/green_fullN_cv_20261011/runs/production_v1'
)
HASHES = {
    'R3': 'e9829c1033314f68f4f5df9fc1c7d1a6b7ce12963404f75e7711369d5444df2c',
    'R10': 'f3df3c08f2dab2b553b2da35887e7018828ed66f7e2cb5012c34f9a994075e07',
    'shipping_R3': '0298b807a44e48796c10c278b2a3869263409a5ffe3c8886feb2f57f9be4c697',
}
SEAL_HASH = 'e745f72675d9bdfb340624b718c25bfbf849989478df9647e11f6adbd0bfa3ee'


def _read_verified(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'Full-N source checksum mismatch: {path}')
    return json.loads(raw)


def load_full_n_data(kind):
    r = 10 if kind == 'R10' else 3
    base = load_shipping_data() if kind == 'shipping_R3' else load_data(r)[0]
    source = RUN / f'published/{kind}/plot_data.json'
    full = _read_verified(source, HASHES[kind])
    validate_plot_data(full, r)
    seal = _read_verified(RUN / 'training_seal.json', SEAL_HASH)
    if seal['status'] != 'sealed' or full['seeds'] != base['seeds']:
        raise ValueError('Unsealed full-N training or mismatched seeds')
    for name in ('xi1', 'xi2'):
        a, b = full['test']['arrays'][name], base['test']['arrays'][name]
        if any(a[key] != b[key] for key in ('array_sha256', 'shape', 'dtype')):
            raise ValueError('Full-N/comparator test arrays differ')
    configs = [row for row in seal['configurations'] if row['kind'] == kind]
    expected = {(n, seed, rho) for n in (50, 200, 800) for seed in range(10)
                for rho in (0., .001, .003, .01, .03, .2, .4, .7, 1.5)}
    if (len(configs) != 270
            or {(c['Ntotal'], c['seed'], c['radius']) for c in configs} != expected):
        raise ValueError('Incomplete full-N training grid')
    for config in configs:
        path = HISTORY / Path(config['summary_path']).relative_to('/data/home/aviyac')
        summary = _read_verified(path, config['summary_sha256'])
        if not (summary['Ntrain'] == summary['Ntotal'] == config['Ntotal']
                and summary['Nval'] == 0 and summary['status'] == 'completed'
                and summary['converged'] is True and summary['complete'] is True
                and summary['selected_candidate_id'] == summary['stopping_candidate_id']):
            raise ValueError(f'Not a completed full-N stopping policy: {path}')
    if [p['Ntotal'] for p in full['panels']] != [50, 200, 800]:
        raise ValueError('Incomplete full-N panels')
    data = deepcopy(base)
    for panel, replacement in zip(data['panels'], full['panels']):
        n = panel['Ntotal']
        for method in ('sro_fa', 'saa_knn'):
            if panel[method]['Ntrain'] != n:
                raise ValueError('Comparator is not fitted on full N')
        panel['green'] = deepcopy(replacement['green'])
        panel['green'].update(Ntotal=n, Ntrain=n)
        panel['training_counts']['green'] = n
    data['selection_rule'] = 'stopping_checkpoint'
    data['full_n_provenance'] = {
        'source_relative_to_experiment': str(source.relative_to(ROOT.parents[1])),
        'sha256': HASHES[kind],
        'training_seal_sha256': SEAL_HASH,
        'verified_full_n_training_summaries': len(configs),
        'protocol': 'Train-only five-fold nominal LR CV, fresh all-N refit, STOP CCG.',
        'original_full_n_footnote': full['footnote'],
        'comparators_and_references': 'Unchanged; test array identities and seeds matched.',
        'new_training_or_evaluation': False,
    }
    return data
