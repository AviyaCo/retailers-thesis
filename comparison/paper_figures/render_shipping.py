"""Apply the paper layout to the existing shipping STOP results only."""
from copy import deepcopy
import hashlib
import json

import matplotlib.pyplot as plt

from .render import ROOT, SIZE_CM, build_figure


SOURCE = ROOT.parents[1] / (
    'workspace/02_results/Moreau/shipping_stopping_revision/'
    'production_v2/published/plot_data.json'
)
SOURCE_SHA256 = '8c13f37d6ac4dbcbca7d18437f38d1feff89b0968d06633178b320aec09f6089'


def load_shipping_data():
    payload = SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError(f'Shipping source checksum mismatch: {SOURCE}')
    data = json.loads(payload)
    if (data['return_rule'] != 'stopping_checkpoint'
            or data['current_revision']['fresh_green_policies'] != 270):
        raise ValueError('Require the completed shipping STOP revision')
    return data


def build_shipping_figure(data):
    if data['return_rule'] != 'stopping_checkpoint':
        raise ValueError('Require shipping stopping checkpoints')
    display = deepcopy(data)
    display['selection_rule'] = data['return_rule']
    return build_figure(display, 3)


def main():
    from .full_n import load_full_n_data

    data = load_full_n_data('shipping_R3')
    fig = build_shipping_figure(data)
    output = ROOT / 'shipping_published'
    try:
        output.mkdir(exist_ok=False)
        for ext in ('pdf', 'png'):
            fig.savefig(output / f'comparison_shipping.{ext}', dpi=300)
        record = {
            'source_relative_to_experiment': str(SOURCE.relative_to(ROOT.parents[1])),
            'source_sha256': SOURCE_SHA256,
            'full_n_nn': data['full_n_provenance'],
            'size_cm': SIZE_CM,
            'nn_color': 'blue (#2465aa); original green Moreau experiment',
            'methods': ['green_moreau', 'sro_fa', 'saa_knn'],
            'nn_training_counts': {str(p['Ntotal']): p['green']['Ntrain']
                                   for p in data['panels']},
            'return_rule': data['return_rule'],
            'original_caption': data['caption'],
            'band_definition': data['band_definition'],
            'is_definition': data['is_definition'],
            'reference': data['panels'][0]['horizontal'][0],
            'new_training_or_evaluation': False,
        }
        (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    finally:
        plt.close(fig)
    print(f'Wrote shipping PDF and PNG to {output}')


if __name__ == '__main__':
    main()
