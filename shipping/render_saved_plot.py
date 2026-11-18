"""Recreate the published shipping PNG/PDF from local saved plot statistics.

No training, policy selection, inference, held-out evaluation or optimization.
Frozen archive files are imported/read, never modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from archive_paths import resolve_archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='New output directory; must not exist')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT / 'archive'):
        raise ValueError('Output must be new and outside the immutable archive')
    start = time.perf_counter()
    campaign = resolve_archive_path('archive/retailers_moreau_training_selected/shipping_comparison_20260928')
    data_path = campaign / 'production_v1/published/plot_data.json'
    renderer = campaign / 'plot_comparison.py'
    spec = importlib.util.spec_from_file_location('_saved_shipping_renderer', renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = data_path.read_bytes()
    data = json.loads(raw)
    output.mkdir(parents=True, exist_ok=False)
    module.render(data, output)
    (output / 'render_provenance.json').write_text(json.dumps({
        'kind': 'presentation-only recreation; not new experiment results',
        'plot_data': str(data_path.relative_to(ROOT)),
        'plot_data_sha256': hashlib.sha256(raw).hexdigest(),
        'renderer': str(renderer.relative_to(ROOT)),
        'renderer_sha256': hashlib.sha256(renderer.read_bytes()).hexdigest(),
        'wall_time_s': time.perf_counter() - start,
        'clock': 'time.perf_counter', 'timing_unit': 'seconds',
        'training_or_evaluation_run': False}, indent=2) + '\n')
    print(f'Recreated shipping PNG/PDF in {output}')


if __name__ == '__main__':
    main()