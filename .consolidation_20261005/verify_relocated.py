"""Source/provenance-only relocation checks; no model, data or solver execution."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_legacy import namespace_command

HERE = Path(__file__).resolve().parent

CHECK = r'''
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

home = Path('/data/home/aviyac')
roots = json.loads(sys.argv[1])
for name in roots:
    p = home / name
    assert p.is_dir() and p.resolve() == p, p
source_hashes = 0
manifest = json.loads((home / 'hydro_moreau_small_radius_grid/results/grid_20260916T090259_289586Z/manifest.json').read_text())
for name, expected in manifest['hashes'].items():
    p = Path(name)
    if p.suffix in ('.py', '.sh', '.md'):
        assert str(p.resolve()) == name, name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
        source_hashes += 1

shipping = home / 'retailers_moreau_training_selected/shipping_comparison_20260928'
manifest = json.loads((shipping / 'production_v1/manifest.json').read_text())
module = ast.parse((shipping / 'common.py').read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'bound_file')
def check(ok, message):
    if not ok:
        raise AssertionError(message)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
namespace = {'Path': Path, 'check': check, 'sha': sha}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(shipping / 'common.py'), 'exec'), namespace)
bound_files = [p for p in manifest['files'] if Path(p).suffix in ('.py', '.sh', '.md')]
for p in bound_files:
    namespace['bound_file'](manifest, p)
assert (shipping.parent / 'shipping_stopping_20261005/production_v2/published/comparison_shipping.png').is_file()
assert (shipping.parent / 'shipping_runtime_comparison_20261005').is_dir()
import config
assert config.SOURCE_DATA_DIR == home / 'hydro_rcr/data/raw'
assert config.SOURCE_DATA_DIR.is_dir()
child = subprocess.check_output([sys.executable, '-B', '-c', 'import os;print(os.getcwd())'], text=True)
assert child.strip() == str(Path.cwd())
print(json.dumps({'virtual_roots': len(roots), 'hydro_frozen_source_hashes': source_hashes,
    'shipping_original_bound_file_checks': len(bound_files), 'new_October5_directories_retained': 2,
    'configuration_import': 'passed', 'child_cwd': 'passed', 'raw_observations_read': False,
    'solvers_training_evaluation_run': False}))
'''


def main():
    start = time.perf_counter()
    mapping = json.loads((HERE / 'paths.json').read_text())
    home = Path(mapping['historical_home'])
    roots = {name: home / rel for name, rel in mapping['physical_roots'].items()}
    command = namespace_command(home, roots, home / 'hydro_rcr_rescaled',
                                [sys.executable, '-B', '-c', CHECK, json.dumps(list(roots))])
    record = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'kind': 'source-only compatibility verification', 'workers': 1}
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        record.update(status='passed', checks=json.loads(result.stdout), stderr=result.stderr)
    except subprocess.CalledProcessError as error:
        record.update(status='failed', stdout=error.stdout, stderr=error.stderr, returncode=error.returncode)
        raise
    finally:
        record.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
        (HERE / 'source_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()