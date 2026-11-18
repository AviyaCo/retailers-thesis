"""Copy audit-discovered gaps without overwriting archived research evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
HERE = Path(__file__).resolve().parent / 'completion_20260930'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    start = time.perf_counter()
    HERE.mkdir(exist_ok=False)
    state = dict(status='running', start_utc=datetime.now(timezone.utc).isoformat(),
                 clock='time.perf_counter', timing_unit='seconds',
                 kind='backup completion, not experiment', verified_files=0,
                 originals_modified=False, archived_evidence_overwritten=False)
    try:
        audit = ROOT / '.archive_migration/shipping_20260930'
        if json.loads((audit / 'status.json').read_text())['status'] != 'completed':
            raise ValueError('Shipping audit incomplete')
        gaps = [json.loads(line) for line in (audit / 'source_differences.jsonl').open()]
        if any(row['status'] != 'uncovered' for row in gaps):
            raise ValueError('Changed/missing-in-scope evidence needs explicit version review')
        state['files_expected'] = len(gaps)
        state['bytes_expected'] = sum(row['bytes'] for row in gaps)
        if shutil.disk_usage(ROOT).free < state['bytes_expected'] * 1.1 + 1024**3:
            raise ValueError('Insufficient copy reserve')
        with (HERE / 'files.sha256.jsonl').open('x') as journal:
            for count, row in enumerate(gaps, 1):
                source = HOME / row['path']
                copy = ROOT / 'archive' / row['path']
                if copy.exists() or source.is_symlink() or source.name == 'gurobi.lic' or source.suffix in ('.pem', '.key'):
                    raise ValueError(f'Unsafe/newly existing destination or source: {source}')
                before = source.stat()
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copy)
                value = digest(source)
                if source.samefile(copy) or digest(copy) != value:
                    raise ValueError(f'Copy mismatch: {source}')
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'Source changed during copy: {source}')
                journal.write(json.dumps(dict(path=row['path'], bytes=before.st_size, sha256=value)) + '\n')
                state['verified_files'] = count
        # Final metadata/inventory audit of every original retailers source root.
        counts = {}
        for source_root in sorted(HOME.glob('retailers_*')):
            if not source_root.is_dir() or source_root == ROOT:
                continue
            count = 0
            for directory, dirs, names in os.walk(source_root, followlinks=False):
                for name in dirs + names:
                    if (Path(directory) / name).is_symlink():
                        raise ValueError('Original source symlink requires review')
                for name in names:
                    source = Path(directory) / name
                    copy = ROOT / 'archive' / source.relative_to(HOME)
                    a, b = source.stat(), copy.stat()
                    if (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns) and digest(source) != digest(copy):
                        raise ValueError(f'Current source differs: {source}')
                    count += 1
            counts[source_root.name] = count
        mapping_path = ROOT / 'archive_paths.json'
        mapping = json.loads(mapping_path.read_text())
        shutil.copy2(mapping_path, HERE / 'archive_paths_before.json')
        mapping['roots'] = [r for r in mapping['roots'] if not r.startswith('retailers_')]
        mapping['roots'] += list(counts)
        mapping['files'] += len(gaps)
        mapping['bytes'] += state['bytes_expected']
        mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
        state.update(status='completed', source_files_by_root=counts,
                     source_files_total=sum(counts.values()),
                     guarantee='All six original retailers roots covered. New gaps SHA256 verified; '
                               'existing copies size/mtime checked, mismatches hashed. '
                               'Original archived bytes never overwritten. Point-in-time only.')
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state.update(end_utc=datetime.now(timezone.utc).isoformat(),
                     wall_time_s=time.perf_counter() - start)
        (HERE / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()