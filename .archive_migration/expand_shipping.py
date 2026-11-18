"""Preserve shipping evidence and audit point-in-time backup coverage.

Originals and existing archived bytes are never overwritten. No experiments run.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
HERE = Path(__file__).resolve().parent / 'shipping_20260930'
ARCHIVE = ROOT / 'archive'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(root):
    if root.is_file():
        yield root
        return
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f'Symlink requires explicit review: {path}')
        for name in names:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f'Not a regular file: {path}')
            yield path


def main():
    started = time.perf_counter()
    HERE.mkdir(exist_ok=False)
    state = {'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
             'kind': 'backup migration and audit; not a research experiment',
             'clock': 'time.perf_counter', 'timing_unit': 'seconds',
             'originals_modified': False, 'existing_archive_overwritten': False,
             'copied_roots': [], 'verified_files': 0}

    def save():
        (HERE / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    save()
    try:
        selected = HOME / 'retailers_moreau_training_selected'
        sources = [HOME / 'retailers_with_shipping']
        sources += sorted(p for p in selected.iterdir()
                          if p.name.lower().startswith('shipping'))
        shipping_files = [p for source in sources for p in inventory(source)]
        for path in shipping_files:
            if path.name == 'gurobi.lic' or path.suffix in ('.pem', '.key'):
                raise ValueError(f'Credential-like file requires review: {path}')
        total = sum(p.stat().st_size for p in shipping_files)
        if shutil.disk_usage(ROOT).free < total * 1.1 + 10 * 1024**3:
            raise ValueError('Insufficient free-space reserve')
        state.update(files_expected=len(shipping_files), bytes_expected=total,
                     phase='copy_shipping')
        save()
        print(f'Shipping: {len(shipping_files)} files, {total} bytes', flush=True)
        stage = HERE / 'staging'
        stage.mkdir()
        for source in sources:
            relative = source.relative_to(HOME)
            if (ARCHIVE / relative).exists():
                raise FileExistsError(f'Will not overwrite {relative}')
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            begin = time.perf_counter()
            subprocess.run(['cp', '-a', '--reflink=auto', '--', str(source),
                            str(destination)], check=True)
            state['copied_roots'].append({'path': str(relative),
                                         'wall_time_s': time.perf_counter() - begin})
            save()
            print(f'Copied {relative}', flush=True)
        state['phase'] = 'verify_shipping_sha256'
        save()
        with (HERE / 'files.sha256.jsonl').open('x') as journal:
            for count, source in enumerate(shipping_files, 1):
                relative = source.relative_to(HOME)
                copy = stage / relative
                before = source.stat()
                digest = sha(source)
                if source.samefile(copy) or digest != sha(copy):
                    raise ValueError(f'Independent copy/hash mismatch: {source}')
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'Source changed during verification: {source}')
                journal.write(json.dumps({'path': str(relative), 'bytes': before.st_size,
                                          'sha256': digest}) + '\n')
                if count % 5000 == 0:
                    state['verified_files'] = count
                    save()
                    print(f'Hash verified {count}/{len(shipping_files)}', flush=True)
        # Detect newly added/deleted files before publishing the copies.
        if set(shipping_files) != {p for source in sources for p in inventory(source)}:
            raise ValueError('Shipping inventory changed during copy')
        for source in shipping_files:
            copy = stage / source.relative_to(HOME)
            a, b = source.stat(), copy.stat()
            if (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns):
                raise ValueError(f'Shipping metadata changed: {source}')
        for source in sources:
            relative = source.relative_to(HOME)
            destination = ARCHIVE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).rename(destination)
        shipping = ROOT / 'shipping'
        shipping.mkdir(exist_ok=False)
        for source in sources:
            link = shipping / source.name
            link.symlink_to(os.path.relpath(ARCHIVE / source.relative_to(HOME), shipping),
                            target_is_directory=source.is_dir())
        map_path = ROOT / 'archive_paths.json'
        mapping = json.loads(map_path.read_text())
        shutil.copy2(map_path, HERE / 'archive_paths_before.json')
        mapping['roots'] += [str(p.relative_to(HOME)) for p in sources]
        mapping['navigation'].update({str((shipping / p.name).relative_to(ROOT)):
                                      str(p.relative_to(HOME)) for p in sources})
        mapping['files'] += len(shipping_files)
        mapping['bytes'] += total
        map_path.write_text(json.dumps(mapping, indent=2) + '\n')
        state.update(verified_files=len(shipping_files), phase='audit_source_changes')
        save()
        covered = changed = missing = uncovered = plots_missing = 0
        plot_suffixes = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.eps'}
        audit_roots = sorted(p for p in HOME.glob('retailers_*')
                             if p.is_dir() and p != ROOT)
        root_counts = {}
        with (HERE / 'source_differences.jsonl').open('x') as journal:
            for root in audit_roots:
                counts = {'files': 0, 'covered': 0, 'uncovered': 0,
                          'missing_within_scope': 0, 'changed': 0, 'uncovered_plots': 0}
                for source in inventory(root):
                    counts['files'] += 1
                    relative = str(source.relative_to(HOME))
                    copy = ARCHIVE / relative
                    in_scope = any(relative == r or relative.startswith(r + '/')
                                   for r in mapping['roots'])
                    issue = None
                    if not copy.is_file():
                        issue = 'missing_within_scope' if in_scope else 'uncovered'
                        counts[issue] += 1
                        missing += int(in_scope)
                        uncovered += int(not in_scope)
                        if source.suffix.lower() in plot_suffixes:
                            plots_missing += 1
                            counts['uncovered_plots'] += 1
                    else:
                        a, b = source.stat(), copy.stat()
                        if (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns):
                            if sha(source) != sha(copy):
                                issue = 'changed'
                                changed += 1
                                counts['changed'] += 1
                        if issue is None:
                            covered += 1
                            counts['covered'] += 1
                    if issue:
                        journal.write(json.dumps({'path': relative, 'status': issue,
                                                  'bytes': source.stat().st_size}) + '\n')
                root_counts[root.name] = counts
                print(f'Audit {root.name}: {counts}', flush=True)
        state.update(status='completed', phase='completed', covered_files=covered,
                     changed_files=changed, missing_within_scope=missing,
                     uncovered_files=uncovered, uncovered_plots=plots_missing,
                     source_audit=root_counts,
                     audit_guarantee='Existing copies: size/mtime comparison; mismatches hashed. '
                                     'Shipping: all source/copy SHA256 pairs verified. '
                                     'Point-in-time, not an atomic snapshot.')
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        print(state['error'], flush=True)
        raise
    finally:
        state.update(end_utc=datetime.now(timezone.utc).isoformat(),
                     wall_time_s=time.perf_counter() - started)
        save()
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()