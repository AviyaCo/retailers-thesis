"""One-time, fail-closed materialization of the approved historical archives.

No experiments are run. Original files are never edited. Frozen records retain
their original bytes; a separate path map locates them in the portable backup.
"""
from concurrent.futures import ThreadPoolExecutor
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
HERE = Path(__file__).resolve().parent
STAGE = HERE / 'archive_staging'
ARCHIVE = ROOT / 'archive'


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    target = HERE / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(pair):
    source, copy = pair
    before = source.stat()
    copied = copy.stat()
    if not copy.is_file() or source.samefile(copy):
        raise ValueError(f'Not an independent copied file: {copy}')
    original_sha = digest(source)
    if before.st_size != copied.st_size or original_sha != digest(copy):
        raise ValueError(f'Copy mismatch: {source}')
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during verification: {source}')
    return {'path': str(source.relative_to(HOME)), 'bytes': before.st_size,
            'sha256': original_sha}


def main():
    started = time.perf_counter()
    state = {'status': 'running', 'start_utc': utc(), 'clock': 'time.perf_counter',
             'timing_unit': 'seconds', 'kind': 'file migration, not experiment',
             'source_home': str(HOME), 'verification_threads': 4,
             'originals_modified': False, 'copied_roots': [], 'verified_files': 0}
    save('status.json', state)
    try:
        if ARCHIVE.exists() or STAGE.exists():
            raise FileExistsError('Archive/staging already exists; no overwrite or silent resume')
        navigation = {}
        for directory, dirs, files in os.walk(ROOT, followlinks=False):
            dirs[:] = [n for n in dirs if n != '.archive_migration']
            for name in dirs + files:
                p = Path(directory) / name
                if p.is_symlink():
                    target = p.resolve(strict=True)
                    if not target.is_relative_to(HOME) or target.is_relative_to(ROOT):
                        raise ValueError(f'Unexpected archive target: {target}')
                    navigation[str(p.relative_to(ROOT))] = str(target.relative_to(HOME))
        if len(navigation) != 26:
            raise ValueError(f'Expected exactly 26 approved links; got {len(navigation)}')
        sources = {HOME / p for p in navigation.values()}
        sources.update(HOME / p for p in (
            'hydro_rcr_rescaled/.moreau_diagnostic_deps',
            'retailers_moreau_N50_R3_net16_v2/baseline.py',
            'retailers_problem/plot_combined_R3_net16v2.py',
            'retailers_problem/R3/N50/nominal_timing_net16_v2',
            'retailers_problem/R3/selected_timing_net16_v2_run2',
        ))
        sources = sorted(p for p in sources
                         if not any(p != q and p.is_relative_to(q) for q in sources))
        files = []
        total = 0
        for source in sources:
            if not source.exists() or source.is_symlink():
                raise ValueError(f'Missing or indirect copy root: {source}')
            if source.is_file():
                files.append(source)
                total += source.stat().st_size
                continue
            for directory, dirs, names in os.walk(source, followlinks=False):
                for name in dirs + names:
                    path = Path(directory) / name
                    if path.is_symlink():
                        raise ValueError(f'Nested symlink requires explicit review: {path}')
                for name in names:
                    path = Path(directory) / name
                    if not path.is_file():
                        raise ValueError(f'Not a regular file: {path}')
                    if path.name == 'gurobi.lic' or path.suffix in ('.pem', '.key'):
                        raise ValueError(f'Credential-like file requires review: {path}')
                    files.append(path)
                    total += path.stat().st_size
        free = shutil.disk_usage(ROOT).free
        if free < total * 1.1 + 10 * 1024**3:
            raise ValueError(f'Insufficient copy reserve: bytes={total}, free={free}')
        plan = {'source_home': str(HOME), 'archive_root': 'archive',
                'navigation': navigation, 'roots': [str(p.relative_to(HOME)) for p in sources],
                'files': len(files), 'bytes': total,
                'frozen_records': 'unchanged; resolve old absolute paths under archive'}
        save('plan.json', plan)
        state.update(files_expected=len(files), bytes_expected=total, phase='copy')
        save('status.json', state)
        print(f'Copying {len(files)} files, {total / 1024**3:.3f} GiB in {len(sources)} roots', flush=True)
        STAGE.mkdir()
        for source in sources:
            relative = source.relative_to(HOME)
            destination = STAGE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            begin = time.perf_counter()
            # Reflinks, where supported, have independent inodes and copy-on-write
            # content. Never hard-link to the original evidence.
            subprocess.run(['cp', '-a', '--reflink=auto', '--', str(source), str(destination)], check=True)
            state['copied_roots'].append({'path': str(relative), 'wall_time_s': time.perf_counter() - begin})
            save('status.json', state)
            print(f'Copied {relative}', flush=True)
        state['phase'] = 'sha256_verification'
        save('status.json', state)
        with (HERE / 'files.sha256.jsonl').open('x') as journal:
            with ThreadPoolExecutor(max_workers=4) as pool:
                pairs = ((p, STAGE / p.relative_to(HOME)) for p in files)
                for count, entry in enumerate(pool.map(verify_file, pairs), 1):
                    journal.write(json.dumps(entry, separators=(',', ':')) + '\n')
                    if count % 10000 == 0:
                        journal.flush()
                        state['verified_files'] = count
                        save('status.json', state)
                        print(f'Hash verified {count}/{len(files)}', flush=True)
        state['verified_files'] = len(files)
        STAGE.rename(ARCHIVE)
        state['phase'] = 'local_navigation'
        save('status.json', state)
        for name, relative in navigation.items():
            link = ROOT / name
            if not link.is_symlink() or link.resolve() != HOME / relative:
                raise ValueError(f'Navigation changed during copy: {link}')
            new_link = link.with_name(link.name + '.local.tmp')
            new_link.symlink_to(os.path.relpath(ARCHIVE / relative, link.parent), target_is_directory=True)
            new_link.replace(link)
            if not link.resolve().is_relative_to(ROOT):
                raise ValueError(f'Link still external: {link}')
        (ROOT / 'archive_paths.json').write_text(json.dumps(plan, indent=2) + '\n')
        state.update(status='completed', phase='completed', archive='archive',
                     external_navigation_links=0,
                     inventory_sha256=digest(HERE / 'files.sha256.jsonl'))
        print('COPY AND FULL HASH VERIFICATION COMPLETE; all navigation now stays inside the backup', flush=True)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        print(state['error'], flush=True)
        raise
    finally:
        state.update(end_utc=utc(), wall_time_s=time.perf_counter() - started)
        save('status.json', state)


if __name__ == '__main__':
    main()