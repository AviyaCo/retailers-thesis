"""Post-copy inventory closure and frozen-manifest dependency verification."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(base, roots):
    files, dirs = set(), set()
    for relative in roots:
        path = base / relative
        if path.is_file():
            files.add(relative)
            continue
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f'Missing regular directory: {path}')
        for directory, subdirs, names in os.walk(path, followlinks=False):
            dirs.add(str(Path(directory).relative_to(base)))
            for name in subdirs + names:
                p = Path(directory) / name
                if p.is_symlink():
                    raise ValueError(f'Unexpected nested symlink: {p}')
            for name in names:
                p = Path(directory) / name
                if not p.is_file():
                    raise ValueError(f'Not a regular file: {p}')
                files.add(str(p.relative_to(base)))
    return files, dirs


def main():
    start = time.perf_counter()
    report = {'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
              'clock': 'time.perf_counter', 'timing_unit': 'seconds',
              'kind': 'backup verification, not experiment'}
    try:
        status = json.loads((HERE / 'status.json').read_text())
        plan = json.loads((ROOT / 'archive_paths.json').read_text())
        assert status['status'] == 'completed'
        assert sha(HERE / 'files.sha256.jsonl') == status['inventory_sha256']
        old = Path(plan['source_home'])
        archive = ROOT / plan['archive_root']
        entries = {}
        with (HERE / 'files.sha256.jsonl').open() as stream:
            for line in stream:
                entry = json.loads(line)
                assert entry['path'] not in entries
                entries[entry['path']] = (entry['bytes'], entry['sha256'])
        assert len(entries) == status['files_expected'] == status['verified_files']
        source_files, source_dirs = inventory(old, plan['roots'])
        copy_files, copy_dirs = inventory(archive, plan['roots'])
        assert source_files == copy_files == set(entries), 'File inventory changed or incomplete'
        assert source_dirs == copy_dirs, 'Directory inventory mismatch'
        end_ns = int(datetime.fromisoformat(status['end_utc']).timestamp() * 1e9)
        for i, (relative, (size, _)) in enumerate(entries.items(), 1):
            a, b = (old / relative).stat(), (archive / relative).stat()
            assert a.st_size == b.st_size == size, relative
            assert a.st_mtime_ns == b.st_mtime_ns <= end_ns, relative
            assert (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), relative
            if i % 100000 == 0:
                print(f'Inventory/metadata checked {i}/{len(entries)}', flush=True)
        report.update(files=len(entries), bytes=sum(v[0] for v in entries.values()),
                      directories=len(copy_dirs), exact_inventory_match=True,
                      metadata_unchanged_since_copy=True, independent_copy_inodes=True)
        links = []
        for directory, dirs, files in os.walk(ROOT, followlinks=False):
            for name in dirs + files:
                p = Path(directory) / name
                if p.is_symlink():
                    assert not Path(os.readlink(p)).is_absolute(), p
                    assert p.resolve(strict=True).is_relative_to(ROOT), p
                    links.append(str(p.relative_to(ROOT)))
        assert len(links) == 26
        report.update(internal_relative_links=links, external_symlinks=0)
        manifests = [
            'retailers_moreau_training_selected/black_continuation_R3_R10_20260925/comparison_20260926/production_v2_20260928/manifest.json',
            'retailers_moreau_training_selected/black_continuation_R3_R10_20260925/production_v1/manifest.json',
            'retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/results/2026-09-24T091945.582854+0000/frozen/manifest.json',
            'retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1/frozen/manifest.json',
        ]
        paths = set()
        bindings = []
        documented_missing = set()

        def visit(value):
            if isinstance(value, str) and value.startswith(str(old) + '/'):
                paths.add(value)
            elif isinstance(value, dict):
                if (value.get('nominal_summary_status') == 'missing'
                        and isinstance(value.get('nominal_summary_path'), str)):
                    documented_missing.add(value['nominal_summary_path'])
                for k, v in value.items():
                    visit(k)
                    visit(v)
                    if (isinstance(k, str) and k.startswith(str(old) + '/')
                            and isinstance(v, str) and len(v) == 64
                            and all(c in '0123456789abcdef' for c in v)):
                        bindings.append((k, v))
            elif isinstance(value, list):
                for v in value:
                    visit(v)

        for name in manifests:
            visit(json.loads((archive / name).read_text()))
        excluded = {str(old / 'ap_env/bin/python')}
        missing = [p for p in paths if p not in excluded
                   and not (archive / Path(p).relative_to(old)).exists()]
        unexpected_missing = [p for p in missing if p not in documented_missing]
        assert not unexpected_missing, unexpected_missing[:20]
        assert all(not Path(p).exists() for p in missing), 'Documented missing source appeared after copying'
        for path, expected in bindings:
            relative = str(Path(path).relative_to(old))
            assert relative in entries and entries[relative][1] == expected, path
        report.update(key_manifests_checked=manifests, archived_references=len(paths - excluded),
                      frozen_hash_bindings_checked=len(bindings), missing_research_dependencies=[],
                      historically_missing_timing_records=sorted(missing),
                      excluded_environment_paths=sorted(paths & excluded),
                      guarantee='Point-in-time verified copy; not a filesystem snapshot or historical execution certification',
                      status='completed')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(end_utc=datetime.now(timezone.utc).isoformat(),
                      wall_time_s=time.perf_counter() - start)
        (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()