"""One-time, journaled same-filesystem relocation; never delete research bytes."""
import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from compare import metadata

HERE = Path(__file__).resolve().parent
HOME = Path('/data/home/aviyac')


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('hydro', 'retailers'))
    args = parser.parse_args()
    mapping = json.loads((HERE / 'paths.json').read_text())['physical_roots']
    selected = {HOME / name: HOME / relative for name, relative in mapping.items()
                if name.startswith(args.group + '_')}
    record_path = HERE / (args.group + '_relocation.json')
    if record_path.exists():
        raise RuntimeError('Relocation record already exists; inspect before any resume')
    if args.group == 'retailers':
        report = json.loads((HERE / 'comparison.json').read_text())
        if report['status'] != 'compared_not_deleted':
            raise RuntimeError('Exact duplicate audit is not complete')
    start = time.perf_counter()
    record = {'status': 'checking', 'started_at': datetime.now(timezone.utc).isoformat(),
              'method': 'same-filesystem rename with complete metadata/inode inventory',
              'workers': 1, 'roots': [], 'navigation': [], 'original_bytes_deleted': 0}
    save(record_path, record)
    try:
        for source, dest in selected.items():
            if not source.is_dir() or source.is_symlink():
                raise RuntimeError(f'Original root missing or not canonical: {source}')
            if not dest.is_symlink() or dest.resolve(strict=True) != source:
                raise RuntimeError(f'Destination is not the verified old navigation alias: {dest}')
            if source.stat().st_dev != dest.parent.stat().st_dev:
                raise RuntimeError('Cross-filesystem copy is not authorized by this rename plan')
            record['roots'].append({'source': str(source), 'destination': str(dest),
                                    'metadata_before': metadata(source), 'status': 'checked'})
        view = HOME / (args.group + '_experiment/workspace')
        for directory, dirs, files in os.walk(view, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if not path.is_symlink() or path in selected.values():
                    continue
                target = path.resolve(strict=True)
                for source, dest in selected.items():
                    if target.is_relative_to(source):
                        record['navigation'].append({'path': str(path), 'old': os.readlink(path),
                            'new': os.path.relpath(dest / target.relative_to(source), path.parent)})
                        break
        record['status'] = 'moving'
        save(record_path, record)
        for item in record['roots']:
            source, dest = Path(item['source']), Path(item['destination'])
            dest.unlink()  # Remove only the already-verified navigation symlink.
            try:
                source.rename(dest)
            except BaseException:
                dest.symlink_to(os.path.relpath(source, dest.parent), target_is_directory=True)
                raise
            item['status'] = 'moved'
            save(record_path, record)
            item['metadata_after'] = metadata(dest)
            if item['metadata_after'] != item['metadata_before']:
                raise RuntimeError(f'Original inventory changed: {dest}')
            item['status'] = 'verified'
            save(record_path, record)
        for item in record['navigation']:
            path = Path(item['path'])
            if os.readlink(path) != item['old']:
                raise RuntimeError(f'Navigation changed concurrently: {path}')
            path.unlink()
            path.symlink_to(item['new'])
            if not path.exists():
                raise RuntimeError(f'Retargeted link is broken: {path}')
        if any(os.path.lexists(source) for source in selected):
            raise RuntimeError('An original top-level path remains unexpectedly')
        record['status'] = 'relocated_verified'
    except BaseException as error:
        record.update(status='failed_inspect_journal_before_resume', error=repr(error))
        raise
    finally:
        record.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
        save(record_path, record)
    print(json.dumps({'status': record['status'], 'roots_moved': len(record['roots']),
                      'links_retargeted': len(record['navigation']), 'wall_seconds': record['wall_seconds']}))


if __name__ == '__main__':
    main()