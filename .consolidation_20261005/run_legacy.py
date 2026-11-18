"""Run frozen research from its relocated files without rewriting provenance.

This is path compatibility, not a security sandbox or a numerical backend.
Historical trees are read-only unless an existing output directory is explicitly
selected with --write. No fallback to old physical directories is supported.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent


def virtual_path(path, home, mappings):
    path = Path(path)
    if not path.is_absolute():
        path = home / path
    # Do not resolve legacy names: they deliberately do not exist on the host.
    for name, physical in mappings.items():
        if path.is_relative_to(physical):
            return home / name / path.relative_to(physical)
    return path


def namespace_command(home, mappings, cwd, command, writes=()):
    bwrap = shutil.which('bwrap')
    if bwrap is None:
        raise RuntimeError('Bubblewrap is required; no path-changing fallback is permitted')
    for name, physical in mappings.items():
        if (home / name).exists() or (home / name).is_symlink():
            raise RuntimeError(f'Unexpected original physical root: {home / name}')
        if not physical.is_dir() or physical.is_symlink():
            raise RuntimeError(f'Missing/noncanonical experiment directory: {physical}')
    cwd = virtual_path(cwd, home, mappings)
    owner = next((name for name in mappings if cwd.is_relative_to(home / name)), None)
    if owner is None or not (mappings[owner] / cwd.relative_to(home / owner)).is_dir():
        raise ValueError('Working directory must identify a relocated experiment directory')
    args = [bwrap, '--bind', '/', '/', '--dev-bind', '/dev', '/dev', '--tmpfs', str(home)]
    collections = {p.parts[len(home.parts)] for p in mappings.values()}
    for path in sorted(home.iterdir()):
        if path.is_symlink():
            args.extend(['--symlink', os.readlink(path), str(path)])
        elif path.is_dir() or path.is_file():
            args.extend(['--ro-bind' if path.name in collections else '--bind', str(path), str(path)])
        else:
            raise RuntimeError(f'Unsupported home entry; not silently omitted: {path}')
    for name, physical in sorted(mappings.items()):
        args.extend(['--ro-bind', str(physical), str(home / name)])
    for selected in writes:
        if '..' in Path(selected).parts:
            raise ValueError('--write cannot contain parent traversal')
        virtual = virtual_path(selected, home, mappings)
        name = next((n for n in mappings if virtual.is_relative_to(home / n)), None)
        if name is None or virtual == home / name:
            raise ValueError('--write must name an existing subdirectory, not a whole experiment')
        physical = mappings[name] / virtual.relative_to(home / name)
        resolved_root = mappings[name].resolve(strict=True)
        resolved = physical.resolve(strict=True)
        relative = physical.relative_to(mappings[name])
        ancestors = [mappings[name].joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]
        if (not physical.is_dir() or any(p.is_symlink() for p in ancestors)
                or resolved == resolved_root or not resolved.is_relative_to(resolved_root)):
            raise ValueError(f'Output must be a strict subdirectory without symlink traversal: {physical}')
        args.extend(['--bind', str(physical), str(physical), '--bind', str(physical), str(virtual)])
    converted = [str(virtual_path(arg, home, mappings)) if arg.startswith('/') else arg for arg in command]
    return args + ['--setenv', 'PWD', str(cwd), '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
                   '--chdir', str(cwd), '--'] + converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', required=True, help='Historical root/subdirectory or its organized physical path')
    parser.add_argument('--write', action='append', default=[], help='Explicit existing output subdirectory; repeatable')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Specify the command after --')
    mapping = json.loads((HERE / 'paths.json').read_text())
    home = Path(mapping['historical_home'])
    mappings = {name: home / relative for name, relative in mapping['physical_roots'].items()}
    invocation = namespace_command(home, mappings, args.cwd, command, args.write)
    record_dir = HERE / 'executions'
    record_dir.mkdir(exist_ok=True)
    record_path = record_dir / (uuid.uuid4().hex + '.json')
    record = {'kind': 'filesystem-compatibility invocation; NOT measured model runtime',
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'cwd': str(virtual_path(args.cwd, home, mappings)), 'command': command,
              'explicit_writable_directories': args.write, 'physical_roots': mapping['physical_roots'],
              'historical_records_modified': False}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    start = time.perf_counter()
    try:
        result = subprocess.run(invocation, check=False)
        record.update(status='completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())