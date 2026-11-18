"""Resolve historical absolute paths against this backup, never the old machine.

Frozen JSON/source bytes are not rewritten: their hashes are evidence. Use
resolve_archive_path() for file access, or local_json() for a clearly labelled
in-memory path-adjusted view. Neither function executes historical code.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_archive_path(value, *, root=None):
    """Resolve a local or recorded historical path, requiring it to exist.

    A recorded old-home path always maps to the backup, even when the original
    happens to exist. This prevents an incomplete backup from passing on the
    source machine. Unrelated absolute paths are rejected by design.
    """
    root = Path(root).resolve() if root is not None else ROOT
    mapping = json.loads((root / 'archive_paths.json').read_text())
    original = Path(mapping['source_home'])
    path = Path(value)
    if path.is_absolute() and path.is_relative_to(root):
        candidate = path
    elif path.is_absolute() and path.is_relative_to(original):
        candidate = root / mapping['archive_root'] / path.relative_to(original)
    elif not path.is_absolute():
        candidate = root / path
    else:
        raise ValueError(f'Path is not part of this backup: {value}')
    candidate = candidate.resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError(f'Path escapes the backup: {value}')
    return candidate


def local_json(value, *, root=None):
    """Return a derived view, keeping unavailable paths explicit, not invented."""
    root = Path(root).resolve() if root is not None else ROOT
    mapping = json.loads((root / 'archive_paths.json').read_text())
    prefix = mapping['source_home'].rstrip('/') + '/'
    path = resolve_archive_path(value, root=root)
    raw = path.read_bytes()
    unavailable = set()

    def translate(item):
        if isinstance(item, str) and item.startswith(prefix):
            try:
                return str(resolve_archive_path(item, root=root))
            except FileNotFoundError:
                unavailable.add(item)
                return item
        if isinstance(item, list):
            return [translate(v) for v in item]
        if isinstance(item, dict):
            return {translate(k): translate(v) for k, v in item.items()}
        return item

    data = translate(json.loads(raw))
    return {'kind': 'derived local path view; NOT the original sealed record',
            'original_record': str(path.relative_to(root)),
            'original_sha256': hashlib.sha256(raw).hexdigest(),
            'unavailable_recorded_paths': sorted(unavailable), 'data': data}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Backup-relative path or original absolute path')
    parser.add_argument('--json-view', action='store_true', help='Print labelled local-path JSON view')
    args = parser.parse_args(argv)
    if args.json_view:
        print(json.dumps(local_json(args.path), indent=2))
    else:
        print(resolve_archive_path(args.path))


if __name__ == '__main__':
    main()