"""Read-only exact duplicate audit; never imports or evaluates research code."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from datetime import datetime, timezone

HOME = Path('/data/home/aviyac')
HERE = Path(__file__).resolve().parent
NAMES = ('retailers_problem', 'retailers_with_shipping',
         'retailers_moreau_training_selected', 'retailers_moreau',
         'retailers_moreau_N50_R3_net16_v2', 'retailers_scs_vs_moreau')
START = time.perf_counter()
COUNTS = {'compared_files': 0, 'equal_files': 0, 'equal_bytes': 0,
          'different_files': 0, 'archive_only_entries': 0, 'current_only_entries': 0}
DIFFERENCES = []


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns,
            s.st_ctime_ns, os.readlink(path) if path.is_symlink() else None]


def sha(path):
    before = identity(path)
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    if before != identity(path):
        raise RuntimeError(f'Concurrent change while hashing: {path}')
    return value.hexdigest()


def metadata(path):
    """Fingerprint including inode; rename changes ctime, so omit ctime here."""
    h = hashlib.sha256()
    def visit(p, relative):
        info = identity(p)
        h.update(json.dumps([relative, info[:5], info[6]], separators=(',', ':')).encode())
        if stat.S_ISDIR(info[2]):
            for child in sorted(p.iterdir()):
                visit(child, relative + '/' + child.name)
    visit(path, '')
    return h.hexdigest()


def candidate(relative, current, archived, files, size):
    return {'relative': relative, 'files': files, 'bytes': size,
            'current_metadata': metadata(current), 'archive_metadata': metadata(archived)}


def verify_equal_now(current, archived, expected_files, expected_bytes):
    """Fresh strict deletion proof, including membership and per-object ctime.

    This does not reuse the planning scan's equality. Symlinks are rejected,
    even if their text matches, because relocation can change relative meaning.
    A second metadata pass binds each digest to the exact tree just verified.
    """
    checked = []
    count = size = 0

    def visit(c, a):
        nonlocal count, size
        ci, ai = identity(c), identity(a)
        if stat.S_IFMT(ci[2]) != stat.S_IFMT(ai[2]):
            raise RuntimeError(f'Type changed: {a}')
        if stat.S_ISLNK(ci[2]) or stat.S_ISLNK(ai[2]):
            raise RuntimeError(f'Never collapse a subtree containing symlinks: {a}')
        if stat.S_IMODE(ci[2]) != stat.S_IMODE(ai[2]):
            raise RuntimeError(f'Mode differs: {a}')
        names = None
        if stat.S_ISDIR(ci[2]):
            names = sorted(p.name for p in c.iterdir())
            if names != sorted(p.name for p in a.iterdir()):
                raise RuntimeError(f'Directory membership differs: {a}')
            for name in names:
                visit(c / name, a / name)
        elif stat.S_ISREG(ci[2]):
            if ci[3:5] != ai[3:5] or sha(c) != sha(a):
                raise RuntimeError(f'Fresh content/mtime mismatch: {a}')
            count += 1
            size += ci[3]
        else:
            raise RuntimeError(f'Unsupported object: {a}')
        if ci != identity(c) or ai != identity(a):
            raise RuntimeError(f'Concurrent change during exact verification: {a}')
        checked.append((c, a, ci, ai, names))

    visit(current, archived)
    if (count, size) != (expected_files, expected_bytes):
        raise RuntimeError('Fresh verified inventory differs from the planned duplicate inventory')
    for c, a, ci, ai, names in checked:
        if ci != identity(c) or ai != identity(a):
            raise RuntimeError(f'Concurrent change after exact verification: {a}')
        if names is not None and (names != sorted(p.name for p in c.iterdir())
                                  or names != sorted(p.name for p in a.iterdir())):
            raise RuntimeError(f'Directory membership changed after verification: {a}')
    return checked


def compare(current, archived, relative):
    c, a = identity(current), identity(archived)
    if stat.S_IFMT(c[2]) != stat.S_IFMT(a[2]):
        DIFFERENCES.append({'path': relative, 'reason': 'type differs'})
        return False, [], 0, 0
    if stat.S_ISDIR(c[2]):
        cs = {p.name: p for p in current.iterdir()}
        ars = {p.name: p for p in archived.iterdir()}
        all_equal = cs.keys() == ars.keys()
        for name in sorted(cs.keys() - ars.keys()):
            COUNTS['current_only_entries'] += 1
            DIFFERENCES.append({'path': relative + '/' + name, 'reason': 'current only'})
        for name in sorted(ars.keys() - cs.keys()):
            COUNTS['archive_only_entries'] += 1
            DIFFERENCES.append({'path': relative + '/' + name, 'reason': 'archive only'})
        plans, files, size = [], 0, 0
        for name in sorted(cs.keys() & ars.keys()):
            same, child_plans, n, b = compare(cs[name], ars[name], relative + '/' + name)
            all_equal = all_equal and same
            plans.extend(child_plans)
            files += n
            size += b
        # Directory permissions remain part of the compatibility requirement.
        all_equal = all_equal and stat.S_IMODE(c[2]) == stat.S_IMODE(a[2])
        if all_equal and files:
            return True, [candidate(relative, current, archived, files, size)], files, size
        return all_equal, plans, files, size
    if stat.S_ISLNK(c[2]):
        same = c[6] == a[6]
        if not same:
            DIFFERENCES.append({'path': relative, 'reason': 'symlink differs'})
        # Do not deduplicate individual symlinks or change their semantics.
        return same, [], 0, 0
    if not stat.S_ISREG(c[2]):
        raise RuntimeError(f'Unsupported filesystem object: {current}')
    COUNTS['compared_files'] += 1
    if COUNTS['compared_files'] % 10000 == 0:
        write(HERE / 'status.json', {'status': 'comparing', 'started_at': STARTED,
              'wall_seconds': time.perf_counter() - START, **COUNTS, 'current_path': relative})
        print(COUNTS, relative, flush=True)
    same = c[3:5] == a[3:5] and stat.S_IMODE(c[2]) == stat.S_IMODE(a[2])
    # Equal timestamps never establish equality. Hash both files independently.
    if c[3] == a[3]:
        ch, ah = sha(current), sha(archived)
        same = same and ch == ah
    else:
        ch, ah = None, None
    if not same:
        COUNTS['different_files'] += 1
        DIFFERENCES.append({'path': relative, 'reason': 'bytes or metadata differ',
                            'current_sha256': ch, 'archive_sha256': ah})
        return False, [], 0, 0
    COUNTS['equal_files'] += 1
    COUNTS['equal_bytes'] += c[3]
    return True, [candidate(relative, current, archived, 1, c[3])], 1, c[3]


if __name__ == '__main__':
    STARTED = utc()
    try:
        plans = []
        for name in NAMES:
            _, entries, _, _ = compare(HOME / name, HOME / 'retailers_experiment/archive' / name, name)
            plans.extend(entries)
        report = {'status': 'compared_not_deleted', 'started_at': STARTED, 'ended_at': utc(),
                  'wall_seconds': time.perf_counter() - START, 'workers': 1,
                  'method': 'independent SHA256 for same-sized regular file pairs; matching mode/mtime; no data parsing',
                  'counts': COUNTS, 'plans': plans, 'differences_preserved': DIFFERENCES}
        write(HERE / 'comparison.json', report)
        write(HERE / 'status.json', {k: v for k, v in report.items() if k not in ('plans', 'differences_preserved')})
        print(json.dumps({'status': report['status'], 'wall_seconds': report['wall_seconds'],
                          'counts': COUNTS, 'maximal_duplicate_entries': len(plans),
                          'differences': len(DIFFERENCES)}), flush=True)
    except BaseException as error:
        write(HERE / 'status.json', {'status': 'failed', 'started_at': STARTED, 'ended_at': utc(),
              'wall_seconds': time.perf_counter() - START, 'error': repr(error), **COUNTS})
        raise