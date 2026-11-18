"""Commit/push code and plots on an isolated branch; preserve WIP/index.

Raw results/checkpoints are deliberately excluded per the user's storage choice.
"""
import json
import os
from pathlib import Path
import subprocess

HOME = Path(__file__).resolve().parents[2]
ROOT = HOME / 'retailers_experiment'
BRANCH = 'backup/retailers-code-plots-20260930'
EXTENSIONS = {'.py', '.md', '.sh', '.tex', '.toml', '.yaml', '.yml',
              '.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg'}


def main():
    report = json.loads((ROOT / '.archive_migration/shipping_20260930/status.json').read_text())
    if report['status'] != 'completed':
        raise ValueError('Shipping migration/audit must complete before Git publication')
    completion = json.loads((ROOT / '.archive_migration/completion_20260930/status.json').read_text())
    if completion['status'] != 'completed':
        raise ValueError('Retailers coverage completion must pass before Git publication')
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(HOME / '.git/retailers-code-plots-20260930.index')

    def git(*args, data=None):
        return subprocess.run(['git', '-c', 'gc.auto=0', *args], cwd=HOME, env=env,
                              input=data, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True).stdout

    if Path(env['GIT_INDEX_FILE']).exists():
        raise FileExistsError('Backup index exists; no implicit resume')
    exists = subprocess.run(['git', 'show-ref', '--verify', '--quiet', 'refs/heads/' + BRANCH],
                            cwd=HOME)
    if exists.returncode != 1:
        raise ValueError('Branch exists or reference check failed')
    identity = git('show', '-s', '--format=%an%n%ae', 'HEAD').decode().splitlines()
    if len(identity) != 2 or not all(identity):
        raise ValueError('Existing commit identity unavailable')
    env.update(GIT_AUTHOR_NAME=identity[0], GIT_AUTHOR_EMAIL=identity[1],
               GIT_COMMITTER_NAME=identity[0], GIT_COMMITTER_EMAIL=identity[1])
    groups = {}
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        # Installed dependency copies and bytecode are not source/plots backups.
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.pytest_cache',
                                               '.moreau_diagnostic_deps', 'staging'}
                   and not (Path(directory) / d).is_symlink()]
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                continue  # Do not publish shortcuts to excluded raw-result folders.
            relative = path.relative_to(ROOT)
            selected = path.suffix.lower() in EXTENSIONS
            if path.suffix == '.json':
                selected = (len(relative.parts) == 1 or
                            relative.as_posix() in {'black/config.json', 'green/config.json'} or
                            (relative.parts[0] == '.archive_migration' and
                             path.stat().st_size < 1024**2))
            selected |= name.startswith('requirements') and path.suffix == '.txt'
            if not selected:
                continue
            size = path.stat().st_size
            if size > 100_000_000:
                raise ValueError(f'Selected file exceeds conservative GitHub limit: {path}')
            if relative.parts[0] == 'archive':
                group = 'archive/' + relative.parts[1]
            else:
                group = relative.parts[0] if len(relative.parts) > 1 else 'overview'
            groups.setdefault(group, []).append((path, size))
    total = sum(size for files in groups.values() for _, size in files)
    plan = {'branch': BRANCH, 'selected_bytes': total,
            'groups': {group: {'files': len(files), 'bytes': sum(s for _, s in files)}
                       for group, files in sorted(groups.items())},
            'excludes': 'raw result JSONs, checkpoints, data arrays, logs, installed dependencies, symlinks',
            'working_branch_and_index_unchanged': True}
    (HOME / '.git/retailers-code-plots-20260930-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2), flush=True)
    if total > 1024**3:
        raise ValueError('Selected files exceed 1 GiB; review storage choice before pushing')
    git('read-tree', '--empty')
    parent = None
    for group, files in sorted(groups.items()):
        # Use Git's standard content hashing and index format; no checkout and no WIP history.
        rows = []
        for path, _ in sorted(files):
            mode = '100755' if path.stat().st_mode & 0o111 else '100644'
            oid = git('hash-object', '-w', '--', str(path)).decode().strip()
            rows.append(f'{mode} {oid}\t{path.relative_to(HOME)}\0')
        git('update-index', '-z', '--index-info', data=''.join(rows).encode())
        tree = git('write-tree').decode().strip()
        args = ['commit-tree', tree]
        if parent:
            args += ['-p', parent]
        commit = git(*args, data=f'Backup retailers code and plots: {group}\n'.encode()).decode().strip()
        if parent:
            git('update-ref', 'refs/heads/' + BRANCH, commit, parent)
        else:
            git('update-ref', 'refs/heads/' + BRANCH, commit, '0' * 40)
        print(f'Commit {commit} {group}: {len(files)} files', flush=True)
        result = git('push', 'origin', f'refs/heads/{BRANCH}:refs/heads/{BRANCH}')
        print(result.decode(), flush=True)
        parent = commit
    remote = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).decode().split()[0]
    if remote != parent:
        raise ValueError('Remote tip does not match final backup commit')
    print(f'VERIFIED REMOTE TIP {parent}', flush=True)


if __name__ == '__main__':
    main()