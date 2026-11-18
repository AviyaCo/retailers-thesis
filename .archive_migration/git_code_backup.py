"""Publish the Retailers code to the public thesis repository AviyaCo/retailers-thesis (branch main).

History: 2026-11-17 written as a backup to branch backup/retailers-code-plots-20260930 of AviyaCo/gold-srv2-homedir
(that branch is kept unchanged); 2026-11-18 retargeted, by user decision, to a dedicated public repository for thesis
inspection: the contents of retailers_experiment/ at the top level, first commit without history ("clean").

Safety model: a private index file and `git commit-tree` in the home repository, no checkout, so the working branch
(WIP) and its index are never touched and WIP (environments, Gurobi license, oversized files) is never pushed.
Commits go to the local branch `retailers-thesis-main`, pushed to `main` of the remote `retailers-thesis`.

Selected: all source code (Python, shell, MATLAB), documentation, configuration, figures, requirements.txt and
setup_env.sh, the bundled Moreau 0.4.0 package folder the code loads (Apache-2.0, licence files included), the
path-compatibility runner records, and a whitelist of small final-result / timing / statistics JSONs (reproduction
targets, each at most 5 MB, never per-job folders). Excluded: checkpoints, raw result arrays and per-job JSONs, logs,
installed dependencies, symlinks, hydro (not final), the archived unused training CLI, unpublished manuscript drafts
and third-party code without a known licence (EXCLUDE), and anything above the size guards.

Usage (from the home repository root /data/home/aviyac):
    ap_env/bin/python -B retailers_experiment/.archive_migration/git_code_backup.py --dry-run
    ap_env/bin/python -B retailers_experiment/.archive_migration/git_code_backup.py --commit   # local commit only
    git push retailers-thesis retailers-thesis-main:main                                       # from a VS Code terminal
    ap_env/bin/python -B retailers_experiment/.archive_migration/git_code_backup.py --push     # commit + push + verify
"""
import argparse
from datetime import datetime, timezone
import json
import os
import re
import subprocess
from pathlib import Path

HOME = Path(__file__).resolve().parents[2]
ROOT = HOME / 'retailers_experiment'
REMOTE = 'retailers-thesis'
REMOTE_URL = 'https://github.com/AviyaCo/retailers-thesis.git'
LOCAL_BRANCH = 'retailers-thesis-main'
REMOTE_BRANCH = 'main'
# Public identity for commits in the public repository (user, 2026-11-18): GitHub no-reply e-mail, not the server's.
AUTHOR_NAME = 'Aviya-Avigail Strauss'
AUTHOR_EMAIL = '86801592+AviyaCo@users.noreply.github.com'
HISTORY = 'workspace/05_history/original_paths/'
# Public-release exclusions (user, 2026-11-18): unpublished manuscript drafts; SRO authors' Julia code (no licence).
EXCLUDE = {HISTORY + 'retailers_problem/' + name for name in (
    'retailers_experiment_revised.tex', 'retailers_experiment_revised.pdf', 'results_thesis_ready.tex',
    'results_thesis_ready_check.pdf', 'experiment_inventory_pooling.jl', 'inventory_pooling_new.jl')}
CODE = {'.py', '.sh', '.jl', '.m', '.md', '.tex', '.bib', '.toml', '.yaml', '.yml', '.cfg', '.ini',
        '.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg'}
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.moreau_diagnostic_deps', 'staging', '.deps', 'site-packages',
             'jobs', 'ccg_results'}
SKIP_PREFIXES = ('archive/hydro', 'archive/unused_training_cli_20261118')
# Moreau 0.4.0 + pinned deps (Apache-2.0/MIT, license files included), loaded from this exact folder by src/config.py.
BUNDLE = 'archive/hydro_rcr_rescaled/.moreau_diagnostic_deps'
RESULT_JSON = re.compile(r'^(config|selections|selection_seal|test_results|selected_test_results|'
                         r'paired_tests[a-z_]*|best_radius[a-z_]*|runtime[a-z_]*|[A-Za-z0-9_]*timing[a-z_]*)\.json$'
                         r'|\.(invocation|launch)\.json$')
TRAINING = 'workspace/05_history/original_paths/retailers_moreau_training_selected/'
# Final per-seed results behind the paper table, figures and paired tests (small; reproduction targets).
FINAL_RESULTS = {
    TRAINING + 'green_validation_R3_R10_20261015/production_v1/results.json',
    TRAINING + 'green_validation_R3_R10_20261015/production_v1/summary.json',
    TRAINING + 'shipping_validation_qpositive_20261006/production_v1/summary.json',
    TRAINING + 'green_fullN_cv_20261011/runs/production_v1/training_seal.json',
    TRAINING + 'green_fullN_cv_20261011/runs/production_v1/published/R3/plot_data.json',
    TRAINING + 'green_fullN_cv_20261011/runs/production_v1/published/R10/plot_data.json',
    TRAINING + 'green_fullN_cv_20261011/runs/production_v1/published/shipping_R3/plot_data.json',
}
RESULT_JSON_MAX = 5 * 1024**2
FILE_MAX = 50 * 1024**2
TOTAL_MAX = 1024**3
SECRET = re.compile(rb'(WLSSECRET|WLSACCESSID|LICENSEID\s*=|ghp_[A-Za-z0-9]{20,}|github_pat_|BEGIN [A-Z ]*PRIVATE KEY)')


def selected(path, relative):
    suffix = path.suffix.lower()
    if relative.as_posix().startswith(BUNDLE + '/'):
        return True
    if suffix in CODE or (path.name.startswith('requirements') and suffix == '.txt'):
        return True
    if suffix != '.json':
        return False
    size = path.stat().st_size
    if relative.parts[0] in ('.archive_migration', '.consolidation_20261005'):
        return size < 1024**2  # migration journals / path-compatibility mapping used by run_legacy.py
    if relative.as_posix() in FINAL_RESULTS or relative.as_posix().startswith('comparison/paper_figures/'):
        return size <= RESULT_JSON_MAX
    return len(relative.parts) == 1 or (RESULT_JSON.search(path.name) is not None and size <= RESULT_JSON_MAX)


def plan():
    files, skipped_large = [], []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        base = Path(directory)
        dirs[:] = [d for d in dirs if not (base / d).is_symlink() and (
            (base / d).relative_to(ROOT).as_posix() in (BUNDLE, 'archive/hydro_rcr_rescaled')
            or (base / d).relative_to(ROOT).as_posix().startswith(BUNDLE + '/') and d != '__pycache__'
            or d not in SKIP_DIRS and not (base / d).relative_to(ROOT).as_posix().startswith(SKIP_PREFIXES))]
        for name in names:
            path = base / name
            if path.is_symlink():
                continue
            relative = path.relative_to(ROOT)
            if relative.as_posix() in EXCLUDE or not selected(path, relative):
                continue
            size = path.stat().st_size
            if size > FILE_MAX:
                skipped_large.append((relative.as_posix(), size))
                continue
            files.append((path, size))
    return sorted(files), skipped_large


def main(push, commit_only=False):
    files, skipped_large = plan()
    total = sum(s for _, s in files)
    leaks = [str(p.relative_to(ROOT)) for p, _ in files if p != Path(__file__).resolve()
             and p.suffix.lower() not in {'.png', '.pdf', '.jpg', '.jpeg', '.eps'} and SECRET.search(p.read_bytes())]
    groups = {}
    for path, size in files:
        key = path.relative_to(ROOT).parts[0] if len(path.relative_to(ROOT).parts) > 1 else 'overview'
        g = groups.setdefault(key, [0, 0])
        g[0] += 1
        g[1] += size
    report = dict(remote=REMOTE_URL, branch=REMOTE_BRANCH, files=len(files), selected_bytes=total, groups=groups,
                  skipped_over_file_max=skipped_large, possible_secrets=leaks)
    print(json.dumps(report, indent=1), flush=True)
    if leaks:
        raise ValueError('possible secrets in selected files; review before publishing')
    if total > TOTAL_MAX:
        raise ValueError('selection exceeds 1 GiB; review before publishing')
    if not (push or commit_only):
        return

    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(HOME / '.git/retailers-code-backup.index')
    env.update(GIT_AUTHOR_NAME=AUTHOR_NAME, GIT_AUTHOR_EMAIL=AUTHOR_EMAIL,
               GIT_COMMITTER_NAME=AUTHOR_NAME, GIT_COMMITTER_EMAIL=AUTHOR_EMAIL)

    def git(*args, data=None):
        return subprocess.run(['git', '-c', 'gc.auto=0', *args], cwd=HOME, env=env, input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout

    if REMOTE not in git('remote').decode().split():
        git('remote', 'add', REMOTE, REMOTE_URL)
    tracking = f'refs/remotes/{REMOTE}/{REMOTE_BRANCH}'
    if push and git('ls-remote', '--heads', REMOTE, REMOTE_BRANCH).strip():
        git('fetch', REMOTE, f'refs/heads/{REMOTE_BRANCH}:{tracking}')
    exists = lambda ref: subprocess.run(['git', 'rev-parse', '--verify', '--quiet', ref], cwd=HOME,
                                        stdout=subprocess.DEVNULL).returncode == 0
    parent = git('rev-parse', f'refs/heads/{LOCAL_BRANCH}').decode().strip() if exists(f'refs/heads/{LOCAL_BRANCH}') else None
    if exists(tracking):
        remote_tip = git('rev-parse', tracking).decode().strip()
        # Local commits not yet pushed (made with --commit) are kept; the branch must not have diverged.
        if parent is None or subprocess.run(['git', 'merge-base', '--is-ancestor', remote_tip, parent],
                                            cwd=HOME).returncode != 0:
            raise ValueError(f'local {LOCAL_BRANCH} has diverged from {REMOTE}/{REMOTE_BRANCH}; reconcile it first')
    Path(env['GIT_INDEX_FILE']).unlink(missing_ok=True)
    git('read-tree', '--empty')
    rows = []
    for path, _ in files:
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        oid = git('hash-object', '-w', '--', str(path)).decode().strip()
        rows.append(f'{mode} {oid}\t{path.relative_to(ROOT).as_posix()}\0')
    git('update-index', '-z', '--index-info', data=''.join(rows).encode())
    tree = git('write-tree').decode().strip()
    Path(env['GIT_INDEX_FILE']).unlink()
    if parent and git('rev-parse', parent + '^{tree}').decode().strip() == tree:
        print(f'No changes since {parent}; nothing committed.', flush=True)
    else:
        message = (f'Retailers thesis code ({datetime.now(timezone.utc):%Y-%m-%d})\n\n'
                   'Code, docs, configs, figures, requirements/setup, bundled Moreau and small final-result/timing/\n'
                   'statistics JSONs for the retailer experiments (protocols A and B: Neural CCG, SRO-FA, SAA-kNN,\n'
                   'clairvoyant references). Start with README.md and REPRODUCE.md. Checkpoints and raw results\n'
                   'are not included.\n\n'
                   'Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>\n')
        commit = git('commit-tree', tree, *(['-p', parent] if parent else []), data=message.encode()).decode().strip()
        git('update-ref', f'refs/heads/{LOCAL_BRANCH}', commit, *([parent] if parent else ['0' * 40]))
        parent = commit
        print(f'COMMITTED {commit} on local {LOCAL_BRANCH}', flush=True)
    (HOME / '.git/retailers-thesis-publish-plan.json').write_text(json.dumps(dict(report, commit=parent), indent=1))
    if commit_only:
        print(f'Push with:  cd {HOME} && git push {REMOTE} {LOCAL_BRANCH}:{REMOTE_BRANCH}', flush=True)
        return
    git('push', REMOTE, f'{parent}:refs/heads/{REMOTE_BRANCH}')
    git('fetch', REMOTE, f'refs/heads/{REMOTE_BRANCH}:{tracking}')
    if git('rev-parse', tracking).decode().strip() != parent:
        raise ValueError('remote tip does not match the new commit')
    print(f'VERIFIED {REMOTE}/{REMOTE_BRANCH} = {parent}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--commit', action='store_true', help=f'commit on local {LOCAL_BRANCH} only (no network)')
    mode.add_argument('--push', action='store_true', help='fetch, commit, push and verify (needs GitHub credentials)')
    args = parser.parse_args()
    main(args.push, args.commit)
