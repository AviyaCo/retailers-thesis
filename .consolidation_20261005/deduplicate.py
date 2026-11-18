"""Remove only archive bytes proven equal to canonical relocated current files.

Different revisions and archive-only entries remain untouched. A durable journal
and temporary holding directory precede removal; never resume blindly on error.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from compare import identity, metadata, verify_equal_now
from relocate import save

HERE = Path(__file__).resolve().parent
HOME = Path('/data/home/aviyac')
ARCHIVE = HOME / 'retailers_experiment/archive'


def main():
    start = time.perf_counter()
    report = json.loads((HERE / 'comparison.json').read_text())
    relocation = json.loads((HERE / 'retailers_relocation.json').read_text())
    if report['status'] != 'compared_not_deleted' or relocation['status'] != 'relocated_verified':
        raise RuntimeError('Required comparison/relocation verification absent')
    mappings = json.loads((HERE / 'paths.json').read_text())['physical_roots']
    journal_path = HERE / 'deduplication.json'
    if journal_path.exists():
        raise RuntimeError('Journal exists: manual inspection required before any resume')
    holding = HERE / 'duplicate_holding'
    holding.mkdir(exist_ok=False)
    journal = {'status': 'checking', 'started_at': datetime.now(timezone.utc).isoformat(),
               'workers': 1, 'entries': [], 'retained_differences': len(report['differences_preserved']),
               'duplicate_files_removed': 0, 'duplicate_logical_bytes_removed': 0}
    save(journal_path, journal)
    try:
        for entry in report['plans']:
            rel = Path(entry['relative'])
            canonical = HOME / mappings[rel.parts[0]] / Path(*rel.parts[1:])
            archived = ARCHIVE / rel
            if canonical.is_symlink() or archived.is_symlink():
                raise RuntimeError(f'Unexpected existing alias: {rel}')
            if metadata(canonical) != entry['current_metadata'] or metadata(archived) != entry['archive_metadata']:
                raise RuntimeError(f'File inventory changed after exact comparison: {rel}')
            journal['entries'].append({**entry, 'canonical': str(canonical), 'archive': str(archived),
                                       'held': str(holding / rel), 'status': 'checked'})
        journal['status'] = 'relinking_verified_duplicates'
        save(journal_path, journal)
        for entry in journal['entries']:
            archived, canonical, held = map(Path, (entry['archive'], entry['canonical'], entry['held']))
            held.parent.mkdir(parents=True, exist_ok=True)
            archived.rename(held)
            try:
                archived.symlink_to(os.path.relpath(canonical, archived.parent), target_is_directory=canonical.is_dir())
            except BaseException:
                held.rename(archived)
                raise
            entry['status'] = 'linked_duplicate_retained_temporarily'
            save(journal_path, journal)
        journal['status'] = 'removing_proven_duplicate_bytes'
        save(journal_path, journal)
        for entry in journal['entries']:
            canonical, archived, held = map(Path, (entry['canonical'], entry['archive'], entry['held']))
            if archived.resolve(strict=True) != canonical.resolve(strict=True):
                raise RuntimeError(f'Wrong replacement target: {archived}')
            # Planning fingerprints are not deletion proof. Compare bytes afresh
            # and reject any changed membership, symlink, or concurrent mutation.
            proof = verify_equal_now(canonical, held, entry['files'], entry['bytes'])
            entry['status'] = 'fresh_bytes_and_ctime_verified'
            save(journal_path, journal)
            # Postorder unlink, not rmtree: never delete an unverified addition.
            # A late new file makes rmdir fail and remains available for review.
            for current, duplicate, ci, ai, names in proof:
                if names is None:
                    if identity(current) != ci or identity(duplicate) != ai:
                        raise RuntimeError(f'File changed immediately before unlink: {duplicate}')
                    duplicate.unlink()
                else:
                    if identity(current) != ci:
                        raise RuntimeError(f'Canonical directory changed: {current}')
                    now = identity(duplicate)
                    if now[:3] != ai[:3]:
                        raise RuntimeError(f'Held directory identity changed: {duplicate}')
                    duplicate.rmdir()
            entry['status'] = 'duplicate_removed_canonical_retained'
            journal['duplicate_files_removed'] += entry['files']
            journal['duplicate_logical_bytes_removed'] += entry['bytes']
            save(journal_path, journal)
        # Only empty staging directories remain; do not remove unexpected files.
        for directory, dirs, files in os.walk(holding, topdown=False):
            if files:
                raise RuntimeError(f'Unexpected retained staging files: {directory}')
            Path(directory).rmdir()
        journal['status'] = 'deduplicated_verified'
    except BaseException as error:
        journal.update(status='failed_inspect_journal_before_resume', error=repr(error))
        raise
    finally:
        journal.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
        save(journal_path, journal)
    print(json.dumps({key: value for key, value in journal.items() if key != 'entries'}, indent=2))


if __name__ == '__main__':
    main()