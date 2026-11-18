"""Tiny local filesystem fixtures only; not experimental data."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import compare
import deduplicate


class DuplicateChecks(unittest.TestCase):
    def test_same_size_and_timestamp_but_different_bytes_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a', root / 'b'
            a.write_bytes(b'abc')
            shutil.copy2(a, b)
            stamp = b.stat().st_mtime_ns
            b.write_bytes(b'xyz')
            os.utime(b, ns=(stamp, stamp))
            same, plans, _, _ = compare.compare(a, b, 'fixture')
            self.assertFalse(same)
            self.assertEqual(plans, [])

    def test_missing_or_changed_revision_kept_and_duplicates_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            here = home / 'records'
            here.mkdir()
            current = home / 'organized/old'
            archived = home / 'archive/old'
            current.mkdir(parents=True)
            archived.mkdir(parents=True)
            (current / 'equal').mkdir()
            (current / 'equal/file').write_bytes(b'same')
            shutil.copytree(current / 'equal', archived / 'equal', copy_function=shutil.copy2)
            (current / 'different').write_bytes(b'new')
            (archived / 'different').write_bytes(b'old')
            (archived / 'only_old').write_bytes(b'keep')
            same, plans, _, _ = compare.compare(current, archived, 'old')
            self.assertFalse(same)
            self.assertEqual([p['relative'] for p in plans], ['old/equal'])
            (here / 'comparison.json').write_text(json.dumps({'status': 'compared_not_deleted',
                'plans': plans, 'differences_preserved': ['different', 'only_old']}))
            (here / 'retailers_relocation.json').write_text(json.dumps({'status': 'relocated_verified'}))
            (here / 'paths.json').write_text(json.dumps({'physical_roots': {'old': 'organized/old'}}))
            with patch.multiple(deduplicate, HERE=here, HOME=home, ARCHIVE=home / 'archive'):
                deduplicate.main()
            self.assertTrue((archived / 'equal').is_symlink())
            self.assertFalse(os.path.isabs(os.readlink(archived / 'equal')))
            self.assertEqual((archived / 'equal/file').read_bytes(), b'same')
            self.assertEqual((archived / 'different').read_bytes(), b'old')
            self.assertEqual((current / 'different').read_bytes(), b'new')
            self.assertEqual((archived / 'only_old').read_bytes(), b'keep')
            self.assertFalse((here / 'duplicate_holding').exists())

    def test_inventory_detects_changed_bytes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'file'
            p.write_bytes(b'before')
            before = compare.metadata(p)
            p.write_bytes(b'after')
            self.assertNotEqual(before, compare.metadata(p))

    def test_fresh_proof_rejects_unverified_addition_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a', root / 'b'
            a.mkdir()
            (a / 'file').write_bytes(b'same')
            shutil.copytree(a, b, copy_function=shutil.copy2)
            (b / 'extra').write_bytes(b'keep')
            with self.assertRaisesRegex(RuntimeError, 'membership'):
                compare.verify_equal_now(a, b, 1, 4)
            (b / 'extra').unlink()
            (a / 'link').symlink_to('../different')
            (b / 'link').symlink_to('../different')
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                compare.verify_equal_now(a, b, 1, 4)


if __name__ == '__main__':
    unittest.main()