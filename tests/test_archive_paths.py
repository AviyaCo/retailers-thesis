"""Synthetic filesystem tests; no solver or historical/test data access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from archive_paths import local_json, resolve_archive_path


class ArchivePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.old = self.home / 'old_machine'
        self.root = self.home / 'restored' / 'experiment'
        (self.root / 'archive' / 'campaign').mkdir(parents=True)
        (self.old / 'campaign').mkdir(parents=True)
        (self.root / 'archive_paths.json').write_text(json.dumps({
            'source_home': str(self.old), 'archive_root': 'archive'}))
        self.copy = self.root / 'archive/campaign/result.json'
        self.copy.write_text('{}')
        self.original = self.old / 'campaign/result.json'
        self.original.write_text('{"different": true}')

    def test_original_path_uses_copy_even_if_original_exists(self):
        self.assertEqual(resolve_archive_path(self.original, root=self.root), self.copy)

    def test_missing_copy_never_falls_back_to_original(self):
        self.copy.unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_archive_path(self.original, root=self.root)

    def test_relative_and_local_absolute_paths(self):
        self.assertEqual(resolve_archive_path('archive/campaign/result.json', root=self.root), self.copy)
        self.assertEqual(resolve_archive_path(self.copy, root=self.root), self.copy)

    def test_external_symlink_and_traversal_rejected(self):
        (self.root / 'escape').symlink_to(self.old, target_is_directory=True)
        for value in ('escape/campaign/result.json', '../../old_machine/campaign/result.json'):
            with self.assertRaises(ValueError):
                resolve_archive_path(value, root=self.root)

    def test_local_json_preserves_original_bytes_and_reports_missing(self):
        missing = str(self.old / 'ap_env/bin/python')
        self.copy.write_text(json.dumps({'checkpoint': str(self.original), 'python': missing,
                                        str(self.original): 'unchanged hash'}))
        before = self.copy.read_bytes()
        view = local_json(self.original, root=self.root)
        self.assertEqual(self.copy.read_bytes(), before)
        self.assertEqual(view['original_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(view['data']['checkpoint'], str(self.copy))
        self.assertEqual(view['data'][str(self.copy)], 'unchanged hash')
        self.assertEqual(view['unavailable_recorded_paths'], [missing])

    def test_restore_to_different_directory(self):
        restored = self.home / 'another_location'
        self.root.rename(restored)
        self.assertEqual(resolve_archive_path(self.original, root=restored),
                         restored / 'archive/campaign/result.json')


if __name__ == '__main__':
    unittest.main()