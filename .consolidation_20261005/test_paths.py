"""Scratch-only migration tests; no observations, checkpoints or solvers."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_legacy import namespace_command, virtual_path


class PathCompatibility(unittest.TestCase):
    def test_virtual_identity_and_child_and_explicit_output(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / 'organized/history/old_experiment'
            source.mkdir(parents=True)
            (source / 'frozen.txt').write_text('unchanged fixture\n')
            (source / 'output').mkdir()
            (home / 'ordinary').mkdir()
            (home / 'ordinary_link').symlink_to('ordinary', target_is_directory=True)
            mapping = {'old_experiment': source}
            virtual = home / 'old_experiment'
            digest = hashlib.sha256((source / 'frozen.txt').read_bytes()).hexdigest()
            code = """
import hashlib, os, pathlib, subprocess, sys
p = pathlib.Path.cwd()
with open('/dev/urandom', 'rb') as random_device:
    assert len(random_device.read(1)) == 1
assert str(p) == sys.argv[1]
assert str((p/'frozen.txt').resolve()) == sys.argv[1]+'/frozen.txt'
assert hashlib.sha256((p/'frozen.txt').read_bytes()).hexdigest() == sys.argv[2]
assert os.environ['PWD'] == str(p)
assert pathlib.Path('/proc/self/cwd').resolve() == p
assert (p.parent/'ordinary_link').is_symlink()
for file in (p/'frozen.txt', pathlib.Path(sys.argv[3])/'frozen.txt'):
    try:
        file.write_text('must fail')
    except OSError:
        pass
    else:
        raise AssertionError('Frozen source writable')
assert subprocess.check_output([sys.executable, '-B', '-c', 'import os;print(os.getcwd())'], text=True).strip() == str(p)
f=p/'output'/'new.tmp'
f.write_text('new output')
f.replace(p/'output'/'new.txt')
"""
            args = namespace_command(home, mapping, source,
                                     [sys.executable, '-B', '-c', code, str(virtual), digest, str(source)],
                                     [source / 'output'])
            # The last physical argument is intentionally not path-translated so
            # the fixture also tests read-only access by the canonical name.
            args[-1] = str(source)
            subprocess.run(args, check=True)
            self.assertFalse(virtual.exists())
            self.assertEqual((source / 'frozen.txt').read_text(), 'unchanged fixture\n')
            self.assertEqual((source / 'output/new.txt').read_text(), 'new output')

    def test_no_old_root_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / 'organized/old'
            source.mkdir(parents=True)
            (home / 'old').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'Unexpected original'):
                namespace_command(home, {'old': source}, home / 'old', ['/bin/true'])

    def test_reject_unmapped_cwd_and_whole_root_write(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / 'organized/old'
            source.mkdir(parents=True)
            with self.assertRaises(ValueError):
                namespace_command(home, {'old': source}, home, ['/bin/true'])
            with self.assertRaises(ValueError):
                namespace_command(home, {'old': source}, home / 'old', ['/bin/true'], [home / 'old'])

    def test_physical_argument_translation(self):
        home = Path('/fixture')
        mapping = {'old': home / 'organized/history/old'}
        self.assertEqual(virtual_path('/fixture/organized/history/old/code.py', home, mapping), home / 'old/code.py')
        self.assertEqual(virtual_path('old/code.py', home, mapping), home / 'old/code.py')

    def test_reject_write_parent_and_symlink_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / 'organized/old'
            (source / 'out').mkdir(parents=True)
            (source / 'alias').symlink_to('out', target_is_directory=True)
            for bad in (home / 'old/out/..', home / 'old/alias'):
                with self.assertRaises(ValueError):
                    namespace_command(home, {'old': source}, home / 'old', ['/bin/true'], [bad])


if __name__ == '__main__':
    unittest.main()