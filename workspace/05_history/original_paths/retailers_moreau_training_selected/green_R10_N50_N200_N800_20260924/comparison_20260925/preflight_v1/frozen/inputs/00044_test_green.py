"""All frozen green tests, with dimension/approved-size fixtures adapted explicitly."""
from source_adapter import GREEN, SOURCE_NAMES, load_into
load_into(globals(), GREEN / 'test_green.py', [
    ("for name in ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py'):",
     'for name in SOURCE_NAMES:'),
    ('self.assertEqual(common.SIZES, (200, 800))', 'self.assertEqual(common.SIZES, (50, 200, 800))'),
    ('(3, 0, 0, 100009)', '(10, 0, 0, 100009)'),
    ('for n in (160, 640):', 'for n in (40, 160, 640):'),
    ('model = Policy()\n        for n', 'model = Policy(n_val=-1)\n        for n'),
    ('(1 + i % 3, 6)', '(1 + i % 3, 20)'),
    ('np.ones(6,', 'np.ones(20,'),
    ('test_saa_rejects_fixed40_without_invoking_solver', 'test_saa_rejects_unapproved_size_without_invoking_solver'),
    ('np.ones((40, 3))', 'np.ones((41, 10))'),
    ('torch.zeros(3,', 'torch.zeros(10,'),
    ('(self.n_train, 6)', '(self.n_train, 20)'),
    ('[:, :3]', '[:, :10]'),
    ('[:, 3:]', '[:, 10:]'),
    ('[sample_id, :3]', '[sample_id, :10]'),
    ('[sample_id, 3:]', '[sample_id, 10:]'),
])
if __name__ == '__main__':
    unittest.main()