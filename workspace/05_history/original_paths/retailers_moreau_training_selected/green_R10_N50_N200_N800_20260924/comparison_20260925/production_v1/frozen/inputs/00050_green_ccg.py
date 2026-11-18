"""Exact frozen live-green controller; only paired demand dimensions change."""
from source_adapter import GREEN, load_into
load_into(globals(), GREEN / 'green_ccg.py', [
    ('centers[sample_id, :3]', 'centers[sample_id, :10]'),
    ('centers[sample_id, 3:]', 'centers[sample_id, 10:]'),
    ('centers.shape == (n_train, 6)', 'centers.shape == (n_train, 20)'),
    ('paired6 training centers', 'paired20 training centers'),
])