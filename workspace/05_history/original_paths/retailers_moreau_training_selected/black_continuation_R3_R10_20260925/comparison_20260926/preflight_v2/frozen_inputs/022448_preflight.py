"""DISPOSABLE ONLY: frozen preflight plus R10 dimensions, all sizes and AP .4/1.5.

CLI: preflight.py OUTPUTDIR, or --size {50,200,800} --outputdir OUTPUTDIR.
Two epochs are integration checks, never production training or research data.
Each directory must be new. Failures propagate and all evidence is retained.
"""
from source_adapter import GREEN, SOURCE_NAMES as ALL_SOURCES, FROZEN, load_into

load_into(globals(), GREEN / 'preflight.py', [
    ("SOURCE_NAMES = ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py',\n                'run_campaign.py', 'test_campaign.py', 'preflight.py', 'launch.sh', 'README.md')",
     'SOURCE_NAMES = ALL_SOURCES'),
    ("'disposable_R3_f0_green_preflight'", "'disposable_R10_f0_green_preflight'"),
    ("'-v', 'test_green', 'test_campaign'", "'-v', 'test_green', 'test_campaign', 'test_r10'"),
    ('for size in (200, 800):', 'for size in (50, 200, 800):'),
    ("core.baseline.cfg.R == 3, 'Expected canonical R3 model'",
     "core.baseline.cfg.R == 10, 'Expected isolated R10 model'"),
    ("compare_path = Path(core.__file__).resolve().parent / 'matched_oos_comparison/compare.py'",
     "compare_path = ROOT / 'r10_strict.py'"),
    ('(size, 3)', '(size, 10)'),
    ('(count, 3)', '(count, 10)'),
    ('centers[:, :3]', 'centers[:, :10]'),
    ('centers[:, 3:]', 'centers[:, 10:]'),
    ('centers[0, :3]', 'centers[0, :10]'),
    ('centers[0, 3:]', 'centers[0, 10:]'),
    ('centers[:1, :3]', 'centers[:1, :10]'),
    ('centers[:1, 3:]', 'centers[:1, 10:]'),
    ('for radius in (0., .2):', 'for radius in (0., .4, 1.5):'),
    ('point = ccg.validate_ap(result, ap, 0, centers[0], pools[0], radius, float(values[0]))',
     'point = ccg.validate_ap(result, ap, 0, centers[0], pools[0], radius, float(values[0]))\n            verify_folded_targets(model, common, ap, row)'),
    ("    with evidence.phase('strict_projection_training_row0', sample_id=0) as row:",
     "    zero_capacity_check(evidence, common)\n    with evidence.phase('strict_projection_training_row0', sample_id=0) as row:"),
    ("spec = importlib.util.spec_from_file_location('_green_preflight_strict_compare', compare_path)\n        require(spec is not None and spec.loader is not None, 'Cannot load strict projection definitions')\n        compare = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(compare)",
     'import r10_strict as compare'),
    ('(41, 6)', '(41, 20)'),
    ('paired6 radius-1.5', 'paired20 radius-1.5'),
    ('in all six coordinates', 'in all twenty coordinates'),
    ('choices=(200, 800)', 'choices=(50, 200, 800)'),
    ('--size {200,800}', '--size {50,200,800}'),
    ("record['sources_sha256'] = evidence.expected",
     "record['sources_sha256'] = evidence.expected\n        record['frozen_sources_sha256'] = FROZEN"),
])

_original_sources = sources
def sources():
    for path, sha in FROZEN.items():
        require(digest(path) == sha, f'Historical source changed: {path}')
    return _original_sources()

_original_parent = parent
def parent(evidence):
    from run_campaign import choose_workers, resources
    with evidence.phase('resource_guard') as row:
        snapshot = resources()
        row.update(snapshot=snapshot, permitted=choose_workers(snapshot,
            {'sizes': [{'peak_rss_bytes': 0}]}), actual_size_workers=1)
    subprocess_phase(evidence, 'bash_syntax', ['bash', '-n', str(ROOT / 'launch.sh')])
    return _original_parent(evidence)

def verify_folded_targets(model, common, ap, row):
    """Real AP incumbent vs exact folded coefficients and native target, train only."""
    np, torch = common.np, common.torch
    w = common.adversary._load_weights(model)
    x = np.asarray(ap['incumbent']['worst1'], dtype=np.float64)
    folded = w['W3'] @ np.maximum(0., w['W2'] @ np.maximum(0., w['W1'] @ x + w['b1']) + w['b2']) + w['b3']
    with torch.no_grad():
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        native = model.fc_layers(torch.cat((torch.tensor(x[None], dtype=torch.float32),
            q1r[None], q10.reshape(1, 1)), dim=1)).numpy()[0]
    target = np.asarray(ap['incumbent']['target'], dtype=np.float64)
    row['folded_targets'] = {}
    for name, value in (('folded', folded), ('native', native)):
        gap = float(np.max(np.abs(value-target)) / max(1., np.max(np.abs(value)), np.max(np.abs(target))))
        row['folded_targets'][name] = dict(target=value.tolist(), floor_scaled_discrepancy=gap)
        require(np.isfinite(gap) and gap <= common.adversary.REPLAY_TOLERANCE,
                f'AP/{name} target discrepancy exceeds unchanged replay tolerance')

def zero_capacity_check(evidence, common):
    torch = common.torch
    with evidence.phase('zero_capacity_projection') as row:
        projection = common.core.NativeProjection()
        z = torch.arange(-5., 5., dtype=torch.float64).reshape(1, 10).requires_grad_()
        capacity = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
        result, = projection(z, capacity, common.core._SOURCE_SOLVER_ARGS)
        result.sum().backward()
        row.update(output=result.detach().tolist(), gradient=z.grad.tolist(),
                   capacity_gradient=capacity.grad.tolist(), projection=projection.drain_summary())
        require(bool(torch.isfinite(result).all() and torch.isfinite(z.grad).all()
                     and torch.isfinite(capacity.grad).all()), 'Nonfinite zero-capacity projection/gradient')
        require(float(result.detach().abs().max()) <= 1e-6, 'Zero capacity physical residual')

if __name__ == '__main__':
    sys.exit(main())