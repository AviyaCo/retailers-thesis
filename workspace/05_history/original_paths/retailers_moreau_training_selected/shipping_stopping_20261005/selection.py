"""Return shipping stopping checkpoints without modifying sealed trajectories."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'shipping_comparison_20260928'
OLD = PARENT / 'production_v1'
MANIFEST_SHA = '57ca39132e8be101a4c4a58095de9593fe401f562c437c3e56919480b5545d77'
SUMMARY_SHA = 'd6d92fc3c852bcc06b7481cd2f59e5ed6e511f3e4ef91f7381078d96a5c9e313'
RETURN_RULE = 'stopping_checkpoint'
sys.path.append(str(PARENT))
import common as c
import freeze as f

c.check(c.ROOT == PARENT and Path(f.__file__).resolve().parent == PARENT,
        'Wrong shipping replay dependency imported')


def stopping_pool(record, training, read=c.read):
    """Validate the historical selector, then return FIRST passing assessment.

    The old minimum-AP fields remain evidence, never rewritten to pretend that
    training originally used the new return rule. No terminal fallback exists.
    """
    c.check(record['status'] == 'completed' and record['complete'] and record['converged']
            and record['terminal_status'] == 'green_gap_satisfied', 'Incomplete green trajectory')
    f.validate_prefix(record, training)
    stop = record['stopping_candidate_id']
    candidate = record['iterations'][stop]
    c.check(candidate['green_pass'] is True, 'Stopping assessment does not pass')
    f.validate_assessment(read(candidate['ap_reference']), candidate)
    path = Path(candidate['checkpoint']).parent / 'pool.json'
    c.check(path.parent == Path(candidate['ap_reference']).parent, 'Checkpoint/pool assessment mismatch')
    c.check(c.sha(path) == candidate['pool_file_sha256'], 'Stopping pool file changed')
    saved = read(path)
    c.check(saved['original_sample_ids'] == training['original_sample_ids'], 'Stopping pool IDs changed')
    pools = f.validate_pools(saved['pools'], training['centers'], record['radius'])
    c.check(c.pool_digest(pools) == saved['pool_sha256'] == candidate['pool_sha256'],
            'Stopping pool digest changed')
    return candidate, path, pools


def validate_grid(jobs):
    expected = {(n, s, r) for n in c.SIZES for s in c.SEEDS for r in c.RADII}
    green = [j for j in jobs if j['method'] == 'green']
    c.check(len(green) == 270 and len(jobs) == 750 and
            {(j['Ntotal'], j['seed'], j['radius']) for j in green} == expected and
            len({j['id'] for j in jobs}) == 750, 'Incomplete or duplicate fixed grid')
    c.check(all(j['selected_candidate_id'] == j['stopping_candidate_id'] and
                j['return_rule'] == RETURN_RULE and j['green_pass'] is True for j in green),
            'A returned policy is not its passing stopping assessment')
    return green


def collect(output):
    """Seal all choices before reading test arrays or parent plot/cost values.

    Already frozen parent files are referenced, not rewritten or recopied.
    Newly consumed stopping checkpoints/pools and all local sources are copied.
    """
    c.check(c.sha(OLD / 'manifest.json') == MANIFEST_SHA and
            (OLD / 'manifest.sha256').read_text().strip() == MANIFEST_SHA,
            'Parent manifest changed')
    c.check(c.sha(OLD / 'summary.json') == SUMMARY_SHA, 'Parent completion record changed')
    old = c.read(OLD / 'manifest.json')
    receipt = c.read(OLD / 'summary.json')
    c.check(receipt['status'] == 'completed' and receipt['manifest_sha256'] == MANIFEST_SHA,
            'Parent comparison did not complete')
    f.verify(old)
    manifest = deepcopy(old)
    manifest.update(experiment='shipping_stopping_20261005', return_rule=RETURN_RULE,
        new_selection=True, new_training=False, test_generated=False,
        evaluation_scope='270 fresh green; 480 legacy and benchmarks retained unchanged',
        parent_manifest_sha256=MANIFEST_SHA, parent_summary_sha256=SUMMARY_SHA)
    manifest.pop('timing', None)
    folder = Path(output) / 'frozen'
    folder.mkdir()
    snap = f.Snapshot(folder, manifest)
    for source in [*sorted(ROOT.glob('*.py')), ROOT / 'README.md']:
        snap.capture(source)
    snap.capture(OLD / 'manifest.json', MANIFEST_SHA)
    snap.capture(OLD / 'summary.json', SUMMARY_SHA)
    changes = []
    for index, previous in enumerate(old['jobs']):
        if previous['method'] != 'green':
            continue
        source = previous['source_summary_path']
        record = snap.read(source, old['files'][source]['sha256'])
        c.check(all(record[k] == previous[k] for k in
                    ('Ntotal', 'Ntrain', 'Nval', 'seed', 'radius', 'selected_candidate_id',
                     'stopping_candidate_id')), 'Parent job differs from trajectory')
        c.check(record['selected_state_sha256'] == previous['state_sha256'] and
                record['selected_checkpoint_sha256'] == previous['checkpoint_sha256'],
                'Parent selected policy differs')
        train_path = Path(source).parent / 'inputs/training_data.json'
        training = snap.read(train_path, record['training_data_sha256'])
        c.check(training['centers'] == previous['centers'] and
                training['original_sample_ids'] == previous['original_sample_ids'] and
                training['data_sha256'] == record['data_sha256'], 'Training identity changed')
        candidate, pool_path, pools = stopping_pool(record, training, snap.read)
        model = snap.model(candidate['checkpoint'], 'green', candidate['checkpoint_sha256'],
                           candidate['state_sha256'])
        inputs = set(previous['input_files'])
        inputs.update((snap.capture(pool_path, candidate['pool_file_sha256']),
                       snap.capture(candidate['ap_reference']), model['checkpoint']))
        job = deepcopy(previous)
        job.update(**model, pools=[p.tolist() for p in pools], sizes=[len(p) for p in pools],
            pool_path=str(pool_path), pool_sha256=c.pool_digest(pools),
            pool_file_sha256=candidate['pool_file_sha256'],
            selected_candidate_id=candidate['candidate_id'], return_rule=RETURN_RULE,
            ap_reference=candidate['ap_reference'], green_gap=candidate['green_gap'], green_pass=True,
            saved_native_finite_pool_mean=candidate['restricted_mean'],
            input_files=sorted(inputs), original_box_certificate=False,
            previous_selected_candidate_id=previous['selected_candidate_id'],
            previous_state_sha256=previous['state_sha256'],
            previous_pool_sha256=previous['pool_sha256'])
        manifest['jobs'][index] = job
        changes.append(dict(id=job['id'], Ntotal=job['Ntotal'], seed=job['seed'], radius=job['radius'],
            previous_candidate_id=previous['selected_candidate_id'], stopping_candidate_id=candidate['candidate_id'],
            candidate_changed=previous['selected_candidate_id'] != candidate['candidate_id'],
            state_changed=previous['state_sha256'] != job['state_sha256'],
            pool_changed=previous['pool_sha256'] != job['pool_sha256'], green_gap=candidate['green_gap']))
    validate_grid(manifest['jobs'])
    c.check([j for j in manifest['jobs'] if j['method'] == 'legacy'] ==
            [j for j in old['jobs'] if j['method'] == 'legacy'], 'Legacy selection changed')
    manifest['selection_changes'] = changes
    # Parent test metadata is copied unchanged; array bytes are not read here.
    manifest['parent_published_hashes'] = receipt['published']
    f.verify(manifest)
    path = Path(output) / 'selection_manifest.json'
    c.write(path, manifest)
    path.chmod(0o444)
    seal = path.with_suffix('.sha256')
    seal.write_text(c.sha(path) + '\n')
    seal.chmod(0o444)
    return manifest