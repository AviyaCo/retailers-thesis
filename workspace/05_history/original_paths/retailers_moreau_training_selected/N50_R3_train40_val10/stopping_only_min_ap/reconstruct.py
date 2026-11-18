"""Training-record-only stopping/selection; never reads validation or test costs."""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct(record, color):
    require(color in ('green', 'black'), 'Unknown stopping rule')
    require(record['status'] == 'completed', 'Incomplete source trajectory')
    prefix = []
    for a in record['iterations']:
        require(a['candidate_id'] == len(prefix) and a['assessed'], 'Incomplete/unordered AP')
        require(len(a['samples']) == 40 and
                [s['sample_id'] for s in a['samples']] == list(range(40)) and
                all(s['controller_status'] == 'completed' for s in a['samples']),
                'Require all forty accepted APs')
        ap, pool = a['ap_incumbent_mean'], a['restricted_mean']
        require(math.isfinite(ap) and math.isfinite(pool), 'Nonfinite stopping costs')
        require(math.isclose(ap, sum(s['incumbent']['cost'] for s in a['samples']) / 40,
                             rel_tol=1e-12, abs_tol=1e-12), 'AP mean mismatch')
        prefix.append(a)
        gap = abs(ap - pool) / max(abs(pool), 1e-9)
        if color == 'green' and gap < .01:
            break
    require(prefix, 'Empty trajectory')
    if color == 'green':
        # A black stop does not logically imply a mean-gap stop. Never substitute it.
        require(gap < .01, 'Green stop missing: continuation requires separate authorization')
        terminal = 'nominal_reference' if record['radius'] == 0 else 'mean_gap_satisfied'
    else:
        require(len(prefix) == record['mp_count'] + 1, 'Black MP count mismatch')
        terminal = record['terminal_status']
        require(terminal in ('separation_satisfied', 'iteration_cap', 'stagnation'),
                'Invalid black termination')
    selected = min(prefix, key=lambda a: a['ap_incumbent_mean'])  # Earliest exact tie.
    phases = [dict(phase='input_verification_and_loading',
                   **record['input_verification_and_loading'])]
    trace = []
    for i, a in enumerate(prefix):
        trace.append({k: a[k] for k in ('candidate_id', 'ap_incumbent_mean',
            'restricted_mean', 'ap_bound_mean', 'state_sha256', 'checkpoint_sha256')})
        for name in ('evaluation', 'ap'):
            require(a[name].get('candidate_id', i) == i, 'Timing candidate identity mismatch')
            phases.append({**a[name], 'candidate_id': i, 'phase': name})
        if i < len(prefix) - 1:
            for name in ('incoming_pool_evaluation', 'mp_call', 'returned_pool_evaluation'):
                require(a[name].get('candidate_id', i) == i, 'Timing candidate identity mismatch')
                phases.append({**a[name], 'candidate_id': i, 'phase': name})
    require(all(p['status'] == 'completed' and math.isfinite(p['wall_time_s'])
                and p['wall_time_s'] >= 0 for p in phases), 'Invalid prefix timing')
    return dict(color=color, seed=record['seed'], radius=record['radius'],
        n_train=40, n_val=10, network_seed=0, split_seed=100000 + record['seed'],
        method='new', training_backend='Moreau', status='reconstructed',
        terminal_status=terminal, source_terminal_status=record['terminal_status'],
        stop_candidate=prefix[-1]['candidate_id'], mp_count=len(prefix)-1,
        ap_sweeps=len(prefix), stop_mean_gap=gap,
        selected_iteration=selected['candidate_id'], checkpoint=selected['checkpoint'],
        checkpoint_sha256=selected['checkpoint_sha256'], state_sha256=selected['state_sha256'],
        selected_separation_status=selected['separation_status'],
        selected_ap_mean=selected['ap_incumbent_mean'],
        selected_ap_bound_mean=selected['ap_bound_mean'],
        selection='earliest minimum mean AP incumbent within own stopping prefix',
        source_summary_path=record['summary_path'], phases=phases, ap_trace=trace,
        reconstructed_phase_s=sum(p['wall_time_s'] for p in phases),
        ap_s=sum(p['wall_time_s'] for p in phases if p['phase'] == 'ap'),
        mp_s=sum(p['wall_time_s'] for p in phases if p['phase'] == 'mp_call'),
        pool_evaluation_s=sum(p['wall_time_s'] for p in phases if 'evaluation' in p['phase']),
        source_start_utc=record['started_at_utc'],
        source_stop_ap_end_utc=prefix[-1]['ap']['ended_at_utc'],
        source_ccg_wall_s=record['wall_time_s'], fresh_protocol_wall_s=None,
        source_ccg_workers=16, threads_per_worker=1,
        timing_scope='saved nonoverlapping phases through STOP, not selected iteration; '
                     'excludes nominal, selection/publication and uninstrumented overhead; '
                     'not a fresh end-to-end or parallel-sweep runtime',
        cached_nominal=record['cached_nominal'])


def identity(row):
    return row['seed'], row['state_sha256'], row['checkpoint_sha256']


def choose_radii(rows):
    """Fixed validation means only, no OOS input to selection."""
    require(len(rows) == 180 and len({(r['color'], r['seed'], r['radius']) for r in rows}) == 180,
            'Need complete two-method grid')
    return {color: [min((r for r in rows if r['color'] == color and r['seed'] == seed),
                       key=lambda r: (r['validation_mean'], r['radius']))
                    for seed in range(10)] for color in ('green', 'black')}