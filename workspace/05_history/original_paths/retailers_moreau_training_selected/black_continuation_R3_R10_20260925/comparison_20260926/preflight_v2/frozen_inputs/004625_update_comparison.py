"""Evaluation-only extension of the existing matched retailer figure.

Freeze terminal and training/AP-selected prefix checkpoints before reading OOS
arrays. Reuse the unchanged strict evaluator; never fit, reselect on OOS, retry,
or substitute an unavailable historical-rule checkpoint.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PREVIOUS = Path('/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison')
TRAIN = HERE / 'results/20260916T080636.766867Z'
OUTPUT = HERE / 'matched_oos_update'
FROZEN = OUTPUT / 'frozen'
sys.path.insert(0, str(PREVIOUS))
import compare as c
import numpy as np


def recover_prefix(record):
    """Chronological mean stop; earliest minimum AP incumbent in that prefix."""
    prefix = []
    reason = None
    for a in record['iterations']:
        if a['candidate_id'] != len(prefix) or not a['assessed']:
            raise ValueError('Incomplete or unordered AP trajectory')
        if len(a['samples']) != 40 or any(r['controller_status'] != 'completed' for r in a['samples']):
            raise ValueError('Incomplete AP coverage')
        mean, pool = a['ap_incumbent_mean'], a['restricted_mean']
        if not np.isfinite([mean, pool]).all():
            raise ValueError('Nonfinite stopping diagnostic')
        gap = abs(mean - pool) / max(abs(pool), 1e-9)
        prefix.append(a)
        if record['radius'] == 0:
            reason = 'nominal_reference_not_historical_stop'
            break
        if a['candidate_id'] >= 1 and gap < .01:
            reason = 'first_eligible_mean_gap_pass'
            break
    if reason is None:
        return dict(recoverable=False, reason='no_eligible_mean_stop_in_saved_trajectory',
                    saved_candidates=len(prefix), last_candidate=prefix[-1]['candidate_id'])
    selected = min(prefix, key=lambda a: a['ap_incumbent_mean'])
    phases = [dict(phase='input_verification_and_loading', **record['input_verification_and_loading'])]
    for a in prefix:
        # AP/evaluation at the stopping candidate are included; its NEXT MP is not.
        for name in ('evaluation', 'ap'):
            phases.append(dict(candidate=a['candidate_id'], phase=name, **a[name]))
        if a is not prefix[-1]:
            for name in ('incoming_pool_evaluation', 'mp_call', 'returned_pool_evaluation'):
                phases.append(dict(candidate=a['candidate_id'], phase=name, **a[name]))
    if any(p['status'] != 'completed' or not np.isfinite(p['wall_time_s']) or p['wall_time_s'] < 0 for p in phases):
        raise ValueError('Missing/failed prefix timings')
    return dict(recoverable=True, reason=reason, stop_candidate=prefix[-1]['candidate_id'],
        selected_candidate=selected['candidate_id'], checkpoint=selected['checkpoint'],
        checkpoint_sha256=selected['checkpoint_sha256'], state_sha256=selected['state_sha256'],
        selected_separation_status=selected['separation_status'], stop_mean_gap=gap,
        selected_ap_mean=selected['ap_incumbent_mean'], mp_count=len(prefix)-1,
        ap_sweeps=len(prefix), source_start_utc=record['started_at_utc'],
        source_prefix_end_utc=prefix[-1]['ap']['ended_at_utc'],
        measured_prefix_phase_sum_s=sum(p['wall_time_s'] for p in phases),
        prefix_ap_s=sum(a['ap']['wall_time_s'] for a in prefix),
        prefix_mp_s=sum(a['mp_call']['wall_time_s'] for a in prefix[:-1]),
        phases=phases, fresh_short_run_wall_s=None,
        timing_scope='sum of persisted nonoverlapping perf_counter phases; excludes uninstrumented controller/persistence and new selection overhead; NOT fresh end-to-end runtime',
        prefix_candidate_evidence=[dict(candidate=a['candidate_id'], ap_mean=a['ap_incumbent_mean'],
            pool_mean=a['restricted_mean'], checkpoint_sha256=a['checkpoint_sha256']) for a in prefix])


def freeze():
    start = time.perf_counter()
    OUTPUT.mkdir(exist_ok=False)
    FROZEN.mkdir()
    m = dict(start_utc=c.utc(), status='freezing', jobs=[], recovery=[], files={},
             workers=c.WORKERS, threads=1, n_test=c.N_TEST, test_seed=c.TEST_SEED,
             radii=list(c.RADII), seeds=list(c.SEEDS), new_training=False,
             selection='first eligible <1% aligned AP-incumbent mean gap, then earliest min-AP within prefix',
             timing_clock='time.perf_counter', timing_unit='seconds',
             source_ccg_workers=16, source_threads=1,
             versions=dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                           torch=c.torch.__version__, gurobi=list(c.gp.gurobi.version())))

    def capture(path, expected=None):
        path = Path(path).resolve()
        digest = c.sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        if str(path) not in m['files']:
            dest = FROZEN / f'{len(m["files"]):04d}_{path.name}'
            shutil.copyfile(path, dest)
            if c.sha(dest) != digest or c.sha(path) != digest:
                raise ValueError('Input changed during freeze')
            dest.chmod(0o444)
            m['files'][str(path)] = dict(copy=dest.name, sha256=digest)
        return m['files'][str(path)]

    try:
        historical = c.read_json(PREVIOUS / 'frozen/manifest.json')
        capture(PREVIOUS / 'compare.py', historical['files'][str(PREVIOUS / 'compare.py')]['sha256'])
        for name in ('update_comparison.py', 'test_update_comparison.py', 'MATCHED_UPDATE_PROTOCOL.md', 'launch_comparison.sh'):
            capture(HERE / name)
        for name in ('summary.json', 'provenance.json', 'pre_validation_terminal_policies.json'):
            capture(TRAIN / name)
        provenance = c.read_json(TRAIN / 'provenance.json')
        for path, expected in provenance['source_sha256'].items():
            capture(path, expected)
        train = c.read_json(TRAIN / 'summary.json')
        if train['status'] != 'completed' or not train['final_selection_published'] or not train['inputs_unchanged']:
            raise ValueError('Training campaign not complete/verified')
        fixed = {(p['seed'], p['radius']): p for p in c.read_json(TRAIN / 'pre_validation_terminal_policies.json')['policies']}
        m['saved_radius_selections'] = train['selections']
        expected = {(s, r) for s in c.SEEDS for r in c.RADII}
        if set(fixed) != expected or len(train['ccg']['jobs']) != 90:
            raise ValueError('Missing fixed-policy grid')
        for j in sorted(train['ccg']['jobs'], key=lambda j: (j['radius'], j['seed'])):
            capture(j['summary_path'], j['summary_sha256'])
            record = c.read_json(j['summary_path'])
            policy = fixed[(j['seed'], j['radius'])]
            for k in ('selected_checkpoint_sha256', 'selected_state_sha256', 'selected_candidate_id', 'terminal_status'):
                if j[k] != policy[k]:
                    raise ValueError('Pre-validation identity changed')
            recovered = recover_prefix(record)
            m['recovery'].append(dict(seed=j['seed'], radius=j['radius'],
                full_mp_count=j['mp_count'], full_ccg_wall_s=j['wall_time_s'],
                full_ap_s=j['ap_total_time_s'], full_mp_s=j['mp_total_time_s'],
                full_terminal_status=j['terminal_status'], **recovered))
            choices = [('terminal', dict(checkpoint=j['selected_checkpoint'],
                checkpoint_sha256=j['selected_checkpoint_sha256'], state_sha256=j['selected_state_sha256'],
                selected_candidate=j['selected_candidate_id'], selected_separation_status=j['selected_separation_status']))]
            if recovered['recoverable']:
                choices.append(('mean_gap', recovered))
            for method, choice in choices:
                info = capture(choice['checkpoint'], choice['checkpoint_sha256'])
                model = c.load_model(FROZEN / info['copy'], 'new')
                if c.state_hash(model) != choice['state_sha256']:
                    raise ValueError('Learned checkpoint identity mismatch')
                m['jobs'].append(dict(id=f'{method}_r{j["radius"]:g}_s{j["seed"]}',
                    method='new', comparison_method=method, radius=j['radius'], seed=j['seed'],
                    n_train=40, n_val=10, network_seed=0, split_seed=100000+j['seed'],
                    selected_iteration=choice['selected_candidate'], checkpoint=choice['checkpoint'],
                    checkpoint_copy=info['copy'], checkpoint_sha256=info['sha256'],
                    state_sha256=choice['state_sha256'], training_backend='Moreau',
                    selected_separation_status=choice['selected_separation_status'],
                    terminal_status=j['terminal_status'],
                    recovery_status=recovered['reason'] if method == 'mean_gap' else 'actual_terminal'))
        # All checkpoint decisions above precede loading cached OOS results/arrays.
        capture(PREVIOUS / 'results/summary.json')
        capture(PREVIOUS / 'results/matched_oos.png')
        capture(PREVIOUS / 'frozen/manifest.json', (PREVIOUS / 'frozen/manifest.sha256').read_text().strip())
        previous = c.read_json(PREVIOUS / 'results/summary.json')
        c.complete_grid(previous, historical)
        m['test_arrays'] = previous['test_arrays']
        for key, info in m['test_arrays'].items():
            capture(PREVIOUS / 'results' / info['file'], info['file_sha256'])
        for source in (c.SRO, c.CACHE, c.COND):
            info = historical['files'][str(source)]
            capture(PREVIOUS / 'frozen' / info['copy'], info['sha256'])
        m.update(status='frozen', end_utc=c.utc(), wall_time_s=time.perf_counter()-start)
        c.write_json(FROZEN / 'manifest.json', m)
        (FROZEN / 'manifest.json').chmod(0o444)
        (FROZEN / 'manifest.sha256').write_text(c.sha(FROZEN / 'manifest.json')+'\n')
        print(f'Frozen {len(m["jobs"])} evaluations; {sum(not r["recoverable"] for r in m["recovery"])} unrecoverable prefixes', flush=True)
    except BaseException:
        c.write_json(OUTPUT / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(), wall_time_s=time.perf_counter()-start))
        raise


def verify():
    if c.sha(FROZEN / 'manifest.json') != (FROZEN / 'manifest.sha256').read_text().strip():
        raise ValueError('Manifest changed')
    m = c.read_json(FROZEN / 'manifest.json')
    versions=dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                  torch=c.torch.__version__, gurobi=list(c.gp.gurobi.version()))
    if m['versions'] != versions:
        raise ValueError('Runtime versions changed since freeze')
    for source, info in m['files'].items():
        if c.sha(source) != info['sha256'] or c.sha(FROZEN / info['copy']) != info['sha256']:
            raise ValueError(f'Frozen input changed: {source}')
    return m


def initialize():
    c.FROZEN = FROZEN
    c.initialize_worker()


def plot(summary, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def saved(path):
        return c.read_json(FROZEN / manifest['files'][str(path)]['copy'])

    previous = saved(PREVIOUS / 'results/summary.json')
    historical = saved(PREVIOUS / 'frozen/manifest.json')
    c.complete_grid(previous, historical)
    c.complete_grid(summary, manifest)
    fig, ax = plt.subplots(figsize=(11, 8))
    curves = []

    def curve(radii, values, label, color, marker='o', style='-'):
        values = np.asarray(values, dtype=float)
        means = values.mean(axis=1)
        lo, hi = np.percentile(values, [20, 80], axis=1)
        ax.plot(radii, means, marker+style, color=color, label=label, ms=4)
        ax.fill_between(radii, lo, hi, color=color, alpha=.13)
        curves.append(dict(label=label, radius=list(radii), per_seed=values.tolist(),
                           mean=means.tolist(), p20=lo.tolist(), p80=hi.tolist()))

    for method, color, label in [('old', '#2166ac', 'Previous SCS (50 train)'),
                                ('new', '#762a83', 'Previous Moreau (40 train / 10 val)')]:
        radii = historical[f'{method}_radii']
        curve(radii, [[next(j['mean_oos'] for j in previous['jobs'] if j['method']==method and j['radius']==r and j['seed']==s) for s in c.SEEDS] for r in radii], label, color)
    for method, color, label, marker in [('terminal', '#111111', 'New Moreau: every-center stop / terminal', 'D'),
                                        ('mean_gap', '#009e73', 'New Moreau: mean stop / min-AP prefix*', '^')]:
        jobs = [j for j in summary['jobs'] if j['comparison_method']==method]
        radii = [r for r in c.RADII if {j['seed'] for j in jobs if j['radius']==r}==set(c.SEEDS)]
        positive = [r for r in radii if r>0] if method=='mean_gap' else radii
        curve(positive, [[next(j['mean_oos'] for j in jobs if j['radius']==r and j['seed']==s) for s in c.SEEDS] for r in positive], label, color, marker)
        if method=='mean_gap':
            curve([0.], [[next(j['mean_oos'] for j in jobs if j['radius']==0 and j['seed']==s) for s in c.SEEDS]], '_short nominal reference', color, marker)
    def reference(source):
        return saved(PREVIOUS / 'frozen' / historical['files'][str(source)]['copy'])
    sro = reference(c.SRO)
    radii = sorted(float(r) for r in sro if 0<float(r)<=1.5)
    curve(radii, [[sro[str(r)][str(s)]['oos'] for s in c.SEEDS] for r in radii], 'SRO-FA (historical cached reference)', '#d62728', 's', '--')
    benchmarks = reference(c.CACHE)['benchmarks']
    horizontal = [('SAA-kNN (historical 10k; LOO k)', float(np.mean(benchmarks['saa_knn']['oos'])), '#e6a817', '-.'),
                  ('SAA-LP LB (cached 2k, not matched 10k)', float(np.mean(benchmarks['saa_lb']['oos'])), '#1a9641', '--'),
                  ('Conditional LB (separate 1M; seed 42)', reference(c.COND)['saa_cond_lb_large'], '#1b7837', '-.')]
    for label, value, color, style in horizontal:
        ax.axhline(value, color=color, ls=style, label=label)
    ax.set_xscale('symlog', linthresh=.001, linscale=1)
    ax.set_xticks([0, .001, .003, .01, .03, .1, .2, .4, .7, 1.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set(xlabel=r'Training radius $\rho$ (linear near zero; logarithmic above .001)',
           ylabel='Mean out-of-sample cost', title='R=3, total N=50: updated retailer comparison\nSame 10,000 evaluation scenarios (seed 9999); means and 20/80 seed percentiles, not CIs')
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=8)
    fig.text(.5, .025, '*Recovered from saved Moreau trajectories, not a fresh historical-SCS rerun. No test-based selection.\n'
        'Mean-rule curve requires all 10 seeds: unavailable at .001/.003/.01/.03; isolated zero is nominal.\n'
        'Terminal curve includes capped seed 9 at .7. Historical protocols differ; old reference lines unchanged.\n'
        'Previously inspected evaluation draw: descriptive, not confirmatory. Cached bounds use different samples.', ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.38)
    c.write_json(OUTPUT / 'plot_data.json', dict(curves=curves, horizontal=horizontal, test_arrays=manifest['test_arrays']))
    fig.savefig(OUTPUT / 'matched_oos_updated.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def report(summary, m):
    missing_ids = sorted({j['id'] for j in m['jobs']} - {j['id'] for j in summary['jobs']})
    summary['missing_evaluation_ids'] = missing_ids
    lines = ['# Updated matched retailer comparison', '',
        'New evaluation only; previous curves cached unchanged. No fitting or OOS selection.',
        'Mean-rule recovery uses aligned training/AP means on the existing Moreau path, not the entire historical SCS algorithm.',
        '31 positive-radius prefixes are unavailable: no eligible post-initial AP. The .03 cohort has only 9/10 and is not drawn.',
        'Prefix phase sums exclude uninstrumented overhead and are NOT newly measured short-run totals or sweep elapsed.', '',
        '| Radius | Recovered n | Mean MPs | Mean AP s | Mean MP s | Prefix phase sum mean / median / min / max s |',
        '|---:|---:|---:|---:|---:|---|']
    for radius in c.RADII:
        rows = [r for r in m['recovery'] if r['radius']==radius and r['recoverable']]
        if rows:
            stats = c.timing_stats([r['measured_prefix_phase_sum_s'] for r in rows])
            lines.append(f'| {radius:g} | {len(rows)} | {np.mean([r["mp_count"] for r in rows]):.2f} | {np.mean([r["prefix_ap_s"] for r in rows]):.3f} | {np.mean([r["prefix_mp_s"] for r in rows]):.3f} | '+ ' / '.join(f'{stats[k]:.3f}' for k in ('mean','median','min','max'))+' |')
    lines += ['', '| Evaluation method | Radius | n | Mean / median / min / max wall s |', '|---|---:|---:|---|']
    summary['evaluation_runtime_aggregates'] = []
    for method in ('terminal', 'mean_gap'):
        for radius in c.RADII:
            rows=[r for r in summary['jobs'] if r['comparison_method']==method and r['radius']==radius and r['status']=='completed']
            if rows:
                stats=c.timing_stats([r['wall_time_s'] for r in rows])
                summary['evaluation_runtime_aggregates'].append(dict(method=method, radius=radius, **stats))
                lines.append(f'| {method} | {radius:g} | {len(rows)} | '+' / '.join(f'{stats[k]:.6f}' for k in ('mean','median','min','max'))+' |')
    training=c.read_json(FROZEN / m['files'][str(TRAIN/'summary.json')]['copy'])
    lines += ['', f'Current evaluation/plot invocation: {c.human(summary["wall_time_s"])}; {summary["status"]}.',
        f'Returned failed/incomplete evaluations: {sum(r["status"]!="completed" for r in summary["jobs"])}; missing/unreturned: {len(missing_ids)}. All excluded from completed aggregates.',
        '| Category | Timing / scope |', '|---|---|',
        f'| Prior nominal LR-search phase | {c.human(training["nominal"]["wall_time_s"])}; not rerun |',
        f'| Prior robust CCG sweep | {c.human(training["ccg"]["wall_time_s"])}; per-iteration/model details in ../RESULTS.md and source summaries |',
        '| Recovered shorter CCG | Persisted phase sums above and manifest; fresh end-to-end and sweep times unavailable, rerun required |',
        '| SRO-FA | Cached result; not rerun, construction/solver timing not imported |',
        '| CV / new hyperparameter selection | Not run |',
        f'| OOS evaluation | {sum(r["status"]=="completed" for r in summary["jobs"])}/{len(m["jobs"])} completed; measured separately, no cache substitution for new curves |',
        f'Environment: {summary["environment"]}; {c.WORKERS} workers, one numerical/solver thread each.',
        'No summed concurrent model duration is represented as sweep elapsed. Full run has 89 separated / 1 capped; reconstruction does not certify every-center separation.',
        '[Frozen selections/timings](frozen/manifest.json)']
    if summary['status']=='completed':
        lines.append('[Updated plot](matched_oos_updated.png) · [Raw plotted values](plot_data.json)')
    else:
        lines.append('Campaign failed/incomplete; no completed comparison is claimed.')
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    start=time.perf_counter()
    summary=dict(status='running', start_utc=c.utc(), jobs=[], phases={}, environment=dict(host=platform.node(), platform=platform.platform(), cpu_count=os.cpu_count()),
                 command=sys.argv, cwd=str(Path.cwd()), workers=c.WORKERS, threads=1, timing_clock='time.perf_counter', timing_unit='seconds')
    m=verify()
    if (OUTPUT/'summary.json').exists():
        raise FileExistsError('No overwrite or silent resume')
    summary.update(manifest_sha256=c.sha(FROZEN/'manifest.json'), test_arrays=m['test_arrays'])
    try:
        with c.phase(summary['phases'], 'copy_fixed_test_arrays'):
            for key, info in m['test_arrays'].items():
                source=PREVIOUS/'results'/info['file']
                frozen=FROZEN/m['files'][str(source)]['copy']
                shutil.copyfile(frozen, OUTPUT/info['file'])
                (OUTPUT/info['file']).chmod(0o444)
        with c.phase(summary['phases'], 'parallel_evaluation'):
            with ProcessPoolExecutor(max_workers=c.WORKERS, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
                futures={pool.submit(c.evaluate_job,j,summary['manifest_sha256'],m['test_arrays'],OUTPUT):j for j in m['jobs']}
                for f in as_completed(futures):
                    try:
                        result=f.result()
                    except BaseException:
                        result=dict(**futures[f], status='worker_failed', wall_time_s=None, mean_oos=None, error=traceback.format_exc())
                    summary['jobs'].append(result)
                    c.write_json(OUTPUT/'summary.json',summary)
                    print(f'{len(summary["jobs"])}/{len(m["jobs"])} {result["id"]}: {result["status"]}',flush=True)
        verify()
        c.complete_grid(summary,m)
        with c.phase(summary['phases'],'plot'):
            plot(summary,m)
        summary['status']='completed'
    except BaseException:
        summary.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        summary.update(end_utc=c.utc(),wall_time_s=time.perf_counter()-start)
        try:
            report(summary,m)
        except BaseException:
            summary.update(status='failed', report_error=traceback.format_exc())
            raise
        finally:
            c.write_json(OUTPUT/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run'))
    args=parser.parse_args()
    if args.command=='freeze':
        c.initialize_worker()
        freeze()
    else:
        run()