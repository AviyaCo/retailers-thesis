"""Fixed-policy finite-pool evaluation and standalone paper figure; no training.

IS = mean over 40 training centers of max cost within the returned checkpoint's
own saved MP pools. This is the old SCS figure's finite-pool definition, applied
to green's actual pools, not AP means or pools collected after that checkpoint.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'results'
OUT = HERE / 'results'
EVALUATOR = Path('/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py')
sys.path.insert(0, str(EVALUATOR.parent))
import compare as c
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pool_digest(pools):
    """The saved controller's exact little-endian pool identity contract."""
    h = hashlib.sha256()
    for sample_id, pool in enumerate(pools):
        h.update(np.asarray([sample_id, len(pool)], dtype='<i8').tobytes())
        h.update(np.asarray(pool, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def finite_pool_cost(costs, sizes):
    costs = np.asarray(costs, dtype=np.float64)
    require(len(sizes) == 40 and all(type(n) is int and n > 0 for n in sizes),
            'Require forty nonempty pools')
    require(costs.ndim == 1 and len(costs) == sum(sizes) and np.isfinite(costs).all(),
            'Invalid cost vector')
    groups = np.split(costs, np.cumsum(sizes)[:-1])
    maxima = np.asarray([group.max() for group in groups])
    return float(maxima.mean()), maxima.tolist()


def stats(values):
    values = list(values)
    require(values and all(np.isfinite(values)), 'Invalid statistics')
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def render(data):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    style = {'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
             'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
             'xtick.labelsize': 8, 'ytick.labelsize': 9, 'legend.fontsize': 8,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7}
    with mpl.rc_context(style):
        fig = Figure(figsize=(7.0, 4.7))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        g, red = data['green'], data['sro_fa']
        ax.plot(g['radius'], g['oos_mean'], color='#00845b', marker='o', ms=3.3,
                lw=1.8, label='Neural CCG (OOS)', zorder=5)
        ax.fill_between(g['radius'], g['oos_p20'], g['oos_p80'], color='#00845b', alpha=.12, lw=0)
        ax.plot(g['radius'], g['is_mean'], color='#70ad92', marker='^', ms=3.4,
                lw=1.6, ls='--', label='Neural CCG (IS, finite pool)', zorder=4)
        ax.plot(red['radius'], red['mean'], color='#ce3339', marker='s', ms=2.6,
                lw=1.5, ls='--', label='SRO-FA (OOS)', zorder=3)
        ax.fill_between(red['radius'], red['p20'], red['p80'], color='#ce3339', alpha=.10, lw=0)
        for i, (label, value, _, _) in enumerate(data['horizontal']):
            color, ls, name = [('#d69a00', '-.', 'SAA-kNN (OOS)'),
                               ('#9c9c9c', ':', 'SAA-LP reference (2k)'),
                               ('#525252', (0, (5, 2)), 'Conditional clairvoyant reference')][i]
            ax.axhline(value, color=color, lw=1.25, ls=ls, label=name, zorder=2)
        ax.set_xscale('symlog', linthresh=.001, linscale=1.)
        ax.set_xlim(-.00012, 1.7)
        ax.set_xticks([0, .001, .003, .01, .03, .1, .2, .4, .7, 1.5])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.set_xlabel(r'Robustness radius $\rho$')
        ax.set_ylabel('Mean cost')
        ax.set_title(r'$R=3$, $f=0$, $N_{\mathrm{total}}=50$ (40 training / 10 validation)', pad=10)
        ax.grid(axis='y', color='#d9d9d9', lw=.55, alpha=.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        # Row-major layout: OOS/IS, SRO-FA/kNN, two references.
        order = [0, 2, 4, 1, 3, 5]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False,
                   handlelength=2.5, columnspacing=2.5, labelspacing=.65)
        fig.subplots_adjust(left=.11, right=.985, top=.91, bottom=.25)
        fig.savefig(OUT / 'green_paper.png', dpi=300, facecolor='white')
        fig.savefig(OUT / 'green_paper.pdf', facecolor='white',
                    metadata={'Title': 'Neural CCG: in-sample and out-of-sample cost',
                              'Subject': 'Initial-eligible mean stop; minimum-AP return; finite-pool IS'})


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    summary = dict(experiment='green_paper_finite_pool_evaluation_20260923', status='running',
        started_at_utc=c.utc(), command=sys.orig_argv, cwd=str(HERE), workers=1, solver_threads=1,
        environment=dict(host=platform.node(), platform=platform.platform(), cpu_count=os.cpu_count(),
                         python=sys.version), timing_clock='time.perf_counter', timing_unit='seconds',
        phases={}, jobs=[], input_sha256={}, new_training=False, new_ap=False, new_mp=False,
        new_validation=False, new_test_evaluation=False,
        is_definition='mean of per-center maxima on returned checkpoint own saved MP pool; strict Gurobi replay')

    def capture(path, expected=None):
        path = Path(path).resolve()
        h = c.sha(path)
        require(expected is None or h == expected, f'Input hash mismatch: {path}')
        require(str(path) not in summary['input_sha256'] or summary['input_sha256'][str(path)] == h,
                f'Input changed while reading: {path}')
        summary['input_sha256'][str(path)] = h
        return h

    def read(path, expected=None):
        h = capture(path, expected)
        value = c.read_json(path)
        require(c.sha(path) == h, 'File changed during read')
        return value

    def save():
        c.write_json(OUT / 'summary.json', summary)

    try:
        c.initialize_worker()
        with c.phase(summary['phases'], 'tests') as phase:
            proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_figure'],
                                  cwd=HERE, capture_output=True, text=True)
            phase.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            require(proc.returncode == 0, 'Fixture tests failed')
        with c.phase(summary['phases'], 'freeze_inputs'):
            for name in ('make_figure.py', 'test_figure.py', 'README.md', 'caption.tex', 'launch.sh'):
                capture(HERE / name)
            source = read(SOURCE / 'summary.json')
            require(source['status'] == 'completed' and source['inputs_unchanged'], 'Source incomplete')
            for path, h in source['input_sha256'].items():
                capture(path, h)
            require(c.sha(EVALUATOR) == source['input_sha256'][str(EVALUATOR)], 'Evaluator changed')
            versions = dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                            torch=c.torch.__version__, gurobi=list(c.gp.gurobi.version()))
            require(versions == source['versions'], 'Evaluation environment changed')
            summary['versions'] = versions
            plot = read(SOURCE / 'plot_data.json')
            choices = sorted((r for r in source['choices'] if r['color'] == 'green'),
                             key=lambda r: (r['radius'], r['seed']))
            require(len(choices) == 90 and {(r['seed'], r['radius']) for r in choices} ==
                    {(s, r) for s in range(10) for r in c.RADII}, 'Incomplete green grid')
            specs = []
            for row in choices:
                record = read(row['source_summary_path'], row['source_summary_sha256'])
                a = record['iterations'][row['selected_iteration']]
                for key in ('checkpoint', 'checkpoint_sha256', 'state_sha256'):
                    require(a[key] == row[key], f'Selected identity mismatch: {key}')
                require(a['candidate_id'] <= row['stop_candidate'] and a['assessed'], 'Unassessed/future candidate')
                pool_path = Path(row['checkpoint']).parent / 'pool.json'
                pool = read(pool_path, a['pool_file_sha256'])
                pools = [np.asarray(p, dtype=np.float64) for p in pool['pools']]
                require(len(pools) == 40 and all(p.ndim == 2 and p.shape[1] == 6 and len(p) > 0
                        and np.isfinite(p).all() for p in pools), 'Invalid saved pools')
                require(pool_digest(pools) == pool['pool_sha256'] == a['pool_sha256'], 'Pool identity mismatch')
                capture(row['checkpoint'], row['checkpoint_sha256'])
                spec = {k: row[k] for k in ('id', 'seed', 'radius', 'checkpoint', 'checkpoint_sha256',
                    'state_sha256', 'selected_iteration', 'stop_candidate', 'test_mean', 'test_origin',
                    'test_wall_s', 'reconstructed_phase_s', 'ap_s', 'mp_s', 'cached_nominal')}
                spec.update(pool_path=str(pool_path), pool_file_sha256=a['pool_file_sha256'],
                            pool_sha256=a['pool_sha256'], source_summary_path=row['source_summary_path'],
                            source_summary_sha256=row['source_summary_sha256'],
                            saved_native_finite_pool_mean=a['restricted_mean'], sizes=[len(p) for p in pools])
                specs.append(spec)
            c.write_json(OUT / 'frozen_jobs.json', dict(jobs=specs, frozen_at_utc=c.utc(),
                selection='unchanged initial-eligible green min-AP checkpoints; no fitting or reselection'))
            summary['frozen_jobs_sha256'] = c.sha(OUT / 'frozen_jobs.json')
        print('Frozen90 green policy/pool identities; evaluating training pools only.', flush=True)
        save()
        with c.phase(summary['phases'], 'finite_pool_evaluation'):
            for spec in specs:
                t = time.perf_counter()
                row = dict(**spec, status='running', started_at_utc=c.utc(), phases={}, workers=1,
                           solver_threads=1, solves_attempted=0, solves_accepted=0,
                           solver_status_counts={}, solver_iterations=0., barrier_iterations=0,
                           max_physical_residual=0.)
                summary['jobs'].append(row)
                try:
                    with c.phase(row['phases'], 'input_loading'):
                        require(c.sha(spec['checkpoint']) == spec['checkpoint_sha256'], 'Checkpoint changed')
                        pool = read(spec['pool_path'], spec['pool_file_sha256'])
                        points = np.concatenate([np.asarray(p, dtype=np.float64) for p in pool['pools']])
                        model = c.load_model(spec['checkpoint'], 'new')
                        require(c.state_hash(model) == spec['state_sha256'], 'Learned state changed')
                    with c.phase(row['phases'], 'targets'):
                        z, q10, q1r = model.targets(points[:, :3])
                    with c.phase(row['phases'], 'projection_qp'):
                        q2 = np.empty((len(points), 3), dtype=np.float64)
                        counts = Counter()
                        with c.gp.Env(empty=True) as env, (OUT / f'{spec["id"]}_solves.jsonl').open('x') as log:
                            env.setParam('OutputFlag', 0)
                            env.setParam('Threads', 1)
                            env.start()
                            with c.gp.Model(env=env) as settings_model:
                                row['solver_settings'] = {k: settings_model.getParamInfo(k)[2] for k in
                                    ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
                            for i in range(len(points)):
                                diagnostic = dict(row=i)
                                row['solves_attempted'] += 1
                                try:
                                    q2[i] = c.strict_project(z[i], q10, env, diagnostic)
                                    row['solves_accepted'] += 1
                                finally:
                                    counts[str(diagnostic['status'])] += 1
                                    row['solver_status_counts'] = dict(counts)
                                    if diagnostic['iterations'] is not None:
                                        row['solver_iterations'] += diagnostic['iterations']
                                    if diagnostic['barrier_iterations'] is not None:
                                        row['barrier_iterations'] += diagnostic['barrier_iterations']
                                    if 'physical_residual' in diagnostic:
                                        row['max_physical_residual'] = max(row['max_physical_residual'], diagnostic['physical_residual'])
                                    log.write(json.dumps(diagnostic, allow_nan=False)+'\n')
                    with c.phase(row['phases'], 'cost_aggregation'):
                        costs = c.scenario_costs(q10, q1r, q2, points[:, :3], points[:, 3:])
                        row['finite_pool_mean'], row['per_center_maxima'] = finite_pool_cost(costs, spec['sizes'])
                        row['difference_from_saved_native'] = row['finite_pool_mean'] - spec['saved_native_finite_pool_mean']
                    require(c.state_hash(model) == spec['state_sha256'], 'Evaluation changed policy')
                    row['status'] = 'completed'
                except BaseException:
                    row.update(status='failed', error=traceback.format_exc())
                    raise
                finally:
                    row.update(ended_at_utc=c.utc(), wall_time_s=time.perf_counter()-t)
                    save()
                print(f'{len(summary["jobs"])}/90 {spec["id"]}: IS {row["finite_pool_mean"]:.8f}, '
                      f'{row["solves_accepted"]} OPTIMAL, {row["wall_time_s"]:.3f}s', flush=True)
        with c.phase(summary['phases'], 'verify_and_aggregate'):
            for path, h in summary['input_sha256'].items():
                require(c.sha(path) == h, f'Protected input changed: {path}')
            require(c.sha(OUT / 'frozen_jobs.json') == summary['frozen_jobs_sha256'], 'Frozen identities changed')
            summary['inputs_unchanged'] = True
            data = dict(green=dict(radius=list(c.RADII), oos_per_seed=[], is_per_seed=[]),
                        sro_fa=deepcopy(plot['curves'][5]), horizontal=deepcopy(plot['horizontal']),
                        is_definition=summary['is_definition'], test_arrays=plot['test_arrays'])
            summary['runtime_by_radius'] = []
            for radius in c.RADII:
                rows = sorted((r for r in summary['jobs'] if r['radius'] == radius), key=lambda r:r['seed'])
                require([r['seed'] for r in rows] == list(range(10)) and all(r['status']=='completed' for r in rows), 'Partial figure forbidden')
                data['green']['oos_per_seed'].append([r['test_mean'] for r in rows])
                data['green']['is_per_seed'].append([r['finite_pool_mean'] for r in rows])
                summary['runtime_by_radius'].append(dict(radius=radius, n_train=40, status='completed',
                    evaluation=stats(r['wall_time_s'] for r in rows),
                    cached_ccg_phases=stats(r['reconstructed_phase_s'] for r in rows)))
            for prefix in ('is','oos'):
                values=np.asarray(data['green'][prefix+'_per_seed'])
                data['green'][prefix+'_mean'] = values.mean(axis=1).tolist()
                data['green'][prefix+'_p20'] = np.percentile(values,20,axis=1).tolist()
                data['green'][prefix+'_p80'] = np.percentile(values,80,axis=1).tolist()
            expected_oos = np.concatenate([np.asarray(plot['curves'][4]['per_seed']),
                                           np.asarray(plot['curves'][3]['per_seed'])],axis=0)
            require(np.array_equal(expected_oos,np.asarray(data['green']['oos_per_seed'])), 'OOS curve changed')
            summary['solves_accepted'] = sum(r['solves_accepted'] for r in summary['jobs'])
            summary['max_abs_native_difference'] = max(abs(r['difference_from_saved_native']) for r in summary['jobs'])
            c.write_json(OUT / 'plot_data.json', data)
        with c.phase(summary['phases'], 'render'):
            render(data)
            summary['figure_sha256'] = {name:c.sha(OUT/name) for name in ('green_paper.png','green_paper.pdf')}
        summary['status'] = 'completed'
        print(f'Completed {summary["solves_accepted"]} training-pool QPs; PDF and PNG saved.', flush=True)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(ended_at_utc=c.utc(), wall_time_s=time.perf_counter()-start)
        summary['wall_time_human'] = c.human(summary['wall_time_s'])
        save()


if __name__ == '__main__':
    main()