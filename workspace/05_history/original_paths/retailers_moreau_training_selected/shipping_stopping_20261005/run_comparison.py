"""Shipping stopping-checkpoint replay; immutable parent comparators retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import selection as s
from selection import c, f
import evaluation as e
import plot_comparison as p

WORKERS = 16
_spec = importlib.util.spec_from_file_location('_shipping_parent_runner', s.PARENT / 'run_comparison.py')
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)


def training_probe(job, output):
    """Actual spawned replay at two training centers, before test-array access."""
    return parent.probe(job, output)


def verify_one(job, result, path):
    seal = c.sha(path)
    c.check(seal == Path(path).with_suffix('.sha256').read_text().strip(), 'Manifest seal changed')
    manifest = c.read(path)
    c.check([j for j in manifest['jobs'] if j['id'] == job['id']] == [job], 'Unsealed job')
    manifest['seal'] = seal
    return e.verify_result(job, result, manifest)


def update_plot(old, manifest, rows):
    """Replace only green curves. Parent comparator objects are exact copies."""
    green = s.validate_grid(manifest['jobs'])
    c.check(len(rows) == 270 and {r['id'] for r in rows} == {j['id'] for j in green},
            'Incomplete green replay')
    indexed = p._index(rows)
    for row in rows:
        p._validate_row(row, manifest['test'])
    data = deepcopy(old)
    for panel in data['panels']:
        panel['green'] = p._neural(indexed, 'green', panel['Ntotal'], c.RADII)
        panel['green']['label'] = 'Green Moreau (stopping)'
        panel['green']['provenance'].update(return_rule=s.RETURN_RULE, current_invocation=True)
    data.update(return_rule=s.RETURN_RULE,
        caption='Green now returns the first passing stopping assessment and its own pool. '
            'Only green is freshly replayed in this revision; all other curves are unchanged parent results. '
            'Unmodified comparator provenance (including cached flags) describes the parent invocation, '
            'not new work. Parent comparison context: ' + old['caption'],
        footnote='Green: stopping checkpoint, fresh common Gurobi replay; blue: unchanged Sept 28 replay. Raw q2 > 0 fees.\n'
            'Training: green 40/160/640; historical 50/200/800. Bands: seed 20–80%, not CI. Green: 265 original + 5 advisory.\n'
            'Cached kNN/SRO-FA: q2 > 1e-6 / 1e-10. Gray: finite 1M MIP estimate, not exact population bound. No test-based selection.',
        current_revision=dict(parent_manifest_sha256=s.MANIFEST_SHA, parent_summary_sha256=s.SUMMARY_SHA,
            manifest_sha256=manifest['seal'], fresh_green_policies=270, cached_legacy_policies=480,
            manifest_relative_path='../manifest.json',
            comparator_metadata_scope='unchanged September28 parent invocation; cached in this revision',
            selection_changes=manifest['selection_changes']))
    for before, after in zip(old['panels'], data['panels']):
        c.check({k: v for k, v in before.items() if k != 'green'} ==
                {k: v for k, v in after.items() if k != 'green'}, 'Comparator changed')
    p._validate_plot_data(data)
    return data


def report(rows, summary, manifest, published):
    groups = []
    for n in c.SIZES:
        for radius in c.RADII:
            cohort = [r for r in rows if (r['Ntotal'], r['radius']) == (n, radius)]
            c.check(len(cohort) == 10 and all(r['status'] == 'completed' for r in cohort), 'Incomplete runtime group')
            groups.append(dict(method='green_stopping', Ntotal=n, radius=radius,
                total_evaluation_s=c.stats([r['wall_time_s'] for r in cohort]),
                oos_s=c.stats([r['phases']['oos']['wall_time_s'] for r in cohort]),
                is_s=c.stats([r['phases']['is']['wall_time_s'] for r in cohort])))
    runtime = dict(unit='seconds', current_evaluation_groups=groups, phases=summary['phases'],
        environment=summary['resource_gate'], workers=WORKERS, threads=1,
        historical_training=[dict(id=r['id'], timing=r['training_runtime_reference']) for r in rows],
        parent_runtime_report=str(s.OLD / 'published/runtime_report.json'),
        note='Historical training and parent legacy evaluation are not current work. No sums of concurrent '
             'walls represent sweep elapsed. Missing historical timings unavailable; rerun required.')
    c.write(published / 'runtime_report.json', runtime)
    changes = manifest['selection_changes']
    lines = ['# Shipping — stopping-checkpoint comparison', '',
        'All 270 green policies freshly replayed at their first passing stopping assessment, with its own pool.',
        f'Changed selections: {sum(r["candidate_changed"] for r in changes)}; changed learned states: '
        f'{sum(r["state_changed"] for r in changes)}; changed pools: {sum(r["pool_changed"] for r in changes)}.',
        'The 480 historical Clarabel policies, kNN, SRO-FA and conditional reference are unchanged cached parent curves.',
        'Same 10,000 historical test scenarios, seed 9999, all ten training seeds and original nine radii.',
        'Common strict Gurobi replay; raw q>0 fees; OPTIMAL and physical residual <=1e-6 required. No clipping or retry.',
        'Training40/160/640; green mixed cohort265 original +5 advisory; no exact/global/every-center robustness claim.',
        'IS is mean per-center maximum on the stopping policy own pre-insertion pool, not a full robust supremum.',
        'This known test and historically test-informed design are descriptive, not fresh confirmatory evidence. '
        'No radius selection or tuning. Earlier minimum-AP artifacts remain unchanged.', '',
        '## Runtime coverage', '', '| Component | Current work |', '|---|---|',
        '| Nominal neural training | Not rerun; original per-seed timings retained |',
        '| Robust CCG / AP / MP | Not rerun; original per-iteration timings retained |',
        '| SRO-FA partition / optimization / evaluation / total | Not rerun; parent plot provenance retained |',
        '| CV / radius validation / refitting | Not run |',
        '| Green OOS and own-pool evaluation | Fresh; per-configuration table below |',
        '| Legacy neural evaluation | Cached September28 results; not current replay time |',
        '', f'Host: {summary["resource_gate"]["environment"]["host"]}; 16 workers, one Torch/Gurobi thread each.',
        'Missing historical times remain unavailable; a rerun is required to measure them.',
        f'Whole-child elapsed and UTC start/end: ../../{published.parent.name}.invocation.json (written at process exit).', '',
        '## Completed evaluation runtimes (seconds)', '',
        '| N | Radius | Runs | Mean | Median | Min | Max | Mean OOS seconds | Mean IS seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        t = g['total_evaluation_s']
        lines.append(f'| {g["Ntotal"]} | {g["radius"]:g} | {t["count"]} | {t["mean"]:.3f} | '
            f'{t["median"]:.3f} | {t["min"]:.3f} | {t["max"]:.3f} | '
            f'{g["oos_s"]["mean"]:.3f} | {g["is_s"]["mean"]:.3f} |')
    lines += ['', '## Current phases (nested; not additive concurrent job times)', '']
    for name, phase in summary['phases'].items():
        lines.append(f'- {name}: {c.c.human(phase["wall_time_s"])}; {phase["status"]}.')
    (published / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = dict(experiment='shipping_stopping_20261005', status='running', start_utc=c.utc(),
        phases={}, results=[], workers=WORKERS, threads=1, training_run=False,
        return_rule=s.RETURN_RULE, radius_selection_run=False, test_generated=False)
    def persist():
        summary['wall_time_s'] = time.perf_counter() - start
        c.write(output / 'summary.json', summary)
    persist()
    try:
        c.initialize_worker()
        with c.phase(summary['phases'], 'tests'):
            with (output / 'tests.log').open('x') as log:
                tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                    'test_stopping'], cwd=s.ROOT, stdout=log, stderr=subprocess.STDOUT)
            c.check(tests.returncode == 0, 'Stopping regression tests failed')
            with (output / 'parent_tests.log').open('x') as log:
                tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                    'test_freeze', 'test_evaluation', 'test_plot', 'test_runner'],
                    cwd=s.PARENT, stdout=log, stderr=subprocess.STDOUT)
            c.check(tests.returncode == 0, 'Unchanged replay regression tests failed')
        with c.phase(summary['phases'], 'selection_and_freeze'):
            manifest = s.collect(output)
        green = s.validate_grid(manifest['jobs'])
        summary['selection_changes'] = manifest['selection_changes']
        persist()
        with c.phase(summary['phases'], 'training_only_preflight'):
            # Exercise a nontrivial stop for each N, including advisory N200/N800.
            keys = [(50, 0, .2), (200, 5, 1.5), (800, 7, .2)]
            chosen = [next(j for j in green if (j['Ntotal'], j['seed'], j['radius']) == key) for key in keys]
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=c.initialize_worker) as pool:
                probes = [pool.submit(training_probe, j, str(output / 'preflight')).result() for j in chosen]
            summary['probes'] = probes
            f.verify(manifest)
        summary['resource_gate'] = parent.resource_gate(max(r['peak_rss_bytes'] for r in probes))
        with c.phase(summary['phases'], 'test_binding'):
            # Read existing array bytes only after all choices and preflight pass.
            for info in manifest['test']['arrays'].values():
                c.check(c.sha(info['file']) == info['file_sha256'], 'Parent test changed')
            path = output / 'manifest.json'
            c.write(path, manifest)
            path.chmod(0o444)
            seal = c.sha(path)
            path.with_suffix('.sha256').write_text(seal + '\n')
            path.with_suffix('.sha256').chmod(0o444)
            summary.update(manifest_path=str(path), manifest_sha256=seal)
        with c.phase(summary['phases'], 'green_evaluation'):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=e.initialize, initargs=(str(path),)) as pool:
                futures = [pool.submit(e.worker, j['id'], str(output / 'evaluations')) for j in green]
                for future in as_completed(futures):
                    result = future.result()
                    summary['results'].append(result)
                    persist()
                    print(f'{len(summary["results"])}/270 {result["id"]} {result["status"]}', flush=True)
        c.check(len(summary['results']) == 270 and all(r['status'] == 'completed' for r in summary['results']),
                'Incomplete or failed green grid; no publication')
        with c.phase(summary['phases'], 'raw_verification'):
            indexed = {r['id']: r for r in summary['results']}
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=c.initialize_worker) as pool:
                futures = [pool.submit(verify_one, j, indexed[j['id']], str(path)) for j in green]
                rows = [future.result() for future in futures]
        with c.phase(summary['phases'], 'final_integrity'):
            f.verify(manifest)
            c.check(c.sha(path) == seal, 'Manifest changed')
            c.check(c.sha(output / 'selection_manifest.json') ==
                    (output / 'selection_manifest.sha256').read_text().strip(), 'Selection seal changed')
            for name, digest in manifest['parent_published_hashes'].items():
                c.check(c.sha(name) == digest, 'Parent published artifact changed')
            old = c.read(s.OLD / 'published/plot_data.json')
            c.check(old['test'] == manifest['test'], 'Test protocol changed')
            manifest['seal'] = seal
        with c.phase(summary['phases'], 'render'):
            data = update_plot(old, manifest, rows)
            staging = output / 'publication_staging'
            staging.mkdir()
            p.render(data, staging)
            c.write(staging / 'plot_data.json', data)
        report(rows, summary, manifest, staging)
        published = output / 'published'
        staging.rename(published)
        summary.update(status='completed', solves_accepted=sum(r['solves_accepted'] for r in rows),
            fresh_green=270, unchanged_legacy=480,
            published={str(file): c.sha(file) for file in published.iterdir() if file.is_file()})
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary['end_utc'] = c.utc()
        persist()


if __name__ == '__main__':
    main()