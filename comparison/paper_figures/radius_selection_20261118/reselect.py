"""Re-select every retailer radius from the stored validation scores under the rule of 2026-11-18 (no retraining).

Rule (src/config.py, RADIUS_SELECTION_RULE): radii whose validation scores lie within 1e-6 (absolute) of the minimum
are tied, and the largest tied radius is selected. It replaces "exact tie -> smaller radius".

Inputs (read-only; the sealed selection records of the original runs are not modified):
  Protocol A (reported): green_validation_R3_R10_20261015 (Neural CCG, R3/R10), shipping_validation_qpositive_20261006
  (Neural CCG green and legacy blue, R3 with shipping), sro_nn_split_20261015 (SRO-FA and SRO-FA+LDR).
  Legacy (not in any current reported result): the blue net16_v2 and legacy SRO-FA validation selections, and the
  blue N50 CV selection.
Test costs of newly selected policies are looked up in the stored evaluations of the same evaluator on the same
seed-9999 test set; nothing is re-evaluated (a missing stored evaluation raises an error).
Outputs (this folder): selections_new_rule.json (every selection, per seed: old/new radius, tied set and its scores,
smallest nonzero gap, test costs), RESULTS.md.
Run from retailers_experiment/: ../ap_env/bin/python -B comparison/paper_figures/radius_selection_20261118/reselect.py
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.config import RADIUS_SELECTION_RULE, RADIUS_TIE_TOLERANCE, select_radius  # noqa: E402

HIST = ROOT / 'workspace/05_history/original_paths'
TRAIN = HIST / 'retailers_moreau_training_selected'
STOP = ROOT / 'comparison/stopping_checkpoint_20260928'
SHIP_EVAL = TRAIN / 'shipping_stopping_20261005/production_v2/evaluations'
OLD_RULE = 'minimum validation score; exact tie -> smaller radius'


def old_select(scores):
    best = min(scores.values())
    return min((k for k, v in scores.items() if v == best), key=float)


def record(source, setting, n, seed, scores, stored_old, reported, **extra):
    scores = {float(k): float(v) for k, v in scores.items()}
    best = min(scores.values())
    values = sorted(set(scores.values()))
    new = float(select_radius(scores))
    tied = sorted(k for k, v in scores.items() if v - best <= RADIUS_TIE_TOLERANCE)
    return dict(source=source, setting=setting, N=n, seed=seed, reported=reported,
                old_radius=None if stored_old is None else float(stored_old),
                old_rule_reproduces_stored=None if stored_old is None else math.isclose(old_select(scores), stored_old),
                new_radius=new, changed=stored_old is not None and not math.isclose(new, stored_old),
                tied_radii=tied, tied_scores=[scores[k] for k in tied],
                exact_ties_only=all(scores[k] == best for k in tied),
                smallest_nonzero_gap=values[1] - values[0] if len(values) > 1 else None,
                validation_scores={str(k): v for k, v in sorted(scores.items())}, **extra)


def protocol_a():
    rows = []
    for c in json.loads((TRAIN / 'green_validation_R3_R10_20261015/production_v1/selections.json').read_text())['choices']:
        rows.append(record('neural_ccg', f"R{c['R']}", c['Ntotal'], c['seed'], c['validation_means'],
                           c['selected_radius'], True))
    base = TRAIN / 'shipping_validation_qpositive_20261006/production_v1'
    for c in json.loads((base / 'selections.json').read_text())['choices']:
        arms = json.loads((base / 'validation' / c['validation_result']['id'] / 'result.json').read_text())['arms']
        rows.append(record('neural_ccg' if c['method'] == 'green' else 'legacy_blue_nn', 'R3_shipping', c['Ntotal'],
                           c['seed'], {a['radius']: a['mean_validation'] for a in arms}, c['selected_radius'],
                           c['method'] == 'green'))
    seal = json.loads((TRAIN / 'sro_nn_split_20261015/runs/production_v1/selection_seal.json').read_text())
    case = {'R3': 'R3', 'R10': 'R10', 'shipping_R3': 'R3_shipping'}
    for c in seal['selections']:
        rows.append(record('sro_fa' if c['method'] == 'FA' else 'sro_fa_ldr', case[c['case']], c['n'], c['seed'],
                           c['candidates'], c['rho'], c['method'] == 'FA'))
    return rows


def legacy():
    rows = []
    files = {('legacy_blue_nn', 'R3'): 'retailers_problem/R3/rho_validation_results_net16_v2.json',
             ('legacy_blue_nn', 'R10'): 'retailers_problem/R10/rho_validation_results_net16_v2.json',
             ('legacy_blue_nn', 'R3_shipping'): 'retailers_with_shipping/rho_validation_results_net16_v2.json',
             ('legacy_sro_fa', 'R3'): 'retailers_problem/R3/sro_fa_rho_validation_results.json',
             ('legacy_sro_fa', 'R10'): 'retailers_problem/R10/sro_fa_rho_validation_results.json',
             ('legacy_sro_fa', 'R3_shipping'): 'retailers_with_shipping/sro_fa_rho_validation_results.json'}
    for (source, setting), rel in files.items():
        for n, block in json.loads((HIST / rel).read_text())['by_N'].items():
            for seed, ps in block['per_seed'].items():
                scores = (ps['validation_costs'] if 'validation_costs' in ps else
                          {k: v['validation_cost'] for k, v in ps['validation_candidates'].items()})
                rows.append(record(source, setting, int(n), int(seed), scores, ps.get('selected_rho'), False,
                                   file=rel, stored_selected_test_mean=ps.get('selected_10k_oos')))
    cv = 'retailers_problem/R3/N50/rho_sweep_net16_v2/cv_rho_results.json'
    for seed, ps in json.loads((HIST / cv).read_text())['per_seed'].items():
        rows.append(record('legacy_blue_nn_cv', 'R3', 50, int(seed), ps['val_costs'], ps['cv_rho'], False, file=cv))
    return rows


def nn_result_path(row, radius):
    if row['setting'] == 'R3_shipping':
        tag = ('%g' % radius).replace('.', 'p')
        return SHIP_EVAL / f"green_N{row['N']}_r{tag}_s{row['seed']}/result.json"
    return STOP / f"production_v1/jobs/green_{row['setting']}_N{row['N']}_s{row['seed']}_rho{'%g' % radius}/result.json"


def attach_tests(rows):
    """Stored test costs of the old and new selected policies (same evaluator, seed-9999, 10,000 rows)."""
    sro = {(r['method'], r['case'], r['n'], r['seed']): r for r in json.loads(
        (TRAIN / 'sro_nn_split_20261015/runs/production_v1/test_results.json').read_text())['results']}
    blue_sweep = json.loads((HIST / 'retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json').read_text())
    manifest = {j['id']: j for j in json.loads((STOP / 'preflight_v1/selection_manifest.json').read_text())['jobs']}
    for row in rows:
        t0 = time.perf_counter()
        if row['source'] == 'neural_ccg':
            old, new = (json.loads(nn_result_path(row, r).read_text()) for r in (row['old_radius'], row['new_radius']))
            if not (math.isclose(old['radius'], row['old_radius']) and math.isclose(new['radius'], row['new_radius'])
                    and old['seed'] == new['seed'] == row['seed']):
                raise RuntimeError(f'stored evaluation identity mismatch: {row}')
            tied_states = {json.loads(nn_result_path(row, r).read_text())['state_sha256'] for r in row['tied_radii']}
            row.update(old_test_mean=old['mean_oos'], new_test_mean=new['mean_oos'],
                       old_test_file=str(nn_result_path(row, row['old_radius']).relative_to(ROOT)),
                       new_test_file=str(nn_result_path(row, row['new_radius']).relative_to(ROOT)),
                       new_equals_old_policy=old['state_sha256'] == new['state_sha256'],
                       distinct_policies_in_tied_set=len(tied_states), test_reevaluated=False)
            if row['setting'] in ('R3', 'R10'):
                for which in ('old', 'new'):
                    job = manifest[f"green_{row['setting']}_N{row['N']}_s{row['seed']}_rho{'%g' % row[which + '_radius']}"]
                    row[f'{which}_ccg_through_stop_active_time_s'] = \
                        job['historical_training']['green_through_stop']['active_phase_time_s']
        elif row['source'] in ('sro_fa', 'sro_fa_ldr'):
            key = ('FA' if row['source'] == 'sro_fa' else 'FA+LDR', {'R3_shipping': 'shipping_R3'}.get(
                row['setting'], row['setting']), row['N'], row['seed'])
            stored = sro[key]
            if not math.isclose(stored['rho'], row['old_radius']) or row['changed']:
                raise RuntimeError(f'SRO selection mismatch or change (only selected policies were evaluated): {key}')
            row.update(old_test_mean=stored['test']['mean'], new_test_mean=stored['test']['mean'],
                       new_equals_old_policy=True, test_reevaluated=False)
        elif row['changed'] and row['source'] == 'legacy_blue_nn' and row['setting'] == 'R3' and row['N'] == 50:
            costs = {r: blue_sweep['sweep'][str(r)]['rob_oos'][row['seed']] for r in row['tied_radii']}
            row.update(old_test_mean=costs[row['old_radius']], new_test_mean=costs[row['new_radius']],
                       tied_test_means={str(k): v for k, v in costs.items()},
                       new_test_file='workspace/05_history/original_paths/retailers_problem/R3/N50/rho_sweep_net16_v2/'
                                     'sweep_eval_net16_v2_10k.json', test_reevaluated=False)
        elif row['changed']:
            raise RuntimeError(f'changed legacy selection without a test lookup: {row}')
        row['lookup_wall_time_s'] = time.perf_counter() - t0


def _count(values):
    counts = defaultdict(int)
    for v in values:
        counts['%g' % v] += 1
    return dict(sorted(counts.items(), key=lambda kv: float(kv[0])))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['source'] in ('neural_ccg', 'sro_fa', 'sro_fa_ldr'):
            groups[(r['source'], r['setting'], r['N'])].append(r)
    out = []
    for (source, setting, n), g in sorted(groups.items()):
        old = np.array([r['old_test_mean'] for r in g]); new = np.array([r['new_test_mean'] for r in g])
        item = dict(source=source, setting=setting, N=n, seeds=len(g), changed=sum(r['changed'] for r in g),
                    old_mean=float(old.mean()), old_sd=float(old.std(ddof=1)),
                    new_mean=float(new.mean()), new_sd=float(new.std(ddof=1)),
                    old_radius_counts=_count(r['old_radius'] for r in g),
                    new_radius_counts=_count(r['new_radius'] for r in g))
        if all('new_ccg_through_stop_active_time_s' in r for r in g):
            item.update(old_ccg_time_mean=float(np.mean([r['old_ccg_through_stop_active_time_s'] for r in g])),
                        new_ccg_time_mean=float(np.mean([r['new_ccg_through_stop_active_time_s'] for r in g])))
        out.append(item)
    return out


def report(rows, summary, wall):
    stored = [r for r in rows if r['old_radius'] is not None]
    gaps = [r['smallest_nonzero_gap'] for r in rows if r['smallest_nonzero_gap'] is not None]
    changed = [r for r in rows if r['changed']]
    label = dict(neural_ccg='Neural CCG', sro_fa='SRO-FA', sro_fa_ldr='SRO-FA+LDR (not reported)')
    lines = ['# Radius re-selection under the rule of 2026-11-18', '',
             f'Rule: **{RADIUS_SELECTION_RULE}** (replaces "{OLD_RULE}"). No policy was retrained or re-evaluated:',
             'every test cost is a stored evaluation of the same policy with the same evaluator on the same',
             '10,000-scenario seed-9999 test set. Per-seed records (old/new radius, tied set and scores, gaps, test',
             'files): `selections_new_rule.json`. Created by `reselect.py`.', '',
             '## Tolerance check', '',
             f'- {len(rows)} selections over all sources and seeds. The stored choice is reproduced by the old rule in '
             f"{sum(bool(r['old_rule_reproduces_stored']) for r in stored)} of {len(stored)} cases "
             f"({len(rows) - len(stored)} legacy selection(s) have no stored choice).",
             f'- Smallest nonzero gap between the best and second-best validation score: {min(gaps):.2e}; '
             f"genuine gaps at or below 1e-6: {sum(g <= RADIUS_TIE_TOLERANCE for g in gaps)}.",
             f"- All ties are exact (bit-identical scores): {all(r['exact_ties_only'] for r in rows)}. No tie arises "
             'from rounding; the old rule did fire, always choosing the smaller radius.', '',
             '## Reported Protocol A selections, per setting', '',
             '| Method | Setting | N | Changed seeds | Test cost, old rule (mean ± sd) | Test cost, new rule (mean ± sd) | '
             'Selected radii, old | Selected radii, new |', '|---|---|---:|---:|---:|---:|---|---|']
    for s in summary:
        lines.append(f"| {label[s['source']]} | {s['setting']} | {s['N']} | {s['changed']}/{s['seeds']} | "
                     f"{s['old_mean']:.4f} ± {s['old_sd']:.4f} | {s['new_mean']:.4f} ± {s['new_sd']:.4f} | "
                     f"{s['old_radius_counts']} | {s['new_radius_counts']} |")
    lines += ['', '## Every changed selection', '',
              '| Source | Setting | N | Seed | Old | New | Tied radii (all at the score) | Tied score | Same policy | '
              'Test cost old → new | Stored test evaluation used |', '|---|---|---:|---:|---:|---:|---|---:|---|---|---|']
    for r in changed:
        lines.append(f"| {r['source']} | {r['setting']} | {r['N']} | {r['seed']} | {r['old_radius']:g} | "
                     f"{r['new_radius']:g} | {', '.join('%g' % x for x in r['tied_radii'])} | {r['tied_scores'][0]:.6f} | "
                     f"{r.get('new_equals_old_policy', 'same test cost at all tied radii')} | "
                     f"{r['old_test_mean']:.6f} → {r['new_test_mean']:.6f} | `{r['new_test_file']}` |")
    t = [s for s in summary if 'new_ccg_time_mean' in s and s['setting'] == 'R3' and s['source'] == 'neural_ccg']
    lines += ['', '## Affected timing entry (Experiment 1, Protocol A, "Neural selected-radius CCG")', '',
              'Mean over ten seeds of the recorded CCG active time through the stopping checkpoint of the selected run',
              '(historical seconds, excluding nominal training). The newly selected ρ=0.03 runs return the same policy',
              'as ρ=0 but are separate CCG runs with their own recorded adversarial-verification time.', '',
              '| N | Old rule | New rule |', '|---:|---:|---:|']
    lines += [f"| {s['N']} | {s['old_ccg_time_mean']:.1f} | {s['new_ccg_time_mean']:.1f} |" for s in t]
    lines += ['', f'Re-selection wall time: {wall:.2f} s, one process; stored-record lookups only.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main():
    start = time.perf_counter()
    rows = protocol_a() + legacy()
    attach_tests(rows)
    summary = summarize(rows)
    wall = time.perf_counter() - start
    (HERE / 'selections_new_rule.json').write_text(json.dumps(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), rule=RADIUS_SELECTION_RULE, previous_rule=OLD_RULE,
        tolerance_absolute=RADIUS_TIE_TOLERANCE, retraining=False, reevaluation=False,
        test_set='seed 9999, 10,000 scenarios (stored evaluations)', wall_time_s=wall,
        summary=summary, selections=rows), indent=1) + '\n')
    report(rows, summary, wall)


if __name__ == '__main__':
    main()
