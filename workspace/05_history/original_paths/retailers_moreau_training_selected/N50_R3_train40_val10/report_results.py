"""Post-run descriptive report; never chooses another model or accesses test data."""
import json
from pathlib import Path
import statistics
import sys


def main():
    path=Path(sys.argv[1]).resolve()
    summary=json.loads((path/'summary.json').read_text())
    if summary['status']!='completed' or not summary['final_selection_published']:
        raise ValueError('Cannot report partial campaign as completed')
    aggregates=json.loads((path/'aggregates.json').read_text())
    root=Path(__file__).resolve().parent
    rel=path.relative_to(root)
    text=['# Retailer training-selected rerun results', '',
        f"Completed {summary['ended_at_utc']}; overall elapsed **{summary['wall_time_s']:.6f}s "
        f"({summary['wall_time_s']/60:.6f}min)**. Fresh30 nominal trainings,90 independent CCG configurations; no test evaluation.", '',
        'Nominal and MP checkpoints, scheduler and early stopping use training loss. Validation chooses the nominal LR after training and the radius after all terminal policies are fixed. The40/10 split, ten seeds, nine radii, zero capacity buffer and native Moreau/AP numerical settings are unchanged. Historical results are not overwritten.', '',
        '## Validation (selection data, not an independent test)', '',
        f"Nominal LR-selected mean: **{aggregates['nominal_validation_mean']:.9f}**. "
        f"Final radius-selected mean: **{aggregates['radius_selected_validation_mean']:.9f}**. "
        'These scores reuse ten validation observations per seed for LR/radius selection; they do not establish generalization improvement or a U-shaped population curve.', '',
        '| Radius | Validation mean | Initial AP mean at this radius | Terminal AP mean | Terminal AP upper mean | MPs | Separated / stagnated / capped | Nominal weights retained |',
        '|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in aggregates['per_radius']:
        text.append(f"| {r['radius']:g} | {r['validation_mean']:.9f} | {r['initial_ap_mean']:.9f} | {r['terminal_ap_mean']:.9f} | {r['terminal_ap_upper_mean']:.9f} | {r['mp_count']} | {r['separated_count']} / {r['stagnation_count']} / {r['iteration_cap_count']} | {r['unchanged_nominal_count']} |")
    text += ['', 'AP comparisons are paired at the same radius and seed. Increasing AP costs across nested boxes is not a failed U-shape. AP bounds concern the encoded numerical policy approximation, not uniform native-policy or global optimization certification. Stagnated/capped terminal policies remain included with their statuses—not relabelled converged.', '',
        '## MP checkpoint audit', '',
        f"{len(aggregates['mp'])} MPs: {aggregates['mp_selected_improved']} selected objectives improved, "
        f"{aggregates['mp_selected_worsened']} worsened, and {aggregates['mp_unchanged']} returned unchanged weights. "
        'Each comparison uses incoming and returned states on the same augmented pool. Selected states are checked against the saved checkpoint and reevaluated before the next AP. This prevents validation restoration from choosing a worse training objective, but does not guarantee full-box or held-out improvement.', '',
        '## Runtime accounting', '',
        'Linux gold-srv2, Intel Xeon Gold6354,72 logical CPU slots. Nominal10 workers then CCG16 workers; Torch/Gurobi single-thread; final validation1 worker. Stored unit: wall seconds, monotonic perf_counter. Concurrent model walls are not sweep elapsed; nested phases are not additive.', '',
        '| Component | Elapsed seconds | Minutes |', '|---|---:|---:|']
    for name in ('nominal','ccg','validation'):
        t=summary[name]['wall_time_s'];text.append(f'| {name} phase | {t:.6f} | {t/60:.6f} |')
    text += ['', '| Nominal training cohort | n | Mean seconds | Median | Min | Max |',
        '|---|---:|---:|---:|---:|---:|']
    for label,s in [('Selected LR winners',aggregates['nominal_winner_training']),
                    *[(f'LR{lr}',s) for lr,s in aggregates['nominal_training_by_lr'].items()]]:
        text.append(f"| {label} | {s['count']} | {s['mean']:.6f} | {s['median']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    text += ['', 'Winner-only training excludes rejected LRs; full nominal-phase elapsed above includes initialization, all30 trainings, subsequent validation and LR selection.', '',
        '| Radius | n | Mean CCG seconds | Median | Min | Max | Mean AP | Mean MP | Mean pool-evaluation | Mean final-validation |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates['per_radius']:
        t=r['runtime_s'];s=t['wall_time_s']
        text.append(f"| {r['radius']:g} | {s['count']} | {s['mean']:.6f} | {s['median']:.6f} | {s['min']:.6f} | {s['max']:.6f} | {t['ap_total_time_s']['mean']:.6f} | {t['mp_total_time_s']['mean']:.6f} | {t['evaluation_time_s']['mean']:.6f} | {r['validation_time_s']['mean']:.6f} |")
    text += ['', 'CCG totals exclude nominal training. AP includes native replay and persistence; MP includes training snapshots/journals. Radius0 is nominal-policy assessment, not additional nominal training. Completed execution is not synonymous with separation: capped/stagnated outcomes are included in the descriptive all-executed table and stratified below.', '',
        '| Radius | Terminal status | n | Mean CCG seconds | Median | Min | Max |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for radius in summary['protocol']['radii']:
        for status in ('separation_satisfied','stagnation','iteration_cap'):
            values=[j['wall_time_s'] for j in summary['ccg']['jobs'] if j['radius']==radius and j['terminal_status']==status]
            if values:
                text.append(f'| {radius:g} | {status} | {len(values)} | {statistics.mean(values):.6f} | {statistics.median(values):.6f} | {min(values):.6f} | {max(values):.6f} |')
    text += ['', '| Other component | Status |','|---|---|',
        '| SRO-FA | Not run |','| CV | Not run; fixed40/10 split |',
        '| Independent test | Not evaluated |','| Failed/interrupted campaign attempts | Separate timestamped artifacts; never counted as completed reruns |', '',
        '## Reproducibility', '',
        f"Started {summary['started_at_utc']}; ended {summary['ended_at_utc']}. Exact command: `{summary['command']}`. "
        'Cwd this folder. Named tmux session `retailers_training_selected_20260916`; launch `tmux new-session -d -s retailers_training_selected_20260916 -c /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 bash launch.sh`. While active attach `tmux attach -t retailers_training_selected_20260916`; check session `tmux has-session -t retailers_training_selected_20260916`. Completed sessions disappear; persistent logs and exit artifacts remain.', '',
        f'- [{rel}/summary.json]({rel}/summary.json)',f'- [{rel}/aggregates.json]({rel}/aggregates.json)',
        f'- [{rel}/provenance.json]({rel}/provenance.json)',
        f'- [{rel}/pre_validation_terminal_policies.json]({rel}/pre_validation_terminal_policies.json)',
        '- [Preflight and protocol review](PRELAUNCH_REVIEW.md)',
        '- [Thesis notes](../../unit_commitment/two_stage/uc_surrogate_mp_experiment/thesis_modeling_notes.tex)', '']
    for p in sorted(root.glob('campaign_*')):
        if p.suffix in ('.log','.exit'):
            text.append(f'- [{p.name}]({p.name})')
    with (root/'RESULTS.md').open('x') as stream:
        stream.write('\n'.join(text)+'\n')
    print(root/'RESULTS.md')


if __name__=='__main__':
    main()