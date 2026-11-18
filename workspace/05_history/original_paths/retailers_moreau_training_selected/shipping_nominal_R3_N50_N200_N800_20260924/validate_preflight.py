"""Full fail-closed nested gate. No rerun/overwrite when evidence changes."""
from source_adapter import sources, verify, check, digest
from pathlib import Path
import json
import math

PHASES = {'initialization','nominal'}


def completed(row):
    check(row['status']=='completed', 'Incomplete status')
    check(math.isfinite(row['wall_time_s']) and row['wall_time_s']>=0, 'Invalid timing')
    check(bool(row['started_at_utc']) and bool(row['ended_at_utc']), 'Missing UTC endpoints')


def source_checks(row, expected):
    check(row['source_sha256']==expected, 'Mismatched source manifest')
    checks = row['source_checks']
    check(len(checks)==2 and [c['label'] for c in checks]==['before','after'], 'Missing source checks')
    check(all(c['status']=='completed' and c['source_sha256']==expected for c in checks),
          'Source check did not pass')


def validate(path, expected_sha):
    path = Path(path).resolve()
    check(digest(path)==expected_sha, 'Preflight SHA mismatch: abort, never rerun')
    row = json.loads(path.read_text())
    completed(row)
    expected = sources()
    verify(expected)
    source_checks(row,expected)
    check(row['preflight_only'] is True and row['production_reuse_forbidden'] is True,
          'Not disposable preflight')
    check(row['heldout_evaluated'] is False and row['workers']==1, 'Invalid preflight scope')
    tests = row['tests']
    completed(tests)
    check(tests['returncode']==0 and tests['tests_run']>=20 and tests['failures']==tests['errors']==0,
          'Incomplete fixture tests')
    test_path = path.parent/'tests.json'
    check(digest(test_path)==tests['sha256'], 'Test evidence changed')
    stored_tests = json.loads(test_path.read_text())
    check(all(stored_tests[k]==tests[k] for k in ('tests_run','failures','errors','status')), 'Test evidence mismatch')
    children = row['children']
    check(len(children)==3 and {c['Ntotal'] for c in children}=={50,200,800}, 'Unique three-size coverage required')
    check(len({c['path'] for c in children})==3, 'Duplicate child paths')
    for child in children:
        completed(child)
        check(child['returncode']==0 and child['peak_rss_bytes']>0, 'Child process failed')
        child_path = path.parent/f'N{child["Ntotal"]}'/'summary.json'
        check(Path(child['path'])==child_path and digest(child_path)==child['sha256'], 'Child path/hash changed')
        data = json.loads(child_path.read_text())
        completed(data)
        source_checks(data,expected)
        check(data['Ntotal']==child['Ntotal'] and data['seed']==0 and data['epochs']==2
              and data['preflight_only'] is True and data['heldout_evaluated'] is False
              and data['workers']==1 and data['peak_rss_bytes']==child['peak_rss_bytes'], 'Invalid child scope')
        check(set(data['phases'])==PHASES, 'Missing/extra child phases')
        for kind, phase in data['phases'].items():
            completed(phase)
            check(phase['complete'] is True and phase['Ntotal']==child['Ntotal'] and phase['seed']==0,
                  'Invalid phase identity/completeness')
            check(phase['actual_torch_threads']==phase['actual_interop_threads']==1, 'Invalid worker threads')
            artifact = child_path.parent/kind/'result.json'
            check(digest(artifact)==data['phase_sha256'][kind], 'Changed phase artifact')
            check(json.loads(artifact.read_text())==phase, 'Phase content mismatch')
        init = data['phases']['initialization']
        completed(init['saa_initialization'])
        check(init['saa_initialization']['solver_status']==2 and
              init['saa_initialization']['solution_count']>0, 'SAA not OPTIMAL')
        completed(init['model_initialization'])
        for key in ('initial','training_data','split'):
            check(digest(init[key+'_path'])==init[key+'_sha256'], 'Initialization artifact changed')
        nom = data['phases']['nominal']
        train = nom['training']
        completed(train)
        check(train['epochs_evaluated']==train['optimizer_steps']==2 and
              train['max_epochs']==2 and train['Ntrain']==4*child['Ntotal']//5 and
              train['best_epoch'] in (1,2), 'Not two-epoch fullbatch preupdate training')
        check(nom['initial_sha256']==init['initial_sha256'] and nom['lr_net']==.01,
              'Wrong nominal initialization/LR')
        check(digest(nom['checkpoint'])==nom['checkpoint_sha256'], 'Changed trained checkpoint')
        replay = nom['training_only_replay']
        completed(replay)
        check(replay['finite_gradients'] is True and replay['same_returned_shipments'] is True
              and replay['real_arithmetic_certificate'] is False, 'Invalid replay')
    return row


def review(path, report_path):
      """Validate real evidence, then prove nested corruptions fail in memory.

      Hashes remain those of the untouched real files. Altered read contents model
      malformed nested evidence; neither production files nor preflight are edited.
      """
      from unittest.mock import patch
      import copy
      import shipping_core as core
      from shipping_resources import resources, choose_workers
      path, report_path = Path(path).resolve(), Path(report_path).resolve()
      check(not report_path.exists(), 'Review output exists')
      sha = digest(path)
      parent = validate(path,sha)
      first = Path(parent['children'][0]['path'])
      original_child = json.loads(first.read_text())
      original_read = Path.read_text
      report = dict(preflight_path=str(path),preflight_sha256=sha,
                          source_count=len(sources()),source_sha256=sources(),rejections=[])
      def reject(label, target, mutate):
            altered = copy.deepcopy(parent if target==path else original_child)
            mutate(altered)
            def read(p, *args, **kwargs):
                  return json.dumps(altered) if p.resolve()==target else original_read(p,*args,**kwargs)
            with patch.object(Path,'read_text',read):
                  try:
                        validate(path,sha)
                  except (ValueError,KeyError) as error:
                        report['rejections'].append(dict(case=label,status='completed',message=str(error)))
                  else:
                        raise AssertionError(f'Malformed evidence accepted: {label}')
      with core.timed(report):
            reject('duplicate children',path,lambda r:r['children'].__setitem__(1,r['children'][0]))
            reject('missing child',path,lambda r:r['children'].pop())
            reject('failed child process',path,lambda r:r['children'][0].update(returncode=1))
            reject('wrong child hash',path,lambda r:r['children'][0].update(sha256='0'*64))
            reject('parent source check failed',path,lambda r:r['source_checks'][1].update(status='failed'))
            reject('tests failed',path,lambda r:r['tests'].update(failures=1))
            reject('child source check failed',first,lambda r:r['source_checks'][1].update(status='failed'))
            reject('missing nominal phase',first,lambda r:r['phases'].pop('nominal'))
            reject('failed initialization phase',first,lambda r:r['phases']['initialization'].update(status='failed'))
            reject('wrong phase artifact hash',first,lambda r:r['phase_sha256'].update(nominal='0'*64))
            reject('invalid epoch count',first,lambda r:r['phases']['nominal']['training'].update(epochs_evaluated=1))
            reject('heldout evaluated',first,lambda r:r.update(heldout_evaluated=True))
            reject('different source bindings',first,lambda r:r['source_sha256'].pop(next(iter(r['source_sha256']))))
            try:
                  validate(path,'0'*64)
            except ValueError:
                  report['rejections'].append(dict(case='wrong explicit SHA',status='completed'))
            else:
                  raise AssertionError('Wrong explicit SHA accepted')
            validate(path,sha)
            snapshot = resources()
            report.update(environment=snapshot,prospective_workers_only=choose_workers(snapshot,
                  max(c['peak_rss_bytes'] for c in parent['children'])),production_launched=False,
                  tests=parent['tests'],preflight_wall_time_s=parent['wall_time_s'],not_run=parent['not_run'])
            report['runtime_table'] = []
            for c in parent['children']:
                  d = json.loads(Path(c['path']).read_text())
                  init,nom = d['phases']['initialization'],d['phases']['nominal']
                  for method,seconds in (('shipping SAA',init['saa_initialization']['wall_time_s']),
                        ('initialization total',init['wall_time_s']),('nominal two epochs',nom['training']['wall_time_s']),
                        ('training-only replay',nom['training_only_replay']['wall_time_s']),('child invocation',c['wall_time_s'])):
                        report['runtime_table'].append(dict(Ntotal=c['Ntotal'],seed=0,lr_net=.01,method=method,
                              status='completed',preflight_only=True,wall_time_s=core.stats([seconds]),
                              human=f'{seconds:.6f} s / {seconds/60:.6f} min'))
            report['not_run_runtime_table'] = {method:None for method in
                  ('production nominal training','robust neural/CCG','SRO-FA','validation/test/CV')}
      with report_path.open('x') as stream:
            json.dump(core._json_value(report),stream,indent=2,allow_nan=False)
      return report


if __name__=='__main__':
      import argparse
      p = argparse.ArgumentParser()
      p.add_argument('--preflight',type=Path,required=True)
      p.add_argument('--report',type=Path,required=True)
      args = p.parse_args()
      result = review(args.preflight,args.report)
      print('VALIDATED',result['preflight_sha256'],'nested rejections',len(result['rejections']))