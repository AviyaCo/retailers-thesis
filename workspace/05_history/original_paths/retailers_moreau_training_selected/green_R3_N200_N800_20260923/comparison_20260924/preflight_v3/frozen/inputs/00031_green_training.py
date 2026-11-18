"""Train-only port: thresholded preupdate selection including incoming weights.

Every evaluated epoch, including the patience-triggering epoch, is updated before
restoration. The caller supplies Ntrain and all objective/input metadata. No
validation evaluation, old trainer, optimizer reuse, or numerical fallback.
"""
from collections import Counter
import json
from pathlib import Path
import random
import time

from green_common import core, np, torch


def train_model(model, loss_fn, lr, folder, record, epochs=3000, patience=300):
    """Restore training-best; retain raw epoch journal and failure snapshots."""
    n_train = record['Ntrain']
    if n_train not in (160, 640):
        raise ValueError('Require Ntrain=160 or 640 in caller record')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / 'epochs.jsonl').open('x')
    optimizer = scheduler = reference = best_state = None
    no_improvement = 0
    history = []
    step_times = []
    projection_statuses = Counter()
    projection = getattr(model, 'proj_layer', None)
    phase_names = ('forward_time_s', 'backward_time_s', 'update_time_s',
                   'checkpoint_time_s', 'optimizer_setup_time_s')
    record.update(n_train=n_train, lr_net=lr, lr_inventory=.01,
                  max_epochs=epochs, patience=patience, improvement_threshold=1e-4,
                  scheduler_metric='training_loss', scheduler_factor=.7,
                  scheduler_patience=150, scheduler_threshold=1e-3,
                  epoch_semantics='1-based preupdate; final evaluated epoch updated before restoration',
                  torch_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(), device='cpu',
                  timing_unit='seconds', timing_clock='time.perf_counter',
                  epochs_attempted=0, epochs_evaluated=0, optimizer_steps=0,
                  best_train=None, first_train=None, last_train=None, best_epoch=None,
                  selected_training_loss=None, no_improvement=0,
                  validation_time_s=0., projection_call_count=0,
                  projection_sample_count=0, projection_iterations=0,
                  projection_statuses={}, projection_iteration_count=0,
                  projection_iteration_min=None, projection_iteration_max=None,
                  epoch_history=history)
    record.update({key: 0. for key in phase_names})

    def drain():
        if not isinstance(projection, core.NativeProjection):
            return None
        summary = projection.drain_summary()
        projection_statuses.update(summary['statuses'])
        record['projection_statuses'] = dict(projection_statuses)
        record['projection_call_count'] += summary['count']
        record['projection_sample_count'] += summary['sample_count']
        iterations = summary['iterations']
        record['projection_iterations'] += iterations['total']
        record['projection_iteration_count'] += iterations['count']
        if iterations['count']:
            for suffix, operation in (('min', min), ('max', max)):
                key = 'projection_iteration_' + suffix
                previous = record[key]
                record[key] = (iterations[suffix] if previous is None
                               else operation(previous, iterations[suffix]))
        return summary

    def snapshot():
        return dict(model=core.weights(model),
                    optimizer=optimizer.state_dict() if optimizer is not None else None,
                    scheduler=scheduler.state_dict() if scheduler is not None else None,
                    rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                             python=random.getstate()),
                    candidate_reference_model_state=reference,
                    selected_model_state=best_state,
                    epochs_attempted=record['epochs_attempted'],
                    epochs_evaluated=record['epochs_evaluated'],
                    optimizer_steps=record['optimizer_steps'],
                    best_epoch=record['best_epoch'], best_train=record['best_train'],
                    first_train=record['first_train'], last_train=record['last_train'],
                    selected_training_loss=record['selected_training_loss'],
                    no_improvement=no_improvement)

    try:
        with stream, core.timed(record):
            training_start = time.perf_counter()
            try:
                with core._phase(record, {}, 'optimizer_setup_time_s'):
                    if not 1 <= epochs <= 3000 or patience < 1:
                        raise ValueError('Require 1..3000 evaluated epochs and positive patience')
                    reference = core.weights(model)
                    core._check_weights(model)
                    if isinstance(projection, core.NativeProjection):
                        record['projection_settings'] = projection.settings
                        record['preexisting_projection_calls'] = projection.drain_summary()
                    optimizer = torch.optim.Adam([
                        {'params': model.fc_layers.parameters(), 'lr': lr},
                        {'params': [model.Q10, model.Q1r], 'lr': .01}])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=.7, patience=150, threshold=1e-3)
                for epoch in range(1, epochs + 1):
                    record['epochs_attempted'] = epoch
                    entry = dict(epoch=epoch, status='running', started_at_utc=core.utc_now(),
                                 preupdate_optimizer_steps=record['optimizer_steps'],
                                 learning_rates_preupdate=[g['lr'] for g in optimizer.param_groups])
                    entry.update({key: 0. for key in phase_names})
                    epoch_start = time.perf_counter()
                    try:
                        model.train()
                        with core._phase(record, entry, 'forward_time_s'):
                            optimizer.zero_grad(set_to_none=True)
                            loss = loss_fn(model)
                            entry['train_loss'] = loss.detach().cpu().tolist()
                            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                                raise FloatingPointError('Training objective must be a finite scalar')
                            train_value = float(loss.detach())
                            record['last_train'] = train_value
                            if record['first_train'] is None:
                                record['first_train'] = train_value
                        record['epochs_evaluated'] += 1
                        improved = (record['best_train'] is None
                                    or train_value < record['best_train'] - 1e-4)
                        if improved:
                            with core._phase(record, entry, 'checkpoint_time_s'):
                                best_state = core.weights(model)
                            record.update(best_train=train_value, best_epoch=epoch,
                                          selected_training_loss=train_value)
                            no_improvement = 0
                        else:
                            no_improvement += 1
                        record['no_improvement'] = no_improvement
                        entry.update(improved=improved, best_epoch=record['best_epoch'],
                                     best_train=record['best_train'], no_improvement=no_improvement)
                        with core._phase(record, entry, 'backward_time_s'):
                            loss.backward()
                        with core._phase(record, entry, 'update_time_s'):
                            for name, parameter in model.named_parameters():
                                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                                    raise FloatingPointError(f'Nonfinite gradient: {name}')
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                                                 error_if_nonfinite=True)
                            entry['gradient_norm_preclip'] = float(norm)
                            optimizer.step()
                            record['optimizer_steps'] += 1
                            core._check_weights(model)
                            scheduler.step(train_value)
                        entry.update(status='completed',
                                     learning_rates=[g['lr'] for g in optimizer.param_groups])
                    except BaseException as error:
                        entry.update(status='failed',
                                     error={'type': type(error).__name__, 'message': str(error)})
                        raise
                    finally:
                        entry['projection'] = drain()
                        entry['optimizer_steps'] = record['optimizer_steps']
                        entry['train_step_time_s'] = sum(entry[key] for key in
                            ('forward_time_s', 'backward_time_s', 'update_time_s'))
                        if entry['status'] == 'completed':
                            step_times.append(entry['train_step_time_s'])
                        entry.update(ended_at_utc=core.utc_now(),
                                     wall_time_s=time.perf_counter() - epoch_start)
                        history.append(entry)
                        stream.write(json.dumps(core._json_value(entry), allow_nan=False) + '\n')
                        stream.flush()
                    if epoch % 100 == 0:
                        record['train_step_times'] = core.stats(step_times)
                        record['current_invocation_elapsed_time_s'] = time.perf_counter() - training_start
                        core.write_json(folder / 'training.json', record)
                    if no_improvement >= patience:
                        record['stop_reason'] = 'training_patience'
                        break
                else:
                    record['stop_reason'] = 'max_evaluated_epochs'
                with core._phase(record, {}, 'checkpoint_time_s'):
                    torch.save(snapshot(), folder / 'final_state.pth')
                    if best_state is None:
                        raise RuntimeError('No finite evaluated checkpoint')
                    torch.save(best_state, folder / 'selected.pth')
                    core._load_learned(model, best_state)
                    model.eval()
            except BaseException:
                with core._phase(record, {}, 'checkpoint_time_s'):
                    record['failure_projection'] = drain()
                    torch.save(snapshot(), folder / 'failure_state.pth')
                raise
    finally:
        record['train_step_times'] = core.stats(step_times)
        record['train_compute_time_s'] = sum(record[key] for key in
            ('forward_time_s', 'backward_time_s', 'update_time_s'))
        record['training_time_s'] = record['wall_time_s']
        record['current_invocation_elapsed_time_s'] = record['wall_time_s']
        record['timing_scope'] = ('current invocation: setup, all attempted epochs, journal writes, '
                                  'checkpoints and restoration; excludes this final JSON write')
        core.write_json(folder / 'training.json', record)
    return model