"""Live-green shipping controller: hard stopping, separate smooth MP replays."""
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
from shipping_live_common import (MAX_MP,PROTOCOL,adversary,check,core,legacy_ccg,
    metadata,np,torch,verify_sources,definitions,GREEN,shipping_training)

train_model=shipping_training.train_model
restricted_objective=legacy_ccg.restricted_objective
hard_objective=legacy_ccg.hard_objective
validate_ap=legacy_ccg.validate_ap
check_membership=legacy_ccg.check_membership
point_membership=legacy_ccg.point_membership
pool_hash,state_hash=legacy_ccg.pool_hash,legacy_ccg.state_hash

definitions(globals(),GREEN/'green_ccg.py',('_error','_finite','terminal_decision',
    'choose_candidate','_copy','_verify_checkpoint','_save_pool'))
definitions(globals(),GREEN/'green_ccg.py',('_pool_mean',),changes=((
    'restricted_objective(pools)(model)','hard_objective(pools)(model)'),))


def _smooth_pool_mean(model,pools,record):
    """Training-loss replay only, never the hard green-gap denominator."""
    frozen=state_hash(model)
    with core.timed(record),torch.no_grad():
        model.eval()
        values=restricted_objective(pools)(model).detach().cpu().numpy()
        check(values.shape==(len(pools),) and np.isfinite(values).all(),'Invalid smooth pool costs')
        record.update(mean=float(values.mean()),per_ball_costs=values.tolist(),
            objective='smooth shipping surrogate',state_sha256=frozen,
            projection=model.proj_layer.drain_summary())
        check(state_hash(model)==frozen,'Smooth evaluation changed model')
    return values


definitions(globals(),GREEN/'green_ccg.py',('_flush',),changes=((
    "c.get('evaluation', {}).get('wall_time_s', 0.)",
    "c.get('evaluation', {}).get('wall_time_s', 0.)\n"
    "                      + c.get('smooth_training_replay', {}).get('wall_time_s', 0.)"),))
definitions(globals(),GREEN/'green_ccg.py',('assess_candidate',),changes=((
    "    candidate.update(pool_costs=maxima.tolist(), restricted_mean=evaluation['mean'])",
    "    evaluation.update(objective='native strict-positive hard shipping cost',\n"
    "        projection=model.proj_layer.drain_summary())\n"
    "    candidate.update(pool_costs=maxima.tolist(), restricted_mean=evaluation['mean'])"),(
    "            candidate.update(assessed=True,",
    "            candidate.update(material_negative_bound_pool_gaps=sum(r['material_negative_bound_pool_gap'] for r in candidate['samples']),\n"
    "                native_support_disagreements=sum(not r['activation_replay']['support_matches'] for r in candidate['samples']),\n"
    "                native_cost_disagreements=sum(not r['native_cost_observation']['cost_matches_previous_tolerance'] for r in candidate['samples']),\n"
    "                original_box_certificate=False)\n"
    "            candidate.update(assessed=True,")))
definitions(globals(),GREEN/'green_ccg.py',('run_configuration',),changes=((
    "nominal_origin_contract='parent supplies fresh per-seed training-only LR-search winner'",
    "nominal_origin_contract='cached shipping smooth fits; parent hard-validation LR selection'"),(
    "            if candidate_id == 0:\n"
    "                check(abs(candidate['restricted_mean'] - record['nominal_expected_train']) <= 1e-6,",
    "            smooth_replay=candidate['smooth_training_replay']={}\n"
    "            _smooth_pool_mean(model,pools,smooth_replay)\n"
    "            if candidate_id == 0:\n"
    "                check(abs(smooth_replay['mean'] - record['nominal_expected_train']) <= 1e-6,"),(
    "check(abs(candidate['restricted_mean'] - previous_training['selected_training_loss']) <= 1e-6,",
    "check(abs(smooth_replay['mean'] - previous_training['selected_training_loss']) <= 1e-6,"),(
    '_pool_mean(model, pools, incoming)','_smooth_pool_mean(model, pools, incoming)'),(
    '_pool_mean(model, pools, returned)','_smooth_pool_mean(model, pools, returned)')))