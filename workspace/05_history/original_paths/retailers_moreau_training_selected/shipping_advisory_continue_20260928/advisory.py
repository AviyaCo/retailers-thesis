"""Activation consistency and exact-duplicate separation are advisory, by approval."""
from fractions import Fraction
import sys
import time
from continue_common import parent,np,torch,check,definitions

prior=parent.adversary
core,gp,GRB=prior.core,prior.gp,prior.GRB
PARAMETERS=dict(prior.PARAMETERS)
EPSILON,REPLAY_TOLERANCE=prior.EPSILON,prior.REPLAY_TOLERANCE
PHYSICAL_TOLERANCE=1e-6
ActivationMismatch,exact_folded_projection=prior.ActivationMismatch,prior.exact_folded_projection
check_replay_cost,finish_replay=prior.check_replay_cost,prior.finish_replay
OBSERVED=parent.ROOT.parent/'shipping_replay_observed_20260925'

definitions(globals(),OBSERVED/'observed_adversary.py',('classify_support',),changes=((
    "    if not np.array_equal(active,exact_support):\n"
    "        raise ActivationMismatch('AP binary fee disagrees with exact folded-policy support')",
    "    replay.update(exact_support_matches=bool(np.array_equal(active,exact_support)),\n"
    "        exact_support_rejection_enabled=False)"),))

definitions(globals(),prior.REPLAY/'epsilon_adversary.py',('validate_activation',),changes=(
    *prior._changes,
    ("        check(link_residual<=tol,'Shipping activation link violation')",
     "        replay.update(activation_link_guard_passed=bool(link_residual<=tol),\n"
     "            activation_link_rejection_enabled=False)"),
    ("        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol) for r in range(3)):\n"
     "            raise ActivationMismatch('AP witness fails internal positive/epsilon shipment condition')",
     "        replay.update(ap_positive_epsilon_matches=not any(active[r] and\n"
     "            (q[r]<=0 or float(exact_q[r])<EPSILON-tol) for r in range(3)),\n"
     "            ap_positive_epsilon_rejection_enabled=False)"),
    ("        check(native_q.min()>=-tol and native_q.sum()<=w['Q10']+tol,\n"
     "              'Native AP witness violates physical constraints')",
     "        replay.update(native_physical_tolerance=PHYSICAL_TOLERANCE,\n"
     "            native_physical_residual=max(0.,float(-native_q.min()),float(native_q.sum()-w['Q10'])),\n"
     "            native_physical_guard_passed=bool(native_q.min()>=-PHYSICAL_TOLERANCE\n"
     "                and native_q.sum()<=w['Q10']+PHYSICAL_TOLERANCE))\n"
     "        check(replay['native_physical_guard_passed'],'Native AP witness violates physical constraints')")))
definitions(globals(),prior.OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),
    changes=(*prior.prior.prior._prior._changes,(prior.prior.prior._prior._replay_block,
        '        validate_activation(model, w, worst1, worst2, record)\n'
        '        finish_replay(record, cost)\n')))

SEPARATION_TOLERANCE=parent.legacy_ccg.SEPARATION_TOLERANCE
MEMBERSHIP_TOLERANCE=parent.legacy_ccg.MEMBERSHIP_TOLERANCE
state_hash,pool_hash=parent.legacy_ccg.state_hash,parent.legacy_ccg.pool_hash
hard_objective,restricted_objective=parent.legacy_ccg.hard_objective,parent.legacy_ccg.restricted_objective
check_membership,point_membership=parent.legacy_ccg.check_membership,parent.legacy_ccg.point_membership
_finite=parent.legacy_ccg._finite
adversary=sys.modules[__name__]
definitions(globals(),OBSERVED/'observed_pool.py',('observed_replay',))
definitions(globals(),prior.OLD/'ccg.py',('validate_ap',),changes=((
    'if discrepancy > tol or abs(discrepancy - reported) > 1e-12:',
    'if abs(discrepancy - reported) > 1e-12 or not observed_replay(row):'),(
    "    if raw_gap < -tol * scale:\n        raise ValueError('Material negative bound/pool gap')",
    "    row.update(negative_bound_pool_rejection_enabled=False,\n"
    "        bound_pool_consistency_status='recorded_only',material_negative_bound_pool_gap=bool(raw_gap < -tol * scale),\n"
    "        exact_duplicate_separation_rejection_enabled=False,\n"
    "        exact_duplicate_material_separation=bool(row['membership']['exact_duplicate']\n"
    "            and row['residual']>SEPARATION_TOLERANCE))"),(
    "        if row['membership']['exact_duplicate'] or row['membership']['float32_collision']:\n"
    "            raise ValueError('Duplicate/colliding point with material separation residual')",
    "        if row['membership']['float32_collision']:\n"
    "            raise ValueError('Float32-colliding point with material separation residual')")))