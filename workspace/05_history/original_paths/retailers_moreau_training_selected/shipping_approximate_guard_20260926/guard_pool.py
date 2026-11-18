"""Log cross-implementation negative gaps; retain all other controller checks."""
from guard_common import np,check,definitions,OLD,parent
import guard_adversary as adversary

_prior=parent.prior_pool
SEPARATION_TOLERANCE=_prior.SEPARATION_TOLERANCE
MEMBERSHIP_TOLERANCE=_prior.MEMBERSHIP_TOLERANCE
state_hash,pool_hash=_prior.state_hash,_prior.pool_hash
hard_objective,restricted_objective=_prior.hard_objective,_prior.restricted_objective
check_membership,point_membership=_prior.check_membership,_prior.point_membership
observed_replay,_finite=_prior.observed_replay,_prior._finite

definitions(globals(),OLD/'ccg.py',('validate_ap',),changes=((
    'if discrepancy > tol or abs(discrepancy - reported) > 1e-12:',
    'if abs(discrepancy - reported) > 1e-12 or not observed_replay(row):'),(
    "    if raw_gap < -tol * scale:\n"
    "        raise ValueError('Material negative bound/pool gap')",
    "    row.update(negative_bound_pool_rejection_enabled=False,\n"
    "        bound_pool_consistency_status='recorded_only',\n"
    "        material_negative_bound_pool_gap=bool(raw_gap < -tol * scale))")))