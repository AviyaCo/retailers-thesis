"""Separate smooth MP and native-hard pool costs; retain signed AP gates."""
from fractions import Fraction
import hashlib
from campaign_common import (np,torch,core,adversary,check,definitions,OLD)

SEPARATION_TOLERANCE=.01
MEMBERSHIP_TOLERANCE=1e-9
definitions(globals(),OLD/'ccg.py',('state_hash','pool_hash','check_membership',
    'point_membership','restricted_objective','_finite'))


def hard_objective(pools):
    """Native hard maximum per original pool; do not clip q or adjust fees."""
    sizes=[len(pool) for pool in pools]
    points=np.concatenate(pools,axis=0)
    x1=torch.tensor(points[:,:3],dtype=torch.float32)
    x2=torch.tensor(points[:,3:],dtype=torch.float32)

    def per_ball(model):
        costs=model.hard_shipping_cost(x1,x2)
        check(costs.shape==(len(points),) and bool(torch.isfinite(costs).all()),
              'Nonfinite/malformed hard pool costs')
        return torch.stack([torch.max(group) for group in torch.split(costs,sizes)])
    return per_ball


def approved_replay(row):
    """Recompute exception eligibility and arithmetic, never trust a flag alone."""
    activation=row['activation_replay']
    check(activation.get('status')=='completed','Incomplete activation/physical checks')
    inc=row['incumbent']
    check(activation['solver_q']==inc['q2'] and activation['binary_values']==inc['zship'],
          'Replay support is not bound to AP incumbent')
    recomputed={}
    adversary.classify_support(np.asarray(inc['zship'])>.5,
        np.asarray(inc['q2']),tuple(Fraction(q) for q in activation['exact_q']),
        np.asarray(activation['native_q']),recomputed)
    for k,v in recomputed.items():
        check(activation[k]==v,f'Changed activation evidence: {k}')
    recomputed['native_hard_cost']=activation['native_hard_cost']
    costcheck={}
    adversary.check_replay_cost(inc['cost'],recomputed,costcheck)
    for k,v in costcheck.items():
        check(row['replay'][k]==v,f'Changed replay cost evidence: {k}')
    check(row['replay_consistency']==costcheck['consistency_status'],
          'Changed replay acceptance status')
    check(row['consistency_exception']==(costcheck['consistency_status']=='accepted_with_replay_tolerance'),
          'Changed replay exception marker')
    return True


# Only the raw replay acceptance branch changes. All following signed ordering,
# membership, duplicate and unresolved-bound checks are the frozen definitions.
definitions(globals(),OLD/'ccg.py',('validate_ap',),changes=((
    'if discrepancy > tol or abs(discrepancy - reported) > 1e-12:',
    'if abs(discrepancy - reported) > 1e-12 or not approved_replay(row):'),))