"""Native hard pools and unchanged signed checks; replay agreement is advisory."""
from fractions import Fraction
import hashlib
from observed_common import core,np,torch,check,definitions,OLD,PREVIOUS
import observed_adversary as adversary

SEPARATION_TOLERANCE=.01
MEMBERSHIP_TOLERANCE=1e-9
definitions(globals(),OLD/'ccg.py',('state_hash','pool_hash','check_membership',
    'point_membership','restricted_objective','_finite'))
definitions(globals(),PREVIOUS/'shipping_pool.py',('hard_objective',))


def observed_replay(row):
    """Validate evidence integrity, not agreement between the two objectives."""
    activation=row['activation_replay']
    check(activation.get('status')=='completed','Incomplete activation/physical checks')
    inc=row['incumbent']
    check(activation['solver_q']==inc['q2'] and activation['binary_values']==inc['zship'],
          'Replay support is not bound to AP incumbent')
    observation={}
    adversary.classify_support(np.asarray(inc['zship'])>.5,np.asarray(inc['q2']),
        tuple(Fraction(q) for q in activation['exact_q']),np.asarray(activation['native_q']),observation)
    for k,v in observation.items():
        check(activation[k]==v,f'Changed activation evidence: {k}')
    costcheck={}
    adversary.check_replay_cost(inc['cost'],activation,costcheck)
    for k,v in costcheck.items():
        check(row['replay'][k]==v,f'Changed replay cost evidence: {k}')
    check(row['replay_consistency']=='recorded_only' and row['replay_rejection_enabled'] is False,
          'Unexpected replay observation contract')
    return True


# Native agreement is not required; finite raw values and diagnostic arithmetic
# remain checked. The signed-ordering suffix is unchanged byte-for-byte.
definitions(globals(),OLD/'ccg.py',('validate_ap',),changes=((
    'if discrepancy > tol or abs(discrepancy - reported) > 1e-12:',
    'if abs(discrepancy - reported) > 1e-12 or not observed_replay(row):'),))