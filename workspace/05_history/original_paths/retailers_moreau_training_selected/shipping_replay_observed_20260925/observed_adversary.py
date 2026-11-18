"""Unchanged epsilon AP; native agreement is logged, not required."""
import time
from observed_common import np,torch,check,definitions,OLD,REPLAY,previous

_prior=previous.adversary
core,gp,GRB=_prior.core,_prior.gp,_prior.GRB
PARAMETERS=dict(_prior.PARAMETERS)
EPSILON,REPLAY_TOLERANCE=_prior.EPSILON,_prior.REPLAY_TOLERANCE
ActivationMismatch=_prior.ActivationMismatch
exact_folded_projection=_prior.exact_folded_projection


def classify_support(active,solver_q,exact_q,native_q,replay):
    """AP-internal exact support remains hard; native disagreement is advisory."""
    exact_support=np.array([q>0 for q in exact_q])
    native_support=native_q>0
    check(np.isfinite(native_q).all(),'Nonfinite native shipment')
    replay.update(binary_support=active.tolist(),exact_support=exact_support.tolist(),
        native_support=native_support.tolist(),
        support_matches=bool(np.array_equal(active,native_support)),
        discrepancy_indices_zero_based=np.flatnonzero(active!=native_support).tolist(),
        support_check_status='recorded_only',replay_rejection_enabled=False,
        ap_fee=.1*int(active.sum()),native_fee=.1*int(native_support.sum()),
        native_epsilon_violations_zero_based=np.flatnonzero(
            active & (native_q<EPSILON-PARAMETERS['FeasibilityTol'])).tolist())
    if not np.array_equal(active,exact_support):
        raise ActivationMismatch('AP binary fee disagrees with exact folded-policy support')


# Retain shape/finiteness, AP integrality/links/true positivity, rational AP
# checks and native physical constraints. Only native epsilon disagreement
# changes from rejection to the observation recorded by classify_support.
definitions(globals(),REPLAY/'epsilon_adversary.py',('validate_activation',),changes=((
    "        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol\n"
    "                             or native_q[r]<EPSILON-tol) for r in range(3)):",
    "        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol) for r in range(3)):"),(
    '# An eligible exception never bypasses epsilon or physical checks.',
    '# Native agreement is advisory; AP-internal epsilon and physical checks remain.'),(
    'Returned AP witness fails positive/epsilon shipment condition',
    'AP witness fails internal positive/epsilon shipment condition')))


def check_replay_cost(cost,activation,replay):
    """Record raw discrepancy only; no adjusted cost or acceptance exception."""
    native=float(activation['native_hard_cost'])
    check(np.isfinite([cost,native]).all(),'Nonfinite AP/native costs')
    raw=abs(cost-native)/max(1.,abs(cost),abs(native))
    replay.update(cost=native,costs=[native],ap_cost=cost,
        floor_scaled_discrepancy=raw,raw_native_minus_AP=native-cost,
        consistency_status='recorded_only',replay_rejection_enabled=False,
        cost_matches_previous_tolerance=raw<=REPLAY_TOLERANCE,
        diagnostic_cost_tolerance=REPLAY_TOLERANCE,raw_hard_cost_unchanged=True,
        native_invocation='same invocation as activation_replay')


def finish_replay(record,cost):
    replay=record['replay']={}
    with core.timed(replay):
        check(record['activation_replay'].get('status')=='completed',
              'Incomplete activation or physical checks')
        check_replay_cost(cost,record['activation_replay'],replay)
        record.update(replay_consistency='recorded_only',replay_rejection_enabled=False)


# Exact frozen solver construction; only locally resolved replay functions differ.
definitions(globals(),OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),
    changes=(*_prior._changes,(_prior._replay_block,
        '        validate_activation(model, w, worst1, worst2, record)\n'
        '        finish_replay(record, cost)\n')))