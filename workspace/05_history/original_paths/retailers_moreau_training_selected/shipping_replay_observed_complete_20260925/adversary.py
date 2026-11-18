"""Same acceptance contract; persist cost observation before physical rejection."""
import time
from common import np,torch,check,definitions,REPLAY,OLD,OBSERVED,prior_adversary as prior

core,gp,GRB=prior.core,prior.gp,prior.GRB
PARAMETERS,EPSILON=prior.PARAMETERS,prior.EPSILON
REPLAY_TOLERANCE=prior.REPLAY_TOLERANCE
ActivationMismatch=prior.ActivationMismatch
exact_folded_projection=prior.exact_folded_projection
classify_support=prior.classify_support
check_replay_cost=prior.check_replay_cost
finish_replay=prior.finish_replay

definitions(globals(),REPLAY/'epsilon_adversary.py',('validate_activation',),changes=((
    "        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol\n"
    "                             or native_q[r]<EPSILON-tol) for r in range(3)):",
    "        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol) for r in range(3)):"),(
    '# An eligible exception never bypasses epsilon or physical checks.',
    '# Native agreement is advisory; AP-internal epsilon and physical checks remain.'),(
    'Returned AP witness fails positive/epsilon shipment condition',
    'AP witness fails internal positive/epsilon shipment condition'),(
    "        check(integer_residual<=PARAMETERS['IntFeasTol'] and ((z>=0)&(z<=1)).all(),",
    "        with core.timed(record.setdefault('native_cost_observation',{})):\n"
    "            check_replay_cost(inc['cost'],replay,record['native_cost_observation'])\n"
    "        check(integer_residual<=PARAMETERS['IntFeasTol'] and ((z>=0)&(z<=1)).all(),")))

definitions(globals(),OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),
    changes=(*prior._prior._changes,(prior._prior._replay_block,
        '        validate_activation(model, w, worst1, worst2, record)\n'
        '        finish_replay(record, cost)\n')))