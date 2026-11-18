"""Only the native physical acceptance tolerance changes; raw policy is untouched."""
import ast
import time
from guard_common import (np,torch,check,definitions,REPLAY,OLD,PARENT,
    PHYSICAL_TOLERANCE,parent_adversary as prior)

core,gp,GRB=prior.core,prior.gp,prior.GRB
PARAMETERS,EPSILON=prior.PARAMETERS,prior.EPSILON
REPLAY_TOLERANCE=prior.REPLAY_TOLERANCE
ActivationMismatch=prior.ActivationMismatch
exact_folded_projection=prior.exact_folded_projection
classify_support=prior.classify_support
check_replay_cost=prior.check_replay_cost
finish_replay=prior.finish_replay

# Reuse the exact prior replay adaptations; keep AP epsilon/integrality
# tolerances at 1e-9 and retain complete cost observations on failed witnesses.
_tree=ast.parse((PARENT/'adversary.py').read_text())
_calls=[node.value for node in _tree.body if isinstance(node,ast.Expr)
    and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
    and node.value.func.id=='definitions'
    and ast.literal_eval(node.value.args[2])==('validate_activation',)]
check(len(_calls)==1,'Expected one parent activation adaptation')
_changes=ast.literal_eval(next(kw.value for kw in _calls[0].keywords if kw.arg=='changes'))
definitions(globals(),REPLAY/'epsilon_adversary.py',('validate_activation',),changes=(*_changes,(
    "        check(native_q.min()>=-tol and native_q.sum()<=w['Q10']+tol,\n"
    "              'Native AP witness violates physical constraints')",
    "        replay.update(native_physical_tolerance=PHYSICAL_TOLERANCE,\n"
    "            native_physical_residual=max(0.,float(-native_q.min()),float(native_q.sum()-w['Q10'])),\n"
    "            native_physical_guard_passed=bool(native_q.min()>=-PHYSICAL_TOLERANCE\n"
    "                and native_q.sum()<=w['Q10']+PHYSICAL_TOLERANCE))\n"
    "        check(replay['native_physical_guard_passed'],\n"
    "              'Native AP witness violates physical constraints')")))

definitions(globals(),OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),
    changes=(*prior.prior._prior._changes,(prior.prior._prior._replay_block,
        '        validate_activation(model, w, worst1, worst2, record)\n'
        '        finish_replay(record, cost)\n')))