"""Epsilon-restricted shipping AP; explicit support checks, no silent repairs."""
import time
from fractions import Fraction as F
from types import SimpleNamespace
from epsilon_common import core as shipping_core, np, torch, check, definitions, OLD, DIAGNOSTIC, EPSILON

gp, GRB = shipping_core.gp, shipping_core.GRB
# A private namespace supplies the original AP's economics without modifying
# the imported shipping model or any on-disk frozen source.
core = SimpleNamespace(timed=shipping_core.timed, utc_now=shipping_core.utc_now,
    _json_value=shipping_core._json_value, baseline=SimpleNamespace(cfg=shipping_core.cfg))
PARAMETERS = dict(Threads=1,MIPGap=1e-4,FeasibilityTol=1e-9,
    OptimalityTol=1e-9,IntFeasTol=1e-9,NumericFocus=3,OutputFlag=0)
REPLAY_TOLERANCE = 1e-4
definitions(globals(), DIAGNOSTIC, ('binary','project'))


class ActivationMismatch(ValueError):
    pass


def exact_folded_projection(w, x1):
    """Replay the real interpretation of the MIP's stored coefficients.

    Reuses the tested standard R3 projection formula with Fraction arithmetic.
    This verifies one returned point, not a global solver bound or native policy.
    """
    values = [binary(v) for v in x1]
    for index in (1,2,3):
        matrix, bias = w[f'W{index}'], w[f'b{index}']
        values = [binary(b) + sum(binary(a)*v for a,v in zip(row,values))
                  for row,b in zip(matrix,bias)]
        if index < 3:
            values = [max(F(0),v) for v in values]
    return values, project(tuple(values),binary(w['Q10']))


def validate_activation(model, w, worst1, worst2, record):
    """Reject ghost fees and componentwise real/native support disagreement."""
    replay = record.setdefault('activation_replay',{})
    with core.timed(replay):
        incumbent = record['incumbent']
        z = np.asarray(incumbent['zship'])
        q = np.asarray(incumbent['q2'])
        mu = np.asarray(incumbent['mu'])
        target = np.asarray(incumbent['target'])
        lam = float(incumbent['lambda'][0])
        check(np.isfinite([*z,*q,*mu,*target,lam]).all(),'Nonfinite AP support evidence')
        active = z > .5
        target_exact, q_exact = exact_folded_projection(w,worst1)
        exact_support = np.asarray([v>0 for v in q_exact])
        with torch.no_grad():
            smooth, hard, native_q = model.costs_and_shipments(
                torch.tensor(worst1[None,:],dtype=torch.float32),
                torch.tensor(worst2[None,:],dtype=torch.float32))
        native_q = native_q.detach().cpu().numpy()[0]
        native_support = native_q > 0
        tol = PARAMETERS['FeasibilityTol']
        integer_residual = float(np.max(np.abs(z-np.rint(z))))
        link_residual = max(0.,float((EPSILON*z-q).max()),float((q-w['Q10']*z).max()))
        stationarity = float(np.max(np.abs(2*q-2*target+lam-mu)))
        replay.update(binary_values=z.tolist(),solver_q=q.tolist(),
            exact_target=[str(v) for v in target_exact],exact_q=[str(v) for v in q_exact],
            exact_support=exact_support.tolist(),native_q=native_q.tolist(),
            native_support=native_support.tolist(),binary_support=active.tolist(),
            native_hard_cost=float(hard.item()),native_smooth_cost=float(smooth.item()),
            link_residual=link_residual,integer_residual=integer_residual,
            stationarity_residual=stationarity,
            complementary_product_residual=max(abs(lam*(w['Q10']-q.sum())),float(np.max(np.abs(mu*q)))),
            projection_error=max(abs(float(v)-u) for v,u in zip(q_exact,q)),
            minimum_active_exact_q=min((float(v) for v in q_exact if v>0),default=None),
            minimum_active_native_q=float(native_q[active].min()) if active.any() else None,
            epsilon=EPSILON,epsilon_feasibility_tolerance=tol,
            unchanged_native_shipments=True,original_box_certificate=False)
        check(integer_residual<=PARAMETERS['IntFeasTol'],'Nonintegral shipping flags')
        check(link_residual<=tol,'Shipping activation link violation')
        if not np.array_equal(active,exact_support):
            raise ActivationMismatch('AP binary fee disagrees with exact folded-policy support')
        if not np.array_equal(active,native_support):
            raise ActivationMismatch('AP binary fee disagrees with unchanged native shipment support')
        if any(active[r] and (q[r]<=0 or float(q_exact[r])<EPSILON-tol
                             or native_q[r]<EPSILON-tol) for r in range(3)):
            raise ActivationMismatch('Returned AP witness fails positive/epsilon shipment condition')
        check(np.isfinite(native_q).all() and native_q.min()>=-tol
              and native_q.sum()<=w['Q10']+tol,'Native AP witness violates physical constraints')
        replay['support_matches'] = True


definitions(globals(),OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),changes=(
    ("method='fixed-policy canonical AP'", "method='epsilon-restricted fixed-policy shipping AP'"),
    ("                v = solver.addVars(3, lb=-GRB.INFINITY, name='max3')",
     "                zship = solver.addVars(3, vtype=GRB.BINARY, name='zship')\n"
     "                for r in range(3):\n"
     "                    solver.addConstr(q2[r] <= q10*zship[r], name=f'shipping_upper_{r}')\n"
     "                    solver.addConstr(q2[r] >= EPSILON*zship[r], name=f'shipping_lower_{r}')\n"
     "                v = solver.addVars(3, lb=-GRB.INFINITY, name='max3')"),
    ("solver.setObjective(stage1 + gp.quicksum(v[r] for r in range(3)), GRB.MAXIMIZE)",
     "solver.setObjective(stage1 + gp.quicksum(v[r] + .1*zship[r] for r in range(3)), GRB.MAXIMIZE)"),
    ("target=target), record)","target=target, zship=zship, mu=mu, **{'lambda': {r: lam for r in range(3)}}), record)"),
    ("        replay = record['replay'] = {}",
     "        validate_activation(model, w, worst1, worst2, record)\n        replay = record['replay'] = {}"),
    ("native = model.per_sample_cost(replay1, replay2)",
     "native = model.hard_shipping_cost(replay1, replay2)"),
))