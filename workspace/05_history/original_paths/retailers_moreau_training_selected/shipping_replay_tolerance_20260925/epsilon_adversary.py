"""Tolerance only for explained replay residues; no cost or shipment changes."""
import time
from fractions import Fraction as F
from types import SimpleNamespace
from epsilon_common import (core as shipping_core,np,torch,check,definitions,OLD,
    DIAGNOSTIC,PREVIOUS,EPSILON,REPLAY_ZERO_TOLERANCE)

gp,GRB=shipping_core.gp,shipping_core.GRB
core=SimpleNamespace(timed=shipping_core.timed,utc_now=shipping_core.utc_now,
    _json_value=shipping_core._json_value,baseline=SimpleNamespace(cfg=shipping_core.cfg))
PARAMETERS=dict(Threads=1,MIPGap=1e-4,FeasibilityTol=1e-9,
    OptimalityTol=1e-9,IntFeasTol=1e-9,NumericFocus=3,OutputFlag=0)
REPLAY_TOLERANCE=1e-4
definitions(globals(),DIAGNOSTIC,('binary','project'))
definitions(globals(),PREVIOUS/'epsilon_adversary.py',('ActivationMismatch','exact_folded_projection'))


def classify_support(active,solver_q,exact_q,native_q,replay):
    """Classify only AP/exact-zero to native-tiny-positive discrepancies."""
    exact_support=np.array([q>0 for q in exact_q])
    native_support=native_q>0
    different=np.flatnonzero(active!=native_support)
    replay.update(exact_support=exact_support.tolist(),native_support=native_support.tolist(),
        binary_support=active.tolist(),support_matches=bool(np.array_equal(active,native_support)),
        discrepancy_indices_zero_based=different.tolist(),
        consistency_zero_tolerance=REPLAY_ZERO_TOLERANCE,
        support_check_status='failed',expected_fee_delta=0.)
    if not np.array_equal(active,exact_support):
        raise ActivationMismatch('AP binary fee disagrees with exact folded-policy support')
    if not np.isfinite(native_q).all():
        raise ActivationMismatch('Nonfinite native shipment')
    for r in different:
        if not (not active[r] and solver_q[r]==0 and exact_q[r]==0
                and 0<native_q[r]<=REPLAY_ZERO_TOLERANCE):
            raise ActivationMismatch('Support mismatch is not an AP/exact-zero native residue')
    replay.update(support_check_status='exception_pending_cost_check' if len(different) else 'passed',
        exception_reason='native_positive_residue_at_exact_inactive_AP_component' if len(different) else None,
        expected_fee_delta=.1*len(different),ap_fee=.1*int(active.sum()),
        native_fee=.1*int(native_support.sum()))


def validate_activation(model,w,worst1,worst2,record):
    replay=record.setdefault('activation_replay',{})
    with core.timed(replay):
        inc=record['incumbent']
        z,q,mu,target=(np.asarray(inc[k],dtype=np.float64) for k in ('zship','q2','mu','target'))
        lam=float(inc['lambda'][0])
        check(all(v.shape==(3,) for v in (z,q,mu,target)),'Malformed AP support evidence')
        check(np.isfinite([*z,*q,*mu,*target,lam]).all(),'Nonfinite AP support evidence')
        active=z>.5
        exact_target,exact_q=exact_folded_projection(w,worst1)
        replay['preexisting_projection_calls']=model.proj_layer.drain_summary()
        try:
            with torch.no_grad():
                smooth,hard,native_q=model.costs_and_shipments(
                    torch.tensor(worst1[None,:],dtype=torch.float32),
                    torch.tensor(worst2[None,:],dtype=torch.float32))
        finally:
            replay['projection']=model.proj_layer.drain_summary()
        check(smooth.shape==hard.shape==(1,) and native_q.shape==(1,3),'Malformed native replay')
        check(bool(torch.isfinite(smooth).all() and torch.isfinite(hard).all()
                   and torch.isfinite(native_q).all()),'Nonfinite native replay')
        native_q=native_q.detach().cpu().numpy()[0]
        tol=PARAMETERS['FeasibilityTol']
        integer_residual=float(np.max(np.abs(z-np.rint(z))))
        link_residual=max(0.,float((EPSILON*z-q).max()),float((q-w['Q10']*z).max()))
        replay.update(binary_values=z.tolist(),solver_q=q.tolist(),
            exact_target=[str(v) for v in exact_target],exact_q=[str(v) for v in exact_q],
            native_q=native_q.tolist(),native_hard_cost=float(hard.item()),
            native_smooth_cost=float(smooth.item()),link_residual=link_residual,
            integer_residual=integer_residual,
            stationarity_residual=float(np.max(np.abs(2*q-2*target+lam-mu))),
            complementary_product_residual=max(abs(lam*(w['Q10']-q.sum())),float(np.max(np.abs(mu*q)))),
            projection_error=max(abs(float(v)-u) for v,u in zip(exact_q,q)),
            epsilon=EPSILON,epsilon_feasibility_tolerance=tol,
            unchanged_native_shipments=True,unchanged_native_hard_cost=True,original_box_certificate=False)
        check(integer_residual<=PARAMETERS['IntFeasTol'] and ((z>=0)&(z<=1)).all(),
              'Nonbinary shipping flags')
        check(link_residual<=tol,'Shipping activation link violation')
        classify_support(active,q,exact_q,native_q,replay)
        # An eligible exception never bypasses epsilon or physical checks.
        if any(active[r] and (q[r]<=0 or float(exact_q[r])<EPSILON-tol
                             or native_q[r]<EPSILON-tol) for r in range(3)):
            raise ActivationMismatch('Returned AP witness fails positive/epsilon shipment condition')
        check(native_q.min()>=-tol and native_q.sum()<=w['Q10']+tol,
              'Native AP witness violates physical constraints')


def check_replay_cost(cost,activation,replay):
    """Fee subtraction is explanatory arithmetic only; raw costs stay intact."""
    native=float(activation['native_hard_cost'])
    check(np.isfinite([cost,native]).all(),'Nonfinite AP/native costs')
    raw=abs(cost-native)/max(1.,abs(cost),abs(native))
    replay.update(cost=native,costs=[native],ap_cost=cost,
        floor_scaled_discrepancy=raw,raw_native_minus_AP=native-cost,
        consistency_status='failed',raw_hard_cost_unchanged=True,
        native_invocation='same invocation as activation_replay')
    if activation['support_check_status']=='passed':
        check(raw<=REPLAY_TOLERANCE,'Unexplained AP/native cost discrepancy')
        replay['consistency_status']='passed'
        return
    check(activation['support_check_status']=='exception_pending_cost_check',
          'Unvalidated support mismatch cannot bypass cost replay')
    delta=activation['expected_fee_delta']
    fee_free_ap=cost-activation['ap_fee']
    fee_free_native=native-activation['native_fee']
    residual=native-cost-delta
    scaled=abs(residual)/max(1.,abs(fee_free_ap),abs(fee_free_native))
    replay.update(expected_fee_delta=delta,fee_free_AP_cost=fee_free_ap,
        fee_free_native_cost=fee_free_native,unexplained_cost_residual=residual,
        unexplained_floor_scaled_discrepancy=scaled,remaining_cost_tolerance=REPLAY_TOLERANCE,
        consistency_zero_tolerance=REPLAY_ZERO_TOLERANCE,
        discrepancy_indices_zero_based=activation['discrepancy_indices_zero_based'],
        exception_reason=activation['exception_reason'])
    check(scaled<=REPLAY_TOLERANCE,'Residue fee does not explain AP/native cost discrepancy')
    replay['consistency_status']='accepted_with_replay_tolerance'


def finish_replay(record,cost):
    replay=record['replay']={}
    with core.timed(replay):
        activation=record['activation_replay']
        check(activation.get('status')=='completed','Incomplete activation or physical checks')
        check_replay_cost(cost,activation,replay)
        record['replay_consistency']=replay['consistency_status']
        record['consistency_exception']=replay['consistency_status']=='accepted_with_replay_tolerance'


# Reuse the exact previous solver-construction adaptations. Replace only its
# validator call's resolution (local function above) and the later replay block.
_old=(OLD/'adversary.py').read_text()
_start=_old.index("        replay = record['replay'] = {}")
_end=_old.index('        return worst1, worst2, cost',_start)
_replay_block=_old[_start:_end]
# Execute only the previous adapter invocation, not its module imports/functions.
import ast
_tree=ast.parse((PREVIOUS/'epsilon_adversary.py').read_text())
_calls=[node for node in _tree.body if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
        and node.value.func.id=='definitions' and len(node.value.args)>2
    and isinstance(node.value.args[2],ast.Tuple)
    and any(kw.arg=='changes' for kw in node.value.keywords)]
check(len(_calls)==1,'Expected one frozen AP-definition adapter')
_kwargs={kw.arg:ast.literal_eval(kw.value) for kw in _calls[0].value.keywords}
_changes=tuple((old,new) for old,new in _kwargs['changes']
    if old not in ("        replay = record['replay'] = {}",
                   'native = model.per_sample_cost(replay1, replay2)'))
definitions(globals(),OLD/'adversary.py',('_load_weights','_solver_snapshot','solve_ap'),
    changes=(*_changes,(_replay_block,
        '        validate_activation(model, w, worst1, worst2, record)\n'
        '        finish_replay(record, cost)\n')))