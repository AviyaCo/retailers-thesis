"""Approved paired data protocol and training-only shipping SAA MILP."""
import shipping_core as core
from shipping_core import np, torch, truncnorm, gp, GRB, check
from source_adapter import GREEN, load_into

SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
LR_GRID = (.005, .01, .05)
PROTOCOL = dict(experiment='shipping_nominal_R3_N50_N200_N800_20260924',
    sizes=list(SIZES), data_seeds=list(SEEDS), network_seed=0, split_seed='100000+s',
    R=3, f=.1, gamma=5., c=.25, h=.05, b=.5, hidden=[16,16],
    train_fraction=.8, batch='full training set', network_dtype='float32',
    projection_dtype='float64', capacity_buffer=0., shipment_clipping=False,
    hard_fee='0.1 per native returned q2 strictly > 0; no epsilon',
    epochs=3000, patience=300, lr_grid=list(LR_GRID), inventory_lr=.01,
    scheduler=dict(factor=.7, patience=150, threshold=1e-3, metric='training_loss'),
    gradient_clip=1., improvement_threshold=1e-4, checkpoint='training-best preupdate',
    fresh_Adam_per_arm=True, initialization_barrier=30, nominal_arms=90,
    not_run=['LR selection','validation','test','CCG','AP','radii','refit','SRO-FA','CV'],
    resume=False, fallback=False, exact_real_arithmetic_certified=False)


def metadata(n_total, seed=None, method=None):
    check(type(n_total) is int and n_total in SIZES, 'Unapproved size')
    return dict(experiment=PROTOCOL['experiment'], method=method, Ntotal=n_total,
        Ntrain=4*n_total//5, Nval=n_total//5, seed=seed, network_seed=0,
        split_seed=None if seed is None else 100000+seed, R=3, f=.1, gamma=5.,
        torch_threads=1, gurobi_threads=1, timing_clock='time.perf_counter', timing_unit='seconds')


load_into(globals(), GREEN / 'green_common.py', ('split_data',))


def solve_saa(train1, train2, record):
    """Minimum MILP; positive binary penalty needs upper link only, no epsilon."""
    record.update(solver_status=None, solver_runtime_s=None, iterations=None,
                  node_count=None, solution_count=None, solution_availability='not optimized')
    with core.timed(record):
        x1, x2 = np.asarray(train1, dtype=np.float64), np.asarray(train2, dtype=np.float64)
        n = len(x1)
        check(n in (40,160,640) and x1.shape == x2.shape == (n,3), 'Train-only SAA shape')
        check(np.isfinite(x1).all() and np.isfinite(x2).all() and
              (x1 >= 0).all() and (x2 >= 0).all(), 'Invalid SAA demand')
        big_m = float((x1+x2).max()) + 1.
        record.update(n_train=n, big_m=big_m, method='training-only shipping SAA MILP', threads=1)
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with gp.Model(env=env) as m:
                m.Params.Threads = 1
                q10 = m.addVar(lb=0., name='Q10')
                q1 = m.addVars(3, lb=0., name='Q1r')
                q2 = m.addVars(n,3,lb=0.,name='Q2')
                z = m.addVars(n,3,vtype=GRB.BINARY,name='z')
                v = m.addVars(n,3,lb=-GRB.INFINITY,name='v')
                for i in range(n):
                    m.addConstr(gp.quicksum(q2[i,r] for r in range(3)) <= q10)
                    for r in range(3):
                        m.addConstr(v[i,r] >= .5*(x1[i,r]+x2[i,r]-q2[i,r]-q1[r])-.05*q2[i,r])
                        m.addConstr(v[i,r] >= .05*(q1[r]-x1[i,r]-x2[i,r]))
                        m.addConstr(v[i,r] >= .5*(x1[i,r]-q1[r])-.05*x2[i,r])
                        m.addConstr(q2[i,r] <= big_m*z[i,r])
                m.setObjective(.25*(q10+gp.quicksum(q1[r] for r in range(3)))+.05*q10+
                    gp.quicksum(v[i,r]+.1*z[i,r] for i in range(n) for r in range(3))/n, GRB.MINIMIZE)
                try:
                    m.optimize()
                finally:
                    record.update(solver_status=int(m.Status), solver_runtime_s=float(m.Runtime),
                        iterations=float(m.IterCount), node_count=float(m.NodeCount), solution_count=int(m.SolCount))
                    if m.SolCount > 0:
                        a = np.array([[q2[i,r].X for r in range(3)] for i in range(n)])
                        b = np.array([[z[i,r].X for r in range(3)] for i in range(n)])
                        vv = np.array([[v[i,r].X for r in range(3)] for i in range(n)])
                        inventory = np.array([q1[r].X for r in range(3)])
                        terms = np.maximum.reduce([.5*(x1+x2-a-inventory)-.05*a,
                            .05*(inventory-x1-x2), .5*(x1-inventory)-.05*x2])
                        record.update(solution_availability='incumbent', q10=float(q10.X), q1r=inventory,
                            objective=float(m.ObjVal), allocations=a, binaries=b, epigraph=vv,
                            residuals=dict(nonneg=float(max(0.,-a.min(),-inventory.min(),-q10.X)),
                                capacity=float(max(0.,(a.sum(1)-q10.X).max())),
                                upper_link=float(max(0.,(a-big_m*b).max())),
                                binary=float(np.abs(b-np.round(b)).max()),
                                epigraph=float(max(0.,(terms-vv).max()))))
                    else:
                        record['solution_availability'] = 'no incumbent; X/ObjVal unavailable and not read'
                check(m.Status == GRB.OPTIMAL, f'SAA requires OPTIMAL, got {m.Status}')
                check(np.isfinite([record['q10'],*record['q1r'],record['objective']]).all(),
                      'Nonfinite SAA solution')
    return record['q10'], np.asarray(record['q1r'])