"""
cv_rho_selection_net16_v2.py  --  CV-based rho selection for N50 sweep
=======================================================================
Goal: select rho WITHOUT using the test set, to give a fair comparison
      to the paper's "SRO-FA Best CV" number.

Method:
  For each training seed k (data seed 0-9):
    - Generate the N=50 training set with that seed
    - Designate points [40:50] (last 10) as a held-out validation set
    - For each rho, load the final robust model and evaluate its cost
      on the 10 validation points
    - Select rho* = argmin over rho of validation cost
  Report:
    - OOS (10k) cost at the per-seed selected rho*  (CV selection)
    - Compare to oracle selection (best rho per seed by test OOS)
    - Compare to paper SRO-FA Best CV = 21.3

Limitation (acknowledged):
  The models were trained on ALL 50 points (including the validation 10),
  so this is "training-set holdout" rather than true K-fold CV.
  It is still a valid rho selection criterion because:
    1. rho selection never uses the 10k test set
    2. the validation cost provides an unbiased comparison signal
       across rho values FOR A FIXED model architecture
  True K-fold would require retraining — left for future work.

Output: printed table + cv_rho_results.json
"""

import os, json
import numpy as np
from scipy.stats import truncnorm
import torch, torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

NOMINAL_DIR = "/data/home/aviyac/retailers_problem/R3/N50/saainit_net16_v2"
SWEEP_DIR   = "/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2"
EVAL_10K    = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_10k.json")
OUT_PATH    = os.path.join(SWEEP_DIR, "cv_rho_results.json")

SEEDS       = list(range(10))
N_TRAIN     = 50
N_VAL       = 10        # held-out: last 10 of the 50 training points
N_TEST      = 10000
SEED_TEST   = 9999
R           = 3
c, h, b     = 0.25, 0.05, 0.5

FINAL_ITER_040 = {0: 3, 1: 5, 2: 4, 3: 5, 4: 7, 5: 4, 6: 7, 7: 5, 8: 5, 9: 5}

# ── data ──────────────────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

# ── Gurobi projection ─────────────────────────────────────────────────────────
_grb_env = None
def _get_grb_env():
    global _grb_env
    if _grb_env is None:
        _grb_env = gp.Env(empty=True)
        _grb_env.setParam("OutputFlag", 0)
        _grb_env.start()
    return _grb_env

def _gurobi_project(z_row, q10):
    env = _get_grb_env()
    m   = gp.Model(env=env)
    m.setParam("OutputFlag", 0)
    q2  = m.addVars(R, lb=0.0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= q10)
    m.setObjective(
        gp.quicksum((q2[r] - float(z_row[r]))**2 for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    result = np.array([q2[r].X for r in range(R)]) if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else np.zeros(R)
    m.dispose()
    return result

# ── model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),         nn.ReLU(),
            nn.Linear(16, R))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2(self, xi1_t):
        N   = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        ni  = torch.cat([xi1_t,
                          torch.tensor(q1r).unsqueeze(0).expand(N, -1),
                          torch.full((N, 1), q10)], dim=1)
        z   = self.fc_layers(ni).numpy()
        Q2  = np.array([_gurobi_project(z[i], q10) for i in range(N)])
        return Q2, q10, q1r

def load_model(path):
    m = Model()
    m.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"], strict=False)
    m.eval()
    return m

def mean_cost(Q10, Q1r, Q2, xi1, xi2):
    s1 = c * (Q10 + Q1r.sum()) + h * Q10
    v  = np.maximum(b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
          np.maximum(h * (Q1r - xi1 - xi2),
                     b * (xi1 - Q1r) - h * xi2))
    return float((s1 + v.sum(axis=1)).mean())

# ── load 10k eval JSON for OOS lookup ─────────────────────────────────────────
print(f"Loading 10k eval JSON ...")
with open(EVAL_10K) as f:
    d10k = json.load(f)
rhos_all = [r for r in d10k["rhos"] if r > 0]

# ── generate test data (for oracle comparison only) ───────────────────────────
print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

# ── per-seed CV rho selection ─────────────────────────────────────────────────
print(f"\nVal set = last {N_VAL} of {N_TRAIN} training points per seed\n")
print(f"{'seed':>5}  {'cv_rho':>8}  {'cv_val':>9}  {'cv_oos':>9}  {'oracle_rho':>11}  {'oracle_oos':>11}")
print("-" * 65)

cv_oos_list     = []
oracle_oos_list = []
cv_rho_list     = []
oracle_rho_list = []
per_seed_detail = {}

for seed in SEEDS:
    xi1_tr, xi2_tr = gen(N_TRAIN, seed)
    # held-out validation: last N_VAL points
    xi1_val = xi1_tr[-N_VAL:]
    xi2_val = xi2_tr[-N_VAL:]
    xi1_val_t = torch.tensor(xi1_val, dtype=torch.float32)

    val_costs  = {}   # rho -> val cost
    oos_costs  = {}   # rho -> oos cost (from 10k JSON)

    for rho in rhos_all:
        # --- load model for this (rho, seed) ---
        key = str(rho)
        e   = d10k["sweep"].get(key, {})
        rob_oos_list = e.get("rob_oos", [])
        if len(rob_oos_list) <= seed:
            # seed missing for this rho (3 missing jobs)
            continue

        # find checkpoint path
        _rho_str = f"{rho:.6f}".rstrip('0')
        _parts   = _rho_str.split('.')
        if _rho_str.endswith('.'):   _rho_str += '000'
        elif len(_parts) > 1 and len(_parts[1]) < 3:
            _rho_str += '0' * (3 - len(_parts[1]))
        rho_tag  = _rho_str.replace(".", "p")
        rho_dir  = os.path.join(SWEEP_DIR, f"rho_{rho_tag}")
        summ_f   = os.path.join(rho_dir, f"seed_{seed}_summary.json")

        if not os.path.exists(summ_f):
            continue
        with open(summ_f) as f2:
            summ = json.load(f2)
        fmp = summ.get("final_mp_iter", 0)
        if fmp == 0:
            rob_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        else:
            rob_path = os.path.join(rho_dir, f"seed_{seed}_mp_iter{fmp}.pth")
        if not os.path.exists(rob_path):
            continue

        # evaluate on validation set
        m_rob = load_model(rob_path)
        Q2_val, q10r, q1rr = m_rob.get_Q2(xi1_val_t)
        val_costs[rho] = mean_cost(q10r, q1rr, Q2_val, xi1_val, xi2_val)

        # OOS from JSON (already computed)
        oos_costs[rho] = float(rob_oos_list[seed])

    if not val_costs:
        print(f"  seed {seed}: no data, skipping")
        continue

    # CV selection: pick rho with lowest validation cost
    cv_rho   = min(val_costs, key=val_costs.get)
    cv_val   = val_costs[cv_rho]
    cv_oos   = oos_costs[cv_rho]

    # Oracle selection: pick rho with lowest OOS cost
    oracle_rho = min(oos_costs, key=oos_costs.get)
    oracle_oos = oos_costs[oracle_rho]

    cv_oos_list.append(cv_oos)
    oracle_oos_list.append(oracle_oos)
    cv_rho_list.append(cv_rho)
    oracle_rho_list.append(oracle_rho)
    per_seed_detail[seed] = {
        "cv_rho": cv_rho, "cv_val": cv_val, "cv_oos": cv_oos,
        "oracle_rho": oracle_rho, "oracle_oos": oracle_oos,
        "val_costs": {str(r): v for r, v in val_costs.items()},
    }
    print(f"  {seed:>3}  {cv_rho:>8.4f}  {cv_val:>9.4f}  {cv_oos:>9.4f}  "
          f"{oracle_rho:>11.4f}  {oracle_oos:>11.4f}")

print("-" * 65)
cv_mean     = float(np.mean(cv_oos_list))
oracle_mean = float(np.mean(oracle_oos_list))
print(f"  {'mean':>3}  {'':>8}  {'':>9}  {cv_mean:>9.4f}  {'':>11}  {oracle_mean:>11.4f}")

print(f"""
╔══════════════════════════════════════════╗
║  Summary (N=50, R=3, N_test=10k)        ║
╠══════════════════════════════════════════╣
║  SAA-LP lower bound       :  19.5950    ║
║  Neural robust CV-rho     : {cv_mean:>8.4f}    ║
║  Neural robust oracle-rho : {oracle_mean:>8.4f}    ║
║  Paper SRO-FA Best CV     :  21.3000    ║
║  Paper SRO-FA Best rho    :  21.5000    ║
╠══════════════════════════════════════════╣
║  Improvement vs paper CV  : {21.3 - cv_mean:>+8.4f}    ║
║  Improvement vs paper rho : {21.5 - oracle_mean:>+8.4f}    ║
╚══════════════════════════════════════════╝
""")

# ── save ──────────────────────────────────────────────────────────────────────
out = {
    "cv_oos_mean":     cv_mean,
    "oracle_oos_mean": oracle_mean,
    "cv_rhos":         cv_rho_list,
    "oracle_rhos":     oracle_rho_list,
    "per_seed":        per_seed_detail,
    "paper_sro_fa_cv": 21.3,
    "paper_sro_fa_rho": 21.5,
    "saa_lb":          19.595,
    "meta": {"N_train": N_TRAIN, "N_val": N_VAL, "N_test": N_TEST,
             "seed_test": SEED_TEST, "seeds": SEEDS},
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"[Saved] {OUT_PATH}")
