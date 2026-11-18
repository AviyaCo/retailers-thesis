"""Native differentiable retailer projection, no custom derivative or fallback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                      "hydro_rcr_rescaled/.moreau_diagnostic_deps"))
import moreau
from moreau.torch import Solver
import numpy as np
from scipy import sparse
import torch


class NativeProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        p = sparse.eye(3, format="csr")
        a = sparse.csr_matrix(np.vstack([-np.eye(3), np.ones((1, 3))]))
        self.p = torch.tensor(p.data, dtype=torch.float64)
        self.a = torch.tensor(a.data, dtype=torch.float64)
        settings = moreau.Settings(solver="active_set", device="cpu", enable_grad=True,
            auto_tune=False, active_set_settings=moreau.ActiveSetSettings(diff_method="exact"))
        self.settings = settings.model_dump(mode="json")
        # Preserve the unlimited setting explicitly in strict JSON metadata.
        # The actual solver settings object is unchanged.
        if self.settings["time_limit"] == float("inf"):
            self.settings["time_limit"] = "Infinity (unlimited)"
        self.solver = Solver(n=3, m=4,
            P_row_offsets=torch.tensor(p.indptr, dtype=torch.int64),
            P_col_indices=torch.tensor(p.indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(a.indptr, dtype=torch.int64),
            A_col_indices=torch.tensor(a.indices, dtype=torch.int64),
            cones=moreau.Cones(num_nonneg_cones=4), settings=settings)
        self.calls = []
        original = self.solver._impl.solve

        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            record = {"status_codes": result["status"].detach().reshape(-1).tolist(),
                      "iterations": result["iterations"].detach().reshape(-1).tolist()}
            self.calls.append(record)
            if not all(s == 1 for s in record["status_codes"]):
                raise RuntimeError(f"Moreau solve failed: {record}")
            if not all(bool(torch.isfinite(result[k]).all()) for k in ("x", "z", "s", "obj_val")):
                raise RuntimeError("nonfinite native solution")
            return result

        self.solver._impl.solve = checked

    def forward(self, z, capacity, solver_args):
        if solver_args != {"solve_method": "SCS", "eps": 1e-4, "max_iters": 200000}:
            raise ValueError("unexpected source solver configuration")
        z, capacity = z.to(torch.float64), capacity.to(torch.float64)
        b = torch.cat([torch.zeros(z.shape[0], 3, dtype=z.dtype), capacity + 1e-6], dim=1)
        return (self.solver.solve(self.p, self.a, -z, b).x,)


ARGS = {"solve_method": "SCS", "eps": 1e-4, "max_iters": 200000}


def derivative_fixture():
    """Separate active/inactive capacity fixtures; central differences, not data."""
    layer = NativeProjection()
    z = torch.tensor([[3., 2., -1.], [.3, -.2, .4]], dtype=torch.float64, requires_grad=True)
    cap = torch.tensor([[2.], [4.]], dtype=torch.float64, requires_grad=True)
    probe = torch.tensor([[.2, .8, .4], [.7, .1, .3]], dtype=torch.float64)
    value = (layer(z, cap, ARGS)[0] * probe).sum()
    grads = torch.autograd.grad(value, (z, cap))
    errors = []
    for index, tensor in enumerate((z, cap)):
        numeric = torch.empty_like(tensor)
        for j in range(tensor.numel()):
            plus, minus = [z.detach().clone(), cap.detach().clone()], [z.detach().clone(), cap.detach().clone()]
            plus[index].view(-1)[j] += 1e-5
            minus[index].view(-1)[j] -= 1e-5
            with torch.no_grad():
                numeric.view(-1)[j] = ((layer(*plus, ARGS)[0] - layer(*minus, ARGS)[0]) * probe).sum() / 2e-5
        errors.append(float((numeric - grads[index]).abs().max()))
    if max(errors) > 1e-6:
        raise RuntimeError(f"fixture derivative mismatch: {errors}")
    return {"z_gradient_max_error": errors[0], "capacity_gradient_max_error": errors[1],
            "finite_difference_step": 1e-5, "fixture_only_not_experiment_data": True}