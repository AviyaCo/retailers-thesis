"""Small fixtures only: never call main, load checkpoints, or generate data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import benchmark_ablation as ab

sys.path.insert(0, str(ab.IMPLEMENTATION))
from baseline import load_baseline
from native_projection import NativeProjection, ARGS as OLD_ARGS
import run_trial


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.baseline = load_baseline()

    def inputs(self):
        z = torch.tensor([[3., 2., -1.], [.3, -.2, .4]], dtype=torch.float64)
        cap = torch.tensor([[2.], [4.]], dtype=torch.float64)
        oracle = np.array([[1.5000005, .5000005, 0.], [.3, 0., .4]])
        upstream = torch.randn(z.shape, generator=torch.Generator().manual_seed(ab.UPSTREAM_SEED),
                               dtype=torch.float64)
        return z, cap, oracle, upstream

    def capture(self):
        z, cap, oracle, upstream = self.inputs()
        model = self.baseline.Model()
        captured, record = [], {}
        q = ab.run_pass("A", 0, model.proj_layer, z, cap, upstream, oracle, record, captured)
        return model, captured, record, q

    def test_actual_canonical_capture_and_compiler_mapping(self):
        model, captured, record, q_a = self.capture()
        mapping = ab.recovery_mapping(model.proj_layer)
        native = ab.NativeSCS(captured, {}, mapping)
        output = native.solve({})
        np.testing.assert_allclose(output, q_a, atol=1e-10, rtol=0)
        # Cross-check against the library's recovery, not hand-coded offsets.
        module = __import__("cvxpylayers.torch.cvxpylayer", fromlist=["_recover_results"])
        for i, call in enumerate(record["scs_calls"]):
            result = module._recover_results(
                torch.tensor([call["vectors"]["x"]], dtype=torch.float64),
                torch.tensor([call["vectors"]["y"]], dtype=torch.float64), model.proj_layer.ctx, (1,))[0]
            np.testing.assert_allclose(ab.recover_primal(call["vectors"]["x"], mapping),
                                       result[0].numpy(), rtol=0, atol=0)
            self.assertEqual(captured[i]["sample"], i)
            self.assertIsNone(captured[i]["data"]["P"])
            self.assertTrue(captured[i]["cone"]["q"])
            self.assertEqual(captured[i]["settings"], ab.SCS_SETTINGS)
            self.assertFalse(call["solution_warm_start_supplied"])
            self.assertIn("setup", call)
            self.assertIn("solve", call)
            self.assertIn("canonical_capture", call)
        labels = {r["phase"] for r in record["extraction_canonicalization_events"]}
        self.assertEqual(labels, {"build_diffcp_matrices", "flatten_parameters",
                                  "parameter_map_matmul", "recover_variables"})
        # Raw canonical serialization preserves exact sparse representation.
        encoded = ab.packed_data(captured[0]["data"])
        matrix = captured[0]["data"]["A"]
        np.testing.assert_array_equal(encoded["A"]["data"], matrix.data)
        np.testing.assert_array_equal(encoded["A"]["indices"], matrix.indices)
        with patch.object(model.proj_layer.ctx.var_recover[0], "source", "dual"):
            with self.assertRaisesRegex(RuntimeError, "unsupported compiler recovery"):
                ab.recovery_mapping(model.proj_layer)

    def test_p2_direct_equivalence_backward_and_repeated_cold_solves(self):
        model, captured, _, q_a = self.capture()
        z, cap, oracle, upstream = self.inputs()
        problems = ab.direct_data(z.numpy(), cap.numpy())
        np.testing.assert_array_equal(problems[0]["data"]["P"].toarray(), 2 * np.eye(3))
        np.testing.assert_array_equal(problems[0]["data"]["c"], -2 * z[0].numpy())
        self.assertEqual(problems[0]["data"]["b"][-1], 2. + ab.BUFFER)
        self.assertEqual(problems[0]["cone"], {"l": 4})
        c = ab.NativeSCS(problems, {})
        b = ab.NativeSCS(captured, {}, ab.recovery_mapping(model.proj_layer))
        # Independent SCS instances follow the same cached setup lifecycle.
        # warm_start=False does not reset adaptive scaling/acceleration state.
        controls = {"B": [ab.scs.SCS(p["data"], p["cone"], **p["settings"]) for p in captured],
                "C": [ab.scs.SCS(p["data"], p["cone"], **p["settings"]) for p in problems]}
        native_setup = {}
        d = ab.NativeMoreau(native_setup)
        old = NativeProjection()
        p1_before = old.p.clone()
        for repeat in range(2):
            records = {}
            for arm, backend in (("A", model.proj_layer), ("B", b), ("C", c), ("D", d)):
                record = records[arm] = {}
                q = ab.run_pass(arm, repeat, backend, z, cap, upstream, oracle, record)
                # SCS relative stopping is NOT an absolute physical error bound.
                # Check actual recorded outputs/residuals, not an invented eps bound.
                self.assertEqual(record["max_error_gurobi"], float(np.abs(q - oracle).max()))
                if arm == "D":
                    np.testing.assert_allclose(q, oracle, atol=1e-6, rtol=0)
                elif arm == "A" or (arm == "B" and repeat == 0):
                    np.testing.assert_allclose(q, q_a, atol=1e-10, rtol=0)
                if arm in controls:
                    raw = [s.solve(warm_start=False)["x"] for s in controls[arm]]
                    expected = np.stack([ab.recover_primal(x, b.mapping) if arm == "B" else x for x in raw])
                    np.testing.assert_array_equal(q, expected)
                if arm in ("A", "D"):
                    self.assertTrue(record["gradients_finite"])
                else:
                    self.assertNotIn("backward", record)
                    self.assertTrue(all(not call["warm_start"] for call in record["scs_calls"]))
            for name in ("target", "capacity"):
                np.testing.assert_allclose(records["A"]["gradients"][name],
                                           records["D"]["gradients"][name], atol=1e-4, rtol=0)
            events = records["D"]["native_events"]
            self.assertEqual(sum(e["phase"] == "native_matrix_setup" for e in events), 1 if repeat == 0 else 0)
            z_leaf, cap_leaf = z.clone().requires_grad_(), cap.clone().requires_grad_()
            old_q = old(z_leaf, cap_leaf, OLD_ARGS)[0]
            old_q.backward(upstream)
            np.testing.assert_allclose(records["D"]["q"], old_q.detach().numpy(), atol=1e-6, rtol=0)
            np.testing.assert_allclose(records["D"]["gradients"]["target"], z_leaf.grad.numpy(), atol=1e-6, rtol=0)
            np.testing.assert_allclose(records["D"]["gradients"]["capacity"], cap_leaf.grad.numpy(), atol=1e-6, rtol=0)
        torch.testing.assert_close(old.p, p1_before, rtol=0, atol=0)
        strict = json.dumps(ab.json_numbers(native_setup), allow_nan=False)
        self.assertIn("+Infinity", strict)
        # The scalar objectives differ by z'z only, preserving original scaling.
        x = oracle[0]
        direct = .5 * x @ problems[0]["data"]["P"] @ x + problems[0]["data"]["c"] @ x
        self.assertAlmostEqual(float(direct + z[0].square().sum()), float(np.square(x-z[0].numpy()).sum()))

    def test_float32_network_preparation_and_existing_gurobi_reference(self):
        torch.manual_seed(0)
        model = self.baseline.Model()
        x1 = torch.tensor([[2.25, 3.75, 1.125], [6., 2., 8.]], dtype=torch.float32)
        before = ab.state_hash(ab.parameter_state(model))
        observed = {}

        class CaptureOnly(torch.nn.Module):
            def forward(self, z, cap, solver_args):
                observed.update(z=z.detach().clone(), cap=cap.detach().clone())
                return (torch.zeros_like(z),)

        with patch.object(model, "proj_layer", CaptureOnly()):
            with torch.no_grad():
                model.per_sample_cost(x1, torch.zeros_like(x1))
        z, cap = ab.prepare_inputs(model, x1)
        torch.testing.assert_close(z, observed["z"].double(), rtol=0, atol=0)
        torch.testing.assert_close(cap, observed["cap"].double(), rtol=0, atol=0)
        self.assertFalse(z.requires_grad)
        self.assertFalse(cap.requires_grad)
        reference_record = {}
        oracle = run_trial.reference_projection(model, x1, reference_record).numpy()
        upstream = torch.randn(z.shape, generator=torch.Generator().manual_seed(ab.UPSTREAM_SEED), dtype=torch.float64)
        record = {}
        q = ab.run_pass("D", 0, ab.NativeMoreau({}), z, cap, upstream, oracle, record)
        np.testing.assert_allclose(q, oracle, atol=1e-6, rtol=0)
        self.assertEqual(before, ab.state_hash(ab.parameter_state(model)))
        self.assertEqual(reference_record["status"], "completed")

    def test_legacy_retry_capture_tracks_sample_not_attempt(self):
        import diffcp.cone_program as cp
        z, cap, _, _ = self.inputs()
        problem = ab.direct_data(z.numpy(), cap.numpy())[0]
        accurate = {"x": np.zeros(3), "y": np.zeros(4), "s": np.zeros(4),
                    "info": {"status": "solved", "status_val": 1, "iter": 2}}
        inaccurate = copy.deepcopy(accurate)
        inaccurate["info"].update(status="solved (inaccurate - reached max_iters)", status_val=2)
        captured, record = [], {}
        # Simulated solver responses test the pre-existing retry branch; no real
        # solver settings or fixture tolerances are changed to force a failure.
        with patch.object(ab.scs, "solve", side_effect=[inaccurate, accurate, accurate]):
            with ab.observe_a(record, captured):
                for _ in range(2):
                    cp.solve_internal(problem["data"]["A"], problem["data"]["b"],
                                      problem["data"]["c"], problem["cone"], solve_method="SCS", **ab.SCS_SETTINGS)
        self.assertEqual([(c["sample"], c["attempt"]) for c in record["scs_calls"]], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual([c["sample"] for c in captured], [0, 1])
        self.assertEqual(record["scs_calls"][1]["settings"]["acceleration_lookback"], 0)
        self.assertNotIn("acceleration_lookback", captured[0]["settings"])

    def test_cached_b_retry_and_c_inaccurate_retention(self):
        model, captured, _, _ = self.capture()
        b = ab.NativeSCS(captured[:1], {}, ab.recovery_mapping(model.proj_layer))
        inaccurate = b.solvers[0].solve(warm_start=False)
        inaccurate["info"].update(status="solved (inaccurate - reached max_iters)", status_val=2)
        with patch.object(b.solvers[0], "solve", return_value=inaccurate):
            for i in range(2):
                record = {}
                b.solve(record)
                self.assertEqual(len(record["scs_calls"]), 2)
                self.assertEqual("legacy_retry_setup" in record, i == 0)
        z, cap, _, _ = self.inputs()
        c = ab.NativeSCS(ab.direct_data(z[:1].numpy(), cap[:1].numpy()), {})
        inaccurate = c.solvers[0].solve(warm_start=False)
        inaccurate["info"].update(status="solved (inaccurate - reached max_iters)", status_val=2)
        inaccurate["x"][0] = -0.25
        with patch.object(c.solvers[0], "solve", return_value=inaccurate):
            record = {}
            q = c.solve(record)
        self.assertEqual(len(record["scs_calls"]), 1)
        self.assertEqual(q[0, 0], -.25)  # no clamping or discarded inaccurate result

    def test_summary_first_excluded_atomic_json_and_exclusive_output(self):
        passes = [{"arm": "A", "repeat": i, "status": "completed", "solver_outcome": "solved",
                   "forward": {"runtime_seconds": seconds}}
                  for i, seconds in enumerate([100., 1., 2., 3., 4., 5.])]
        passes.append({"arm": "A", "repeat": 6, "status": "failed", "forward": {"runtime_seconds": 1000.}})
        passes.append({"arm": "A", "repeat": 7, "status": "completed",
                   "solver_outcome": "nonstandard_status_or_nonfinite_retained",
                   "forward": {"runtime_seconds": 500.}})
        result = ab.summary(passes)["A"]
        self.assertEqual(result["forward"]["count"], 5)
        self.assertEqual(result["forward"]["mean"], 3.)
        self.assertEqual(result["forward"]["median"], 3.)
        self.assertEqual(result["forward"]["minimum"], 1.)
        self.assertEqual(result["forward"]["maximum"], 5.)
        self.assertEqual(result["failed_or_incomplete_repeats"], [6])
        self.assertEqual(result["nonstandard_solver_repeats"], [7])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = {}
            with patch.object(ab, "persist", side_effect=AssertionError("I/O in phase")):
                with ab.timed(record):
                    pass
            ab.persist({"timing": record, "infinity": float("inf")}, output)
            loaded = json.loads((output / "ablation.json").read_text())
            self.assertEqual(loaded["infinity"], "+Infinity")
            self.assertFalse((output / "ablation.tmp").exists())
            # Existing directory stops before any data access or solve.
            with patch.object(ab, "OUTPUT", output):
                with self.assertRaises(FileExistsError):
                    ab.main()


if __name__ == "__main__":
    unittest.main()