"""
Moreau JAX integration with unified CPU/CUDA support.

Provides differentiable convex optimization for JAX with automatic
device selection between CPU and CUDA backends.

Example (full signature):
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from moreau.jax import Solver
    >>> import moreau
    >>>
    >>> cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    >>> solver = Solver(n=2, m=3, P_row_offsets=..., P_col_indices=...,
    ...                 A_row_offsets=..., A_col_indices=..., cones=cones)
    >>>
    >>> solution = solver.solve(P_values, A_values, q, b)
    >>> print(solution.x, solver.info.status)

Example (two-step API):
    >>> solver = Solver(n=2, m=3, ...)
    >>> solver.setup(P_values, A_values)  # Set matrices once
    >>> solution = solver.solve(q, b)  # Solve with 2 args
    >>>
    >>> # User can vmap/jit as needed
    >>> batched = jax.vmap(solver.solve)
    >>> solutions = batched(q_batch, b_batch)

Example (differentiation):
    >>> def loss_fn(q):
    ...     return jnp.sum(solver.solve(q, b).x)
    >>> grad_q = jax.grad(loss_fn)(q)
"""

from typing import Optional, Sequence
import time
import warnings

from moreau._backend import (
    default_device,
    get_device_component,
    jax_available,
)
from moreau._types import (
    Cones,
    Settings,
    IPMSettings,
    WarmStart,
    BatchedWarmStart,
)
from ._types import JaxSolution, JaxSolveInfo
from moreau._backend import (
    benchmark_and_select_winner as _benchmark_and_select_winner,
    needs_auto_tune as _needs_auto_tune_shared,
)

                                                                        
                                                                    
                                                                          
from moreau._backend import _device_registry as _dr

if "cpu" in _dr and _dr["cpu"].get("jax_solver") is None:
    from ._cpu_impl import JaxSolverCpu

    _dr["cpu"]["jax_solver"] = JaxSolverCpu
del _dr


class Solver:
    """JAX solver with automatic device selection and gradient support.

    Provides a solve method compatible with jax.vmap, jax.grad, and jax.jit.
    Solve metadata (timing, status) is available via solver.info after solve().

    Supports two usage patterns:
    1. Full signature: solve(P_values, A_values, q, b)
    2. Two-step: setup(P_values, A_values) then solve(q, b)

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (array-like).
            P must be full symmetric (both upper and lower triangles).
        P_col_indices: CSR column indices for P matrix (array-like)
        A_row_offsets: CSR row pointers for A matrix (array-like)
        A_col_indices: CSR column indices for A matrix (array-like)
        cones: Cone specification (moreau.Cones object)
        settings: Optional solver settings (moreau.Settings object).
                  If device='auto' (default), uses CUDA if available, else CPU.
                  Note: jax.Solver always enables gradients (enable_grad=True)
                  internally regardless of this setting.
        jit: If True (default), JIT-compile the solve method for better performance.
        b_sparsity_pattern: Optional boolean mask of length m indicating which
                  entries of b are structurally nonzero. Enables more aggressive
                  chordal decomposition for PSD cones.

    Example (full signature):
        >>> solver = Solver(n=2, m=3, P_row_offsets=..., P_col_indices=...,
        ...                 A_row_offsets=..., A_col_indices=..., cones=cones)
        >>> solution = solver.solve(P_values, A_values, q, b)

    Example (two-step):
        >>> solver = Solver(n=2, m=3, ...)
        >>> solver.setup(P_values, A_values)  # Set matrices once
        >>> solution = solver.solve(q, b)  # Solve with 2 args
        >>>
        >>> # User can vmap/jit as needed
        >>> batched = jax.vmap(solver.solve)
        >>> solutions = batched(q_batch, b_batch)
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones: Cones,
        settings: Optional[Settings] = None,
        jit: bool = True,
        b_sparsity_pattern: Optional[Sequence[bool]] = None,
    ):
                                            
        from moreau import _warn_cvxpy_soc_if_needed, _require_dir_cones_compatible

        _warn_cvxpy_soc_if_needed(cones)
        _require_dir_cones_compatible(cones, settings)

                                                                         
                                                                       
                                                                  
                                                                       
                                                                          
                     

        if settings is None:
            settings = Settings()

                                                                           
        device = settings.device
        if device == "auto":
            device = default_device()

                                                                
        from moreau import _resolve_solver_type

        nnz_P = len(P_col_indices)
        settings, device = _resolve_solver_type(settings, n, m, cones, device, nnz_P=nnz_P)

                                                                              
                                                                                
                                                                              
                                                                             
                                                                            
                                                                               
                                                            
        JaxSolverClass = get_device_component(device, "jax_solver")
        if JaxSolverClass is None and device != "cpu":
            cpu_class = get_device_component("cpu", "jax_solver")
            if cpu_class is not None:
                warnings.warn(
                    f"JAX extension for device='{device}' not available; "
                    f"falling back to CPU. Pass device='cpu' explicitly to silence this warning.",
                    stacklevel=2,
                )
                device = "cpu"
                                                                           
                                                
                if settings.ipm_settings is not None:
                    method = (settings.ipm_settings.direct_solve_method or "").lower()
                    if method in ("cudss",):
                        settings = settings.model_copy(deep=True)
                        settings.ipm_settings.direct_solve_method = "auto"
                JaxSolverClass = cpu_class

        if JaxSolverClass is None:
            raise ImportError(
                f"JAX extension for device '{device}' not available. "
                f"Ensure JAX is installed and the appropriate moreau backend is built with JAX support. "
                f"Available devices with JAX support: {[d for d in ['cpu', 'cuda'] if jax_available(d)]}"
            )

        self._device = device
        self._n = n
        self._m = m
        self._settings = settings
        self._cones = cones
        self._P_row_offsets = P_row_offsets
        self._P_col_indices = P_col_indices
        self._A_row_offsets = A_row_offsets
        self._A_col_indices = A_col_indices
        self._jit = jit
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

                                 
        construction_start = time.perf_counter()

                                       
        self._impl = JaxSolverClass(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )

        self._construction_time = time.perf_counter() - construction_start

                                              
        self._solve_raw = self._impl.solve

                                     
        if jit:
            import jax

            self._solve_raw = jax.jit(self._solve_raw)

                                                                  
                                                                             
                                                                          
                                                                                  
                                                         
        self._original_device = settings.device
        ipm = settings.ipm_settings
        self._original_method = ipm.direct_solve_method if ipm else "auto"
        if self._original_method == "auto" and settings.device != "auto":
            from moreau._backend import _rank_method
            import numpy as _np

            nnz_A = len(_np.asarray(A_col_indices))
            ranked = _rank_method(device, n, m, nnz_A, settings.batch_size)
            initial_method = ranked[0] if ranked else "auto"
            if ipm:
                ipm.direct_solve_method = initial_method

                         
        self._auto_tuned = False
        self._tune_result = None

                                                   
        self._info = None

                                          
        self._P_values = None
        self._A_values = None

    def setup(self, P_values, A_values):
        """Set P and A matrix values for subsequent solve() calls.

        After calling setup(), solve() can be called with just (q, b).
        Gradients w.r.t. P/A are still computed when using the 2-arg solve().

        Args:
            P_values: P matrix values, shape (nnzP,)
            A_values: A matrix values, shape (nnzA,)
        """
        import jax.numpy as jnp

        self._P_values = jnp.asarray(P_values, dtype=jnp.float64)
        self._A_values = jnp.asarray(A_values, dtype=jnp.float64)

    def _needs_auto_tune(self) -> bool:
        return _needs_auto_tune_shared(
            self._settings,
            self._auto_tuned,
            self._original_device,
            self._original_method,
        )

    def _auto_tune(self, P_values, A_values, q, b):
        """Benchmark solver configs and lock in the fastest.

        Delegates the benchmark+winner-selection to the shared
        `_benchmark_and_select_winner` helper, then rebuilds this
        Solver's JAX impl with the winning device/method.
        """
        import warnings

        warnings.warn(
            "First solve is benchmarking solver configurations "
            "(auto_tune=True). Subsequent solves will be faster. "
            "Set auto_tune=False (default) to use heuristic selection "
            "without benchmarking.",
            stacklevel=3,
        )
        import numpy as np

                                                                       
                                                     
        stored_P_values = self._P_values
        stored_A_values = self._A_values

        q_np = np.asarray(q, dtype=np.float64)
        b_np = np.asarray(b, dtype=np.float64)
        P_np = np.asarray(P_values, dtype=np.float64)
        A_np = np.asarray(A_values, dtype=np.float64)
        P_ro = np.asarray(self._P_row_offsets, dtype=np.int64)
        P_ci = np.asarray(self._P_col_indices, dtype=np.int64)
        A_ro = np.asarray(self._A_row_offsets, dtype=np.int64)
        A_ci = np.asarray(self._A_col_indices, dtype=np.int64)

                                      
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        if b_np.ndim == 1:
            b_np = b_np.reshape(1, -1)

        ipm = self._settings.ipm_settings
        current_method = ipm.direct_solve_method if ipm else "auto"

        out = _benchmark_and_select_winner(
            settings=self._settings,
            cones=self._cones,
            b_sparsity_pattern=self._b_sparsity_pattern,
            n=self._n,
            m=self._m,
            P_ro=P_ro,
            P_ci=P_ci,
            A_ro=A_ro,
            A_ci=A_ci,
            P_np=P_np,
            A_np=A_np,
            q_np=q_np,
            b_np=b_np,
            original_device=self._original_device,
            original_method=self._original_method,
            current_device=self._device,
            component_key="jax_solver",
            component_cpu_fallback=True,
        )
        if out is None:
            self._auto_tuned = True
            return

        best_device, best_method, time_limit, _results, tune_result = out
        self._tune_result = tune_result

        if (best_device, best_method) != (self._device, current_method):
            winning_settings = self._settings.model_copy(deep=True)
            winning_settings.device = best_device
            if winning_settings.ipm_settings is None:
                winning_settings.ipm_settings = IPMSettings()
            winning_settings.ipm_settings.direct_solve_method = best_method
            winning_settings.time_limit = time_limit
            self._settings = winning_settings
            self._device = best_device

            JaxSolverClass = get_device_component(best_device, "jax_solver")
            if JaxSolverClass is None and best_device != "cpu":
                JaxSolverClass = get_device_component("cpu", "jax_solver")

            self._impl = JaxSolverClass(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._cones,
                self._settings,
                b_sparsity_pattern=self._b_sparsity_pattern,
            )
            self._solve_raw = self._impl.solve
            if self._jit:
                import jax

                self._solve_raw = jax.jit(self._solve_raw)

                                                                 
            if stored_P_values is not None:
                self._P_values = stored_P_values
                self._A_values = stored_A_values

        self._auto_tuned = True

    def solve(self, *args, warm_start=None) -> JaxSolution:
        """Solve the optimization problem.

        Two signatures supported:
        - solve(q, b): Uses P/A from setup()
        - solve(P_values, A_values, q, b): Full signature

        Args (2-arg form):
            q: Linear cost, shape (n,) or (batch, n)
            b: Constraint RHS, shape (m,) or (batch, m)

        Args (4-arg form):
            P_values: P matrix values, shape (nnzP,) or (batch, nnzP)
            A_values: A matrix values, shape (nnzA,) or (batch, nnzA)
            q, b: Same as 2-arg form.

        Keyword Args:
            warm_start: Optional WarmStart or BatchedWarmStart from a previous
                solve (e.g. ``solution.to_warm_start()``). Gradients do NOT
                flow through warm start values.

        Returns:
            JaxSolution: NamedTuple with x, z, s

        Note:
            Solver metadata (status, timing) is available via solver.info
            after calling solve().
        """
        if len(args) == 2:
            if self._P_values is None:
                raise RuntimeError("setup() must be called before solve(q, b)")
            q, b = args
            P_values, A_values = self._P_values, self._A_values
        elif len(args) == 4:
            P_values, A_values, q, b = args
        else:
            raise TypeError(f"solve() takes 2 or 4 arguments, got {len(args)}")

                                                                  
        if self._needs_auto_tune():
            self._auto_tune(P_values, A_values, q, b)

        if warm_start is not None:
            if not isinstance(warm_start, (WarmStart, BatchedWarmStart)):
                raise TypeError(
                    f"warm_start must be a WarmStart or BatchedWarmStart, "
                    f"got {type(warm_start).__name__}"
                )

                                                                           
                                                                            
                                                  
            import jax.numpy as jnp

            solve_warm_fn = self._impl.solve_warm
            if solve_warm_fn is not None:
                warm_x = jnp.asarray(warm_start.x, dtype=jnp.float64)
                warm_z = jnp.asarray(warm_start.z, dtype=jnp.float64)
                warm_s = jnp.asarray(warm_start.s, dtype=jnp.float64)
                                                                        
                                                                            
                total_xn = getattr(self._impl, "_total_xn", 0)
                if warm_start.z_x is not None:
                    warm_z_x = jnp.asarray(warm_start.z_x, dtype=jnp.float64)
                elif warm_x.ndim == 1:
                    warm_z_x = jnp.zeros((total_xn,), dtype=jnp.float64)
                else:
                    warm_z_x = jnp.zeros((warm_x.shape[0], total_xn), dtype=jnp.float64)
                solution, info = solve_warm_fn(
                    P_values, A_values, q, b, warm_x, warm_z, warm_s, warm_z_x
                )
            else:
                                                                     
                import numpy as np

                self._impl._pending_warm_start = {
                    "warm_x": np.asarray(warm_start.x, dtype=np.float64),
                    "warm_z": np.asarray(warm_start.z, dtype=np.float64),
                    "warm_s": np.asarray(warm_start.s, dtype=np.float64),
                    "warm_z_x": (
                        np.asarray(warm_start.z_x, dtype=np.float64)
                        if warm_start.z_x is not None
                        else None
                    ),
                }
                solution, info = self._solve_raw(P_values, A_values, q, b)
                self._impl._pending_warm_start = None
        else:
            solution, info = self._solve_raw(P_values, A_values, q, b)

                                                                    
                                                                 
                                                                    
        if warm_start is not None:
            import jax.numpy as jnp
            from moreau._types import _normalize_status, _should_retry_cold, _warn_warm_retry

            status_enum = _normalize_status(int(jnp.asarray(info.status)))
            ipm = self._settings.ipm_settings
            if _should_retry_cold(status_enum, ipm):
                _warn_warm_retry([0], status_enum)
                solution, info = self._solve_raw(P_values, A_values, q, b)

                                                            
                                                                            
        self._info = info

        return solution

    @property
    def info(self) -> Optional[JaxSolveInfo]:
        """Metadata from the last solve() call.

        Returns None if solve() has not been called yet.

        Contains:
            - status: Solver status (as int, SolverStatus enum value)
            - obj_val: Objective value at solution
            - iterations: Number of IPM iterations
            - solve_time: Time in IPM iterations (seconds)
            - setup_time: Time setting matrix values (seconds)
            - construction_time: Time constructing solver (seconds)

        Note: For vmap calls, this returns info from the last single call,
        not the batched result. Access batched info via the returned tuple.
        """
        return self._info

    @property
    def device(self) -> str:
        """Return the active device name ('cpu' or 'cuda')."""
        return self._device

    @property
    def n(self) -> int:
        """Number of primal variables."""
        return self._n

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self._m

    @property
    def construction_time(self) -> float:
        """Time spent constructing solver structure (seconds)."""
        return self._construction_time

    @property
    def tune_result(self):
        """Result from auto-tuning on the first solve() call.

        Returns None if auto-tune has not run yet (e.g. device and method
        were set explicitly, or solve() has not been called).
        """
        return self._tune_result


__all__ = ["Solver"]
