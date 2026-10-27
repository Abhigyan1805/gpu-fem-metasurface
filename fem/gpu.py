"""GPU sparse direct solvers. Measured on a GTX 1650 (REPORT.md section 6).

``cudss``: NVIDIA cuDSS through nvmath-python (``pip install nvmath-python[cu12] cupy-cuda12x``).
    Complex-symmetric matrix type, upper-triangle view, symbolic analysis reused across
    frequencies (operand buffers updated in place, then factorize + solve).
``torch``: ``torch.sparse.spsolve`` on a CUDA CSR tensor (needs a CUDA build of PyTorch
    linked with cuDSS; the stock wheel is not); one-shot factor+solve per frequency, full matrix.
``cudss32``: cuDSS factorisation in complex64 (matrix combined in double, then rounded), no refinement.
``cudss32ir``: the same complex64 factor plus iterative refinement: residual r = b - A x with A and x in
    complex128 on the GPU, correction solved with the complex64 factor, until ||r||/||b|| < ``ir_tol``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class TorchSpsolve:
    def __init__(self, device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.a = None

    def factorize(self, upper: sp.csr_matrix) -> None:
        full = (upper + sp.triu(upper, k=1).T).tocsr()
        t = self.torch
        self.a = t.sparse_csr_tensor(
            t.as_tensor(full.indptr, dtype=t.int64, device=self.device),
            t.as_tensor(full.indices, dtype=t.int64, device=self.device),
            t.as_tensor(full.data, dtype=t.complex128, device=self.device),
            size=full.shape,
        )

    def solve(self, b: np.ndarray) -> np.ndarray:
        t = self.torch
        x = t.sparse.spsolve(self.a, t.as_tensor(b, dtype=t.complex128, device=self.device))
        return x.cpu().numpy()

    memory_mb = float("nan")
    nnz_factor = -1


class CuDSS:
    def __init__(self, device: str = "cuda", precision: str = "complex128", refine: int = 0, ir_tol: float = 1e-10):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import nvmath

        self.cp, self.csp, self.nvmath = cp, csp, nvmath
        self.dtype = cp.dtype(precision)
        self.refine, self.ir_tol = refine, ir_tol
        self.solver = None
        self.pattern = None
        self.ir_steps = []  # refinement steps taken per solve
        self.memory_mb = 0.0

    def factorize(self, upper: sp.csr_matrix) -> None:
        # The operands live in persistent GPU buffers that are updated in place: handing cuDSS new
        # buffers (reset_operands) invalidates the plan, so the symbolic analysis is done only once.
        cp, csp, nvm = self.cp, self.csp, self.nvmath
        data = cp.asarray(upper.data, dtype=cp.complex128)
        key = (upper.shape[0], upper.nnz)
        if self.solver is None or self.pattern != key:
            indices = cp.asarray(upper.indices, dtype=cp.int32)
            indptr = cp.asarray(upper.indptr, dtype=cp.int32)
            self._a = csp.csr_matrix((data.astype(self.dtype), indices, indptr), shape=upper.shape)
            self._b = cp.zeros(upper.shape[0], dtype=self.dtype)
            if self.refine:
                self._u64 = csp.csr_matrix((data.copy(), indices, indptr), shape=upper.shape)
                self._u64t = self._u64.T.tocsr()
            adv = nvm.sparse.advanced
            opts = adv.DirectSolverOptions(
                sparse_system_type=adv.DirectSolverMatrixType.SYMMETRIC,
                sparse_system_view=adv.DirectSolverMatrixViewType.UPPER,
            )
            self.solver = adv.DirectSolver(self._a, self._b, options=opts)
            self.solver.plan()
            self.pattern = key
        else:
            self._a.data[...] = data
            if self.refine:
                self._u64.data[...] = data
                self._u64t = self._u64.T.tocsr()
        if self.refine:
            self._d64 = self._u64.diagonal()
        self.solver.factorize()
        free, total = cp.cuda.runtime.memGetInfo()
        self.memory_mb = max(self.memory_mb, (total - free) / 2**20)  # device memory in use (all processes)

    def _solve(self, b):
        self._b[...] = b
        return self.solver.solve().astype(self.cp.complex128)  # astype copies

    def solve(self, b: np.ndarray) -> np.ndarray:
        cp = self.cp
        bb = cp.asarray(b, dtype=cp.complex128)
        x = self._solve(bb)
        steps = 0
        if self.refine:
            nb = float(cp.linalg.norm(bb))
            for steps in range(self.refine + 1):
                r = bb - (self._u64 @ x + self._u64t @ x - self._d64 * x)
                if float(cp.linalg.norm(r)) / nb < self.ir_tol or steps == self.refine:
                    break
                x = x + self._solve(r)
        self.ir_steps.append(steps)
        return cp.asnumpy(x)

    def release(self) -> None:
        """Free cuDSS's device memory (dropping the Python object alone does not, within a process)."""
        if self.solver is not None:
            self.solver.free()
            self.solver = self.pattern = None
        self._a = self._b = self._u64 = self._u64t = self._d64 = None
        self.cp.get_default_memory_pool().free_all_blocks()

    nnz_factor = -1


def make_gpu_solver(name: str, device: str = "cuda", **_):
    if name == "torch":
        return TorchSpsolve(device)
    if name == "cudss":
        return CuDSS(device)
    if name == "cudss32":
        return CuDSS(device, precision="complex64")
    if name == "cudss32ir":
        return CuDSS(device, precision="complex64", refine=20)
    raise ValueError(name)
